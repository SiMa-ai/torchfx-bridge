"""
Relay semantic graph simplification for quantization.
"""
import numpy as np
import tvm

import tvm.relay.transform as rt
from rich import print
from tqdm.auto import tqdm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_op,
    is_tuple_get_item,
    rewrite,
    wildcard,
)


@rt.function_pass(opt_level=0)
class CanonicalizeBiasAdd(DFPatternCallback):
    """
    TVM bug: canonicalize converts bias_add(x,bias) as add(x,bias) instead of add(x,[1,bias])
    """

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        x = wildcard()
        b = is_constant()
        y = is_op("nn.bias_add")(x, b)
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        x = post.args[0]
        x_shape = x.checked_type.concrete_shape

        b = post.args[1].data.numpy()
        b_shape = b.shape[0]

        new_shape = [1] * len(x_shape)
        new_shape[0] = x_shape[0]
        if b_shape == x_shape[1]:  # NCHW case
            new_shape[1] = b_shape
        else:  # NHWC case
            new_shape[-1] = b_shape
        new_b = b.reshape(new_shape)

        y = relay.add(x, relay.const(new_b))
        return y


@rt.function_pass(opt_level=0)
class CanonicalizeBatchNorm(DFPatternCallback):
    """
    TVM bug: canonicalize converts batch_norm to constants without batch dimension
    """

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        x = wildcard()
        gamma = is_constant()
        beta = is_constant()
        mean = is_constant()
        var = is_constant()
        y = is_op("nn.batch_norm")(x, gamma, beta, mean, var)
        y = is_tuple_get_item(y)
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        bn = post.tuple_value
        x = bn.args[0]
        gamma = bn.args[1].data.numpy()
        beta = bn.args[2].data.numpy()
        mean = bn.args[3].data.numpy()
        var = bn.args[4].data.numpy()
        epsilon = bn.attrs.epsilon
        gamma_norm = gamma / np.sqrt(var + epsilon)

        x_shape = tuple(-1 if type(x) == tvm.tir.Any else int(x) for x in x.checked_type.shape)
        dims = len(x_shape)
        gamma_norm = gamma_norm.reshape((1, -1) + (dims - 2) * (1,))
        mean = mean.reshape((1, -1) + (dims - 2) * (1,))
        beta = beta.reshape((1, -1) + (dims - 2) * (1,))

        if dims > 2 and x_shape[-1] == gamma.shape[0]:
            gamma_norm = gamma_norm.transpose(0, 2, 3, 1)
            mean = mean.transpose(0, 2, 3, 1)
            beta = beta.transpose(0, 2, 3, 1)

        y = relay.add(x, relay.const(-mean))
        y = relay.multiply(y, relay.const(gamma_norm))
        y = relay.add(y, relay.const(beta))
        return y


@rt.function_pass(opt_level=0)
class SimplifyReshapeSqueeze(DFPatternCallback):
    """
    reshape(0,-1,1,1) -> squeeze([2,3]) = batch_flatten
    """

    # Note: global avg/max pooling always ends up as shape (N, C, 1, 1) so it's usually followed by batch_flatten
    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        x = wildcard()
        y = self.reshape = is_op("reshape")(x)
        y = self.squeeze = is_op("squeeze")(y)
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        reshape = node_map[self.reshape][0]
        squeeze = node_map[self.squeeze][0]
        x = reshape.args[0]
        newshape = reshape.attrs["newshape"]
        axis = squeeze.attrs["axis"]
        if newshape[-2] == newshape[-1] == 1 and axis[0] == 2 and axis[1] == 3:
            return relay.nn.batch_flatten(x)

        return post  # no change


@rt.function_pass(opt_level=0)
class BackwardFoldScaleWeights(DFPatternCallback):
    """
    conv(x,w) + add(x, bias) + mul(x,S) -> conv(x,w')+bias'
    """

    # Note: global avg/max pooling always ends up as shape (N, C, 1, 1) so it's usually followed by batch_flatten
    def __init__(self):
        super().__init__(False, False)
        x = wildcard()
        w = is_constant()
        b1 = is_constant()
        b2 = is_constant()

        # TODO: extend to more linear ops
        y = self.conv = (is_op("nn.conv2d") | is_op("nn.dense"))(x, w)
        y = self.bias1 = y.optional(lambda x: is_op("add")(x, b1))
        y = self.mult = is_op("multiply")(y, is_constant())
        y = self.bias2 = y.optional(lambda x: is_op("add")(x, b2))
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        conv = node_map[self.conv][0]
        mul = node_map[self.mult][0]
        bias1 = node_map[self.bias1][0]
        bias2 = node_map[self.bias2][0]
        x = conv.args[0]
        w = conv.args[1].data.numpy()
        scale = mul.args[1].data.numpy()

        K = w.shape[0]
        scaler = scale if scale.shape == () else scale.reshape(K, -1)
        wn = w.reshape(K, -1) * scaler
        wn = wn.reshape(w.shape)

        wn = relay.const(wn, dtype="float32")

        relay_op = _relay_creator[conv.op.name]
        attrs = {**conv.attrs}
        y = relay_op(x, wn, **attrs)
        bn = None
        if bias1 != conv:
            b = bias1.args[1].data.numpy()
            bn = b * scale

        if bias2 != mul:
            b = bias2.args[1].data.numpy()
            bn = b if bn is None else bn + b

        if bn is not None:
            y = y + relay.const(bn)

        return y


@rt.function_pass(opt_level=0)
class RewriteMultiply(DFPatternCallback):
    """
    Some models like efficientnet have multiply(const, x) instead of multiply(x,const)
    This screws other passes
    """

    # Note: global avg/max pooling always ends up as shape (N, C, 1, 1) so it's usually followed by batch_flatten
    def __init__(self):
        super().__init__(False, False)
        x = wildcard()

        y = self.mul = is_op("multiply")(x, is_constant())
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        mul = node_map[self.mul][0]

        x = mul.args[0]
        K = mul.args[1]
        if isinstance(K, relay.Call):  # weird that is_constant() pass here!
            x, K = K, x
        assert isinstance(K, relay.Constant)
        K = relay.const(K.data.numpy())
        # logger.info(f"rewrite multiply with constant {K=}")
        y = relay.multiply(x, K)
        return y


_relay_creator = {
    "nn.conv2d": relay.nn.conv2d,
    "nn.dense": relay.nn.dense,
}


@rt.function_pass(opt_level=0)
class ForwardFoldScaleWeights(DFPatternCallback):
    """
    add(x, bias) + mul(x,S) + conv(x,w) -> conv(x,w')+bias
    """

    # Note: global avg/max pooling always ends up as shape (N, C, 1, 1) so it's usually followed by batch_flatten
    def __init__(self):
        super().__init__(False, False)
        x = wildcard()
        w = is_constant()
        # b1 = is_constant()
        b2 = is_constant()

        # TODO: extend to more linear ops
        y = self.mult = is_op("multiply")(x, is_constant())
        y = self.bias2 = y.optional(lambda x: is_op("add")(x, b2))
        y = self.conv = (is_op("nn.conv2d") | is_op("nn.dense"))(y, w)
        # y = self.bias1 = y.optional(lambda x: is_op("add")(x, b1))
        self.pattern = y

    def transform_function(
        self, func: relay.function.Function, mod: tvm.IRModule, ctx: tvm.transform.PassContext
    ) -> relay.function.Function:
        return rewrite(self, func)

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: tvm.ir.container.Map) -> relay.Expr:
        conv = node_map[self.conv][0]
        mul = node_map[self.mult][0]
        # bias1 = node_map[self.bias1][0]
        bias2 = node_map[self.bias2][0]
        x = mul.args[0]
        w = conv.args[1].data.numpy()
        scale = mul.args[1].data.numpy()

        K, C, R, S = w.shape
        if scale.shape == ():
            wn = w * scale
        else:
            wn = np.empty(w.shape)
            for k in range(K):
                for c in range(C):
                    wn[k, c] = w[k, c] * scale[0, c, 0, 0]

        wn = relay.const(wn, dtype="float32")

        relay_op = _relay_creator[conv.op.name]
        attrs = {**conv.attrs}
        y = relay_op(x, wn, **attrs)

        bn = None
        # if bias1 != conv:
        #     b = bias1.args[1].data.numpy()
        #     bn = b * scale

        if bias2 != mul:
            b = bias2.args[1].data.numpy()
            conv_attrs = {**conv.attrs}
            x1 = relay.var("x", shape=b.shape)
            y1 = relay.nn.conv2d(x1, relay.const(w), **conv_attrs)
            func = relay.Function(relay.analysis.free_vars(y1), y1)
            mod = tvm.IRModule.from_expr(func)

            res = run_tvm(mod, {"x": b})  # noqa: F841
            res = relay.const(res[0])
            bn = res if bn is None else bn + res

        if bn is not None:
            y = y + bn

        return y


def convert_layout(mod: tvm.IRModule, params=None):
    func = mod["main"]
    if params:
        func = relay.build_module.bind_params_by_name(func, params)
        mod = tvm.IRModule.from_expr(func)

    mod = rt.InferType()(mod)

    input_dict = {p: 0 for p in mod["main"].params}
    mod = rt.ChangeBatch(input_dict, batch_size=1)(mod)
    mod = rt.InferType()(mod)

    ops_freq = relay.analysis.list_op_freqs(tvm.IRModule.from_expr(func))
    desired_layouts = {
        op_name: ["NCDHW" if "3d" in op_name else "NCHW", "default"] for op_name, count in ops_freq.items()
    }

    mod = rt.ConvertLayout(desired_layouts)(mod)

    mod = rt.InferType()(mod)
    return mod


def simplify_relay(mod: tvm.IRModule, params = None) -> relay.Function:

    mod = convert_layout(mod, params)

    transforms = [
        rt.InferType(),
        rt.DynamicToStatic(),
        CanonicalizeBatchNorm(),
        CanonicalizeBiasAdd(),
        rt.InferType(),
        rt.SimplifyInference(),
        RewriteMultiply(),
        rt.FoldConstant(),
        rt.SimplifyExpr(),
        BackwardFoldScaleWeights(), 
        ForwardFoldScaleWeights(), 
        SimplifyReshapeSqueeze(),
        rt.InferType(),
    ]

    with tvm.transform.PassContext(opt_level=3):
        for xf in (pbar := tqdm(transforms)):
            pbar.set_description(f"{xf}") 
            try:
                mod = xf(mod)
            except:
                print('Skipping ', xf)

    return mod