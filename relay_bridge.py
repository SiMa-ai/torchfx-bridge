"""
RelayModule is a Pytorch nn.Module from a relay dataflow graph.
"""
import math
import sys
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvt
import tvm
from torch import fx
from torch import nn as tnn
from torch.fx.passes.shape_prop import ShapeProp  # noqa: F401
from torch.utils.dlpack import from_dlpack, to_dlpack
from tvm import relay
from tvm.relay import nn as rnn  # noqa: F401


def torch_compatible_name(name: str):
    import re

    name = name.replace("nn.", "")
    name = re.sub("[:./-]", "_", name)
    if name.isnumeric():
        name = f"_{name}"
    return name

def run_opt_pass(expr: relay.Expr, opt_pass: list[tvm.transform.Pass]):
    """Run an optimization pass or a list of optimization passes"""
    if not isinstance(opt_pass, list):
        opt_pass = [opt_pass]
    for pass_ in opt_pass:
        assert isinstance(pass_, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential([relay.transform.InferType(), *opt_pass])
    mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def run_infer_type(expr: relay.Expr):
    """Runs InferType pass on relay expression."""
    return run_opt_pass(expr, relay.transform.InferType())

def shape_of(x: relay.Expr, dtype="int32"):
    """Returns the shape of a relay object from symbols."""
    try:
        if not hasattr(x, "checked_type"):
            ttype = run_infer_type(x).checked_type
            if not relay.ty.is_dynamic(ttype):
                return list(ttype.shape)
            return list(relay.shape_of(x, dtype).data.numpy())

        shape_type = x.checked_type
    except ValueError:
        shape_type = x.type_annotation if hasattr(x, "type_annotation") else x.ret_type

    shape = [int(v) if isinstance(v, tvm.tir.IntImm) else -1 for v in shape_type.shape]

    return shape
INT_MAX = sys.maxsize

"""
Register each const as nn.Parameters if weights or as const Tensor
Add relay op as functional in forward or, register modules but it's more complex mapping and same behavior

Notes:
- onnx2torch has a similar architecture where each ONNX op is represented as a nn.Module.
- tf2onnx uses simplifications/optimizers similar to relay but sooo much faster graph transforms and way more extensive
especially with capabilities like searching patterns with more than 2 nodes (relay limited there)
"""

__all__ = ["RelayModule"]

# TODO: test pb models without tf2onnx to see what ops is left and missing here
# TODO: test accuracy of ops whatever attribute values: ORT vs relay vs torch

# weighted layers have known nn.Parameters but any torch op can have parameters.
# however, if parameters come from constant, eg weights for conv, then they are
# in a constant buffer and we can remove it.
# Otherwise, the parameter becomes another input.
linear_layers = [
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
    "nn.prelu",
    "nn.batch_norm",
]

# mapping pytorch quantized types to relay/numpy
t2r_types = {
    torch.quint8: "uint8",
    torch.qint8: "int8",
    torch.quint2x4: "uint8",
    torch.quint4x2: "uint8",
    torch.qint32: "int32",
    torch.int32: "int32",
    torch.float32: "float32",
}

# mapping relay dtypes to pytorch dtypes
r2t_types = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "float32": torch.float32,
    "": torch.float32,  # default dtype
}
from abc import ABC, abstractmethod
from torch import fx, nn

class Transformation(ABC):
    """
    A torch.fx graph transformation.

    (copied from openai optimum.utils.fx)
    """

    preserves_computation: bool = False

    @abstractmethod
    def transform(self, graph_module: fx.GraphModule | nn.Module, concrete_args=None) -> fx.GraphModule:
        """
        Args:
            graph_module (`torch.fx.GraphModule | torch.nn.Module`):
                The module to transform.
            concrete_args (`dict`): Optional.
                Concrete args that may need to be used to trace symbolically the model

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.
        """
        raise NotImplementedError("The transform method needs to be implemented.")

    def __call__(
        self, graph_module: fx.GraphModule, lint_and_recompile: bool = True, concrete_args=None
    ) -> fx.GraphModule:
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The module to transform.
            lint_and_recompile (`bool`, defaults to `True`):
                Whether the transformed module should be linted and recompiled.
                This can be set to `False` when chaining transformations together to perform this operation only once.

        Returns:
            `torch.fx.GraphModule`:
                The transformed module.
        """
        graph_module = self.transform(graph_module, concrete_args=concrete_args)
        if lint_and_recompile:
            graph_module.graph.lint()
            graph_module.recompile()
        return graph_module

    @property
    def signature(self):
        """
        Returns a hash that can be used to identify the transformation.
        """
        attributes_to_use_for_hashing = vars(self)
        attributes_to_use_for_hashing[""] = self.__class__
        hash_str = "_".join(f"{k}_{hash(v)}" for k, v in attributes_to_use_for_hashing.items())
        return hash(hash_str)

    def mark_as_transformed(self, node: fx.Node):
        """
        Marks a node as transformed by this transformation.

        Args:
            node (`torch.fx.Node`):
                The node to mark as transformed.
        """
        node_transformations = getattr(node, "transformations", set())
        node_transformations.add(self.signature)
        node.transformations = node_transformations

    def transformed(self, node: fx.Node) -> bool:
        """
        Args:
            node (`torch.fx.Node`):
                The node to check.

        Returns:
            `bool`:
                Specifies whether the node was transformed by this transformation or not.
        """
        return self.signature in getattr(node, "transformations", set())

    def get_transformed_nodes(self, graph_module: fx.GraphModule) -> list[fx.Node]:
        """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The graph_module to get the nodes from.

        Returns:
            `List[torch.fx.Node]`:
                Gives the list of nodes that were transformed by the transformation.
        """

        return [node for node in graph_module.graph.nodes if self.transformed(node)]




class LintAndRecompile(Transformation):
    """
    Transformation that does nothing except linting and recompiling the graph module.
    """

    preserves_computation = True

    def transform(self, graph_module: fx.GraphModule, concrete_args=None) -> fx.GraphModule:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module

def t2r_arr(torch_arr: torch.Tensor) -> tvm.nd.NDArray:
    """Reads PyTorch tensor to TVM"""
    return tvm.nd.from_dlpack(to_dlpack(torch_arr))
    # return tvm.nd.NDArray(torch_arr.detach().contiguous().cpu().numpy())


def r2t_arr(tvm_arr: tvm.nd.NDArray) -> torch.Tensor:
    """Reads TVM tensor to PyTorch"""
    return from_dlpack(tvm_arr) if isinstance(tvm_arr, tvm.nd.NDArray) else tvm_arr
    # return torch.from_numpy(tvm_arr.numpy())


@fx.wrap
def fx_wrapped_reshape(x, shape):
    return x.reshape(shape)


@fx.wrap
def fx_wrapped_to_pil_image(x):
    x = tvt.ToPILImage(None)(x)  # x.reshape(shape)
    return x


# @fx.wrap
# def fx_wrapped_resize(x):
#     # x = tvt.Resize(**tattrs)(x)
#     x = Ftvt.resize(**tattrs)
#     return x


@fx.wrap
def fx_wrapped_resize(resize, x):
    return resize(x)


@fx.wrap
def fx_wrapped_index_select(x, ax, indices):
    return torch.index_select(x, ax, indices)


@fx.wrap
def fx_wrapped_arange(rank, begin, end, stride):
    begin = (rank + begin) if begin < 0 else begin
    end = (rank + end + 1) if end < 0 else end
    return torch.arange(begin, end, stride)


@fx.wrap
def fx_wrapped_min(x, **attrs):
    return torch.min(x, **attrs)


@fx.wrap
def fx_wrapped_max(x, **attrs):
    return torch.max(x, **attrs)


fx.wrap("len")
# fx.wrap("range")
# fx.wrap("sqrt")
# fx.wrap("rsqrt")


# for tagging our converter classes so FX symbolic tracing doesn't trace them
class RelayFX_Module:
    pass


_CONVERTER_REGISTRY = {}


def register_converter(relay_op_name: str):
    def wrapper(converter: tnn.Module):
        if relay_op_name in _CONVERTER_REGISTRY:
            raise ValueError(f'Operation "{relay_op_name}" already registered')

        _CONVERTER_REGISTRY[relay_op_name] = converter
        #logger.debug(f"Operation converter registered {relay_op_name}")

        return converter

    return wrapper


def get_converter(relay_op: relay.Call):
    """Returned equivalent pytorch functional op to relay op."""
    assert isinstance(relay_op, relay.Call), "Wrong type, relay_op must be relay.Call"
    op_name = relay_op.op.name
    # print(f"Converting {op_name}")
    assert op_name in _CONVERTER_REGISTRY, f"Unsupported relay {op_name} to pytorch"
    return _CONVERTER_REGISTRY[op_name](relay_op)


###############################################################################
#
# Weighted
#
###############################################################################

CONV_OPS = {
    "nn.conv2d": tnn.Conv2d,
    "nn.conv3d": tnn.Conv3d,
    "nn.conv2d_transpose": tnn.ConvTranspose2d,
    "nn.conv3d_transpose": tnn.ConvTranspose3d,
}


@register_converter("nn.conv3d")
@register_converter("nn.conv2d")
@register_converter("nn.conv3d_transpose")
@register_converter("nn.conv2d_transpose")
class _conv_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # rnn.conv2d
        # rnn.conv2d_transpose
        # tnn.Conv2d
        node_name = node.op.name
        is_transpose = node_name.endswith("transpose")
        op = CONV_OPS[node_name]
        w = node.args[1]
        # x_shape = shape_of(node.args[0])
        w_shape = shape_of(w)

        attrs = {**node.attrs}
        tattrs = {}
        if is_transpose:
            tattrs["in_channels"], tattrs["out_channels"], *kernel_size = w_shape
        else:
            tattrs["out_channels"], tattrs["in_channels"], *kernel_size = w_shape

        tattrs["kernel_size"] = kernel_size
        tattrs["bias"] = False  # TODO: handle by fusing conv2d+add(bias)

        dim = len(kernel_size)  # number of spatial dimensions 2 for 2d, 3 for 3d
        groups = 1
        for attr, val in attrs.items():
            match attr:
                case "strides":
                    tval = tuple(int(v) for v in val)
                    tattrs["stride"] = tval[:dim] if len(tval) >= dim else tval[0]
                # case "padding":
                #     tval = tuple(int(v) for v in val)
                #     tattrs["padding"] = tval[dim:] if len(tval) >= dim else tval[0]
                #     print(f'conv2d padding {tval} -> {tattrs["padding"]}')
                case "dilation":
                    tval = tuple(int(v) for v in val)
                    tattrs["dilation"] = tval[:dim] if len(tval) >= dim else tval[0]
                case "groups":
                    tattrs["groups"] = groups = int(val)
                case "out_dtype":
                    tattrs["dtype"] = r2t_types[val]
                case "output_padding":  # conv transpose only
                    opad = convert_relay_padding(val)
                    same_value = all(element == opad[0] for element in opad)
                    tattrs["output_padding"] = opad[0] if same_value else opad

        # in/out channels must always be multiple of groups
        if is_transpose:
            tattrs["out_channels"] *= groups
        else:
            tattrs["in_channels"] *= groups
        #logger.debug(
            # f"{node_name} {w_shape=} {kernel_size=} {tattrs['in_channels']=} {tattrs['out_channels']=} {tattrs['groups']=}"
            # f"{node_name} {x_shape=} {w_shape=} {kernel_size=} {tattrs['in_channels']=} {tattrs['out_channels']=} {tattrs['groups']=}"
        # )

        self.pad = ipad = convert_relay_padding(attrs["padding"])
        same_value = all(element == ipad[0] for element in ipad)
        self.is_pad_needed = not same_value
        tattrs["padding"] = ipad[0] if same_value else 0
        #logger.debug(f"{node_name} {attrs} -> {tattrs} {self.pad=} {self.is_pad_needed=}")
        self.conv = op(**tattrs)
        # self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, *args):
        x = args[0]
        if self.is_pad_needed:
            x = F.pad(x, pad=self.pad)
        x = self.conv(x, *args[1:])
        return x


# BUG - wrong results if asym padding is not done with F.pad
# def _conv_converter(node: relay.Call):
#     node_name = node.op.name
#     is_transpose = node_name.endswith("transpose")
#     op = CONV_OPS[node_name]
#     w = node.args[1]
#     # x_shape = shape_of(node.args[0])
#     w_shape = shape_of(w)

#     attrs = {**node.attrs}
#     tattrs = {}
#     if is_transpose:
#         tattrs["in_channels"], tattrs["out_channels"], *kernel_size = w_shape
#     else:
#         tattrs["out_channels"], tattrs["in_channels"], *kernel_size = w_shape

#     tattrs["kernel_size"] = kernel_size
#     tattrs["bias"] = False  # TODO: handle by fusing conv2d+add(bias)

#     dim = len(kernel_size)  # number of spatial dimensions 2 for 2d, 3 for 3d
#     groups = 1
#     for attr, val in attrs.items():
#         match attr:
#             case "strides":
#                 tval = tuple(int(v) for v in val)
#                 tattrs["stride"] = tval[:dim] if len(tval) >= dim else tval[0]
#             # case "padding":
#             #     tval = tuple(int(v) for v in val)
#             #     tattrs["padding"] = tval[dim:] if len(tval) >= dim else tval[0]
#             #     print(f'conv2d padding {tval} -> {tattrs["padding"]}')
#             case "dilation":
#                 tval = tuple(int(v) for v in val)
#                 tattrs["dilation"] = tval[:dim] if len(tval) >= dim else tval[0]
#             case "groups":
#                 tattrs["groups"] = groups = int(val)
#             case "out_dtype":
#                 tattrs["dtype"] = r2t_types[val]
#             case "output_padding":  # conv transpose only
#                 opad = convert_relay_padding(val)
#                 same_value = all(element == opad[0] for element in opad)
#                 tattrs["output_padding"] = opad[0] if same_value else opad

#     # in/out channels must always be multiple of groups
#     if is_transpose:
#         tattrs["out_channels"] *= groups
#     else:
#         tattrs["in_channels"] *= groups
#     #logger.debug(
#         f"{node_name} {w_shape=} {kernel_size=} {tattrs['in_channels']=} {tattrs['out_channels']=} {tattrs['groups']=}"
#         # f"{node_name} {x_shape=} {w_shape=} {kernel_size=} {tattrs['in_channels']=} {tattrs['out_channels']=} {tattrs['groups']=}"
#     )

#     pad = ipad = convert_relay_padding(attrs["padding"])
#     same_value = all(element == ipad[0] for element in ipad)
#     is_pad_needed = not same_value
#     tattrs["padding"] = ipad[0] if same_value else 0
#     logger.debug(f"{node_name} {attrs} -> {tattrs} {pad=} {is_pad_needed=}")
#     conv = op(**tattrs)
#     # conv = conv.to(memory_format=torch.channels_last)

#     if is_pad_needed:
#         _pad = tnn.ZeroPad2d(pad)
#         return tnn.Sequential(
#             OrderedDict(
#                 {
#                     "pad": _pad,
#                     "conv": conv,
#                 }
#             )
#         )
#     return conv


"""
@register_converter("nn.conv3d_transpose")
@register_converter("nn.conv2d_transpose")
class _conv_transpose_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # rnn.conv2d_transpose
        # tnn.ConvTranspose2d
        node_name = node.op.name
        op = CONV_OPS[node_name]
        w = node.args[1]
        w_shape = shape_of(w)

        attrs = {**node.attrs}
        tattrs = {}
        tattrs["in_channels"], tattrs["out_channels"], *kernel_size = w_shape
        tattrs["kernel_size"] = kernel_size
        tattrs["bias"] = False  # TODO: handle by fusing conv2d+add(bias)
        tattrs["padding"] = 0
        dim = len(kernel_size)  # number of dimensions 2 for 2d, 3 for 3d...
        for attr, val in attrs.items():
            match attr:
                case "strides":
                    tval = tuple(int(v) for v in val)
                    tattrs["stride"] = tval[:dim] if len(tval) >= dim else tval[0]
                # case "padding":
                #     tval = tuple(int(v) for v in val)
                #     tattrs["padding"] = tval[dim:] if len(tval) >= dim else tval[0]
                case "dilation":
                    tval = tuple(int(v) for v in val)
                    tattrs["dilation"] = tval[:dim] if len(tval) >= dim else tval[0]
                case "groups":
                    tattrs["groups"] = g = int(val)
                case "out_dtype":
                    tattrs["dtype"] = r2t_types[val]

        # in/out channels must always be multiple of groups
        tattrs["out_channels"] *= g
        logger.debug(
            f"{node_name} {w_shape=} {tattrs['in_channels']=} {tattrs['out_channels']=} {tattrs['groups']=} {tattrs['padding']=}"
        )

        self.pad = convert_relay_padding(attrs["padding"])
        same_value = all(element == self.pad[0] for element in self.pad)
        self.is_pad_needed = not same_value
        tattrs["padding"] = self.pad[0] if same_value else 0
        # self.is_pad_needed = any(self.pad)
        # logger.debug(f"{node_name} {attrs} -> {tattrs} {self.pad=} {self.is_pad_needed=}")
        self.conv = op(**tattrs)
        # self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, *args):
        x = args[0]
        if self.is_pad_needed:
            x = F.pad(x, pad=self.pad)
        x = self.conv(x, *args[1:])
        return x
"""


@register_converter("nn.dense")
def _dense_converter(node: relay.Call):
    w = node.args[1]
    w_shape = shape_of(w)
    out_features, in_features = w_shape
    # logger.debug(f"dense {out_features=}, {in_features=}")
    return tnn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=r2t_types[node.attrs["out_dtype"]],
    )


@register_converter("nn.bias_add")
# class _bias_add_converter(tnn.Module, RelayFX_Module):
#     def __init__(self, node: relay.Call) -> None:
#         super().__init__()
#         x_shape = shape_of(node.args[0])
#         bias = r2t_arr(node.args[1].data)
#         bias = bias.reshape(1, -1)  # use N=1 so it is broadcast over batch dimension
#         while len(bias.shape) < len(x_shape):
#             bias = bias.unsqueeze(-1)
#         self.bias = bias
#         logger.info(f"bias_add {self.bias.shape=} {x_shape=}")


#     def forward(self, x, b) -> Any:
#         # b = fx_wrapped_reshape(b, (x.shape[0], -1, 1, 1))
#         # b = b.flatten()
#         # b = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#         # logger.info(f"bias_add -> add {b.shape=}")
#         b = self.bias
#         return torch.add(x, b)
def _bias_add_converter(node: relay.Call):
    x_shape = shape_of(node.args[0])
    bias = r2t_arr(node.args[1].data)
    bias = bias.reshape(1, -1)  # use N=1 so it is broadcast over batch dimension
    while len(bias.shape) < len(x_shape):
        bias = bias.unsqueeze(-1)
    return operator.add


@register_converter("nn.batch_matmul")
class _batch_matmul_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # rnn.batch_matmul
        # relay bmm assumes y is transposed, not torch
        self.transpose_a = bool(node.attrs["transpose_a"])
        self.transpose_b = bool(node.attrs["transpose_b"])
        x_shape = shape_of(node.args[0])
        y_shape = shape_of(node.args[1])
        # logger.debug(f"batch_matmul {self.transpose_a=} {self.transpose_b=} {x_shape=} {y_shape=}")
        # assert (
        #     self.transpose_a is False and self.transpose_b is True
        # ), f"Only defaults transpose settings supported. Found {self.transpose_a=} {self.transpose_b=}"

    def forward(self, x, y) -> Any:
        if self.transpose_a:
            x = x.transpose(-2, -1)
        if self.transpose_b:
            y = y.transpose(-2, -1)

        return torch.bmm(x, y)
    
@register_converter("einsum")
class _einsum_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        self.equation = node.attrs['equation']
        super().__init__()

    def forward(self, *args):
        return torch.einsum(self.equation, args)


###############################################################################
#
# Pooling
#
###############################################################################
# POOL_OPS = {
#     "nn.max_pool2d": F.max_pool2d,
#     "nn.max_pool3d": F.max_pool3d,
#     "nn.avg_pool2d": F.avg_pool2d,
#     "nn.avg_pool3d": F.avg_pool3d,
# }

POOL_OPS = {
    "nn.max_pool2d": tnn.MaxPool2d,
    "nn.max_pool3d": tnn.MaxPool3d,
    "nn.avg_pool2d": tnn.AvgPool2d,
    "nn.avg_pool3d": tnn.AvgPool3d,
}


@register_converter("nn.max_pool2d")
@register_converter("nn.max_pool3d")
@register_converter("nn.avg_pool2d")
@register_converter("nn.avg_pool3d")
class _max_pool_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # rnn.max_pool2d
        # tnn.MaxPool2d
        # F.max_pool2d
        attrs = {**node.attrs}
        node_name = node.op.name
        is_avg = "avg" in node_name
        op = POOL_OPS[node_name]
        dims = int(node_name[-2:-1])
        # logger.debug(f"{node_name} {dims=} {attrs}")
        tattrs = {}
        tattrs["padding"] = 0
        for attr, val in attrs.items():
            match attr:
                case "pool_size":
                    tval = tuple(int(v) for v in val)
                    tattrs["kernel_size"] = tval[:dims] if len(tval) >= dims else tval[0]
                case "strides":
                    tval = tuple(int(v) for v in val)
                    tattrs["stride"] = tval[:dims] if len(tval) >= dims else tval[0]
                case "dilation":
                    if not is_avg:
                        # torch avg_pool don't have dilation
                        tval = tuple(int(v) for v in val)
                        tattrs["dilation"] = tval[:dims] if len(tval) >= dims else tval[0]
                    else:
                        assert val[0] == 1 and val[1] == 1, f"{node_name} must have dilation=(1,1). Found {val}"
                # case "padding": # this padding is symmetric
                #     tval = tuple(int(v) for v in val)
                #     tattrs["padding"] = convert_relay_padding(tval)
                #     print(f'max_pool2d padding {tval} -> {tattrs["padding"]}')
                case "ceil_mode":
                    tattrs["ceil_mode"] = bool(val)

        self.pad = convert_relay_padding(attrs["padding"])
        same_value = all(element == self.pad[0] for element in self.pad)
        self.is_pad_needed = not same_value
        tattrs["padding"] = self.pad[0] if same_value else 0

        # #logger.debug(f"{node_name} {attrs} -> {tattrs} {self.pad=} {self.is_pad_needed=}")
        self.pool = op(**tattrs)

    def forward(self, *args):
        x = args[0]
        if self.is_pad_needed:
            x = F.pad(x, pad=self.pad)
        x = self.pool(x, *args[1:])
        return x


@register_converter("nn.global_avg_pool2d")
def _global_avg_pool2d_converter(node: relay.Call):
    return tnn.AdaptiveAvgPool2d(output_size=(1, 1))


@register_converter("nn.global_max_pool2d")
def _global_max_pool2d_converter(node: relay.Call):
    return tnn.AdaptiveMaxPool2d(output_size=(1, 1))


@register_converter("nn.adaptive_avg_pool1d")
def _adaptive_avg_pool1d_converter(node: relay.Call):
    output_size = node.attrs.output_size[0]
    x_shape = shape_of(node.args[0])
    output_size = x_shape[-1] if output_size is None else int(output_size)
    return tnn.AdaptiveAvgPool1d(output_size=output_size)


@register_converter("nn.adaptive_avg_pool2d")
def _adaptive_avg_pool2d_converter(node: relay.Call):
    output_size = tuple([int(v) for v in node.attrs["output_size"]])
    #logger.debug(f"adaptive_avg_pool2d {output_size=}")
    return tnn.AdaptiveAvgPool2d(output_size=output_size)


@register_converter("nn.adaptive_max_pool1d")
def _adaptive_max_pool1d_converter(node: relay.Call):
    output_size = node.attrs.output_size[0]
    x_shape = shape_of(node.args[0])
    output_size = x_shape[-1] if output_size is None else int(output_size)
    return tnn.AdaptiveMaxPool1d(output_size=output_size)


@register_converter("nn.adaptive_max_pool2d")
def _adaptive_max_pool2d_converter(node: relay.Call):
    output_size = tuple([int(v) for v in node.attrs["output_size"]])
    return tnn.AdaptiveMaxPool2d(output_size=output_size)


###############################################################################
#
# Normalization
#
###############################################################################


@register_converter("nn.batch_norm")
def _batch_norm_converter(node: relay.Call):
    # relay.nn.batch_norm
    # tnn.BatchNorm2d
    gamma, beta, moving_mean, moving_var = node.args[1:5]
    center = node.attrs["center"]
    scale = node.attrs["scale"]
    num_features = shape_of(gamma)[0]
    eps = float(node.attrs["epsilon"])
    #logger.debug(f"batch_norm {num_features=} {eps=}")
    bn = tnn.BatchNorm2d(num_features, eps, affine=True, track_running_stats=False)
    if scale:
        bn.weight = tnn.Parameter(r2t_arr(gamma.data), requires_grad=False)
    if center:
        bn.bias = tnn.Parameter(r2t_arr(beta.data), requires_grad=False)
    bn.running_mean = r2t_arr(moving_mean.data)
    bn.running_var = r2t_arr(moving_var.data)
    bn.training = False
    return bn


@register_converter("nn.lrn")
def _lrn_converter(node: relay.Call):
    attrs = {**node.attrs}
    size = int(attrs["size"])
    k = float(attrs["bias"])
    # axis=1 for C channel in NCHW
    alpha = float(attrs["alpha"])
    beta = float(attrs["beta"])
    return tnn.LocalResponseNorm(size, alpha, beta, k)


###############################################################################
#
# Memory
#
###############################################################################


def convert_relay_padding(padding):
    # relay is (axis 0..k min, axis 0..k max)
    # torch is (axis k min, axis k max, ..., axis 0 min, axis 0 max)
    # print(type(padding))
    if isinstance(padding, (tuple, list, tvm.ir.Array)):
        padding = tuple(int(v) for v in padding)
        if len(padding) == 2:  # 1d
            padding = (padding[1], padding[0])
        elif len(padding) == 4:  # 2d
            padding = (padding[1], padding[3], padding[0], padding[2])
        elif len(padding) == 6:  # 3d
            padding = (padding[2], padding[5], padding[1], padding[4], padding[0], padding[3])
        return padding
    return padding


@register_converter("nn.depth_to_space")
class _depth_to_space_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        in_shape = shape_of(node.args[0])
        self.n, self.c, self.h, self.w = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
        self.block_size = node.attrs["block_size"]
        # new_shape = in_shape
        # new_shape[1] = new_shape[1]//(block_size*block_size)
        # new_shape[2] = new_shape[2]*block_size
        # new_shape[3] = new_shape[3]*block_size
        # self.new_shape = new_shape
    
    def forward(self, x) -> Any :
        tmp = torch.reshape(x, [self.n, self.c // (self.block_size ** 2), self.block_size, self.block_size, self.h, self.w])
        tmp = torch.permute(tmp, (0, 1, 4, 2, 5, 3))
        return torch.reshape(tmp, [self.n, self.c // (self.block_size ** 2), self.h * self.block_size, self.w * self.block_size])
      
    
@register_converter("reshape")
class _reshape_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.reshape
        # torch.reshape
        # torch doesn't understand reshape with dim 0 (means copy dim)
        in_shape = shape_of(node.args[0])
        new_shape = [int(s) for s in node.attrs["newshape"]]

        # in relay, if a dim=0, then dim is copied from input
        # for idx, (in_s, out_s) in enumerate(zip(in_shape, new_shape, strict=False)):
        #     new_shape[idx] = int(in_s) if out_s == 0 else out_s
        # self.new_shape = tuple(new_shape)
        def check_shape(input_shape, new_shape):
            new_shape = [*new_shape]
            while 0 in new_shape:
                idx0 = new_shape.index(0)
                # print(f"{idx0=}")
                new_shape[idx0] = input_shape[idx0]
            if -1 in new_shape:
                idx1 = new_shape.index(-1)
                val = math.prod([v for idx, v in enumerate(new_shape) if idx != idx1])
                new_shape[idx1] = math.prod(input_shape) // val
            new_shape[0] = -1
            return tuple(new_shape)

        self.new_shape = check_shape(in_shape, new_shape)
        #logger.debug(f"reshape {in_shape=}, {new_shape=} -> {self.new_shape=}")

    def forward(self, x) -> Any:
        # note: if arg 'shape=' is used, value will appear
        # twice during MCT execution: 1) as arg, 2) as kwarg
        # conclusion: do NOT use kwargs!
        return torch.reshape(x, self.new_shape)


@register_converter("squeeze")
class _squeeze_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.squeeze
        # torch.squeeze
        self.dim = tuple(int(v) for v in node.attrs["axis"])
        # print(f'squeeze axis={node.attrs["axis"]} -> {self.dim}')

    def forward(self, x) -> Any:
        return torch.squeeze(x, self.dim)


@register_converter("split")
class _split_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.split
        # torch.tensor_split
        # torch.chunk
        # torch.split
        attrs = {**node.attrs}
        self.dim = int(attrs["axis"])
        self.sections = None
        self.is_index = False
        if isinstance(attrs["indices_or_sections"], (tuple, list, tvm.ir.Array)):
            self.sections = [int(v) for v in attrs["indices_or_sections"]]
            self.is_index = True
            if len(self.sections) == 1:
                self.sections = self.sections[0]
                self.is_index = False
        else:
            self.sections = int(attrs["indices_or_sections"])
            self.is_index = True
        x_shape = shape_of(node.args[0])
        #logger.debug(f'split {x_shape=} axis={attrs["axis"]} -> {self.dim=}, {self.sections=} {self.is_index=}')

    def forward(self, x) -> Any:
        if self.is_index:
            return torch.tensor_split(x, self.sections, self.dim)
        return torch.split(x, self.sections, self.dim)


# TODO: this doesn't do strided_slice correctly [:,begin:end:step,:]
# : = slice(None) | begin:end:step = slice(begin, end, step) | -1 = slice(None,-1,None) != slice(0,-1)
# x[slices] with an array of slice() object per dimension works same as python slicing
@register_converter("strided_slice")
class _strided_slice_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.strided_slice
        attrs = {**node.attrs}
        mode = attrs["slice_mode"]
        assert mode == "end", "Only slice_mode = 'end' supported"
        # begin = [int(v) if int(v) != INT_MAX else -1 for v in attrs["begin"]]
        # end = [int(v) if int(v) != INT_MAX else -1 for v in attrs["end"]]
        begin = [int(v) for v in attrs["begin"]]
        end = [int(v) for v in attrs["end"]]
        strides = [int(v) for v in attrs["strides"]]
        axes = [int(v) for v in attrs["axes"]] if "axes" in attrs and attrs["axes"] is not None else None
        # #logger.debug(f"strided_slice init: {begin=} {end=} {strides=} {axes=} {mode=}")

        AnyVal = slice(None)
        shape = shape_of(node.args[0])
        dims = len(shape)
        slices = [AnyVal] * dims
        if axes is not None:
            for ax_, begin_, end_, step_ in zip(axes, begin, end, strides, strict=True):
                assert ax_ < dims, f"invalid dimension {ax_}, must be <{dims}"
                slices[ax_] = slice(begin_, end_, step_)
        else:
            assert (
                len(begin) == dims and len(end) == dims and len(strides) == dims
            ), "slices must be specified for all dimensions"
            for ax_, (begin_, end_, step_) in enumerate(zip(begin, end, strides, strict=True)):
                slices[ax_] = slice(begin_, end_, step_)
        #logger.debug(f"strided_slice: {slices=}")
        self.slices = slices

    def forward(self, x) -> Any:
        return x[self.slices]


@register_converter("nn.batch_flatten")
def _batch_flatten_converter(node: relay.Call):
    # print("batch_flatten")
    return tnn.Flatten(start_dim=1)


###############################################################################
#
# Imaging
#
###############################################################################


def _upsampling_method_converter(relay_method: str | None) -> str:
    if relay_method is None or relay_method == "" or relay_method == "nearest_neighbor":
        method = "nearest"
    elif relay_method == "linear" or relay_method == "bilinear":
        method = "bilinear"
    elif relay_method == "cubic":
        method = "cubic"
    else:
        raise NotImplementedError(f"upsampling relay method: {relay_method}")
    return method


@register_converter("nn.upsampling")
def _upsampling_converter(node: relay.Call):
    # rnn.upsampling
    # F.upsample
    tattrs = {}
    attrs = {**node.attrs}
    for attr, val in attrs.items():
        match attr:
            case "scale_h":
                scale_h = float(val)
            case "scale_w":
                scale_w = float(val)
            case "method":
                tattrs["mode"] = _upsampling_method_converter(val)
            case "align_corners":
                tattrs["align_corners"] = bool(val)
    # tattrs["scale_factor"] = (scale_h, scale_w)
    x_shape = shape_of(node.args[0])
    tattrs["size"] = (int(x_shape[-2] * scale_h), int(x_shape[-1] * scale_w))
    #logger.debug(f'upsampling {x_shape=} to {tattrs["size"]}')
    return tnn.Upsample(**tattrs)


@register_converter("image.resize2d")
class _resize2d_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.image.resize2d
        # F.interpolate
        # tvt.Resize

        tattrs = {}
        attrs = {**node.attrs}
        #logger.debug(f"image resize: {attrs=}")
        for attr, val in attrs.items():
            match attr:
                case "size":
                    tattrs["size"] = tuple(int(v) for v in val)
                case "method":
                    tattrs["mode"] = _upsampling_method_converter(val)
                # case "coordinate_transformation_mode":
                # case "rounding_method":
        tattrs["antialias"] = tattrs["mode"] != "nearest"
        tattrs["align_corners"] = True if tattrs["mode"] != "nearest" else None
        self.tattrs = tattrs

    def forward(self, x) -> Any:
        x = F.interpolate(x, **self.tattrs)
        return x


def flatten_list(x):
    return tuple(chain(*x))


@register_converter("nn.pad")
class _pad_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        attrs = {**node.attrs}
        # print(f"pad {attrs=}")
        # rnn.pad
        # F.pad

        padding = flatten_list(reversed(attrs["pad_width"]))
        self.padding = tuple(int(v) for v in padding)
        # print(f"nn.pad {attrs['pad_width']=} -> {self.padding=}")
        # self.value = float(attrs.get("pad_value", 0))
        self.mode = attrs.get("pad_mode", "constant")
        if self.mode == "edge":
            self.mode = "replicate"

    def forward(self, x, val) -> Any:
        # print(f"pad {val=}")
        return F.pad(x, self.padding, self.mode, float(val))


@register_converter("transpose")
class _transpose_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        self.axes = tuple(int(v) for v in node.attrs["axes"])

    def forward(self, x) -> Any:
        return torch.permute(x, self.axes)


@register_converter("layout_transform")
class _layout_transform_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        src_layout = node.attrs["src_layout"]
        dst_layout = node.attrs["dst_layout"]
        # print(f"layout_transform {src_layout=} -> {dst_layout=}")

        self.axes = (0, 1, 2, 3)  # no change
        if src_layout == "NCHW" and dst_layout == "NHWC":
            self.axes = (0, 2, 3, 1)
        elif src_layout == "OIHW" and dst_layout == "HWIO":
            self.axes = (2, 3, 1, 0)
        elif src_layout == "NHWC" and dst_layout == "NCHW":
            self.axes = (0, 3, 1, 2)
        elif src_layout == "HWIO" and dst_layout == "OIHW":
            self.axes = (3, 2, 0, 1)

    def forward(self, x) -> Any:
        return torch.permute(x, self.axes)


@register_converter("concatenate")
class _concatenate_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.concatenate
        # torch.stack
        # torch.concatenate=torch.cat is inverse of torch.split
        # print(f'concatenate axis={node.attrs["axis"]}')
        self.dim = int(node.attrs["axis"])
        x = node.args[0]
        #logger.debug(f"concatenate #inputs={len(x)} {self.dim=}")

    def forward(self, *x) -> Any:
        return torch.cat(*x, self.dim)


@register_converter("expand_dims")
class _expand_dims_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        self.axis = int(node.attrs["axis"])
        # print(f"expand_dims axis={self.axis}")

    def forward(self, x) -> Any:
        return torch.unsqueeze(x, self.axis)


###############################################################################
#
# Activations and Math
#
###############################################################################

import operator  # noqa: E402


BINARY_OP = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "negative": operator.neg,
    "power": torch.pow,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
}


@register_converter("add")
@register_converter("subtract")
@register_converter("multiply")
@register_converter("divide")
@register_converter("negative")
@register_converter("power")
@register_converter("minimum")
@register_converter("maximum")
# class _binary_op_converter(tnn.Module, RelayFX_Module):
#     def __init__(self, node: relay.Call) -> None:
#         super().__init__()
#         # self.ff = nnq.FloatFunctional()
#         # self.op = getattr(self.ff, node.op.name)
#         self.op = BINARY_OP[node.op.name]
#         # self.rhs=r2t_arr(node.args[1]) if isinstance(node.args[1],relay.Constant) else None


#     def forward(self, lhs, rhs) -> Any:
#         # lhs=args[0]
#         # rhs=args[1] if self.rhs is None else self.rhs
#         return self.op(lhs, rhs)
def _binary_op_converter(node: relay.Call):
    return BINARY_OP[node.op.name]


@register_converter("mean")
class _mean_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.mean
        # torch.mean
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in node.attrs["axis"])
        tattrs["keepdim"] = bool(node.attrs["keepdims"])
        if node.attrs["exclude"]:
            pass
            #logger.warning("mean.exclude=True not supported yet")

    def forward(self, x) -> Any:
        return torch.mean(x, **self.tattrs)


@register_converter("variance")
class _variance_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.variance
        # torch.var
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in node.attrs["axis"])
        tattrs["keepdim"] = bool(node.attrs["keepdims"])
        tattrs["correction"] = int(node.attrs["unbiased"])
        if node.attrs["exclude"]:
            pass
            #logger.warning("variance.exclude=True not supported yet")

    def forward(self, x) -> Any:
        return torch.var(x, **self.tattrs)


@register_converter("sum")
class _sum_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.sum
        # torch.sum
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in node.attrs["axis"])
        tattrs["keepdim"] = bool(node.attrs["keepdims"])
        if node.attrs["exclude"]:
            pass
            #logger.warning("sum.exclude=True not supported yet")

    def forward(self, x) -> Any:
        return torch.sum(x, **self.tattrs)


@register_converter("min")
class _min_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.min
        # torch.min
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in node.attrs["axis"])[0]
        tattrs["keepdim"] = bool(node.attrs["keepdims"])
        if node.attrs["exclude"]:
            pass
            #logger.warning("min.exclude=True not supported yet")

    def forward(self, x) -> Any:
        return fx_wrapped_min(x, **self.tattrs)[0]


@register_converter("max")
class _max_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.max
        # torch.max
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in node.attrs["axis"])[0]
        tattrs["keepdim"] = bool(node.attrs["keepdims"])
        if node.attrs["exclude"]:
            pass
            #logger.warning("max.exclude=True not supported yet")

    def forward(self, x) -> Any:
        return fx_wrapped_max(x, **self.tattrs)[0]


@register_converter("clip")
class _clip_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.clip
        # torch.clip
        self.tattrs = tattrs = {}
        tattrs["min"] = r2t_arr(node.attrs["a_min"])
        tattrs["max"] = r2t_arr(node.attrs["a_max"])

    def forward(self, x) -> Any:
        return torch.clip(x, **self.tattrs)


DTYPE_MAP = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bool": torch.bool,
}


@register_converter("cast")
class _cast_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.cast
        dtype = node.attrs["dtype"]
        # if dtype == "int64":  # no reason to use int64
        #     dtype = "int32"
        self.dtype = DTYPE_MAP[dtype]  # intentionally crash if unsupported type

    def forward(self, x) -> Any:
        return x.type(self.dtype)


@register_converter("argmax")
class _argmax_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.argmax
        attrs = {**node.attrs}
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in attrs["axis"])[0]
        tattrs["keepdim"] = bool(attrs["keepdims"])

    def forward(self, x) -> Any:
        return torch.argmax(x, **self.tattrs)


@register_converter("argmin")
class _argmin_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.argmin
        attrs = {**node.attrs}
        self.tattrs = tattrs = {}
        tattrs["dim"] = tuple(int(v) for v in attrs["axis"])[0]
        tattrs["keepdim"] = bool(attrs["keepdims"])

    def forward(self, x) -> Any:
        return torch.argmin(x, **self.tattrs)


@register_converter("take")
class _take_converter(tnn.Module, RelayFX_Module):
    # relay.take behaves like np.take, unlike torch.take
    # relay/np take(x,indices=0,axis=1) is like x[:,index]
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.take  # similar to np.take
        # torch.index_select

        self.dim = int(node.attrs["axis"])
        #logger.debug(f"take dim={self.dim}")
        # tattrs['index']=r2t_arr(node.attrs['indices'])

    def forward(self, x, indices) -> Any:
        #logger.info(f"take {indices=} {self.dim=}")
        # need to squeeze the dimension because torch returns as 1
        # but relay/np squeezes it
        x = fx_wrapped_index_select(x, self.dim, indices)
        # x = torch.index_select(x, self.dim, indices)
        return x.squeeze(self.dim)


@register_converter("broadcast_to")
class _broadcast_to_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        # relay.broadcast_to
        # shape can be constant or dynamic
        # static shape
        self.shape = tuple(int(v) for v in node.attrs["shape"])

    def forward(self, *args) -> Any:
        x = args[0]
        # dynamic shape
        # shape = args[1] if len(args) > 1 else self.shape

        return torch.broadcast_to(x, self.shape)

@register_converter("nn.leaky_relu")
def _leaky_relu_converter(node: relay.Call):
    # rnn.leaky_relu
    return tnn.LeakyReLU(negative_slope=float(node.attrs["alpha"]))


@register_converter("nn.prelu")
# class _prelu_converter(tnn.Module, RelayFX_Module):
#     def __init__(self, node: relay.Call) -> None:
#         super().__init__()
#         # rnn.prelu
#         # tnn.PReLU
#         # F.prelu

#         self.dim = int(node.attrs["axis"])
#         # alpha=node.args[1].data
#         x_shape = shape_of(node.args[0])
#         alpha_shape = shape_of(node.args[1])  # alpha is not an attribute, it's learned weights
#         #logger.debug(f"[prelu] {self.dim=} {x_shape=} {alpha_shape=}")

#         self.prelu = tnn.PReLU(num_parameters=x_shape[self.dim])


#     def forward(self, x) -> Any:
#         # x=torch.unsqueeze(x,0)
#         return self.prelu(x)
class _prelu_converter(tnn.Module, RelayFX_Module):
    def __init__(self, node: relay.Call) -> None:
        super().__init__()
        self.dim = int(node.attrs["axis"])
        # alpha=node.args[1].data
        self.x_shape = x_shape = shape_of(node.args[0])
        self.batch = 1 if len(x_shape) == 1 else int(x_shape[0])
        alpha_shape = shape_of(node.args[1])  # alpha is not an attribute, it's learned weights
        self.channels = int(np.prod(alpha_shape))
        #logger.info(f"[prelu] {self.dim=} {x_shape=} {alpha_shape=}")

        self.prelu = tnn.PReLU(num_parameters=self.channels)

    def forward(self, x) -> Any:
        # x=torch.unsqueeze(x,0)
        x = torch.reshape(x, (self.batch, self.channels))
        x = self.prelu(x)
        x = torch.reshape(x, self.x_shape)
        return x


@register_converter("nn.softmax")
def _softmax_converter(node: relay.Call):
    return tnn.Softmax(int(node.attrs["axis"]))


@register_converter("nn.dropout")
def _dropout_converter(node: relay.Call):
    return tnn.Dropout(float(node.attrs["rate"]))


UNARY_OP_MODULES = {
    # "nn.relu": tnn.ReLU,
    # "sigmoid": tnn.Sigmoid,
    # "tanh": tnn.Tanh,
}

UNARY_OP_FUNC = {
    "log": torch.log,
    "log2": torch.log2,
    "log10": torch.log10,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "erf": torch.erf,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "sigmoid": F.sigmoid,
    "nn.relu": F.relu,
    "tanh": F.tanh,
}


# @register_converter("nn.relu")
# @register_converter("sigmoid")
# @register_converter("tanh")
def _unary_op_module_converter(node: relay.Call):
    return UNARY_OP_MODULES[node.op.name]()


@register_converter("nn.relu")
@register_converter("tanh")
@register_converter("log")
@register_converter("log2")
@register_converter("log10")
@register_converter("exp")
@register_converter("sqrt")
@register_converter("rsqrt")
@register_converter("erf")
@register_converter("floor")
@register_converter("ceil")
@register_converter("sigmoid")
def _unary_op_converter(node: relay.Call):
    return UNARY_OP_FUNC[node.op.name]


class RelayModule(tnn.Module):
    def __init__(self, mod: tvm.IRModule, params: dict[str, np.ndarray] = None, use_observer=False) -> None:
        super().__init__()
        params = {} if params is None else params
        self.mod = mod
        self.params = params
        self.observer_enabled = use_observer
        self.r2t = {}  
        self.t2r = {} 
        self.torch_nodes = {} 

        self.gm: fx.GraphModule = None

    def forward(self, x) -> Any:
        if not self.gm:
            return

        return self.gm(x)

    def prepare_fx(self) -> fx.GraphModule:
        r2fx = {} 
        tgraph = fx.Graph()
        tmodules = tnn.Module()

        def fvisit(node: Any) -> None:
            match node:
                case relay.Var():
                    name = torch_compatible_name(node.name_hint)
                    fx_node = tgraph.placeholder(name)
                    r2fx[node] = fx_node

                case relay.Constant():
                    torch_buffer_name = f"initializer_{fvisit.const_id}"
                    fvisit.const_id += 1
                    tmodules.register_buffer(torch_buffer_name, r2t_arr(node.data))
                    fx_node = tgraph.get_attr(torch_buffer_name)
                    r2fx[node] = fx_node

                case relay.Call():
                    op_name = node.op.name
                    targs = [r2fx[arg] for arg in node.args]  # if arg in r2fx]
                    name = torch_compatible_name(f"{op_name}/{fvisit.node_id}")
                    fvisit.node_id += 1

                    tmod = get_converter(node)
                    tmod_attrs = {}


                    if op_name in linear_layers:

                        if targs[1].op == "get_attr":
                            fx_w = targs[1].target
                            w = tmodules.get_buffer(fx_w)

                            if "nn.conv" in op_name: 
                                tmod.conv.weight = tnn.Parameter(w)
                            elif op_name == "nn.prelu":
                                tmod.weight = tnn.Parameter(w.flatten(), requires_grad=False)
                            elif op_name == "nn.dense":
                                tmod.weight = tnn.Parameter(w, requires_grad=False)
                            else:
                                pass

                            del targs[1:] 
                        else:
                            fields_to_keep = ("stride", "padding", "dilation", "groups")
                            tmod_attrs = {k: v for k, v in vars(tmod).items() if k in fields_to_keep}
                            tmod = F.conv2d
                            del targs[2:]

                    elif op_name == "variance":
                        del targs[1:]

                    if isinstance(tmod, tnn.Module):
                        tmodules.add_module(name, tmod)
                        args = tuple(targs) if isinstance(targs, list) else targs
                        fx_node = tgraph.call_module(name, args)
                    else:
                        fx_node = tgraph.call_function(tmod, tuple(targs), tmod_attrs)
                    r2fx[node] = fx_node

                case tvm.ir.op.Op(): 
                    op_name = node.name

                case relay.Tuple():
                    targs = [r2fx[arg] for arg in node.fields]
                    r2fx[node] = tuple(targs)

                case relay.TupleGetItem():
                    source = r2fx[node.tuple_value]
                    if node.tuple_value.op.name != "nn.batch_norm":
                        fx_node = tgraph.call_function(operator.getitem, (source, node.index))
                        r2fx[node] = fx_node
                    else:
                        r2fx[node] = source

                case relay.Function():
                    body = r2fx.get(node.body, None)
                    fx_node = tgraph.output(body)
                    r2fx[node] = fx_node

                case _:
                    pass

        mod = self.mod
        mod = relay.transform.InferType()(mod) 

        fvisit.const_id = 0
        fvisit.node_id = 0
        with torch.no_grad():
            relay.analysis.post_order_visit(mod["main"], fvisit)
  
        if self.gm is not None:
            del self.gm

        gm = fx.GraphModule(tmodules, tgraph, "RelayModule")
        tracer = fx.Tracer()
        tgraph = tracer.trace(gm)
        gm = fx.GraphModule(tmodules, tgraph, "RelayModule")
        gm.eval()  

        gm = LintAndRecompile()(gm)
        gm.graph.eliminate_dead_code() 

        input_for_shape_infer = [torch.randn(shape_of(p), dtype=torch.float32) for p in mod["main"].params]
        ShapeProp(gm).propagate(*input_for_shape_infer)

        self.gm = gm
        return gm

    def rebuild_elu(self, gm):
        """replace decomposed ELU by tvm back to torch.elu"""
        from torch.fx import replace_pattern

        def pattern(x, one=1.0):  # x is a Proxy
            x_pos = F.relu(x)
            x_neg = F.relu(-x)
            x_neg = torch.exp(-x_neg) - one
            return x_pos + x_neg

        def replacement(x, one=1.0):
            return F.elu(x)  # DO NOT do in place, this screws quant results

        def pattern_alpha(x, alpha, one=1):
            x_pos = F.relu(x)
            x_neg = F.relu(-x)
            x_neg = alpha * (torch.exp(-x_neg) - one)
            return x_pos + x_neg

        def replacement_alpha(x, alpha, one=1):
            return F.elu(x, alpha)  # DO NOT do in place, this screws quant results

        from torch.fx import symbolic_trace

        # print(gm.graph)
        pattern_graph = symbolic_trace(pattern).graph
        # print(f"ELU {pattern_graph}")

        matches = replace_pattern(gm, pattern, replacement)
        #logger.info(f"ELU {matches=}")
        matches = replace_pattern(gm, pattern_alpha, replacement_alpha)
        #logger.debug(f"ELU alpha {matches=}")
        return gm

    def rebuild_swish(self, gm):  # TODO: use nn.Sigmoid instead of F.sigmoid?
        """replace x*sigmoid(x) with silu(x)."""
        from torch.fx import replace_pattern

        def pattern(x):
            return x * F.sigmoid(x)

        def replacement(x):
            return F.silu(x)  # DO NOT do in place, this screws quant results

        swish_matches = replace_pattern(gm, pattern, replacement)
        #logger.debug(f"swish {swish_matches=}")
        return gm
    

    # we are obliged to do it here because hardXXX ops don't exist relay and were recreated while parsing torch or tf
    # hardtanh(x) is just clip(x, tanh_min, tanh_max)
    def rebuild_hardsigmoid(self, gm):
        from torch.fx import replace_pattern

        def pattern(x):
            # y = torch.add(x, 3)  # TODO: this may be fused in bias of layer before
            y = torch.clip(x, 0, 6)
            y = operator.truediv(y, 6)
            return y

        def replacement(x):
            return F.hardsigmoid(x - 3)

        swish_matches = replace_pattern(gm, pattern, replacement)
        #logger.debug(f"{swish_matches=}")
        return gm

    def rebuild_hardswish(self, gm):
        from torch.fx import replace_pattern

        def pattern(x):
            y = torch.add(x, 3)
            y = torch.clip(y, 0, 6)
            y = operator.truediv(y, 6)  # up to here, it's hardsigmoid pattern
            return operator.mul(y, x)

        def replacement(x):
            return F.hardswish(x)

        swish_matches = replace_pattern(gm, pattern, replacement)
        return gm
    
    # def rebuild_layernorm(self, gm):
    #     from torch.fx import replace_pattern

    #     def pattern(x):
    #         y = torch.add(x, 3)
    #         y = torch.clip(y, 0, 6)
    #         y = operator.truediv(y, 6)  # up to here, it's hardsigmoid pattern
    #         return operator.mul(y, x)

    #     def replacement(x):
    #         return F.hardswish(x)

    #     swish_matches = replace_pattern(gm, pattern, replacement)
    #     return gm