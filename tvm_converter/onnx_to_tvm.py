from tvm import relay
import onnx

from pathlib import Path
import cloudpickle as pkl

from simplify_relay import simplify_relay

def onnx_to_tvm(path: str, simplify = True, output_path: str = None, save_tvm: bool = False) -> relay.Function:

    onnx_model = onnx.load(path)
    
    mod, params = relay.frontend.from_onnx(onnx_model)

    if simplify:
        mod = simplify_relay(mod, params)

    if save_tvm:
        relay_path = Path(output_path)

        with relay_path.open("wb") as f:
            pkl.dump(mod, f, protocol=-1)
    
    return mod