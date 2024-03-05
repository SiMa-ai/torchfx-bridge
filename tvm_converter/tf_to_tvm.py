from tvm import relay

from pathlib import Path
import cloudpickle as pkl

from simplify_relay import simplify_relay

def tf_to_tvm(path: str, simplify = True, output_path: str = None, save_tvm: bool = False) -> relay.Function:
    from tensorflow import keras as tf_keras

    keras_model = tf_keras.models.load_model(path)

    name = keras_model.layers[0].name
    shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
    shape = [1 if dim is None else dim for dim in shape]
    shape = [shape[0], shape[3], shape[1], shape[2]]

    shape_dict = {name: shape}
    mod, params = relay.frontend.from_keras(keras_model, shape=shape_dict, layout="NCHW")

    del keras_model
    del tf_keras

    if simplify:
        mod = simplify_relay(mod, params)
    
    if save_tvm:
        relay_path = Path(output_path)

        with relay_path.open("wb") as f:
            pkl.dump(mod, f, protocol=-1)

    return mod