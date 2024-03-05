import argparse

from tvm_converter.onnx_to_tvm import onnx_to_tvm
from tvm_converter.tf_to_tvm import tf_to_tvm
from tvm_to_torchfx import tvm_to_torchfx

import torch

from pathlib import Path
import cloudpickle as pkl

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default=None,
                        help='model path.')
    parser.add_argument('--format', type=str, required=True, default=None,
                        help='model format.')
    parser.add_argument('--save_path', type=str, required=True, default=None,
                        help='save path.')
    parser.add_argument('--simplify', type=str, required=False, default=True,
                        help='apply TVM simplifications.')
    parser.add_argument('--save_tvm', type=str, required=False, default=False,
                        help='save TVM pkl file.')
    
    return parser.parse_args()

def save_node_meta(torch_fx_model, save_path):
    meta_list = []
    for node in torch_fx_model.graph.nodes:

        node.meta.pop('nn_module_stack', None)
        meta_list.append(node.meta)
    
    meta_path = save_path[:-4]+'_meta.pkl'
    meta_path = Path(meta_path)

    with meta_path.open("wb") as f:
        pkl.dump(meta_list, f, protocol=-1)

if __name__ == '__main__':

    # Parse arguments
    args = argument_handler()
    
    if args.format == 'onnx':
        tvm_model = onnx_to_tvm(args.model_path, simplify = args.simplify, save_tvm = args.save_tvm, output_path = args.save_path[:-4]+'_tvm.pkl')
    elif args.format == 'tf':
        tvm_model = tf_to_tvm(args.model_path, simplify = args.simplify, save_tvm = args.save_tvm, output_path = args.save_path[:-4]+'_tvm.pkl')

    torch_fx_model = tvm_to_torchfx(tvm_model)

    torch.save(torch_fx_model, args.save_path)

    save_node_meta(torch_fx_model, args.save_path)
