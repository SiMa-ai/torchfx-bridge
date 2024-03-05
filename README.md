Converts models from onnx/tensorflow to torch fx. 

Usage:

```
python main.py --model_path path_to_model (required) --format onnx/tf (required) --save_path path_to_saved_model (with .pth extension) (required) --simplify True/False (optional, default True) --save_tvm True/False (optional, default False)
```

Saves a torch fx .pth model and a .pkl with meta information.

To reload a saved model use:


```
model = torch.load(path_to_model.pth)

meta_path = Path(path_to_model_meta.pkl')

with meta_path.open("rb") as f:

    meta_list = pkl.load(f)

for i, node in enumerate(model.graph.nodes):

    node.meta = meta_list[i]
```