```
model = torch.load('/home/shreyas.kera/saved_torch_fx_models/mnist_model.pth')

meta_path = Path('/home/shreyas.kera/saved_torch_fx_models/mnist_model_meta.pkl')

with meta_path.open("rb") as f:

    meta_list = pkl.load(f)

for i, node in enumerate(model.graph.nodes):

    node.meta = meta_list[i]
```