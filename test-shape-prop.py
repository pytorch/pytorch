import torch

m = torch.jit.load("/home/ivankobzarev/rsync/nnc/mnetv3-large/mobilenet_v3_large_opt.pt").eval()

inputs = list(m.graph.inputs())
size = [1, 3, 224, 224]
inputs[1].setType(inputs[1].type().with_sizes(size))
torch._C._jit_pass_propagate_shapes_on_graph(m.graph)
print(m.graph)
