import torch

m = torch.nn.ReLU()
m = torch.jit.script(m)
x = torch.Tensor(1)
m(x)
