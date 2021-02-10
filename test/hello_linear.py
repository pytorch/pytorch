import torch

x = torch.rand([20])
y = torch.rand([20, 20])
x = x.reshape(-1, x.size(x.dim() - 1))
out = torch._C._nn.linear(x, y)
