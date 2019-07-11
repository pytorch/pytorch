import torch

with torch.no_grad():
    a = torch.rand([1])
    b = torch.ones([1])
    for _ in range(1000000):
        a = torch.add(a, b)
