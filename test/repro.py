import torch
a = torch.rand(3, 3)
a.mul_(2)
print(a._version)
