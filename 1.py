import torch

device = 'cuda'
a = torch.tensor([1, 1, 3, 1], device=device, dtype=torch.float)
b = torch.tensor([10, 2, 12, 4], device=device, dtype=torch.float)
c = torch.tensor([1, 2, 1, 4], device=device, dtype=torch.float)
d = torch.tensor([1, 11, 5, 13], device=device, dtype=torch.float)

res = torch._foreach_max([a, b], [c, d])
print(res)