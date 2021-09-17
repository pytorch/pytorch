import torch

device = 'cpu'
t = torch.randn((1,), device=device)
t //= 1

print('meta')
device = 'meta'
t = torch.randn((1,), device=device)
t //= 1
