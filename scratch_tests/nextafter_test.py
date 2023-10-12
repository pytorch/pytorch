import torch
from tqdm import tqdm

dt = torch.float32
device = 'cpu'
size = 100000000
a = torch.rand((size), dtype=dt).to(device) * 1000
b = torch.rand((size), dtype=dt).to(device) * 1000
a[0] = float('nan')

print(torch.nextafter(a, b))
print(a)
print(b)

a = torch.rand((size), dtype=dt).to(device)
b = torch.rand((size), dtype=dt).to(device)

for i in tqdm(range(1000)):
    out = torch.nextafter(a, b)

print((out == a).sum())
