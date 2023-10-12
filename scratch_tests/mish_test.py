import torch
from tqdm import tqdm

torch.manual_seed(0)

device = 'mps'

a = torch.rand((1000, 1000, 5)).to(device).requires_grad_(True)

mish = torch.nn.Mish()

for i in tqdm(range(1000)):
    b = mish(a)

print(b[0, 0:2], a[0, 0:2])

loss = b.sum(dim=-1)
loss.backward()
print(a.grad[0, 0:2])
