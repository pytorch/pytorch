import torch
from tqdm import tqdm

device = 'mps'

t = torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j, 5 - 1j, 0], device='cpu')
# t = torch.tensor([1.1, 2.1, 3])

# make t a 2d tensor with each row repeating
t = torch.hstack([t] * 2)
print(t.reshape((2, 2, 3)).sgn())

t = torch.hstack([t] * 2000)
t = torch.vstack([t] * 1000)
t = t.to(device)

for i in tqdm(range(5000)):
    a = t.sgn()


