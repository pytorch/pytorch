import torch
from tqdm import tqdm

torch.manual_seed(0)

device = 'cpu'

a = torch.tensor([-1, -2, 0, 5, 5, 0, 5, 1, -10, 1.5]).to(device)
b = a + 50
print(a)

print(torch.heaviside(a, b))

layer = torch.nn.Linear(1000, 10).to(device)
layer2 = torch.nn.Linear(10, 1).to(device)

optimizer = torch.optim.Adam(layer.parameters())

inputs = torch.rand((1000, 1000, 1000)).to(device)
values = torch.rand((1000)).to(device)
small_value = torch.rand((10)).to(device)

for input in tqdm(inputs):
    a = input.to(device)
    out = layer(a)
    out = torch.heaviside(out, small_value)
    out = layer2(out)
    loss = (out.sum())**2
    loss.backward()
    optimizer.step()

print(out)
