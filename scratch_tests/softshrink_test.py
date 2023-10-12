import torch
from tqdm import tqdm

torch.manual_seed(0)

device = 'mps'

a = torch.tensor([-1, -2, 0, 5, 5, 0, 5, 1, -10, 1.5]).to(device)
b = a + 50
print(a)

softshrink = torch.nn.Softshrink(0.5)

print(softshrink(a))

layer = torch.nn.Linear(1000, 1000).to(device)
layer2 = torch.nn.Linear(1000, 1).to(device)

optimizer = torch.optim.Adam(layer.parameters())

inputs = torch.rand((1000, 1000, 1000)).to(device)

for input in tqdm(inputs):
    a = input.to(device)
    out = layer(a)
    out = softshrink(out)
    out = layer2(out)
    loss = (out.sum())**2
    loss.backward()
    optimizer.step()
