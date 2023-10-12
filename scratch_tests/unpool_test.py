import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm

def test_max_unpool2d():
    batch = torch.randn(16, 3, 15, 15)

    max_pooler2d = torch.nn.MaxPool2d(3, 3, return_indices=True)

    pooled_cpu, indices_cpu = max_pooler2d(batch)
    pooled_mps = pooled_cpu.detach().clone().to("mps")
    indices_mps = indices_cpu.detach().clone().to("mps")

    max_unpooler2d_cpu = torch.nn.MaxUnpool2d(3, stride=3)
    max_unpooler2d_mps = torch.nn.MaxUnpool2d(3, stride=3).to('mps')

    output_cpu = max_unpooler2d_cpu(pooled_cpu, indices_cpu)
    output_mps = max_unpooler2d_mps(pooled_mps, indices_mps)

    assert torch.equal(output_cpu, output_mps.to('cpu'))
    assert output_cpu.size() == output_mps.size()

def loss_fn(output):
    return output.sum()

class Unpooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 2)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.lin = nn.Linear(1200, 1200)

    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        # x = self.unpool(x, indices, output_size=(19, 19))
        # x = x.flatten(start_dim=1)
        # x = self.lin(x)
        return x

unpool1 = nn.MaxUnpool1d(2, 2).to('mps')
input = torch.Tensor([[7, 10, 5], [2, 2, 2]]).to('mps')
indices = torch.Tensor([0, 2]).to(torch.long).to('mps')

pooler = Unpooler()
pooler = pooler.to('mps')

optimizer = torch.optim.Adam(pooler.parameters())
inputs = torch.rand((10000, 128, 3, 20, 20)).to(torch.float)

losses = []

for input in tqdm(inputs):
    input = input.to('mps')
    result = pooler(input)
    # pooler.zero_grad()
    # loss = loss_fn(result)
    # loss.backward()
    # optimizer.step()
    # losses.append(loss.item())
