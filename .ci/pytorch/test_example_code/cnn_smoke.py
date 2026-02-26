r"""
It's used to check basic rnn features with cuda.
For example, it would throw exception if some components are missing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        output = self.pool(F.relu(self.conv(inputs)))
        output = output.view(1)
        return output


# Mock one infer
device = torch.device("cuda:0")
net = SimpleCNN().to(device)
net_inputs = torch.rand((1, 1, 5, 5), device=device)
outputs = net(net_inputs)
print(outputs)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

# Mock one step training
label = torch.full((1,), 1.0, dtype=torch.float, device=device)
loss = criterion(outputs, label)
loss.backward()
optimizer.step()
