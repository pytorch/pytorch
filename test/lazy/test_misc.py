# Owner(s): ["oncall: jit"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.testing._internal.common_utils import run_tests, TestCase

import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
torch._lazy.ts_backend.init()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TestMisc(TestCase):
    def test_cache(self):
        device = "lazy"
        model = Net().to(device)
        model.train()
        lr = 0.01
        momentum = 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        def run_once(model, optimizer):
            input = torch.ones(1, 1, 28, 28, device=device, requires_grad=True)
            target = torch.ones(1, device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            torch._lazy.mark_step()
            _ = loss.cpu()

        for i in range(3):
            run_once(model, optimizer)
            self.assertEqual(None, torch._lazy.metrics.counter_value('UncachedCompile'))
            self.assertEqual(i, torch._lazy.metrics.counter_value('CachedCompile'))

if __name__ == '__main__':
    run_tests()
