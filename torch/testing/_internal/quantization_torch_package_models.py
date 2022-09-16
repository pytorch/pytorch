import math

import torch
import torch.nn as nn


class LinearReluFunctionalChild(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(N, N))
        self.b1 = nn.Parameter(torch.zeros(N))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        x = torch.nn.functional.relu(x)
        return x

class LinearReluFunctional(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.child = LinearReluFunctionalChild(N)
        self.w1 = nn.Parameter(torch.empty(N, N))
        self.b1 = nn.Parameter(torch.zeros(N))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = self.child(x)
        x = torch.nn.functional.linear(x, self.w1, self.b1)
        x = torch.nn.functional.relu(x)
        return x
