import torch
from torch.fx import wrap

wrap("a_non_torch_leaf")


class SimpleTest(torch.nn.Module):
    def forward(self, x):
        x = a_non_torch_leaf(x, x)
        return torch.relu(x + 3.0)


def a_non_torch_leaf(a, b):
    return a + b
