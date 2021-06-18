import torch
import os  # noqa: F401
import os.path  # noqa: F401


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
