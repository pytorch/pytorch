import os  # noqa: F401
import os.path  # noqa: F401

import torch


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return os.path.abspath("test")
