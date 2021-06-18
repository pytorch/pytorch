import torch
import os
import os.path


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
