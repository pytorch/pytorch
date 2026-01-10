import torch
from .subpackage_0.subsubpackage_0 import important_string


class ImportsDirectlyFromSubSubPackage(torch.nn.Module):
    key = important_string

    def forward(self, inp):
        return torch.sum(inp)
