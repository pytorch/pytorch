import yaml

import torch


class SumMod(torch.nn.Module):
    def forward(self, inp):
        return torch.sum(inp)
