import torch
import yaml


class SumMod(torch.nn.Module):
    def forward(self, inp):
        return torch.sum(inp)
