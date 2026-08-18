# Owner(s): ["oncall: package/deploy"]

import torch
from torch.fx import wrap


wrap("a_non_torch_leaf")


class ModWithSubmod(torch.nn.Module):
    def __init__(self, script_mod):
        super().__init__()
        self.script_mod = script_mod

    def forward(self, x):
        return self.script_mod(x)


class ModWithTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return self.tensor * x


class ModWithSubmodAndTensor(torch.nn.Module):
    def __init__(self, tensor, sub_mod):
        super().__init__()
        self.tensor = tensor
        self.sub_mod = sub_mod

    def forward(self, x):
        return self.sub_mod(x) + self.tensor


class ModWithTwoSubmodsAndTensor(torch.nn.Module):
    def __init__(self, tensor, sub_mod_0, sub_mod_1):
        super().__init__()
        self.tensor = tensor
        self.sub_mod_0 = sub_mod_0
        self.sub_mod_1 = sub_mod_1

    def forward(self, x):
        return self.sub_mod_0(x) + self.sub_mod_1(x) + self.tensor


class ModWithMultipleSubmods(torch.nn.Module):
    def __init__(self, mod1, mod2):
        super().__init__()
        self.mod1 = mod1
        self.mod2 = mod2

    def forward(self, x):
        return self.mod1(x) + self.mod2(x)


class SimpleTest(torch.nn.Module):
    def forward(self, x):
        x = a_non_torch_leaf(x, x)
        return torch.relu(x + 3.0)


def a_non_torch_leaf(a, b):
    return a + b
