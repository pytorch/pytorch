import torch
from typing import Union

class TestVersionedDivTensorExampleV7(torch.nn.Module):
    def __init__(self):
        super(TestVersionedDivTensorExampleV7, self).__init__()

    def forward(self, a, b):
        result_0 = a / b
        result_1 = torch.div(a, b)
        result_2 = a.div(b)
        return result_0, result_1, result_2

class TestVersionedLinspaceV7(torch.nn.Module):
    def __init__(self):
        super(TestVersionedLinspaceV7, self).__init__()

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        c = torch.linspace(a, b, steps=5)
        d = torch.linspace(a, b)
        return c, d

class TestVersionedLinspaceOutV7(torch.nn.Module):
    def __init__(self):
        super(TestVersionedLinspaceOutV7, self).__init__()

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
        return torch.linspace(a, b, out=out)


class TestVersionedSvdV7(torch.nn.Module):
    def __init__(self):
        super(TestVersionedSvdV7, self).__init__()

    def forward(self, a):
        u, s, v = torch.svd(a)
        return u
