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

class TestVersionedLogspaceV8(torch.nn.Module):
    def __init__(self):
        super(TestVersionedLogspaceV8, self).__init__()

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex]):
        c = torch.logspace(a, b, steps=5)
        d = torch.logspace(a, b)
        return c, d

class TestVersionedLogspaceOutV8(torch.nn.Module):
    def __init__(self):
        super(TestVersionedLogspaceOutV8, self).__init__()

    def forward(self, a: Union[int, float, complex], b: Union[int, float, complex], out: torch.Tensor):
        return torch.logspace(a, b, out=out)

class TestVersionedGeluV9(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch._C._nn.gelu(x)

class TestVersionedGeluOutV9(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.zeros_like(x)
        return torch._C._nn.gelu(x, out=out)

class TestVersionedStftV10(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_fft: int, window):
        # calling aten::stft direct instead of torch.functional.stft
        return torch.ops.aten.stft(x, n_fft=n_fft, window=window, return_complex=True)
