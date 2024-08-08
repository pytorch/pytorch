# mypy: allow-untyped-defs
import functools

import torch

def test_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1

    return wrapper

class Decorator(torch.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """

    @test_decorator
    def forward(self, x, y):
        return x + y

example_args = (torch.randn(3, 2), torch.randn(3, 2))
model = Decorator()
