import functools

import torch

from torch._export.db.case import export_case


def test_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1

    return wrapper


@export_case(
    example_inputs=(torch.ones(3, 2), torch.ones(3, 2)),
)
class Decorator(torch.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """

    @test_decorator
    def forward(self, x, y):
        return x + y
