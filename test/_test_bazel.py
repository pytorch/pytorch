# Owner(s): ["module: bazel"]

"""
This test module contains a minimalistic "smoke tests" for the bazel build.

Currently it doesn't use any testing framework (i.e. pytest)
TODO: integrate this into the existing pytorch testing framework.

The name uses underscore `_test_bazel.py` to avoid globbing into other non-bazel configurations.
"""

import torch


def test_sum() -> None:
    assert torch.eq(
        torch.tensor([[1, 2, 3]]) + torch.tensor([[4, 5, 6]]), torch.tensor([[5, 7, 9]])
    ).all()


def test_simple_compile_eager() -> None:
    def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    opt_foo1 = torch.compile(foo, backend="eager")
    # just check that we can run without raising an Exception
    assert opt_foo1(torch.randn(10, 10), torch.randn(10, 10)) is not None


test_sum()
test_simple_compile_eager()
