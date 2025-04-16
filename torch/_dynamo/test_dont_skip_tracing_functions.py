"""
Functions used to test torch._dynamo.dont_skip_tracing.
This file is located in torch/_dynamo so that it is skipped by trace rules.
There is a special rule in trace_rules that doesn't skip this file when
dont_skip_tracing is active.
"""

import torch


def f1(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def f2(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def f3(x: torch.Tensor) -> torch.Tensor:
    return f2(x)


def f4(x: torch.Tensor) -> torch.Tensor:
    x = f5(x, 1)
    x = torch._dynamo.dont_skip_tracing(f6)(x)
    x = f5(x, 8)
    return x


def f5(x: torch.Tensor, n: int) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return x + n
    return x


def f6(x: torch.Tensor) -> torch.Tensor:
    x = f5(x, 2)
    torch._dynamo.graph_break()
    x = f5(x, 4)
    return x
