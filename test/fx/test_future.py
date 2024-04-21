# Owner(s): ["module: fx"]

from __future__ import annotations  # type: ignore[attr-defined]

import typing

import torch
from torch.fx import symbolic_trace


class A:
    def __call__(self, x: torch.Tensor):
        return torch.add(x, x)


# No forward references
class M1(torch.nn.Module):
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        return a(x)


# Forward references
class M2(torch.nn.Module):
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        return a(x)


# Non-torch annotation with no internal forward references
class M3(torch.nn.Module):
    def forward(self, x: typing.List[torch.Tensor], a: A) -> torch.Tensor:
        return a(x[0])


# Non-torch annotation with internal forward references
class M4(torch.nn.Module):
    def forward(self, x: typing.List[torch.Tensor], a: A) -> torch.Tensor:
        return a(x[0])


x = torch.rand(2, 3)

ref = torch.add(x, x)

traced1 = symbolic_trace(M1())
res1 = traced1(x, A())
assert torch.all(torch.eq(ref, res1))

traced2 = symbolic_trace(M2())
res2 = traced2(x, A())
assert torch.all(torch.eq(ref, res2))

traced3 = symbolic_trace(M3())
res3 = traced3([x], A())
assert torch.all(torch.eq(ref, res3))

traced4 = symbolic_trace(M4())
res4 = traced4([x], A())
assert torch.all(torch.eq(ref, res4))
