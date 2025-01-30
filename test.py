# mypy: allow-untyped-defs
from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from functorch import make_fx
from torch._higher_order_ops.flat_apply import plop_in_graph


@plop_in_graph
def distance(a, b, norm):
    if norm.typ == "l2":
        return torch.sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    elif norm.typ == "l1":
        return (a.x - b.x).abs() + (a.y - b.y).abs()


@dataclass
class Norm:
    typ: str


pytree.register_constant(Norm)


@dataclass
class Point:
    x: torch.Tensor
    y: torch.Tensor


pytree.register_dataclass(Point)


def f(ax, ay, bx, by):
    a = Point(ax, ay)
    b = Point(bx, by)
    return distance(a, b, Norm("l2"))


ax = torch.tensor(0.0)
ay = torch.tensor(0.0)
bx = torch.tensor(3.0)
by = torch.tensor(4.0)

result = f(ax, ay, bx, by)
assert torch.allclose(result, torch.tensor(5.0))

gm = make_fx(f)(ax, ay, bx, by)
print(gm.code)
"""
def forward(self, ax_1, ay_1, bx_1, by_1):
    func_spec0 = self.func_spec0
    in_spec0 = self.in_spec0
    flat_apply = torch.ops.higher_order.flat_apply(func_spec0, in_spec0, ax_1, ay_1, bx_1, by_1);
    return flat_apply
"""

result = gm(ax, ay, bx, by)
assert torch.allclose(result, torch.tensor(5.0))
