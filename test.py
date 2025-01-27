from dataclasses import dataclass

import torch
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


@dataclass
class Point:
    x: torch.Tensor
    y: torch.Tensor


def point_flatten(p):
    return (p.x, p.y), ()


def point_unflatten(values, _):
    x, y = values
    return Point(x, y)


import torch.utils._pytree as pytree


pytree.register_pytree_node(Point, point_flatten, point_unflatten)


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
    flat_apply = torch.ops.higher_order.flat_apply(SideTableKey(2), SideTableKey(3), ax_1, ay_1, bx_1, by_1);
    return flat_apply
"""

result = gm(ax, ay, bx, by)
assert torch.allclose(result, torch.tensor(5.0))
