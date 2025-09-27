import dataclasses
import torch

from torch._dynamo.source import GuardSource, LocalSource, Source, UserSpecifiedSymIntSource
from torch._dynamo.testing import CompileCounter
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._sympy.numbers import int_oo


"""
- naming: overwrite symbol name
- base: min/max
- create custom source type
- propagate guards? (future work)
"""


def create_user_symint(shape_env, name, val, min=None, max=None):
    sym = shape_env.create_symbol(
        val,
        UserSpecifiedSymIntSource(name),
        dynamic_dim=DimDynamic.DYNAMIC,
        positive=False,
    )
    sym.name = name
    shape_env._constrain_range(sym, min=int_oo if min is None else min, max=int_oo if max is None else max)
    symint = shape_env.create_symintnode(
        sym,
        hint=val,
    )
    return symint


if __name__ == "__main__":
    cnt = CompileCounter()

    @torch.compile(dynamic=True, backend=cnt)
    def f(x):
        z = abs(x)
        return z + 1, torch.empty(z, z*2)

    shape_env = ShapeEnv()
    val = 4
    s0 = create_user_symint(shape_env, "s0", val, min=-16, max=16)
    assert f(s0)[0] == 5
    assert list(f(s0)[1].shape) == [4, 8]
    assert f(-16)[0] == 17
    assert f(10)[0] == 11
    assert list(f(10)[1].shape) == [10, 20]
    assert cnt.frame_count == 1

    f(32)  # out of range
    assert cnt.frame_count == 2
