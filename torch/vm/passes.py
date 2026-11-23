"""The passes of the engine of torch.vm: what its list of passes, ``passes=``, is made of.

A pass rewrites a basic block once its first runs were watched, in the order of
the list, and again until no pass changes the block anymore.
:data:`torch.vm.DEFAULT_PASSES` is the default list.
"""

import dataclasses
from typing import ClassVar


__all__ = [
    "BuiltinPass",
    "DCE",
    "Fold",
    "MethodCalls",
    "NarrowDims",
    "ResolveDispatch",
    "TensorAccessors",
]


class BuiltinPass:
    """A pass of the engine. What it was made with are its fields."""

    # The name it goes by in the engine.
    engine: ClassVar[str]
    # What it is over: one basic block after it was profiled.
    kind: ClassVar[str] = "block"

    def __repr__(self) -> str:
        return f"torch.vm.passes.{type(self).__name__}()"


@dataclasses.dataclass(frozen=True, repr=False)
class ResolveDispatch(BuiltinPass):
    """Operations the profile found to be a single dispatcher call become direct
    calls of the post-dispatch kernel."""

    engine: ClassVar[str] = "resolve_dispatch"


@dataclasses.dataclass(frozen=True, repr=False)
class TensorAccessors(BuiltinPass):
    """``x.shape[i]``, ``x.size(i)``, ``x.dim()`` and ``x.ndim`` on tensors become
    plain SSA integers."""

    engine: ClassVar[str] = "tensor_accessors"


@dataclasses.dataclass(frozen=True, repr=False)
class MethodCalls(BuiltinPass):
    """``x.f(...)`` calls that stay generic are made without creating a bound
    method object."""

    engine: ClassVar[str] = "method_calls"


@dataclasses.dataclass(frozen=True, repr=False)
class NarrowDims(BuiltinPass):
    """A comparison of a dimension with a constant is folded to the outcome last
    observed, under a guard that pins the dimension to the range of sizes with
    that outcome. In front of :class:`Fold`; not a default."""

    engine: ClassVar[str] = "narrow_dims"


@dataclasses.dataclass(frozen=True, repr=False)
class Fold(BuiltinPass):
    """Folds truthiness tests and branches on constants."""

    engine: ClassVar[str] = "fold"


@dataclasses.dataclass(frozen=True, repr=False)
class DCE(BuiltinPass):
    """Drops instructions whose result is not needed."""

    engine: ClassVar[str] = "dce"
