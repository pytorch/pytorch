"""The passes of the engine of torch.vm: what its list of passes, ``passes=``, is made of.

A pass says what it is over. One over the control-flow graph of a function
(``kind`` is ``"cfg"``) rewrites the function once, before it runs. One over a
basic block (``"block"``) rewrites a block once its first runs were watched, in
the order of the list, and again until no pass changes the block anymore.
:data:`torch.vm.DEFAULT_PASSES` is the default list.
"""

import dataclasses
from typing import ClassVar


__all__ = [
    "BuiltinPass",
    "DCE",
    "Fold",
    "Graphs",
    "Inline",
    "MethodCalls",
    "ModuleShortcuts",
    "NarrowDims",
    "ResolveDispatch",
    "TensorAccessors",
    "Unroll",
]


class BuiltinPass:
    """A pass of the engine. What it was made with are its fields."""

    # The name it goes by in the engine.
    engine: ClassVar[str]
    # What it is over: the control-flow graph of the function, once, or one
    # basic block after it was profiled.
    kind: ClassVar[str] = "block"
    # A pass of the engine leaves every result what the interpreter computes.
    exact: ClassVar[bool] = True

    def __repr__(self) -> str:
        fields = dataclasses.fields(self)  # type: ignore[arg-type]
        made = [f for f in fields if getattr(self, f.name) != f.default]
        inner = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in made)
        return f"torch.vm.passes.{type(self).__name__}({inner})"


@dataclasses.dataclass(frozen=True, repr=False)
class Inline(BuiltinPass):
    """Calls of Python functions defined in the same module are replaced by their
    graph, under a guard on the callee; the call itself stays as the other side
    of that guard."""

    engine: ClassVar[str] = "inline"
    kind: ClassVar[str] = "cfg"


@dataclasses.dataclass(frozen=True, repr=False)
class Unroll(BuiltinPass):
    """``for i in range(k)`` with a constant ``k`` of at most ``max_trip_count``
    is replaced by ``k`` copies of its body, under a guard on ``range``. A loop
    whose copies would come to more than 600 instructions is left as it is.
    Applied after :class:`Inline`."""

    engine: ClassVar[str] = "unroll"
    kind: ClassVar[str] = "cfg"

    max_trip_count: int = 8

    def __post_init__(self) -> None:
        most = self.max_trip_count
        if isinstance(most, bool) or not isinstance(most, int):
            raise TypeError(
                f"Unroll takes a number of iterations, not {type(most).__name__}"
            )
        if most < 1:
            raise ValueError(
                f"Unroll({most}): at least one iteration; to unroll nothing, leave the pass out of the list"
            )


@dataclasses.dataclass(frozen=True, repr=False)
class ModuleShortcuts(BuiltinPass):
    """The code of ``nn.Module`` in front of every call of a module and every read
    of a parameter, buffer or submodule is skipped whenever a few checks show
    what it would do: a call of a module without hooks goes straight to its
    ``forward``, and the three dicts are looked at without calling
    ``Module.__getattr__``. ``Module.__call__`` then does not appear on the
    stack."""

    engine: ClassVar[str] = "shortcuts"
    kind: ClassVar[str] = "cfg"


@dataclasses.dataclass(frozen=True, repr=False)
class Graphs(BuiltinPass):
    """Makes graphs of the tensor work of each basic block.

    A graph is a run of the operations of a block that the engine calls
    directly (:class:`ResolveDispatch`), with what sits between them: reads of
    globals and attributes, constants, arithmetic on sizes. It ends in front
    of everything else: a call of a Python function, an operation that writes
    to one of its arguments or draws random numbers, a number read from a
    tensor. ``torch.vm.regions(fn)`` reports the graphs, and what was left out
    of them with the reason.

    The engine runs a graph itself: the same kernels, on the same values, in
    the same order, on the autograd tape of eager mode. With this pass alone a
    function computes what it computes without it, bit for bit.

    Args:
        min_ops: a run with fewer operations than this is no graph, and is not
            reported as one.
    """

    engine: ClassVar[str] = "graphs"

    min_ops: int = 4

    def __post_init__(self) -> None:
        given = self.min_ops
        if isinstance(given, bool) or not isinstance(given, int):
            kind = type(given).__name__
            raise TypeError(f"Graphs takes min_ops as a number, not {kind}")
        if given < 1:
            raise ValueError(f"Graphs(min_ops={given}): at least 1")


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
