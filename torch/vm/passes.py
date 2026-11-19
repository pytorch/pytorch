"""The passes of the engine of torch.vm: what its list of passes, ``passes=``, is made of.

A pass says what it is over. One over the control-flow graph of a function
(``kind`` is ``"cfg"``) rewrites the function once, before it runs. One over a
basic block (``"block"``) rewrites a block once its first runs were watched, in
the order of the list, and again until no pass changes the block anymore.
:data:`torch.vm.DEFAULT_PASSES` is the default list.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar


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
    """``for i in range(k)`` with a constant ``k`` of at most 8 is replaced by
    ``k`` copies of its body, under a guard on ``range``. A loop whose copies
    would come to more than 600 instructions is left as it is. Applied after
    :class:`Inline`."""

    engine: ClassVar[str] = "unroll"
    kind: ClassVar[str] = "cfg"


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
    """Hands the tensor work of each basic block to a graph compiler.

    ``"inductor"`` traces every run of instructions of a block that a trace can
    capture, together with what those instructions call, into one ATen graph,
    and compiles it with AOTAutograd and Inductor: with gradients enabled, the
    tensors such a region returns have one compiled autograd function as their
    ``grad_fn``. The outermost region that traces is the one that is compiled;
    where an instruction does not trace, the VM runs it and the blocks of what
    it calls are tried in turn. ``"aot_eager"`` builds the same graphs and runs
    them with the eager kernels. A callable
    ``backend(graph_module, example_inputs)`` returning a callable is used as
    given. ``torch.vm.regions(fn)`` reports what was compiled and what was left
    to the VM, with the reason. See ``torch/vm/_regions.py`` for what a
    compiled region checks on every call and what it does not.

    This is the pass that makes a function faster than its kernels allow, and
    the one that may change its numerics: what a graph compiler makes of the
    operations is not what eager runs. It is not in
    :data:`torch.vm.DEFAULT_PASSES`: it is asked for by putting it in the list.
    Wherever it stands there, it sees the instructions of a block as the
    function wrote them, and the other passes over a block get what it leaves
    to the VM.

    Args:
        backend: see above.
        min_ops: a region with fewer tensor operations than this is left to the
            VM, whose direct kernel calls are cheaper than entering a compiled
            graph.
        max_variants: how many graphs a region gives the backend before further
            kinds of inputs (dtypes, ranks, modules, modes) run its
            instructions as they are. Inputs that trace to the same graph, as
            the layers of a model do when a region is called with one after the
            other, share what the backend made of it, and sizes that differ
            from call to call are left open in one graph rather than compiled
            one by one.
        freeze: with ``"inductor"``, build the parameters and buffers of the
            modules a region reads into its graph when the call is one of
            inference (no gradients recorded, every module in eval mode), so
            that Inductor folds what follows from them alone (a batch norm into
            the convolution before it) and lays convolution weights out for the
            kernels. Such a graph is checked on every call: a tensor that was
            written to in place since (by an optimizer, say) or replaced makes
            the region compile again with the tensors of that call, and a
            region this happens to twice takes its weights as inputs from then
            on. A tensor the region itself writes is not built in. A write
            through ``.data`` does not count as one to the tensor, and is not
            seen. ``regions()`` reports ``frozen`` per region. On by default
            (``TORCH_VM_FREEZE=0`` turns it off).
    """

    engine: ClassVar[str] = "graphs"

    backend: str | Callable[..., Any] = "inductor"
    min_ops: int = 4
    max_variants: int = 32
    freeze: bool | None = None


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
