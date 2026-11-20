"""The passes of the engine of torch.vm: what its list of passes, ``passes=``, is made of.

A pass says what it is over. One over the control-flow graph of a function
(``kind`` is ``"cfg"``) rewrites the function once, before it runs. One over a
basic block (``"block"``) rewrites a block once its first runs were watched, in
the order of the list, and again until no pass changes the block anymore. One
over a graph (``"graph"``) is given the ``torch.fx.GraphModule`` of a graph
that :class:`Graphs` made, and returns what runs in its place.
:data:`torch.vm.DEFAULT_PASSES` is the default list.
"""

import dataclasses
from typing import Any, ClassVar


__all__ = [
    "AOTAutograd",
    "BuiltinPass",
    "DCE",
    "Fold",
    "Graphs",
    "Inductor",
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
    # What it is over: the control-flow graph of the function, once; one
    # basic block after it was profiled; or a graph of the tensor work of a
    # block.
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
    """Makes graphs of the tensor work of a function.

    A graph starts at an operation that the engine calls directly
    (:class:`ResolveDispatch`), or with the function, and goes on along the
    way a call takes through the function: through the branches as they go,
    round the loops as often as they go round, with what sits between the
    operations (reads of globals and attributes, constants, arithmetic on
    sizes, lists that the function fills). Each way a branch went is checked
    on later calls; a call that goes another way gets a path of its own. A
    graph ends in front of everything else: a call of a Python function, an
    operation that writes to one of its arguments or draws random numbers, a
    number read from a tensor and the branch on it, a store into an object.
    ``torch.vm.regions(fn)`` reports the graphs, and what was left out of
    them with the reason. Graphs start in one's own code and in the code of
    models in torch (``torch.nn``, ``torch.nn.functional``, ``torch.optim``),
    not in the rest of the library.

    The engine runs a graph itself: the same kernels, on the same values, in
    the same order, on the autograd tape of eager mode. With this pass alone a
    function computes what it computes without it, bit for bit.

    A graph leaves the engine only for a pass over graphs, which stands behind
    this one in the list: a function ``pass_(gm, input_types) -> gm`` of
    one's own, :class:`AOTAutograd` or :class:`Inductor`. The graph is then
    exported as a ``torch.fx.GraphModule`` of the operators the function
    called (``torch.ops.aten.linear.default``, not what a kernel decomposes
    into; a loop is as many copies of its operations as it went round), for
    the tensors of one call of it. The passes are applied in the
    order of the list, and what the last one returns runs in place of the
    operations for every call whose tensors have the dtypes, devices, sizes,
    strides and ``requires_grad`` of that one, with the same modes, and with
    the same numbers and objects where the operations take any. Another call
    is exported in its turn; so is a call under another dtype of autocast,
    or with deterministic algorithms asked for. A call under a torch
    function mode, a dispatch mode or a transform of ``torch.func`` runs the
    operations as they are, so that those see them. With a pass over graphs
    in the list and this pass missing, it is taken to be there, in front of
    the first of them.

    What an operation takes of the values the function reads (a number kept
    by a module, a global, a variable of an enclosing function) is checked
    on every call, where the graph takes it: a call that finds another
    value is not run by what was exported for the old one. A number that is
    another on every call (a rate on a schedule) is exported three times and
    no more: the calls of that kind then run the operations as they are. What the
    function does between its operations runs in front of what was exported
    or behind it only where that runs no Python code; a read that would call
    a property or a ``__getattr__`` leaves the graph to the engine.

    Args:
        min_ops: a path with fewer operations than this is no graph: it is
            not reported as one, and not exported.
        max_variants: how many exported forms a graph has, one per kind of
            call; the form that ran last is tried first. The calls of
            further kinds run its operations as they are.
    """

    engine: ClassVar[str] = "graphs"

    min_ops: int = 4
    max_variants: int = 8

    def __post_init__(self) -> None:
        for name, least in (("min_ops", 1), ("max_variants", 0)):
            given = getattr(self, name)
            if isinstance(given, bool) or not isinstance(given, int):
                kind = type(given).__name__
                raise TypeError(
                    f"Graphs takes {name} as a number, not {kind}: what compiles "
                    "its graphs is a pass of its own, behind it in the list "
                    "(torch.vm.passes.Inductor())"
                )
            if given < least:
                raise ValueError(f"Graphs({name}={given}): at least {least}")


@dataclasses.dataclass(frozen=True, repr=False)
class AOTAutograd(BuiltinPass):
    """A pass over a graph: AOTAutograd traces the graph and its backward ahead
    of time, and the eager kernels run both. With gradients enabled, the
    tensors a graph returns have one compiled autograd function as their
    ``grad_fn``.

    It is called as every pass over a graph is, ``pass_(gm, input_types)``,
    and returns a graph whose forward calls what it compiled. It compiles
    for the tensors of the call the graph was exported on. A pass that
    compiles is the last pass over a graph in the list, and there is one at
    most. ``torch._functorch.aot_autograd`` is imported when the pass first
    runs.
    """

    engine: ClassVar[str] = "aot_autograd"
    kind: ClassVar[str] = "graph"
    # What it runs is a trace of the operations, decomposed.
    exact: ClassVar[bool] = False
    compiles: ClassVar[bool] = True

    def __call__(self, gm: Any, input_types: list[Any]) -> Any:
        from torch._dynamo.backends.debugging import boxed_nop
        from torch._functorch.aot_autograd import aot_module_simplified

        from . import _export

        # What the graph writes to its inputs is written back behind it,
        # once: kept in the graph, the write moves the version counter of
        # an input twice.
        compiled = aot_module_simplified(
            gm,
            _export.examples(input_types),
            fw_compiler=boxed_nop,
            keep_inference_input_mutations=False,
        )
        return _export.calling(gm, compiled)


@dataclasses.dataclass(frozen=True, repr=False)
class Inductor(BuiltinPass):
    """A pass over a graph: Inductor compiles the graph, and with gradients
    enabled its backward, which AOTAutograd traces for it.

    This is the pass that makes a function faster than its kernels allow, and
    one that may change its numerics: what a graph compiler makes of the
    operations is not what eager runs. It is called as every pass over a
    graph is, ``pass_(gm, input_types)``, and returns a graph whose forward
    calls what it compiled. It compiles for the tensors of the call the graph
    was exported on. A pass that compiles is the last pass over a graph in
    the list, and there is one at most. ``torch._inductor`` is imported when
    the pass first runs.
    """

    engine: ClassVar[str] = "inductor"
    kind: ClassVar[str] = "graph"
    # The kernels of a graph compiler are not the ones eager runs.
    exact: ClassVar[bool] = False
    compiles: ClassVar[bool] = True

    def __call__(self, gm: Any, input_types: list[Any]) -> Any:
        import torch._inductor

        from . import _export

        compiled = torch._inductor.compile(gm, _export.examples(input_types))
        return _export.calling(gm, compiled)


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
