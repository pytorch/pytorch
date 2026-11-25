# mypy: allow-untyped-defs
"""torch.vm's own passes.

Each is a callable whose signature says its kind, like any pass: a cfg pass
takes ``(cfg)``, a block pass ``(gm, input_types)``.

The VM's optimizer (``Split``, ``PinCalls``, ``UseAten``, ``LiftGuard``, and
the opt-in ``Elide``) and region formation (``PureCalls``, ``Spans``,
``Regions``, ``BypassCold``) run inside the executor; their cfg passes select
and configure them. ``Autograd`` and ``Partition`` mark where block passes stop
seeing a block's forward and start seeing its joint graph, then its forward
and backward graphs. ``Inductor`` is a block pass that compiles the graph.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch.vm._cfg import Cfg
from torch.vm._graph import below_autocast, lower_to


__all__ = [
    "Autograd",
    "BypassCold",
    "Elide",
    "Inductor",
    "LiftGuard",
    "OPTIMIZER",
    "Partition",
    "PinCalls",
    "PureCalls",
    "Regions",
    "Skip",
    "Spans",
    "Split",
    "ThreadRng",
    "UseAten",
    "static_parameters",
]


@dataclasses.dataclass(frozen=True)
class _Builtin:
    def __call__(self, cfg: Cfg) -> None:
        cfg.builtin[self.name] = self

    @property
    def name(self) -> str:
        return "".join(
            f"_{c.lower()}" if c.isupper() else c for c in type(self).__name__
        ).lstrip("_")


@dataclasses.dataclass(frozen=True)
class Split(_Builtin):
    """Splits blocks at calls that aren't pure, so the pure parts can be regions."""


@dataclasses.dataclass(frozen=True)
class Elide(_Builtin):
    """Inlines calls to Python functions whose runs only did torch ops."""


@dataclasses.dataclass(frozen=True)
class PinCalls(_Builtin):
    """Guards a call's callee by identity, so its target is known."""


@dataclasses.dataclass(frozen=True)
class UseAten(_Builtin):
    """Calls the aten op a torch call dispatched to, skipping the Python layer."""


@dataclasses.dataclass(frozen=True)
class LiftGuard(_Builtin):
    """Moves a version's checks to its entry."""


# Together, the executor's optimizer: in this version all of them run or none.
OPTIMIZER = ("split", "pin_calls", "use_aten", "lift_guard")


@dataclasses.dataclass(frozen=True)
class PureCalls(_Builtin):
    """Widens pure code through calls whose callees' runs were pure, one call
    level deeper per re-optimization; ``rounds`` re-optimizations run in the
    calls just before ``Regions`` starts observing.

    ``pure``: callables (functions, builtins, classes) every call of which is
    taken as one that only computes, without looking at what its runs did:
    ``PureCalls(pure=(inspect.signature,))``. A region's trace then runs such
    a call once and records the torch ops it runs; what it reads of the
    program's state is checked before the region runs, as for any call in a
    region. Only sound if the callable has no effect a later call would miss
    (it writes nothing that outlives it) and its result depends on nothing
    but its arguments and the state it reads."""

    rounds: int = 5
    pure: tuple[Callable, ...] = ()


@dataclasses.dataclass(frozen=True)
class Spans(_Builtin):
    """Groups pure blocks across branches and loops into single-entry spans,
    each one region."""


@dataclasses.dataclass(frozen=True)
class Regions(_Builtin):
    """Records the types of each pure block's (and span's) inputs over the
    ``observe`` calls before call ``at``, then from call ``at`` traces each
    into a graph, runs the block passes on it and swaps the result in: a
    region for each state the place is traced in (a variant).

    A place whose variants differ by which object an argument is, or by the
    value of an int (the body of a loop over layers, a method called on each
    of them), keeps them by that key and finds the one for a call without
    asking the others. It holds up to ``variants`` per key, one per input
    specialization, and up to ``keys`` keys; a place without such a key holds
    ``variants``. Past either bound the code runs as it is."""

    variants: int = 16
    at: int = 24
    observe: int = 4
    keys: int = 128


@dataclasses.dataclass(frozen=True)
class BypassCold(_Builtin):
    """From call ``at``, runs code that has no regions on CPython."""

    at: int = 35


@dataclasses.dataclass(frozen=True, init=False)
class Skip(_Builtin):
    """Runs these functions, and everything they call, on CPython."""

    fns: tuple[Callable, ...]

    def __init__(self, *fns: Callable) -> None:
        object.__setattr__(self, "fns", fns)

    def __call__(self, cfg: Cfg) -> None:
        cfg.skip(*self.fns)


@dataclasses.dataclass(frozen=True)
class Autograd:
    """Differentiates each autograd unit (in this version, each region):
    block passes after it see the joint graph (forward and backward
    together), until ``Partition``. Regions that don't need gradients skip
    the passes between the two."""

    def __call__(self, cfg: Cfg) -> None:
        cfg.autograd = self


@dataclasses.dataclass(frozen=True)
class ThreadRng:
    """Threads the generator state through regions as an effect, so a
    recompute replays the same draws. Not implemented yet."""

    def __call__(self, cfg: Cfg) -> None:
        raise NotImplementedError("vm.passes.ThreadRng isn't implemented yet")


def _inductor_partition(joint, joint_inputs, **kwargs):
    from torch._functorch.partitioners import min_cut_rematerialization_partition
    from torch._inductor.fx_passes.joint_graph import joint_graph_passes

    joint_graph_passes(joint)
    return min_cut_rematerialization_partition(
        joint, joint_inputs, compiler="inductor", **kwargs
    )


@dataclasses.dataclass(frozen=True)
class Partition:
    """Splits each block's joint graph into forward and backward graphs, which
    block passes after it see one at a time. ``fn`` is an AOT Autograd
    partitioner; the default is Inductor's (its joint-graph passes, then the
    min-cut partitioner)."""

    fn: Callable | None = None

    def __call__(self, cfg: Cfg) -> None:
        cfg.partition = self

    def split(self, joint, joint_inputs, **kwargs):
        return (self.fn or _inductor_partition)(joint, joint_inputs, **kwargs)


def _callable(compiled: Any) -> Callable:
    """What compile_fx gave, which is a callable unless it was asked for
    something else (a list of output code, a weights file)."""
    if not callable(compiled):
        raise TypeError(f"compile_fx gave a {type(compiled).__name__}, not a callable")
    return compiled


class Inductor:
    """Compiles the graph with Inductor and returns a graph that calls the
    result. Before ``Autograd`` (or without it), Inductor differentiates the
    block itself (AOT Autograd); after ``Partition``, it compiles the forward
    and backward graphs it's given. ``config`` patches
    ``torch._inductor.config`` while it compiles."""

    def __init__(self, config: dict[str, Any] | None = None, **options: Any) -> None:
        self.config = {**(config or {}), **options}

    def __call__(
        self, gm: torch.fx.GraphModule, input_types: list[Any]
    ) -> torch.fx.GraphModule:
        from torch._inductor import config
        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        stage = gm.meta.get("vm_stage", "forward")
        inputs = gm.meta["vm_example_inputs"]
        if stage == "joint":
            raise TypeError(
                "Inductor can't compile a joint graph: put it after vm.passes.Partition()"
            )
        if stage == "forward" and any(isinstance(x, float) for x in inputs):
            return _compile_without_floats(gm, inputs, self.config)
        if stage == "forward":
            patches = dict(self.config) or None
            compiled = compile_fx(below_autocast(gm), inputs, config_patches=patches)
            return lower_to(gm, _callable(compiled))
        with config.patch(self.config):
            boxed = compile_fx_inner(
                gm,
                inputs,
                is_backward=stage == "backward",
                is_inference=stage == "inference",
            )

        def inductor(*args):
            return boxed(list(args))

        return lower_to(gm, inductor)

    def step_begin(self) -> None:
        # CUDA graph trees start a new step on each Dynamo frame; regions
        # aren't Dynamo frames, so each call of the VM's function is one.
        if self.config.get("triton.cudagraphs"):
            torch.compiler.cudagraph_mark_step_begin()

    def __repr__(self) -> str:
        return f"Inductor({self.config!r})" if self.config else "Inductor()"


def _compile_without_floats(
    gm: torch.fx.GraphModule, inputs: list[Any], config: dict[str, Any]
) -> torch.fx.GraphModule:
    """Inductor on a graph with float inputs: specialized on their values
    (see vm.specialize_floats), then compiled without them; outputs the
    folding made constants are put back around the compiled call."""
    from torch._inductor.compile_fx import compile_fx
    from torch.vm._graph import copy_graph_module, specialize_floats
    from torch.vm._guards import guard_node

    special = specialize_floats(gm, inputs)
    inner = copy_graph_module(special)
    for node in list(inner.graph.find_nodes(op="call_function", target=guard_node)):
        node.replace_all_uses_with(node.args[0])
        inner.graph.erase_node(node)
    keep = [i for i, x in enumerate(inputs) if not isinstance(x, float)]
    for i, ph in enumerate(list(inner.graph.find_nodes(op="placeholder"))):
        if i not in keep:
            inner.graph.erase_node(ph)
    out = next(iter(inner.graph.find_nodes(op="output")))
    results = out.args[0]
    seq = isinstance(results, (list, tuple))
    items = list(results) if seq else [results]
    nodes = [k for k, x in enumerate(items) if isinstance(x, torch.fx.Node)]
    out.args = ([items[k] for k in nodes],)
    inner.recompile()
    kept, patches = [inputs[i] for i in keep], dict(config) or None
    code = compile_fx(below_autocast(inner), kept, config_patches=patches)
    compiled = _callable(code)

    def inductor(*args):
        got = iter(compiled(*[args[i] for i in keep]))
        made = [next(got) if k in nodes else x for k, x in enumerate(items)]
        if not seq:
            return made[0]
        return tuple(made) if isinstance(results, tuple) else list(made)

    lowered = lower_to(special, inductor)
    for ph, special_ph in zip(
        lowered.graph.find_nodes(op="placeholder"),
        special.graph.find_nodes(op="placeholder"),
    ):
        for user in special_ph.users:
            if user.target is guard_node:
                from torch.vm._guards import guard

                guard(lowered, ph, *user.meta["vm_checks"])
    lowered.recompile()
    return lowered


def static_parameters(
    gm: torch.fx.GraphModule, input_types: list[Any]
) -> torch.fx.GraphModule:
    """Marks a block's weights as static inputs, so Inductor's CUDA graphs read
    them in place rather than copying them on every replay (sound: a region
    passes the same weight tensors on every call, and an optimizer updates
    them in place). AOT Autograd reads static inputs off a Dynamo graph's
    metadata, so this gives the graph that metadata."""
    from torch._dynamo.source import LocalSource

    gm.meta["dynamo_compile_id"] = None
    # Not through Module.__setattr__, which is typed for modules and tensors.
    vars(gm)["_param_name_to_source"] = {}
    for i, (node, t) in enumerate(
        zip(gm.graph.find_nodes(op="placeholder"), input_types)
    ):
        node._dynamo_source = LocalSource(f"vm_input_{i}")
        if t.is_const:
            node.meta.setdefault("tensor_dict", {})["_dynamo_static_input_type"] = (
                "unguarded"
            )
    return gm
