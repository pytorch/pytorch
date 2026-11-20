# mypy: allow-untyped-defs
"""Helpers for block passes: plain functions over a ``torch.fx.GraphModule``."""

import copy

import dataclasses
import inspect
import itertools
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.vm._types import Interval, TensorType, Varies


def placeholder(gm: torch.fx.GraphModule, name: str | int) -> torch.fx.Node:
    """The input named ``name`` (or at index ``name``)."""
    nodes = list(gm.graph.find_nodes(op="placeholder"))
    if isinstance(name, int):
        return nodes[name]
    for n in nodes:
        if n.name == name or n.target == name:
            return n
    raise KeyError(f"no input {name!r}; inputs: {', '.join(n.name for n in nodes)}")


def _fake(mode, x: Any) -> Any:
    """A fake tensor like ``x`` (a tensor, TensorType or graph node)."""
    if isinstance(x, torch.fx.Node):
        x = x.meta.get("val")
    if isinstance(x, TensorType):
        if x.is_const:
            return mode.from_tensor(x._const)
        sizes = [d.seen for d in x.dims]
        with mode:
            return torch.empty_strided(sizes, x.strides or _contiguous(sizes), dtype=x.dtype, device=x.device)
    if isinstance(x, FakeTensor):
        with mode:
            return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    if isinstance(x, torch.Tensor):
        return mode.from_tensor(x)
    return x


def _contiguous(sizes: Sequence[int]) -> list[int]:
    out, n = [0] * len(sizes), 1
    for d in reversed(range(len(sizes))):
        out[d] = n
        n *= max(sizes[d], 1)
    return out


@dataclasses.dataclass
class Match:
    """One match of a pattern: ``inputs`` are the graph's nodes bound to the
    pattern's tensor arguments, ``outputs`` those standing for its results.
    ``m.node("g")`` is the graph node for the pattern's local ``g``, and
    ``m["eps"]`` the value a float argument with a default (a pattern
    constant) took here."""

    gm: torch.fx.GraphModule
    inputs: list[torch.fx.Node]
    outputs: list[torch.fx.Node]
    nodes: list[torch.fx.Node]
    named: dict[str, torch.fx.Node] = dataclasses.field(default_factory=dict)
    constants: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Set by replace: (op, atol, rtol, collectives), for verify mode.
    replaced: tuple[Any, ...] | None = None

    def node(self, name: str) -> torch.fx.Node:
        if name not in self.named:
            raise KeyError(f"the pattern has no tensor named {name!r} that's in the match; it has {', '.join(self.named)}")
        return self.named[name]

    def __getitem__(self, name: str) -> Any:
        if name not in self.constants:
            raise KeyError(f"the pattern has no constant {name!r}; it has {', '.join(self.constants)}")
        return self.constants[name]

    def replace(
        self,
        op: Callable,
        args: Sequence[Any],
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        writes: Sequence[Any] = (),
        collectives: str = "same",
    ) -> torch.fx.Node:
        """Replaces the match with ``op(*args)`` (args may be graph nodes, such
        as ``inputs``). ``atol``/``rtol``: how far its results may drift from
        the ops it replaces (verify mode). ``collectives="changes"``: it
        issues different collectives than the ops it replaces, so the
        version's guards must give the same answer on every rank. Returns the
        new node."""
        if writes:
            raise NotImplementedError("replace(writes=...): effect nodes need cfg.merge_across, which isn't implemented yet")
        if collectives not in ("same", "changes"):
            raise ValueError(f"collectives is 'same' or 'changes', not {collectives!r}")
        g = self.gm.graph
        with g.inserting_before(self.outputs[0] if len(self.outputs) == 1 else _first(g, self.outputs)):
            new = g.call_function(op, tuple(args))
        if len(self.outputs) == 1:
            self.outputs[0].replace_all_uses_with(new)
        else:
            for k, out in enumerate(self.outputs):
                with g.inserting_after(new):
                    item = g.call_function(operator.getitem, (new, k))
                out.replace_all_uses_with(item)
        for n in reversed(self.nodes):
            if not n.users:
                g.erase_node(n)
        g.lint()
        self.gm.recompile()
        self.replaced = (op, atol, rtol, collectives)
        _replacements(self.gm).append(self.replaced)
        return new


def _first(g: torch.fx.Graph, nodes: Iterable[torch.fx.Node]) -> torch.fx.Node:
    want = set(nodes)
    return next(n for n in g.nodes if n in want)


def _replacements(gm: torch.fx.GraphModule) -> list[Any]:
    return gm.meta.setdefault("vm_replacements", [])


# Traced in place of a pattern's float constants, to find where they land.
_SENTINEL = 0.7316421978125


@dataclasses.dataclass
class _Pattern:
    gm: torch.fx.GraphModule
    # The pattern's locals and tensor arguments, by name: nodes of gm.
    named: dict[str, torch.fx.Node]
    # Float constants traced as _SENTINEL: name -> (pattern node, arg index or kwarg name).
    slots: dict[str, list[tuple[torch.fx.Node, int | str]]]


def _float_constants(pattern: Callable) -> dict[str, float]:
    return {q.name: q.default for q in inspect.signature(pattern).parameters.values() if type(q.default) is float}


def _trace(pattern: Callable, examples: Sequence[Any], constants: dict[str, Any] | None = None) -> _Pattern:
    """Traces ``pattern`` on fakes like ``examples``, its float constants set
    to ``constants`` (to sentinels when None, to find where they're used)."""
    import functools
    import sys

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import get_proxy_mode, get_proxy_slot, make_fx

    floats = _float_constants(pattern)
    sentinels = {k: _SENTINEL + i for i, k in enumerate(floats)} if constants is None else {}
    fn = functools.partial(pattern, **(sentinels or constants or {}))
    params = [q.name for q in inspect.signature(pattern).parameters.values() if q.default is q.empty]
    found: dict[str, torch.fx.Node] = {}

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is pattern.__code__:
            mode = get_proxy_mode()
            for name, v in frame.f_locals.items():
                if isinstance(v, torch.Tensor) and mode is not None:
                    proxy = get_proxy_slot(v, mode.tracer, None)
                    if proxy is not None:
                        found[name] = proxy.proxy.node
        return profile

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fakes = [_fake(mode, x) for x in examples]
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with mode:
            pg = make_fx(fn)(*fakes)
    finally:
        sys.setprofile(previous)
    pg.graph.eliminate_dead_code()
    alive = set(pg.graph.nodes)
    phs = list(pg.graph.find_nodes(op="placeholder"))
    # Locals found while tracing are nodes of make_fx's graph, which is pg's.
    named = {k: n for k, n in found.items() if n in alive and n.op != "placeholder"}
    named.update({k: p for k, p in zip(params, phs)})
    slots: dict[str, list[tuple[torch.fx.Node, int | str]]] = {}
    by_value = {v: k for k, v in sentinels.items()}
    for n in pg.graph.nodes:
        for where, a in [*enumerate(n.args), *n.kwargs.items()]:
            if type(a) is float and a in by_value:
                slots.setdefault(by_value[a], []).append((n, where))
    return _Pattern(pg, named, slots)


def _constants(pattern: Callable, p: _Pattern, m: Any) -> dict[str, Any] | None:
    """The values ``pattern``'s float constants took in structural match
    ``m`` of ``p`` (their defaults where the pattern doesn't use them), or
    None if one is bound to something other than a number."""
    out = dict(_float_constants(pattern))
    for name, uses in p.slots.items():
        n, where = uses[0]
        g = m.nodes_map[n]
        v = g.args[where] if isinstance(where, int) else g.kwargs.get(where)
        if type(v) not in (int, float):
            return None
        out[name] = v
    return out


def _matches(gm: torch.fx.GraphModule, p: _Pattern, ignore_literals: bool) -> list[Any]:
    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

    matcher = SubgraphMatcher(p.gm.graph, match_output=False, match_placeholder=False, remove_overlapping_matches=True, ignore_literals=ignore_literals)
    return matcher.match(gm.graph)


def _candidates(gm: torch.fx.GraphModule, n: int, limit: int = 256) -> Iterator[list[Any]]:
    """Example inputs to trace a pattern of ``n`` tensors on, to find its
    structure: the graph's tensor metadata, by (rank, dtype, device), most
    common first, each also as a square shape (broadcasts and multiplies with
    anything of its rank)."""
    shapes: dict[tuple[Any, ...], dict[tuple[int, ...], int]] = {}
    for node in gm.graph.nodes:
        v = node.meta.get("val")
        if isinstance(v, torch.Tensor):
            by = shapes.setdefault((v.dim(), v.dtype, v.device), {})
            by[tuple(v.shape)] = by.get(tuple(v.shape), 0) + 1
    options = []
    for (rank, dtype, device), by in sorted(shapes.items(), key=lambda kv: -sum(kv[1].values())):
        for shape in [(8,) * rank, *sorted(by, key=by.__getitem__, reverse=True)[:3]]:
            options.append(TensorType(torch.Tensor, dtype, device, tuple(Interval(s, s, s) for s in shape), Varies))
    yield from ([o] * n for o in options)
    for combo in itertools.islice(itertools.product(options, repeat=n), limit):
        if len(set(map(id, combo))) > 1:
            yield list(combo)


def match(gm: torch.fx.GraphModule, pattern: Callable, example_inputs: Sequence[Any] | None = None) -> list[Match]:
    """Where ``pattern`` (a function of tensors, written as the model writes
    it) occurs in ``gm``, matched on aten ops. It's traced on each match's own
    input metadata, so sizes it reads must agree too; its float arguments
    with defaults are constants, matched to whatever value the graph has
    there. ``example_inputs`` (tensors, TensorTypes or nodes) skips
    searching for how to trace it.

    A match whose internal values are used outside it isn't one."""
    if example_inputs is not None:
        tries: Iterable[Sequence[Any]] = [example_inputs]
    else:
        n = sum(1 for q in inspect.signature(pattern).parameters.values() if q.default is q.empty)
        tries = _candidates(gm, n)
    found: list[tuple[Any, _Pattern, dict[str, Any]]] = []
    tried = set()
    for examples in tries:
        try:
            structure = _trace(pattern, examples)
            structural = _matches(gm, structure, ignore_literals=True)
        except Exception:
            if example_inputs is not None:
                raise
            continue
        for m in structural:
            constants = _constants(pattern, structure, m)
            if constants is None:
                continue
            key = (tuple(m.placeholder_nodes), tuple(sorted(constants.items())))
            if key in tried:
                continue
            tried.add(key)
            try:
                exact = _trace(pattern, m.placeholder_nodes, constants)
                found.extend((e, exact, constants) for e in _matches(gm, exact, ignore_literals=False))
            except Exception:
                continue
    out, taken = [], set()
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    for m, p, constants in found:
        nodes = [v for k, v in m.nodes_map.items() if k.op == "call_function"]
        if taken.intersection(nodes):
            continue
        taken.update(nodes)
        named = {k: m.nodes_map[v] for k, v in p.named.items() if v in m.nodes_map}
        out.append(Match(gm, list(m.placeholder_nodes), list(m.returning_nodes), sorted(nodes, key=order.__getitem__), named, constants))
    return out


def lower_to(gm: torch.fx.GraphModule, fn: Callable) -> torch.fx.GraphModule:
    """A graph with ``gm``'s inputs whose body is one call, ``fn(*inputs)``,
    returning its outputs: how a pass that compiles the block (to a callable)
    returns a graph."""
    g = torch.fx.Graph()
    phs = [g.placeholder(p.name) for p in gm.graph.find_nodes(op="placeholder")]
    for new, old in zip(phs, gm.graph.find_nodes(op="placeholder")):
        new.meta = dict(old.meta)
    if not hasattr(fn, "__name__"):
        # FX names a call by its target's __name__.
        target = fn

        def fn(*args):
            return target(*args)

    call = g.call_function(fn, tuple(phs))
    g.output(call)
    out = torch.fx.GraphModule(torch.nn.Module(), g)
    out.meta.update(gm.meta)
    return out


def copy_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """A copy of gm with its own graph that keeps the Dynamo metadata AOT
    Autograd reads static inputs from (see passes.static_parameters): graph
    deepcopy keeps node meta, not node attributes."""
    out = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    if hasattr(gm, "_param_name_to_source"):
        out._param_name_to_source = gm._param_name_to_source
    for a, b in zip(gm.graph.nodes, out.graph.nodes):
        if hasattr(a, "_dynamo_source"):
            b._dynamo_source = a._dynamo_source
    return out


def specialize_floats(gm: torch.fx.GraphModule, inputs: list[Any]) -> torch.fx.GraphModule:
    """``gm`` with each float input (a Python float among ``inputs``)
    replaced by its value, the Python arithmetic on it folded, and a
    ``vm.Value`` guard on it: so the result runs only on calls with these
    values. For compilers that bake floats in (Inductor, AOT Autograd); a
    float that changes between calls then runs the version below."""
    from torch.vm._guards import guard, Value

    phs = list(gm.graph.find_nodes(op="placeholder"))
    values = {ph: x for ph, x in zip(phs, inputs) if isinstance(x, float)}
    if not values:
        return gm
    gm = copy_graph_module(gm)
    phs = list(gm.graph.find_nodes(op="placeholder"))
    known: dict[torch.fx.Node, Any] = {ph: x for ph, x in zip(phs, inputs) if isinstance(x, float)}
    folding = {v for k, v in vars(operator).items() if not k.startswith("_") and callable(v)}
    for node in list(gm.graph.nodes):
        if node.op == "placeholder":
            continue
        node.args = torch.fx.node.map_arg(node.args, lambda n: known.get(n, n))
        node.kwargs = torch.fx.node.map_arg(node.kwargs, lambda n: known.get(n, n))
        if node.op == "call_function" and node.target in folding and not node.all_input_nodes:
            known[node] = node.target(*node.args, **node.kwargs)
    for node in reversed(list(gm.graph.nodes)):
        if node in known and node.op != "placeholder" and not node.users:
            gm.graph.erase_node(node)
    for ph in phs:
        if ph in known:
            guard(gm, ph, Value(known[ph]))
    gm.recompile()
    return gm


def direct(gm: torch.fx.GraphModule) -> Callable | None:
    """``fn`` if ``gm`` is just a call ``fn(*inputs)`` (from lower_to)."""
    nodes = list(gm.graph.nodes)
    phs = [n for n in nodes if n.op == "placeholder"]
    rest = nodes[len(phs) :]
    if len(rest) == 2 and rest[0].op == "call_function" and list(rest[0].args) == phs and not rest[0].kwargs and rest[1].op == "output" and rest[1].args[0] is rest[0]:
        return rest[0].target
    return None


@dataclasses.dataclass
class Segment:
    """A run of a graph's nodes between split points."""

    gm: torch.fx.GraphModule
    nodes: list[torch.fx.Node]
    inputs: list[torch.fx.Node]
    outputs: list[torch.fx.Node]

    def as_callable(self) -> torch.fx.GraphModule:
        """The segment as a graph from its inputs to its outputs (as a tuple)."""
        g = torch.fx.Graph()
        env: dict[torch.fx.Node, torch.fx.Node] = {}
        for x in self.inputs:
            env[x] = g.placeholder(x.name)
            env[x].meta = dict(x.meta)
        for n in self.nodes:
            env[n] = g.node_copy(n, env.__getitem__)
        g.output(tuple(env[o] for o in self.outputs))
        return torch.fx.GraphModule(self.gm, g)

    def static_inputs(self) -> set[int]:
        """Indices among ``inputs`` of the block's weights (same tensor on
        every call)."""
        consts = set(self.gm.meta.get("vm_const_inputs", ()))
        return {i for i, x in enumerate(self.inputs) if x.op == "placeholder" and x.name in consts}


def segments(gm: torch.fx.GraphModule, split_at: Iterable[Any] = ()) -> list[Segment]:
    """The graph's compute, split at nodes whose target is in ``split_at``
    (those nodes are in no segment)."""
    split = set(split_at)
    runs: list[list[torch.fx.Node]] = [[]]
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target in split:
            runs.append([])
        else:
            runs[-1].append(n)
    out = []
    for run in runs:
        if not run:
            continue
        inside = set(run)
        inputs: dict[torch.fx.Node, None] = {}
        for n in run:
            for a in n.all_input_nodes:
                if a not in inside:
                    inputs.setdefault(a)
        outputs = [n for n in run if any(u not in inside for u in n.users)]
        out.append(Segment(gm, run, list(inputs), outputs))
    return out


def replace_segment(gm: torch.fx.GraphModule, seg: Segment, fn: Callable) -> torch.fx.Node:
    """Replaces the segment's nodes with ``fn(*seg.inputs)``, which returns its
    outputs as a sequence."""
    g = gm.graph
    with g.inserting_after(seg.nodes[-1]):
        call = g.call_function(fn, tuple(seg.inputs))
    last = call
    for k, out in enumerate(seg.outputs):
        with g.inserting_after(last):
            last = g.call_function(operator.getitem, (call, k))
        out.replace_all_uses_with(last)
    for n in reversed(seg.nodes):
        g.erase_node(n)
    g.lint()
    gm.recompile()
    return call


def run_until(gm: torch.fx.GraphModule, node: torch.fx.Node) -> Any:
    """``node``'s value in the traced run (the block's graphs are pure, so
    rerunning them changes nothing)."""
    inputs = gm.meta.get("vm_example_inputs")
    if inputs is None:
        raise ValueError("run_until needs the graph the VM gave the pass (it carries the traced run's inputs)")
    result: dict[str, Any] = {}

    class Until(torch.fx.Interpreter):
        def run_node(self, n):
            v = super().run_node(n)
            if n is node:
                result["v"] = v
                raise StopIteration
            return v

    try:
        with torch.no_grad():
            Until(gm).run(*inputs)
    except StopIteration:
        return result["v"]
    raise ValueError(f"{node} isn't in this graph")


def effect(fn: Callable) -> Callable:
    """An opaque call as a graph node, ordered by the tensors it writes. Needs
    ``cfg.merge_across``, which isn't implemented yet."""
    raise NotImplementedError("vm.effect needs vm.Cfg.merge_across, which isn't implemented yet")


# Joint graphs (what passes between vm.passes.Autograd and Partition see):
# forward inputs are primals_*, the backward's incoming gradients tangents_*.


def _backward(gm: torch.fx.GraphModule) -> set[torch.fx.Node]:
    bw: set[torch.fx.Node] = set()
    for n in gm.graph.nodes:
        if (n.op == "placeholder" and n.name.startswith("tangents")) or any(a in bw for a in n.all_input_nodes):
            bw.add(n)
    return bw


def forward_values_used_in_backward(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """The forward's values the backward reads: what Partition would save
    (or recompute)."""
    bw = _backward(gm)
    return [n for n in gm.graph.nodes if n not in bw and n.op == "call_function" and any(u in bw and u.op != "output" for u in n.users)]


def last_forward_user(gm: torch.fx.GraphModule, node: torch.fx.Node) -> torch.fx.Node:
    bw = _backward(gm)
    return max((u for u in node.users if u not in bw and u.op != "output"), key=list(gm.graph.nodes).index, default=node)


def first_backward_user(gm: torch.fx.GraphModule, node: torch.fx.Node) -> torch.fx.Node:
    bw = _backward(gm)
    return min((u for u in node.users if u in bw and u.op != "output"), key=list(gm.graph.nodes).index)


def replace_backward_uses(gm: torch.fx.GraphModule, old: torch.fx.Node, new: torch.fx.Node) -> None:
    bw = _backward(gm)
    for u in list(old.users):
        if u in bw and u is not new and u.op != "output":
            u.replace_input_with(old, new)


def mark_recompute(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Has Partition recompute ``node`` in the backward rather than save it."""
    from torch.utils.checkpoint import CheckpointPolicy

    node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    node.meta.setdefault("ac_graph_id", 0)


class async_ops:
    """Ops whose results are in flight until another op completes them
    (metadata for cfg passes; the VM doesn't check it yet)."""

    registered: dict[Any, Any] = {}

    @staticmethod
    def register(start: Any, *, completes_with: Any) -> None:
        async_ops.registered[start] = completes_with
