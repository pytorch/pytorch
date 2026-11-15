"""Share the backward's recompute with the forward.

The partitioner recomputes a checkpointed region, and anything else it chooses
to recompute (e.g. under activation_memory_budget), by copying the forward ops
into the backward graph. The forward and backward are compiled separately, so
the two copies can lower to different kernels and the recompute can drift
numerically from the forward (gh-186572). With
torch._functorch.config.deterministic_recompute, each convex piece of the
recompute (per checkpointed region, plus one region for the rest) is outlined
into a subgraph that both graphs invoke, compiled with Inductor
deterministic=True so both lowerings match.

The pass runs after partitioning because that is the first point where the
recomputed ops are known: a selective-checkpointing policy is evaluated per aten
op while AOT autograd traces the joint graph, and the partitioner then makes the
final save/recompute choices (saving region boundaries, skipping ops the backward
never reads, overriding PREFER_* preferences). Outlining regions before AOT
tracing would give explicit boundaries, but would have to know those choices in
advance, and would make each region a HOP in the traced program (its outputs may
not alias its inputs, re-invoking it re-draws RNG, and backends without AOT
autograd get no recompute).
"""

import copy
import functools
import heapq
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import sympy

import torch
from torch._functorch import config
from torch._inductor.fx_utils import get_mutated_input_nodes, get_node_storage
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    optimization_hint,
)
from torch.fx.passes.reinplace import _is_view_op
from torch.fx.traceback import (
    _get_memory_budget_annotation,
    MEMORY_BUDGET_ANNOTATION_KEY,
)


log = logging.getLogger(__name__)


def deterministic_recompute(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_fwd_outputs: int,
    prefix: str = "",
) -> None:
    """Rewrite the partitioned fw_module and bw_module in place so each piece of
    the backward's recompute runs as a shared subgraph. ``prefix`` keeps the
    pieces' invoke_subgraph identifiers unique within the compile, and tells in
    the log which graph a node is in."""
    # Both trace the graphs again and can't trace the shared calls (compiled
    # autograd would graph-break on each backward call and compile its subgraph
    # without deterministic mode).
    if config.enable_complex_wrapper or torch._dynamo.config.compiled_autograd:
        msg = "not applied under enable_complex_wrapper or compiled_autograd"
        torch._logging.warning_once(log, "deterministic_recompute: %s", msg)
        return
    # Nodes in a user memory pool (torch.cuda.use_mem_pool) carry it in their
    # metadata, which a call replacing them wouldn't, and the call could land
    # after the pool's end marker: their allocations would leave the pool (and
    # Inductor would CUDA-graph the parent). (A nested_compile_region body
    # called in a pool has only the metadata.)
    pool_markers = ("mempool::begin", "mempool::end")
    for gm in (fw_module, bw_module):
        for n in gm.graph.nodes:
            op = n.target
            marker = isinstance(op, torch._ops.OpOverload) and op.name() in pool_markers
            if marker or "mempool" in n.meta.get("custom", {}):
                msg = "deterministic_recompute: %s%s is in a memory pool"
                log.info(msg, prefix, n.name)
                return
    fw_outs = fw_module.graph.find_nodes(op="output")[0].args[0]
    saved = {n for n in fw_outs[num_fwd_outputs:] if isinstance(n, torch.fx.Node)}
    copies, regions = _recomputed_copies(fw_module, bw_module, saved, prefix)
    bw_body = {n.name: n for n in bw_module.graph.nodes}

    # A recomputed node that reads or produces a value carrying an unbacked
    # symbol (e.g. from .item() or nonzero, including those nodes themselves)
    # stays out of pieces: Inductor can't bind one coming out of a
    # subgraph, nor in general one passed in, and the symbol almost always
    # leaves the piece (runtime asserts and the backward use it too). So does
    # everything computed from it: a piece holding that would read the dropped
    # value, which each graph computes on its own and which it couldn't
    # recompute itself.
    # (unbacked_bindings may name a symbol before a rename the values see, so
    # the symbols are read off the values, minus those of graph inputs, which
    # are fine.)
    unbacked = set()
    for n in fw_module.graph.nodes:
        if n.meta.get("unbacked_bindings"):
            unbacked |= free_unbacked_symbols(n.meta.get("val"))
    for p in fw_module.graph.find_nodes(op="placeholder"):
        unbacked -= free_unbacked_symbols(p.meta.get("val"))
    if unbacked:
        dropped: set[torch.fx.Node] = set()
        for n in fw_module.graph.nodes:
            if n in copies and (
                any(a in dropped for a in n.all_input_nodes)
                or any(
                    free_unbacked_symbols(m.meta.get("val")) & unbacked
                    for m in (n, *n.all_input_nodes)
                )
            ):
                dropped.add(n)
        copies -= dropped
        # Say so once, where it starts.
        for n in fw_module.graph.nodes:
            if n in dropped and not any(a in dropped for a in n.all_input_nodes):
                log.info(
                    "deterministic_recompute: %s%s left as partitioned, with what "
                    "is computed from it: it reads or computes a value carrying "
                    "an unbacked symbol",
                    prefix,
                    n.name,
                )
        for nodes in regions.values():
            nodes[:] = [n for n in nodes if n not in dropped]

    # A constant is a value computed from no tensor graph input: an arange, an
    # attribute such as a tensor constant, math on those, or SymInt
    # arithmetic. An impure op isn't one, since copying it would run it again.
    # A piece copies the recomputed constants and the attributes it reads, so
    # they fuse into its kernels instead of being operands; a submodule (e.g.
    # flex_attention's score_mod) couldn't be an operand anyway. A constant the
    # backward reads directly must be the value the forward used, so it is
    # shared as a piece of its own instead, which other pieces read; except one
    # that is the same in any lowering, which stays copied: a scalar, which a
    # piece couldn't return to Inductor anyway, or an integer or bool tensor
    # computed only from such values and sizes (e.g. a causal mask; a threshold
    # on float math could differ).
    constants: set[torch.fx.Node] = set()
    for n in fw_module.graph.nodes:
        if n.op == "get_attr" or (
            n.op == "call_function"
            and not _is_rng(n)
            and not n.is_impure()
            and all(
                a in constants
                for a in n.all_input_nodes
                if not isinstance(a.meta.get("val"), _SCALARS)
            )
        ):
            constants.add(n)
    copy_names = {n.name for n in copies if _draw_of(n) is None}
    in_regions = {n for nodes in regions.values() for n in nodes}
    direct = {
        c
        for c in in_regions & constants
        if any(u.name not in copy_names for u in bw_body[c.name].users)
    }
    # Reading a view of a constant reads the constant: the piece returns the
    # constant and the view stays inline. A multi-output op goes together with
    # all of its getitems, since a piece can't take or return a tuple.
    stack = list(direct)
    while stack:
        c = stack.pop()
        group = list(c.all_input_nodes) if _is_view(c.target) else []
        if isinstance(c.meta.get("val"), (tuple, list)):
            group += c.users
        for a in group:
            if a in in_regions and a in constants and a not in direct:
                direct.add(a)
                stack.append(a)

    exact: set[torch.fx.Node] = set()
    sizes = (int, bool, torch.SymInt, torch.SymBool)
    for c in fw_module.graph.nodes:
        val = c.meta.get("val")
        vals = val if isinstance(val, (tuple, list)) else (val,)
        tensors = [v for v in vals if isinstance(v, torch.Tensor)]
        inexact = any(t.is_floating_point() or t.is_complex() for t in tensors)
        if c not in constants or inexact:
            continue
        exact_inputs = all(
            a in exact or isinstance(a.meta.get("val"), sizes)
            for a in c.all_input_nodes
        )
        if not tensors or exact_inputs:
            exact.add(c)
    direct -= exact
    copied = (copies & constants) - direct
    copied |= set(fw_module.graph.find_nodes(op="get_attr"))

    # A node that another region reads (e.g. a weight cast that CSE merged
    # across checkpointed calls of one layer) isn't private to its region: the
    # backward needs it whenever any of those regions runs. It gets a piece of
    # its own, together with what it's computed from in its region (which has
    # to run first anyway), instead of tying the rest of its region to the
    # earliest of those times.
    # (A recomputed input from another region is shared itself; a multi-output
    # op keeps its readers. A reader that isn't recomputed is in another region
    # only if it has an ac_graph_id; otherwise it's an ordinary forward op.)
    region_of = {n: key for key, nodes in regions.items() for n in nodes}

    def _in_other_region(n: torch.fx.Node, u: torch.fx.Node) -> bool:
        if u in region_of:
            return region_of[u] != region_of[n]
        key = u.meta.get("ac_graph_id")
        return key is not None and key != region_of[n]

    candidates = in_regions - constants
    shared = {n for n in candidates if any(_in_other_region(n, u) for u in n.users)}

    def _close(stack: list[torch.fx.Node]) -> None:
        while stack:
            n = stack.pop()
            readers = n.users if isinstance(n.meta.get("val"), (tuple, list)) else ()
            for a in (*n.all_input_nodes, *readers):
                if a in candidates and a not in shared:
                    shared.add(a)
                    stack.append(a)

    _close(list(shared))
    order = {n: i for i, n in enumerate(fw_module.graph.nodes)}

    # The partitioner places recompute right before its first use in the
    # backward, so a region's recompute falls into runs separated by gradient
    # ops (those computed from a tangent, views aside). A piece runs all at
    # once at its first reader and its outputs stay alive until their last, so
    # a piece spanning runs would compute the later runs' values early; pieces
    # stay within one run (except those of constants the backward reads
    # directly, which are usually small). A view is in the run of what it
    # views.
    bw_run: dict[str, int] = {}
    run_start: dict[int, int] = {}
    grads: set[torch.fx.Node] = set()
    run, gap = 0, False
    for i, n in enumerate(bw_module.graph.nodes):
        if n.op == "placeholder" and n.name.startswith("tangents"):
            grads.add(n)
        if n.op in ("placeholder", "output"):
            continue
        if any(a in grads for a in n.all_input_nodes):
            grads.add(n)
        if n.name in copy_names:
            run += gap
            gap = False
            bw_run[n.name] = run
            run_start.setdefault(run, i)
        elif n in grads and not _is_view(n.target):
            gap = True
    run_of = {n: bw_run[n.name] for n in in_regions if n.name in bw_run}
    # A node run after its inputs' run is better computed with them if it is
    # their only remaining reader and smaller than them (e.g. a cast of an fp32
    # value to bf16): the gap then holds its output instead of its inputs. Not
    # if it's just as big: that only grows the piece, and in the forward a
    # piece writes out everything the backward reads from it at once. Only
    # inputs of its own split count (it couldn't join another's piece), and
    # neither a view nor a copied constant frees anything: a view's base may
    # stay alive, an expand is smaller than it looks, and a constant is copied
    # into every piece that reads it. (An input the backward computes but no
    # run labels, e.g. a replayed draw, may come only in the node's own run.)
    bw_index = {n.name: i for i, n in enumerate(bw_module.graph.nodes)}
    split_of = {
        n: (key, n in shared, n in direct)
        for key, nodes in regions.items()
        for n in nodes
    }

    def _readers(name: str) -> list[torch.fx.Node]:
        # (A view keeps its base alive while it's read.)
        stack = list(bw_body[name].users)
        readers: list[torch.fx.Node] = []
        while stack:
            u = stack.pop()
            readers.append(u)
            if _views(u):
                stack.extend(u.users)
        return readers

    for n in fw_module.graph.nodes:
        inputs = [a for a in n.all_input_nodes if a in run_of]
        if n not in run_of or not inputs:
            continue
        if _views(n):
            run_of[n] = max(run_of[a] for a in inputs)
            continue
        unlabeled = (a for a in n.all_input_nodes if a not in run_of)
        if any(
            a.op not in ("placeholder", "get_attr") and a not in saved
            for a in unlabeled
        ):
            continue
        start = run_start[run_of[n]]
        own = [a for a in inputs if split_of[a] == split_of[n] and a not in copied]
        freed = sum(
            _nbytes(a)
            for a in own
            if not _views(a)
            and all(
                u.name == n.name or bw_index[u.name] < start for u in _readers(a.name)
            )
        )
        # (In a run none of its own inputs is in, it would be a piece of its own.)
        prev = max(run_of[a] for a in inputs)
        if prev < run_of[n] and prev in {run_of[a] for a in own}:
            if 0 < _nbytes(n) < freed:
                run_of[n] = prev

    changed = False
    # The backward node holding each value a shared piece returns, so later
    # pieces can read it.
    bw_of: dict[torch.fx.Node, torch.fx.Node] = {}
    # Name pieces by their order in this graph, not by ac_graph_id: that is a
    # process-global counter, so recompiling the same graph would get new names
    # and miss the FX graph cache.
    index = itertools.count()

    def _share(pieces: list[list[torch.fx.Node]]) -> list[list[torch.fx.Node]]:
        """Share ``pieces`` in order; returns those left as partitioned."""
        nonlocal changed
        skipped = []
        for piece in pieces:
            # (An earlier piece may have recomputed some of its nodes, reached
            # through a node in no piece, and erased them.)
            piece = [n for n in piece if not n._erased]
            if not piece:
                continue
            identifier = f"{prefix}recompute_{next(index)}"
            outputs = _share_region(
                fw_module, bw_module, piece, saved, copies, copied, bw_of, identifier
            )
            if outputs is None:
                skipped.append(piece)
            else:
                bw_of.update(outputs)
                changed = True
        return skipped

    # Constant pieces go first, so the pieces that read them find their outputs
    # in bw_of; the constants of one left as partitioned are copied into those
    # pieces instead.
    direct_pieces = _split_region(sorted(direct, key=order.__getitem__), copied, order)
    for piece in _share(_dependency_order(direct_pieces, order)):
        copied |= direct & set(piece)
    # (Sharing added nodes to the graph, and erased some.)
    order = {n: i for i, n in enumerate(fw_module.graph.nodes)}
    pieces: list[list[torch.fx.Node]] = []
    for nodes in regions.values():
        nodes = [n for n in nodes if n not in direct and n in order]
        for part in (
            [n for n in nodes if n in shared],
            [n for n in nodes if n not in shared],
        ):
            pieces += _split_region(part, copied, order, run_of)
    _share(_dependency_order(pieces, order))
    if changed:
        for gm in (fw_module, bw_module):
            _call_cheap_pieces_at_readers(gm, inputs_held=gm is fw_module)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()


_SCALARS = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)


def _nbytes(n: torch.fx.Node) -> int:
    val = n.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return 0
    return optimization_hint(val.numel(), fallback=1) * val.element_size()


def _recomputed_copies(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    saved: set[torch.fx.Node],
    prefix: str = "",
) -> tuple[set[torch.fx.Node], dict[Any, list[torch.fx.Node]]]:
    """The forward nodes whose copies the backward recomputes, given the values
    the forward ``saved`` for it, and those of them in each checkpoint region
    (keyed by ac_graph_id, in graph order; None for those outside regions).
    Logs where the backward's recompute stops matching the forward, naming the
    node after ``prefix``."""
    # Both graphs are extracted from the joint graph, so the backward's
    # recomputed copy of a forward node keeps its name; checkpointed nodes carry
    # their region's ac_graph_id. A matching name alone isn't trusted: a copy
    # must also be the same op on the same inputs (primals, saved values,
    # attributes, or other copies).
    bw_body = {
        n.name: n
        for n in bw_module.graph.nodes
        if n.op not in ("placeholder", "output")
    }
    # The backward replays a graphsafe RNG draw (e.g. dropout's) from its own
    # generator, bwd_rng_state_i, which the runtime keeps in sync with the
    # forward's fwd_rng_state_i (see initialize_rng_states), so the replay is a
    # copy.
    rng_state_names = {
        p.name: "fwd" + p.name.removeprefix("bwd")
        for p in bw_module.graph.find_nodes(op="placeholder")
        if p.name.startswith("bwd_rng_state_")
    }

    def _names(n: torch.fx.Node) -> Any:
        return torch.fx.node.map_arg(
            (n.args, n.kwargs), lambda a: rng_state_names.get(a.name, a.name)
        )

    copies: set[torch.fx.Node] = set()
    not_copies: set[torch.fx.Node] = set()
    regions: dict[Any, list[torch.fx.Node]] = defaultdict(list)
    for n in fw_module.graph.nodes:
        bw_n = bw_body.get(n.name)
        if n.op != "call_function" or n in saved or bw_n is None:
            continue
        missing = [
            a
            for a in n.all_input_nodes
            if a.op not in ("placeholder", "get_attr")
            and a not in saved
            and a not in copies
        ]
        if missing or bw_n.target != n.target or _names(bw_n) != _names(n):
            # Everything computed from n is left unshared too; say so once,
            # where that starts (an input the backward doesn't copy, which may
            # have other readers, or n itself).
            if not any(a in not_copies for a in missing):
                log.info(
                    "deterministic_recompute: %s%s left as partitioned: the "
                    "backward recomputes it differently",
                    prefix,
                    missing[0].name if missing else n.name,
                )
            not_copies.add(n)
            not_copies.update(missing)
            continue
        copies.add(n)
        # The replayed draw is never part of a piece (a piece holding any other
        # RNG op is left as partitioned); pieces read its result instead.
        if _draw_of(n) is not None:
            continue
        # Recompute the partitioner chose outside checkpoint regions (e.g. under
        # activation_memory_budget) forms one more region. A getitem goes with
        # its op, even if a custom pass moved only one of them out of a region.
        op = n.args[0] if n.target is operator.getitem else n
        regions[op.meta.get("ac_graph_id")].append(n)
    return copies, regions


def _split_region(
    nodes: list[torch.fx.Node],
    copied: set[torch.fx.Node],
    order: dict[torch.fx.Node, int],
    run_of: dict[torch.fx.Node, int] | None = None,
) -> list[list[torch.fx.Node]]:
    """Split a region's recomputed nodes (in topological order) into convex
    pieces, e.g. two recomputed stretches separated by a saved matmul. A piece
    must be convex: if a path left it and came back, no single call could
    replace it. Each piece grows from the first unassigned node, taking a
    neighboring node whenever every node on a path between the piece and it is
    an unassigned node of the region, and then all of those nodes; a
    multi-output op joins together with its readers. With ``run_of``, a piece
    only grows by nodes of its seed's run. Each piece gets its own shared
    subgraph; a piece may read another's outputs (see _dependency_order). Views
    of values outside the region, scalars and ``copied`` constants are free to
    replay, so they join every piece that reads them, whatever its run, instead
    of joining pieces together. Each piece's nodes are sorted by ``order``."""
    run_of = run_of or {}
    in_region = set(nodes)

    replayed = set(copied)
    for n in nodes:
        outside = all(a not in in_region or a in replayed for a in n.all_input_nodes)
        if (_views(n) and outside) or isinstance(n.meta.get("val"), _SCALARS):
            replayed.add(n)

    free = {n for n in nodes if n not in replayed}

    def _unit(v: torch.fx.Node) -> set[torch.fx.Node]:
        # A multi-output op and its readers stay together: a call can't take a
        # tuple operand.
        op = v.all_input_nodes[0] if v.target is operator.getitem else v
        if op in free and isinstance(op.meta.get("val"), (tuple, list)):
            return {op, *(u for u in op.users if u in free)}
        return {v}

    groups = []
    for seed in nodes:
        if seed not in free:
            continue
        group = _unit(seed)
        free -= group
        run = run_of.get(seed)
        lo = min(order[n] for n in group)
        hi = max(order[n] for n in group)
        frontier: list[tuple[int, torch.fx.Node]] = []

        def _push(added: set[torch.fx.Node]) -> None:
            for n in added:
                for m in (*n.all_input_nodes, *n.users):
                    if m in free:
                        heapq.heappush(frontier, (order[m], m))

        _push(group)
        # A rejection means some path between the piece and the neighbor goes
        # through a node that is neither free nor replayed, or is of another
        # run; growing the piece keeps that path, so a rejected neighbor stays
        # rejected.
        rejected: set[torch.fx.Node] = set()
        while frontier:
            _, v = heapq.heappop(frontier)
            if v not in free or v in rejected:
                continue
            unit = _unit(v)
            unit_lo = min(lo, *(order[n] for n in unit))
            unit_hi = max(hi, *(order[n] for n in unit))
            added = _hull_additions(group, unit, order, unit_lo, unit_hi)
            # Replayed nodes may sit on such a path; they are copied into the
            # piece anyway.
            if all(n in free or n in replayed for n in added) and all(
                run_of.get(n, run) == run for n in added - replayed
            ):
                added -= replayed
                free -= added
                group |= added
                lo, hi = unit_lo, unit_hi
                _push(added)
            else:
                rejected |= unit
        groups.append(group)
    pieces = []
    for group in groups:
        stack = [a for n in group for a in n.all_input_nodes if a in replayed]
        while stack:
            v = stack.pop()
            if v not in group:
                group.add(v)
                stack.extend(a for a in v.all_input_nodes if a in replayed)
        pieces.append(sorted(group, key=order.__getitem__))
    return pieces


def _dependency_order(
    pieces: list[list[torch.fx.Node]], order: dict[torch.fx.Node, int]
) -> list[list[torch.fx.Node]]:
    """``pieces`` reordered so each comes after the pieces whose results it
    reads, and so finds their outputs in bw_of instead of recomputing them, and
    otherwise by its first node's ``order``; pieces reading each other both
    ways go by that order (each then has a path back through the other, so
    both are left as partitioned). (A node in several pieces is replayed into
    each, and belongs to none.)"""
    owner: dict[torch.fx.Node, int | None] = {}
    for i, piece in enumerate(pieces):
        for n in piece:
            owner[n] = None if n in owner else i

    def _inputs(n: torch.fx.Node) -> Iterable[torch.fx.Node]:
        # The backward's RNG replay is in no piece: reading one reads what it
        # reads.
        stack = list(n.all_input_nodes)
        while stack:
            a = stack.pop()
            if (draw := _draw_of(a)) is not None:
                stack.extend(draw.all_input_nodes)
            else:
                yield a

    deps = [
        {owner[a] for n in piece for a in _inputs(n) if owner.get(a, i) != i} - {None}
        for i, piece in enumerate(pieces)
    ]

    def _first(i: int) -> int:
        return min(order[n] for n in pieces[i] if owner[n] == i)

    remaining = sorted(range(len(pieces)), key=_first)
    done: set[int] = set()
    result = []
    while remaining:
        ready = next((i for i in remaining if deps[i] <= done), remaining[0])
        remaining.remove(ready)
        done.add(ready)
        result.append(pieces[ready])
    return result


def _hull_additions(
    group: set[torch.fx.Node],
    unit: set[torch.fx.Node],
    order: dict[torch.fx.Node, int],
    lo: int,
    hi: int,
) -> set[torch.fx.Node]:
    """``unit`` plus every node outside the convex ``group`` that lies on a path
    between ``group | unit`` and ``unit``. Such paths stay within [``lo``,
    ``hi``] in ``order``, and the searches stop at members."""

    def _member(n: torch.fx.Node) -> bool:
        return n in group or n in unit

    def _on_paths(
        step: Callable[[torch.fx.Node], Iterable[torch.fx.Node]],
        back: Callable[[torch.fx.Node], Iterable[torch.fx.Node]],
        inside: Callable[[torch.fx.Node], bool],
    ) -> set[torch.fx.Node]:
        # Nodes reachable from the unit by ``step`` without passing a member...
        seen: set[torch.fx.Node] = set()
        stack = [m for u in unit for m in step(u) if not _member(m) and inside(m)]
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(m for m in step(x) if not _member(m) and inside(m))
        # ...that reach a member.
        reach = {x for x in seen if any(_member(m) for m in step(x))}
        stack = list(reach)
        while stack:
            for p in back(stack.pop()):
                if p in seen and p not in reach:
                    reach.add(p)
                    stack.append(p)
        return reach

    down = _on_paths(
        lambda n: n.users, lambda n: n.all_input_nodes, lambda n: order[n] < hi
    )
    up = _on_paths(
        lambda n: n.all_input_nodes, lambda n: n.users, lambda n: order[n] > lo
    )
    return unit | down | up


def _keep_budget(dst: torch.fx.Node, *srcs: torch.fx.Node) -> None:
    """A region_activation_memory_budget must annotate every op of the graph it
    covers, including HOP bodies, so a node standing in for ``srcs`` keeps
    theirs. Nothing else is copied: e.g. compile_with_inductor on a call but not
    on its get_attr would break regional Inductor."""
    for src in srcs:
        budget = _get_memory_budget_annotation(src)
        if budget is not None:
            dst.meta.setdefault("custom", {})[MEMORY_BUDGET_ANNOTATION_KEY] = budget
            return


@functools.cache
def _waits() -> tuple[Any, ...]:
    if not torch.distributed.is_available():
        return ()
    return (
        torch.ops._c10d_functional.wait_tensor.default,
        torch.ops._c10d_functional.wait_tensors.default,
    )


def _views(n: torch.fx.Node) -> bool:
    """Whether ``n`` is a view, counting a getitem only of a multi-output view
    op (e.g. split)."""
    op = n.args[0] if n.target is operator.getitem else n
    return _is_view(n.target) and _is_view(getattr(op, "target", None))


def _is_view(target: Any) -> bool:
    # The unsafe_ ops are views whose schemas don't say so. A getitem counts
    # too: one of a multi-output view op (e.g. split) is a view, and callers
    # that care tell the others apart by their op. A collective's wait
    # counts as a view: it has no kernel, and Inductor lowers it to return its
    # input's buffer (although wait_tensor's fake output has fresh storage), so
    # a piece returning it would return its own input.
    unsafe = (
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsafe_split.Tensor,
        torch.ops.aten.unsafe_split_with_sizes.default,
    )
    return _is_view_op(target) or target in (operator.getitem, *unsafe, *_waits())


@functools.cache
def _cheap_ops() -> Any:
    from torch._functorch.partitioners import get_default_op_list

    return get_default_op_list()


def _reads(n: torch.fx.Node) -> set[int]:
    return {s for a in n.all_input_nodes if (s := get_node_storage(a)) is not None}


# Ops that end the use of their tensors' storage without a write in their
# schema: resize_storage_bytes_ frees it, a sync_dealloc must stay its tensor's
# last use, and activation offloading's wait frees its keepalive.
_ENDS_STORAGE = (
    "inductor::resize_storage_bytes_",
    "streams::sync_dealloc",
    "ao::wait_tensor",
)


def _writes(n: torch.fx.Node) -> set[int]:
    """The storages ``n`` writes in place (e.g. an input mutation's copy_ into
    an input or its base), which no data edge orders against other reads and
    writes of them. An op that ends the use of a storage (see _ENDS_STORAGE)
    counts as writing all of its tensor inputs."""
    written = get_mutated_input_nodes(n)
    if isinstance(n.target, torch._ops.OpOverload) and n.target.name() in _ENDS_STORAGE:
        written = n.all_input_nodes
    return {s for a in written if (s := get_node_storage(a)) is not None}


def _accumulates(n: torch.fx.Node) -> bool:
    """Whether ``n``'s kernel accumulates floating-point values in an order that
    can change run to run (in Inductor, and in eager on CUDA). This has to track
    the ops Inductor lowers to atomic adds, the floating-point scans (Inductor
    splits a long one, and ATen's CUDA cumsum isn't deterministic either), and
    the fallbacks whose kernels use atomics (e.g. index_reduce, histc on
    floats); Inductor's deterministic mode changes none of them. Integer
    accumulation is exact in any order. (An op that overwrites with duplicate
    indices, e.g. index_put without accumulate, races too, but its result is
    undefined anyway. Backward ops, which a recompute holds only when traced
    with trace_autograd_ops, aren't listed.)"""
    aten = torch.ops.aten
    val = n.meta.get("val")
    if not isinstance(val, torch.Tensor) or not (
        val.is_floating_point() or val.is_complex()
    ):
        return False
    if n.target in (
        aten.index_add.default,
        aten.scatter_add.default,
        aten.index_reduce.default,
        aten._unsafe_masked_index_put_accumulate.default,
        aten.histc.default,
        aten.cumsum.default,
        aten.cumprod.default,
        aten.logcumsumexp.default,
    ):
        return True
    if n.target in (
        aten.index_put.default,
        aten._unsafe_index_put.default,
        aten.put.default,
    ):
        return bool(n.kwargs.get("accumulate", n.args[3] if len(n.args) > 3 else False))
    if n.target is aten.scatter_reduce.two:
        return n.kwargs.get("reduce", n.args[4] if len(n.args) > 4 else None) in (
            "sum",
            "mean",
            "prod",
        )
    return n.target in (aten.scatter.reduce, aten.scatter.value_reduce)


def _draw_of(n: torch.fx.Node) -> torch.fx.Node | None:
    """The graphsafe RNG draw ``n`` is, or is a getitem of (on the aot backend
    it returns a tuple), if any."""
    from torch._prims.rng_prims import graphsafe_run_with_rng_state

    op = n.all_input_nodes[0] if n.target is operator.getitem else n
    return op if op.target is graphsafe_run_with_rng_state else None


def _is_rng(n: torch.fx.Node) -> bool:
    from torch._functorch.partitioners import is_rng_op
    from torch._prims.rng_prims import (
        graphsafe_run_with_rng_state,
        run_and_save_rng_state,
        run_dtensor_rng_op,
        run_with_rng_state,
    )

    rng_hops = (
        run_and_save_rng_state,
        run_with_rng_state,
        graphsafe_run_with_rng_state,
        run_dtensor_rng_op,
    )
    return is_rng_op(n) or n.target in rng_hops


def _share_region(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    s_nodes: list[torch.fx.Node],
    saved: set[torch.fx.Node],
    copies: set[torch.fx.Node],
    copied: set[torch.fx.Node],
    bw_of: dict[torch.fx.Node, torch.fx.Node],
    identifier: str,
) -> dict[torch.fx.Node, torch.fx.Node] | None:
    """Outline one piece of recomputed forward nodes ``s_nodes`` (in topological
    order) into a subgraph invoked from both graphs, and return each forward
    output's backward counterpart. If the piece can't be expressed as a shared
    subgraph, logs why and returns None, leaving both graphs untouched."""

    def _skip(reason: str) -> None:
        log.info(
            "deterministic_recompute: %s left as partitioned: %s", identifier, reason
        )

    fw_graph = fw_module.graph
    bw_graph = bw_module.graph
    bw_by_name = {n.name: n for n in bw_graph.nodes}
    node_index = {n: i for i, n in enumerate(fw_graph.nodes)}

    # The piece's two calls read their inputs from different graphs. An input
    # each graph computes separately (a copy no earlier piece shares, e.g. one
    # of a piece left as partitioned) could differ between them, so the piece
    # recomputes it itself instead, with the attributes it reads, and returns
    # it if something else reads it, so that later pieces read it too. That
    # moves its work into the piece, which runs wherever the call is placed and
    # fuses with nothing outside it, so only what the partitioner considers
    # cheap to recompute (views, pointwise ops and the like) is recomputed so.
    s_set = set(s_nodes)
    stack = [a for n in s_nodes for a in n.all_input_nodes]
    while stack:
        a = stack.pop()
        if a in s_set or (a not in copies and a.op != "get_attr"):
            continue
        if (draw := _draw_of(a)) is not None:
            # The backward's replay of the draw reproduces the forward's (see
            # _recomputed_copies), so the piece reads it, if the draw reads the
            # same values in both graphs: primals, saved values, outputs of
            # earlier pieces, or views or draws of those.
            reads = list(draw.all_input_nodes)
            while reads:
                d = reads.pop()
                tensor = isinstance(d.meta.get("val"), (torch.Tensor, tuple, list))
                if not tensor or d.op == "placeholder" or d in saved or d in bw_of:
                    continue
                if _draw_of(d) is None and not _is_view(d.target):
                    return _skip(f"{draw.name} reads {d.name}, computed in each graph")
                reads.extend(d.all_input_nodes)
            continue
        scalar = not isinstance(a.meta.get("val"), (torch.Tensor, tuple, list))
        cheap = a.op == "get_attr" or a in copied or _is_view(a.target) or scalar
        if not cheap and not _cheap_ops().is_recomputable(a):
            return _skip(f"its input {a.name} is computed in each graph")
        s_set.add(a)
        stack.extend(a.all_input_nodes)
    s_nodes = sorted(s_set, key=node_index.__getitem__)

    # A recompute that re-draws RNG can't share the forward's draw, and one
    # whose kernel accumulates in an order that can change run to run (see
    # _accumulates) can't match it bit for bit.
    for n in s_nodes:
        if _is_rng(n):
            return _skip(f"its recompute contains RNG ({n.name})")
        if _accumulates(n):
            return _skip(f"{n.name} accumulates in an order that can change")

    # Slice inputs: values the slice reads from outside it, in first-seen order
    # for a stable signature.
    def _inputs(nodes: list[torch.fx.Node]) -> list[torch.fx.Node]:
        inside = set(nodes)
        inputs = (a for n in nodes for a in n.all_input_nodes if a not in inside)
        return list(dict.fromkeys(inputs))

    s_inputs = _inputs(s_nodes)

    def _is_tuple_val(n: torch.fx.Node) -> bool:
        return isinstance(n.meta.get("val"), (tuple, list))

    # Inductor treats invoke_subgraph outputs as fresh buffers and may reuse one
    # in place, so the slice must not return a view of one of its inputs (e.g.
    # w.t() of a weight, which would be overwritten with a gradient) or of
    # another of its outputs. Views are free, so they stay inline in both graphs
    # (and are copied into the slice for its own consumers) instead of being
    # returned. A view of values from outside the slice stays inline even when
    # its storage doesn't show it (e.g. a collective's wait, see _is_view).
    seen_storages = {get_node_storage(n) for n in s_inputs} - {None}
    inline: set[torch.fx.Node] = set()
    for n in s_nodes:
        storage = get_node_storage(n)
        of_outside = all(a not in s_set or a in inline for a in n.all_input_nodes)
        if storage in seen_storages or (_is_view(n.target) and of_outside):
            if not _is_view(n.target):
                return _skip(f"{n.name} aliases a value but isn't a view op")
            inline.add(n)
        if storage is not None:
            seen_storages.add(storage)
    # A multi-output view op (e.g. split) stays inline, and so do its getitems
    # (whether or not they are in the piece). (A getitem with a list value, of
    # an op returning a nested list, isn't a view op.)
    for n in s_nodes:
        if not _is_tuple_val(n):
            continue
        if n.target is not operator.getitem and _is_view(n.target):
            inline.add(n)
        elif any(u in inline for u in n.users):
            return _skip(f"{n.name} aliases a value but isn't a view op")
    # Copied constants are computed inside the slice and never returned, and
    # neither are scalars (e.g. a SymInt size), which Inductor can't return from
    # a subgraph: each graph keeps its own for other readers.
    # (Another value, e.g. a user Triton kernel's dict of results, would be
    # computed in each graph.)
    inline |= s_set & copied
    for n in s_nodes:
        val = n.meta.get("val")
        if isinstance(val, _SCALARS):
            inline.add(n)
        elif n not in inline and not isinstance(val, (torch.Tensor, tuple, list)):
            return _skip(f"{n.name} isn't a tensor or a scalar")
    if len(inline) == len(s_nodes):
        # Only views and scalars are recomputed: no kernels, nothing to drift.
        return _skip("it recomputes only views and scalars: nothing to share")
    contracted_names = {n.name for n in s_nodes if n not in inline}

    # A node is used outside the slice if a node outside it reads it, in either
    # graph, or a live inline node does. An inline node that isn't live is dead
    # once the slice is outlined. The slice copies only the inline nodes it
    # reads.
    s_names = {n.name for n in s_nodes}
    live: set[str] = set()

    def _used_outside(n: torch.fx.Node) -> bool:
        readers = (*n.users, *bw_by_name[n.name].users)
        return any(u.name not in s_names or u.name in live for u in readers)

    needed: set[torch.fx.Node] = set()
    for n in reversed(s_nodes):
        if n in inline and _used_outside(n):
            live.add(n.name)
        if n not in inline or n in needed:
            needed.update(a for a in n.all_input_nodes if a in inline)
    erase_names = contracted_names | ({n.name for n in inline} - live)
    s_nodes = [n for n in s_nodes if n not in inline or n in needed]
    s_set = set(s_nodes)
    s_inputs = _inputs(s_nodes)
    # A symbol no input binds (see _unbound_symbols) is bound by a SymInt graph
    # input that is the symbol, which the piece then also takes; if there is
    # none (Dynamo lifts each backed symbol as one, so only defensively), the
    # piece is left as partitioned.
    if unbound := _unbound_symbols(s_inputs, s_nodes):
        defining = {
            val.node.expr: p
            for p in fw_graph.find_nodes(op="placeholder")
            if isinstance(val := p.meta.get("val"), torch.SymInt)
        }
        if missing := unbound - defining.keys():
            names = ", ".join(sorted(map(str, missing)))
            return _skip(f"no input binds the symbols its sizes use ({names})")
        s_inputs += [defining[sym] for sym in sorted(unbound, key=str)]

    # The backward must supply every input: primals and saved values (its
    # placeholders), outputs of pieces shared earlier, or a replayed RNG draw,
    # which it computes itself.
    bw_operands: list[torch.fx.Node] = []
    for inp in s_inputs:
        if inp in bw_of:
            m = bw_of[inp]
        elif inp.op == "placeholder" or inp in saved:
            m = bw_by_name.get(inp.name)
            m = m if m is not None and m.op == "placeholder" else None
        else:
            m = bw_by_name.get(inp.name) if _draw_of(inp) is not None else None
        if m is None:
            return _skip(f"its input {inp.name} isn't available in the backward")
        bw_operands.append(m)

    # A multi-output op stays whole in the slice (invoke_subgraph outputs must
    # be flat, so its getitems carry the values), and the slice returns whichever
    # of its results either graph reads: the forward may read some outside the
    # piece (e.g. it saves var_mean's variance for a later region), and the
    # backward may read results the forward never uses (e.g. the mean and rstd
    # of native_layer_norm, or SDPA's logsumexp).
    fw_extra: dict[str, torch.fx.Node] = {}
    bw_extra: dict[str, torch.fx.Node] = {}
    extra_items: dict[str, tuple[torch.fx.Node, int]] = {}
    for n in s_nodes:
        if not _is_tuple_val(n) or n.name not in contracted_names:
            continue
        outside = [(fw_extra, u) for u in n.users if u not in s_set or u.name in live]
        bw_users = bw_by_name[n.name].users
        outside += [
            (bw_extra, u) for u in bw_users if u.name not in s_names or u.name in live
        ]
        for extra, u in outside:
            idx = u.args[1] if u.target is operator.getitem else None
            if not isinstance(idx, int):
                return _skip(f"{u.name} reads multi-output op {n.name} directly")
            if not isinstance(n.meta["val"][idx], torch.Tensor):
                return _skip(f"{u.name} isn't a tensor")
            extra[u.name] = u
            extra_items[u.name] = (n, idx)
    # (The piece can't both read such a getitem and return it.)
    if both := next((u for u in fw_extra.values() if u in s_inputs), None):
        return _skip(f"its input {both.name} depends on its own recompute")

    # Return only values read outside the slice, in either graph.
    s_out_nodes = [
        n
        for n in s_nodes
        if not _is_tuple_val(n) and n.name in contracted_names and _used_outside(n)
    ]
    # Two getitems of one result (only a custom pass would leave them) would
    # return one buffer as two outputs; that is left as partitioned rather than
    # relied on.
    items = [tuple(n.args) for n in s_out_nodes if n.target is operator.getitem]
    items += extra_items.values()
    if len(items) != len(set(items)):
        return _skip("it would return one result twice")

    # _place says where each call goes; the forward's fallback is right after
    # the last slice input, or the last placeholder when the slice reads
    # nothing (a piece of constants).
    if s_inputs:
        last_input = max(s_inputs, key=node_index.__getitem__)
    else:
        last_input = fw_graph.find_nodes(op="placeholder")[-1]
    fw_replaced = {n for n in s_nodes if n not in inline} | set(fw_extra.values())
    fw_recomputed = [n for n in s_nodes if n.op != "get_attr"]
    fw_recomputed += fw_extra.values()
    insert_after, fw_moved, reason = _place(
        fw_recomputed, fw_replaced, s_inputs, last_input, node_index
    )
    if reason is not None:
        return _skip(reason)

    # The backward falls back to where the partitioner placed the piece's
    # recompute, but not before its operands.
    bw_index = {n: i for i, n in enumerate(bw_graph.nodes)}
    first_recompute = next(n for n in bw_graph.nodes if n.name in contracted_names)
    last_operand = max([first_recompute.prev, *bw_operands], key=lambda n: bw_index[n])
    bw_replaced = contracted_names | bw_extra.keys()
    bw_replaced_nodes = {bw_by_name[name] for name in bw_replaced}
    bw_recomputed = [
        m
        for name in [*(n.name for n in s_nodes), *bw_extra]
        if (m := bw_by_name.get(name)) is not None
        and m.op not in ("placeholder", "get_attr")
    ]
    bw_insert_after, bw_moved, reason = _place(
        bw_recomputed, bw_replaced_nodes, bw_operands, last_operand, bw_index
    )
    if reason is not None:
        return _skip(reason)

    # Inductor rewrites a graph's subgraph modules in place while compiling it,
    # so each graph gets its own (identical) copy rather than one shared object.
    extras = list(extra_items.values())
    fw_slice_gm = _slice_module(fw_module, s_inputs, s_nodes, s_out_nodes, extras)
    bw_slice_gm = _slice_module(fw_module, s_inputs, s_nodes, s_out_nodes, extras)
    out_names = [n.name for n in s_out_nodes] + list(extra_items)
    slice_out_vals = [n.meta.get("val") for n in s_out_nodes] + [
        n.meta["val"][i] for n, i in extra_items.values()
    ]
    slice_index = {name: i for i, name in enumerate(out_names)}

    fw_outs = _splice_slice_call(
        fw_module,
        fw_slice_gm,
        f"{identifier}_fw",
        slice_out_vals,
        operands=s_inputs,
        replace={n.name: n for n in s_out_nodes},
        slice_index=slice_index,
        erase_names=erase_names,
        insert_after=insert_after,
        move_below=fw_moved,
        repoint=fw_extra,
    )
    bw_outs = _splice_slice_call(
        bw_module,
        bw_slice_gm,
        f"{identifier}_bw",
        slice_out_vals,
        operands=bw_operands,
        replace={n.name: bw_by_name[n.name] for n in s_out_nodes},
        slice_index=slice_index,
        erase_names=erase_names,
        insert_after=bw_insert_after,
        move_below=bw_moved,
        repoint=bw_extra,
    )
    return {fw_outs[n.name]: bw_outs[n.name] for n in s_out_nodes}


def _unbound_symbols(
    inputs: list[torch.fx.Node], nodes: list[torch.fx.Node]
) -> set[sympy.Symbol]:
    """The symbols that the values of ``inputs`` and ``nodes`` use but that a
    subgraph taking ``inputs`` may not bind. This counts as bound a symbol that
    a SymInt input is, or that a size or stride of a tensor input is (not one in
    a size s0*s1): Inductor binds the latter only under config.size_asserts,
    while it emits those asserts, and not for a tensor with a static size of
    0."""
    from torch._inductor import config as inductor_config

    bound = set()
    for inp in inputs:
        val = inp.meta.get("val")
        if not isinstance(val, torch.Tensor):
            dims = (val,)
        elif inductor_config.size_asserts and not any(
            type(d) is int and d == 0 for d in val.shape
        ):
            dims = (*val.shape, *val.stride())
        else:
            dims = ()
        for d in dims:
            if isinstance(d, torch.SymInt) and isinstance(d.node.expr, sympy.Symbol):
                bound.add(d.node.expr)
    used = set()
    for n in (*inputs, *nodes):
        if (val := n.meta.get("val")) is not None:
            used |= free_symbols(val)
    return used - bound


def _place(
    recomputed: list[torch.fx.Node],
    replaced: set[torch.fx.Node],
    operands: list[torch.fx.Node],
    fallback: torch.fx.Node,
    index: dict[torch.fx.Node, int],
) -> tuple[torch.fx.Node, list[torch.fx.Node], str | None]:
    """The node after which the call replacing ``replaced`` goes, and the
    readers that move below it, or the reason it can't go anywhere.
    ``recomputed`` are the nodes the call computes: ``replaced``, plus the
    inline nodes it copies for itself (views, copied constants, scalars),
    which each graph keeps for its other readers; the call's copies read their
    storages where it goes. The call goes right before the first node that
    reads one of ``replaced``, so it computes them no earlier than the first
    of them is needed, but keeps its order with in-place writes (see _writes)
    to storages the piece reads: before a later write that follows the reads,
    after one that precedes them. If an operand only comes then or later, it
    goes after ``fallback`` (at or after every operand) or such a write
    instead, and the readers before that move below it."""
    start = min(index[n] for n in replaced)
    readers = [u for n in replaced for u in n.users if u not in replaced]
    after = list(operands)
    reads: dict[int, list[int]] = defaultdict(list)
    for n in recomputed:
        for s in _reads(n):
            reads[s].append(index[n])
    for w in fallback.graph.nodes:
        if index[w] > start and w not in replaced:
            for s in _writes(w) & reads.keys():
                if min(reads[s]) < index[w]:
                    readers.append(w)
                else:
                    after.append(w)
    first = min(readers, key=index.__getitem__, default=None)
    if first is not None and all(index[o] < index[first] for o in after):
        point = first.prev
    else:
        point = max([fallback, *after], key=index.__getitem__)
    moved, reason = _readers_to_move(recomputed, replaced, operands, point, index)
    return point, moved, reason


def _readers_to_move(
    recomputed: list[torch.fx.Node],
    replaced: set[torch.fx.Node],
    operands: list[torch.fx.Node],
    insert_after: torch.fx.Node,
    index: dict[torch.fx.Node, int],
) -> tuple[list[torch.fx.Node], str | None]:
    """A call inserted after ``insert_after`` will replace the nodes in
    ``replaced``, so whatever reads them (directly or through each other) up to
    that point has to move below the call. Returns those readers in graph order,
    or the reason they can't move."""
    tainted = set(replaced)
    readers: list[torch.fx.Node] = []
    started = False
    for n in insert_after.graph.nodes:
        if n in replaced:
            started = True
        elif started and any(a in tainted for a in n.all_input_nodes):
            tainted.add(n)
            readers.append(n)
        if n is insert_after:
            break
    for inp in operands:
        if inp in tainted:
            # A path leaves the piece and comes back, e.g. through another
            # region's recompute.
            return [], f"its input {inp.name} depends on its own recompute"
    # Moving a random op could reorder it with another, which changes what
    # each draws.
    if rng := next((n for n in readers if _is_rng(n)), None):
        return [], f"it would move the random op {rng.name}"
    # Data edges order everything else (stream syncs are wrapped in
    # control_deps), except in-place writes: computing ``recomputed`` at the
    # call and moving the readers below it must not reorder a write with a read
    # or write of the same storage.
    point = index[insert_after]
    new_index = {n: (point, 1, index[n]) for n in recomputed}
    new_index |= {n: (point, 2, index[n]) for n in readers}
    lo = min(point, *(index[n] for n in new_index))
    hi = max(point, *(index[n] for n in new_index))
    window = [n for n in insert_after.graph.nodes if lo <= index[n] <= hi]

    def _before(a: torch.fx.Node, b: torch.fx.Node) -> bool:
        return new_index.get(a, (index[a], 0, 0)) < new_index.get(b, (index[b], 0, 0))

    # (A write reads the storage it writes.)
    for w in window:
        written = _writes(w)
        if not written:
            continue
        for n in window:
            reordered = n is not w and (index[w] < index[n]) != _before(w, n)
            if reordered and written & _reads(n):
                return [], f"it would reorder {w.name} and {n.name}"
    return readers, None


def _slice_module(
    owner: torch.fx.GraphModule,
    inputs: list[torch.fx.Node],
    nodes: list[torch.fx.Node],
    outputs: list[torch.fx.Node],
    items: Sequence[tuple[torch.fx.Node, int]] = (),
) -> torch.fx.GraphModule:
    """A subgraph computing ``nodes`` (in topological order) from placeholders
    for ``inputs``, returning ``outputs`` and then result ``i`` of each
    multi-output node ``n`` in ``items``."""
    graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    for inp in inputs:
        env[inp] = graph.placeholder(inp.name)
        env[inp].meta = copy.copy(inp.meta)
    for n in nodes:
        env[n] = graph.node_copy(n, lambda x: env[x])
    outs = [env[n] for n in outputs]
    for n, i in items:
        outs.append(graph.call_function(operator.getitem, (env[n], i)))
        outs[-1].meta["val"] = n.meta["val"][i]
        _keep_budget(outs[-1], n)
    graph.output(tuple(outs))
    graph.lint()
    return torch.fx.GraphModule(owner, graph)


def _splice_slice_call(
    parent_gm: torch.fx.GraphModule,
    slice_gm: torch.fx.GraphModule,
    identifier: str,
    slice_out_vals: list[Any],
    operands: list[torch.fx.Node],
    replace: dict[str, torch.fx.Node],
    slice_index: dict[str, int],
    erase_names: set[str],
    insert_after: torch.fx.Node,
    move_below: list[torch.fx.Node],
    repoint: dict[str, torch.fx.Node] | None = None,
) -> dict[str, torch.fx.Node]:
    """Insert ``invoke_subgraph(slice_gm, identifier, *operands)`` after
    ``insert_after``, swap each node in ``replace`` for the matching output,
    point each getitem in ``repoint`` (outside the slice, of a multi-output op
    in it) at the call instead, move the nodes in ``move_below`` (in order)
    right after the call's outputs, and erase the nodes named in
    ``erase_names``. Returns the output getitems by the name of the node they
    stand for."""
    from torch._higher_order_ops.invoke_subgraph import (
        invoke_subgraph,
        NestedCompileRegionOptions,
    )

    g = parent_gm.graph
    # Erased in reverse topological order, as before the move, so no node is
    # erased before its users (a moved view the call copies in may still be
    # read by a replaced node above the call).
    erase = [n for n in g.nodes if n.name in erase_names]
    attr_name = f"{identifier}_mod"
    parent_gm.register_module(attr_name, slice_gm)
    with g.inserting_after(insert_after):
        get_attr_node = g.get_attr(attr_name)
    with g.inserting_after(get_attr_node):
        call = g.call_function(
            invoke_subgraph, args=(get_attr_node, identifier, *operands)
        )
    call.meta["val"] = tuple(slice_out_vals)
    # Each Inductor compile autotunes the slice on its own and could pick
    # numerically different kernel configs; deterministic mode uses fixed
    # heuristics instead, so both calls lower identically. It doesn't allow
    # benchmark_fusion or combo-kernel benchmarking, which may be on globally.
    patches = {
        "deterministic": True,
        "benchmark_fusion": False,
        "benchmark_combo_kernel": False,
    }
    options = NestedCompileRegionOptions(inductor_config_patches=patches)
    call.meta.setdefault("custom", {})["nested_region_config"] = options
    # The forward's compile keeps an invoke_subgraph's output strides as traced
    # and the backward's may pad them, and the backward's operands may come
    # padded too (e.g. a saved value); with different strides the two copies
    # can accumulate in different orders. So both keep the strides the
    # forward traced, which slice_gm's placeholders carry.
    slice_gm.graph.output_node().meta["user_visible_output_idxs"] = list(
        range(len(slice_out_vals))
    )
    operand_vals = [p.meta["val"] for p in slice_gm.graph.find_nodes(op="placeholder")]
    no_kwargs: dict[str, Any] = {}
    call.meta["eager_input_vals"] = ((None, identifier, *operand_vals), no_kwargs)
    _keep_budget(call, *replace.values(), *(repoint or {}).values())
    anchor = call
    getitems = {}
    for name, tgt in (repoint or {}).items():
        # It keeps its name and meta: a saved getitem is paired with its
        # backward placeholder by name.
        tgt.args = (call, slice_index[name])
        anchor.append(tgt)
        getitems[name] = anchor = tgt
    for name, tgt in replace.items():
        idx = slice_index[name]
        with g.inserting_after(anchor):
            gi = g.call_function(operator.getitem, args=(call, idx))
        gi.meta["val"] = slice_out_vals[idx]
        _keep_budget(gi, tgt)
        tgt.replace_all_uses_with(gi)
        getitems[name] = anchor = gi
    for n in move_below:
        anchor.append(n)
        anchor = n
    for n in reversed(erase):
        g.erase_node(n)
    return getitems


def _is_cheap_piece(slice_gm: torch.fx.GraphModule) -> bool:
    """Whether a piece is only ops Inductor would inline into each reader's
    kernel instead of storing: pointwise ops, views and same-device casts, with
    no random ops."""
    for n in slice_gm.graph.nodes:
        if n.op != "call_function":
            continue
        val = n.meta.get("val")
        if _is_rng(n):
            return False
        if not isinstance(val, torch.Tensor) or _is_view(n.target):
            continue
        if n.target is torch.ops.aten._to_copy.default:
            src = n.args[0].meta.get("val")
            if isinstance(src, torch.Tensor) and src.device == val.device:
                continue
        tags = getattr(n.target, "tags", ())
        if torch.Tag.pointwise not in tags:
            return False
    return True


def _call_cheap_pieces_at_readers(gm: torch.fx.GraphModule, inputs_held: bool) -> None:
    """Call each cheap piece (see _is_cheap_piece) again right before a later
    reader, instead of keeping its outputs alive from the first reader to the
    last: unshared, Inductor would recompute such values in each reader's
    kernel and never store them (e.g. a cast or table from an input that the
    backward reads at its start and again in a late layer's recompute). Every
    call runs the same compiled subgraph on the same operands, so the copies
    are bit-identical.

    A call keeps its operands alive until it runs, so one is added only if the
    outputs it lets go (bytes times nodes since their previous reader) outweigh
    the operands it keeps alive longer (bytes times nodes past the last other
    reader; a graph input the caller holds, ``inputs_held``, costs nothing):
    not e.g. for a product of two saved matmul outputs feeding a later matmul,
    which Inductor would store too. A reader otherwise reads the latest call,
    and so does one right after the previous reader or after an in-place write
    to an operand."""
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

    g = gm.graph
    calls = [
        n
        for n in g.find_nodes(op="call_function", target=invoke_subgraph)
        if _is_cheap_piece(getattr(gm, n.args[0].target))
    ]
    # Later calls first, so the calls this adds are readers of earlier pieces.
    for call in reversed(calls):
        getitems = list(call.users)
        # A returned output (e.g. saved for the backward) is kept by name.
        if any(u.op == "output" for gi in getitems for u in gi.users):
            continue
        index = {n: i for i, n in enumerate(g.nodes)}
        readers = sorted(
            {u for gi in getitems for u in gi.users}, key=index.__getitem__
        )
        reads = _reads(call)
        identifier = call.args[1]
        alive_until: dict[torch.fx.Node, int] = {}
        for a in call.args[2:]:
            if not isinstance(a, torch.fx.Node) or not _nbytes(a):
                continue
            if inputs_held and a.op == "placeholder":
                continue
            others = [
                index[u]
                for u in a.users
                if not (u.target is invoke_subgraph and u.args[1] == identifier)
            ]
            alive_until[a] = max(others, default=-1)
        provider, provider_at = call, index[call]
        items: dict[torch.fx.Node, torch.fx.Node] = {}
        for i, r in enumerate(readers):
            used = [gi for gi in getitems if gi in r.all_input_nodes]
            if i > 0 and index[r] - index[readers[i - 1]] > 1:
                at = index[r]
                held = sum(map(_nbytes, used)) * (at - index[readers[i - 1]])
                extended = [a for a, until in alive_until.items() if until < at]
                kept = sum(
                    _nbytes(a) * (at - max(alive_until[a], provider_at))
                    for a in extended
                )
                n = provider.next
                while n is not r and not (_writes(n) & reads):
                    n = n.next
                if kept < held and n is r:
                    with g.inserting_before(r):
                        provider = g.call_function(invoke_subgraph, args=call.args)
                    provider.meta = dict(call.meta)
                    provider_at = at
                    items = {}
                    alive_until.update(dict.fromkeys(extended, at))
            if provider is call:
                continue
            for gi in used:
                if gi not in items:
                    with g.inserting_after(provider):
                        items[gi] = g.call_function(
                            operator.getitem, args=(provider, gi.args[1])
                        )
                    items[gi].meta = dict(gi.meta)
                r.replace_input_with(gi, items[gi])
