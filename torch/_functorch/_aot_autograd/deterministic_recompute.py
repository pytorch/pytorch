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
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.reinplace import _is_view_op
from torch.fx.traceback import (
    _get_memory_budget_annotation,
    MEMORY_BUDGET_ANNOTATION_KEY,
)
from torch.multiprocessing.reductions import StorageWeakRef


log = logging.getLogger(__name__)


def deterministic_recompute(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_fwd_outputs: int,
    prefix: str = "",
) -> None:
    """Rewrite the partitioned fw_module and bw_module in place so each piece of
    the backward's recompute runs as a shared subgraph. ``prefix`` keeps the
    pieces' invoke_subgraph identifiers unique within the compile."""
    from torch._prims.rng_prims import graphsafe_run_with_rng_state

    fw_outs = fw_module.graph.find_nodes(op="output")[0].args[0]
    saved = {n for n in fw_outs[num_fwd_outputs:] if isinstance(n, torch.fx.Node)}
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
            # Everything computed from n is left unshared too; say so where that
            # starts.
            if not any(a in not_copies for a in missing):
                log.info(
                    "deterministic_recompute: %s left as partitioned: the "
                    "backward recomputes it differently",
                    n.name,
                )
            not_copies.add(n)
            continue
        copies.add(n)
        # The replayed draw stays out of pieces like any RNG op (see
        # _share_region), and so does a getitem of it, since on the aot backend
        # it returns a tuple; pieces read its result instead.
        draw = n.all_input_nodes[0] if n.target is operator.getitem else n
        if draw.target is graphsafe_run_with_rng_state:
            continue
        # Recompute the partitioner chose outside checkpoint regions (e.g. under
        # activation_memory_budget) forms one more region.
        regions[n.meta.get("ac_graph_id")].append(n)

    # A recomputed node that reads or produces a value carrying an unbacked
    # symbol (e.g. from .item() or nonzero, including those nodes themselves)
    # stays out of pieces like the draw: Inductor can't bind one coming out of a
    # subgraph, nor in general one passed in, and the symbol almost always
    # leaves the piece (runtime asserts and the backward use it too).
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
        for nodes in regions.values():
            nodes[:] = [
                n
                for n in nodes
                if not any(
                    free_unbacked_symbols(m.meta.get("val")) & unbacked
                    for m in (n, *n.all_input_nodes)
                )
            ]

    # A constant depends on no tensor graph input (e.g. an arange, an attribute
    # such as a tensor constant, or math on one); SymInt sizes don't count. A
    # piece copies the recomputed constants and the attributes it reads instead
    # of taking them as inputs, so they fuse into its kernels (and a submodule,
    # e.g. flex_attention's score_mod, can't be an operand anyway). A constant
    # the backward reads directly must be the value the forward used, so it is
    # shared as a piece of its own instead, which other pieces read; except a
    # scalar (e.g. a slice bound computed from SymInt inputs), which is exact in
    # either graph and which a piece can't return to Inductor.
    recomputed = {n for nodes in regions.values() for n in nodes}
    scalar_types = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)
    constants: set[torch.fx.Node] = set()
    for n in fw_module.graph.nodes:
        if n.op == "get_attr" or (
            n.op == "call_function"
            and not _is_rng(n)
            and all(
                a in constants
                for a in n.all_input_nodes
                if not isinstance(a.meta.get("val"), scalar_types)
            )
        ):
            constants.add(n)
    recomputed_names = {n.name for n in recomputed}
    direct = {
        c
        for c in recomputed & constants
        if any(u.name not in recomputed_names for u in bw_body[c.name].users)
    }
    # Reading a view of a constant reads the constant: the slice returns the
    # constant and the view stays inline.
    stack = list(direct)
    while stack:
        c = stack.pop()
        if _is_view(c.target):
            for a in c.all_input_nodes:
                if a in recomputed and a in constants and a not in direct:
                    direct.add(a)
                    stack.append(a)
    direct = {c for c in direct if not isinstance(c.meta.get("val"), scalar_types)}
    copied = (recomputed & constants) - direct
    copied |= set(fw_module.graph.find_nodes(op="get_attr"))

    # A node that another region reads (e.g. a weight cast that CSE merged
    # across checkpointed calls of one layer) isn't private to its region: the
    # backward needs it whenever any of those regions runs. It gets a piece of
    # its own, together with what it's computed from in its region (which has
    # to run first anyway), instead of tying the rest of its region to the
    # earliest of those times.
    # (A recomputed input from another region is shared itself; a multi-output
    # op keeps its readers. A reader without an ac_graph_id is in another region
    # only if it is recomputed; otherwise it's an ordinary forward op.)
    candidates = recomputed - constants
    shared = {
        n
        for n in candidates
        if any(
            u.meta.get("ac_graph_id") != n.meta.get("ac_graph_id")
            and (u.meta.get("ac_graph_id") is not None or u in recomputed)
            for u in n.users
        )
    }

    def _close(stack: list[torch.fx.Node]) -> None:
        while stack:
            n = stack.pop()
            readers = n.users if isinstance(n.meta.get("val"), (tuple, list)) else ()
            for a in (*n.all_input_nodes, *readers):
                if a in candidates and a not in shared:
                    shared.add(a)
                    stack.append(a)

    _close(list(shared))
    # Recompute outside checkpoint regions isn't needed at one time in the
    # backward the way a checkpointed region is, so the same applies between its
    # pieces: a node another of its pieces reads (e.g. a residual add the next
    # layer's recompute reads) gets a piece of its own, until none does.
    order = {n: i for i, n in enumerate(fw_module.graph.nodes)}
    while True:
        rest = [n for n in regions.get(None, ()) if n not in shared and n not in direct]
        # (A node replayed into several pieces, e.g. w.t() of a weight, maps to
        # None.)
        piece_of: dict[torch.fx.Node, int | None] = {}
        for i, piece in enumerate(_split_region(rest, copied, order)):
            for n in piece:
                piece_of[n] = None if n in piece_of else i
        new = {
            n
            for n, i in piece_of.items()
            if i is not None and any(piece_of.get(u, i) != i for u in n.users)
        }
        if not new:
            break
        shared |= new
        _close(list(new))

    # Constant and shared pieces go first, so the pieces that read them find
    # their outputs in bw_of.
    pieces = _split_region(sorted(direct, key=order.__getitem__), copied, order)
    for nodes in regions.values():
        pieces += _split_region([n for n in nodes if n in shared], copied, order)
    for nodes in regions.values():
        rest = [n for n in nodes if n not in direct and n not in shared]
        pieces += _split_region(rest, copied, order)

    changed = False
    # The backward node holding each value a shared piece returns, so later
    # pieces can read it.
    bw_of: dict[torch.fx.Node, torch.fx.Node] = {}
    # Name pieces by their order in this graph, not by ac_graph_id: that is a
    # process-global counter, so recompiling the same graph would get new names
    # and miss the FX graph cache.
    for i, piece in enumerate(pieces):
        identifier = f"{prefix}recompute_{i}"
        outputs = _share_region(
            fw_module, bw_module, piece, saved, copies, copied, bw_of, identifier
        )
        if outputs is not None:
            bw_of.update(outputs)
            changed = True
    if changed:
        for gm in (fw_module, bw_module):
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()


def _split_region(
    nodes: list[torch.fx.Node],
    copied: set[torch.fx.Node],
    order: dict[torch.fx.Node, int],
) -> list[list[torch.fx.Node]]:
    """Split a region's recomputed nodes (in topological order) into convex
    pieces, e.g. two recomputed stretches separated by a saved matmul. A piece
    must be convex: if a path left it and came back, no single call could replace
    it. Each piece grows from the first unassigned node, taking a neighboring
    node whenever every node on a path between the piece and it is an unassigned
    node of the region, and then all of those nodes; a multi-output op joins
    together with its readers. Each piece gets its own
    shared subgraph; a later piece may read an earlier one's outputs. Views of
    values outside the region and ``copied`` constants are free to replay, so
    they join every piece that reads them instead of joining pieces together.
    Pieces and their nodes are sorted by ``order``."""
    in_region = set(nodes)
    outside_views: set[torch.fx.Node] = set()
    for n in nodes:
        if _is_view(n.target) and all(
            a not in in_region or a in outside_views for a in n.all_input_nodes
        ):
            outside_views.add(n)
    replayed = outside_views | copied

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
        # through a node that is neither free nor replayed; growing the piece
        # keeps that path, so a rejected neighbor stays rejected.
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
            if all(n in free or n in replayed for n in added):
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
    return sorted(pieces, key=lambda piece: order[piece[0]])


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


def _is_view(target: Any) -> bool:
    # The unsafe_ ops are views whose schemas don't say so. A collective's wait
    # counts as a view too: it has no kernel, and Inductor lowers it to return
    # its input's buffer (although wait_tensor's fake output has fresh storage),
    # so a piece returning it would return its own input.
    unsafe = (
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsafe_split.Tensor,
        torch.ops.aten.unsafe_split_with_sizes.default,
    )
    return _is_view_op(target) or target in (operator.getitem, *unsafe, *_waits())


def _is_rng(n: torch.fx.Node) -> bool:
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
    return torch._functorch.partitioners.is_rng_op(n) or n.target in rng_hops


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

    # A recompute that re-draws RNG can't share the forward's draw.
    for n in s_nodes:
        if _is_rng(n):
            return _skip(f"its recompute contains RNG ({n.name})")
    fw_graph = fw_module.graph
    bw_graph = bw_module.graph
    bw_by_name = {n.name: n for n in bw_graph.nodes}
    s_set = set(s_nodes)

    # Slice inputs: values the slice reads from outside it, in first-seen order
    # for a stable signature.
    s_inputs: list[torch.fx.Node] = []
    seen: set[torch.fx.Node] = set()
    for n in s_nodes:
        for inp in n.all_input_nodes:
            if inp not in s_set and inp not in seen:
                seen.add(inp)
                s_inputs.append(inp)

    # Inductor treats invoke_subgraph outputs as fresh buffers and may reuse one
    # in place, so the slice must not return a view of one of its inputs (e.g.
    # w.t() of a weight, which would be overwritten with a gradient) or of
    # another of its outputs. Views are free, so they stay inline in both graphs
    # (and are copied into the slice for its own consumers) instead of being
    # returned.
    def _storage(n: torch.fx.Node) -> StorageWeakRef | None:
        val = n.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return None
        return StorageWeakRef(val.untyped_storage())

    def _is_tuple_val(n: torch.fx.Node) -> bool:
        return isinstance(n.meta.get("val"), (tuple, list))

    seen_storages = {_storage(n) for n in s_inputs} - {None}
    inline: set[torch.fx.Node] = set()
    for n in s_nodes:
        storage = _storage(n)
        # (wait_tensor's fake output has fresh storage; see _is_view.)
        op = n.all_input_nodes[0] if n.target is operator.getitem else n
        if storage in seen_storages or op.target in _waits():
            if not _is_view(n.target):
                return _skip(f"{n.name} aliases a value but isn't a view op")
            inline.add(n)
        if storage is not None:
            seen_storages.add(storage)
    # A multi-output view op (e.g. split) stays inline with its getitems.
    for n in s_nodes:
        if _is_tuple_val(n) and any(u in inline for u in n.users):
            if not _is_view(n.target):
                return _skip(f"{n.name} aliases a value but isn't a view op")
            inline.add(n)
    # Copied constants are computed inside the slice and never returned.
    inline |= s_set & copied
    if len(inline) == len(s_nodes):
        # Only views are recomputed: no kernels, nothing to drift.
        return _skip("it recomputes only views, so there is nothing to share")
    contracted_names = {n.name for n in s_nodes if n not in inline}

    # The backward must supply every input: primals and saved values (its
    # placeholders), outputs of pieces shared earlier, or recomputed copies it
    # computes itself (e.g. a view another piece keeps inline).
    bw_operands: list[torch.fx.Node] = []
    for inp in s_inputs:
        if inp in bw_of:
            m = bw_of[inp]
        elif inp.op == "placeholder" or inp in saved:
            m = bw_by_name.get(inp.name)
            m = m if m is not None and m.op == "placeholder" else None
        else:
            m = bw_by_name.get(inp.name) if inp in copies else None
        if m is None:
            return _skip(f"its input {inp.name} isn't available in the backward")
        bw_operands.append(m)

    # A multi-output op stays whole in the slice (invoke_subgraph outputs must
    # be flat, so its getitems carry the values), and the slice returns whichever
    # of its results either graph reads: the forward may read some outside the
    # piece (e.g. it saves native_layer_norm's mean and rstd), and the backward
    # may read results the forward never uses (e.g. SDPA's logsumexp).
    all_s_names = {n.name for n in s_nodes}
    fw_extra: dict[str, torch.fx.Node] = {}
    bw_extra: dict[str, torch.fx.Node] = {}
    extra_items: dict[str, tuple[torch.fx.Node, int]] = {}
    for n in s_nodes:
        if not _is_tuple_val(n) or n.name not in contracted_names:
            continue
        outside = [(fw_extra, u) for u in n.users if u not in s_set]
        outside += [
            (bw_extra, u) for u in bw_by_name[n.name].users if u.name not in all_s_names
        ]
        for extra, u in outside:
            idx = u.args[1] if u.target is operator.getitem else None
            if not isinstance(idx, int):
                return _skip(f"{u.name} reads multi-output op {n.name} directly")
            extra[u.name] = u
            extra_items[u.name] = (n, idx)

    # Return only values read outside the slice, in either graph.
    def _used_outside(n: torch.fx.Node) -> bool:
        users = (*n.users, *bw_by_name[n.name].users)
        return any(u.name not in contracted_names for u in users)

    s_out_nodes = [
        n
        for n in s_nodes
        if not _is_tuple_val(n) and n.name in contracted_names and _used_outside(n)
    ]

    # Each call goes right before the first node that reads the piece (see
    # _place). The forward otherwise falls back to right after the last
    # slice input, or the last placeholder when the slice reads nothing (e.g. a
    # recomputed factory).
    node_index = {n: i for i, n in enumerate(fw_graph.nodes)}
    if s_inputs:
        last_input = max(s_inputs, key=lambda n: node_index[n])
    else:
        last_input = fw_graph.find_nodes(op="placeholder")[-1]
    fw_replaced = {n for n in s_nodes if n not in inline} | set(fw_extra.values())
    insert_after, fw_moved, reason = _place(
        fw_replaced, s_inputs, last_input, node_index
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
    bw_insert_after, bw_moved, reason = _place(
        bw_replaced_nodes, bw_operands, last_operand, bw_index
    )
    if reason is not None:
        return _skip(reason)

    # Inductor rewrites a graph's subgraph modules in place while compiling it,
    # so each graph gets its own (identical) copy rather than one shared object.
    def _build_slice_gm() -> torch.fx.GraphModule:
        s_graph = torch.fx.Graph()
        env: dict[torch.fx.Node, torch.fx.Node] = {}
        for inp in s_inputs:
            p = s_graph.placeholder(inp.name)
            p.meta = copy.copy(inp.meta)
            env[inp] = p
        for n in s_nodes:
            env[n] = s_graph.node_copy(n, lambda x: env[x])
        outs = [env[n] for n in s_out_nodes]
        for n, i in extra_items.values():
            outs.append(s_graph.call_function(operator.getitem, (env[n], i)))
            outs[-1].meta["val"] = n.meta["val"][i]
            _keep_budget(outs[-1], n)
        s_graph.output(tuple(outs))
        s_graph.lint()
        return torch.fx.GraphModule(fw_module, s_graph)

    fw_slice_gm, bw_slice_gm = _build_slice_gm(), _build_slice_gm()
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
        replace={n.name: n for n in s_out_nodes} | fw_extra,
        slice_index=slice_index,
        erase_names=contracted_names | fw_extra.keys(),
        insert_after=insert_after,
        move_below=fw_moved,
    )
    bw_outs = _splice_slice_call(
        bw_module,
        bw_slice_gm,
        f"{identifier}_bw",
        slice_out_vals,
        operands=bw_operands,
        replace={n.name: bw_by_name[n.name] for n in s_out_nodes} | bw_extra,
        slice_index=slice_index,
        erase_names=bw_replaced,
        insert_after=bw_insert_after,
        move_below=bw_moved,
    )
    outputs = {fw_outs[n.name]: bw_outs[n.name] for n in s_out_nodes}
    # A saved result now comes from the forward call, while the backward still
    # reads it from its placeholder.
    for name in fw_extra:
        m = bw_by_name.get(name)
        if m is not None and m.op == "placeholder":
            outputs[fw_outs[name]] = m
    return outputs


def _place(
    replaced: set[torch.fx.Node],
    operands: list[torch.fx.Node],
    fallback: torch.fx.Node,
    index: dict[torch.fx.Node, int],
) -> tuple[torch.fx.Node, list[torch.fx.Node], str | None]:
    """The node after which the call replacing ``replaced`` goes, and the
    readers that move below it. The call goes right before the first node that
    reads one of them, so its outputs don't live longer than needed, and before
    any side effect on an operand (e.g. an input mutation's copy_), which no
    data edge orders it against. If an operand only comes later, or moving
    there isn't possible, it goes after ``fallback`` and the readers before
    that move below it. Returns the reason if neither works."""
    readers = [u for n in replaced for u in n.users if u not in replaced]
    readers += [u for o in operands for u in o.users if u.is_impure()]
    first = min(readers, key=index.__getitem__, default=None)
    points = [fallback]
    if first is not None and all(index[o] < index[first] for o in operands):
        points.insert(0, first.prev)
    reason = None
    for point in points:
        moved, reason = _readers_to_move(replaced, operands, point)
        if reason is None:
            return point, moved, None
    return fallback, [], reason


def _readers_to_move(
    replaced: set[torch.fx.Node],
    operands: list[torch.fx.Node],
    insert_after: torch.fx.Node,
) -> tuple[list[torch.fx.Node], str | None]:
    """A call inserted after ``insert_after`` will replace the nodes in
    ``replaced``, so whatever reads them (directly or through each other) up to
    that point has to move below the call. Returns those readers in graph order,
    or the reason they can't move."""
    tainted = set(replaced)
    readers: list[torch.fx.Node] = []
    crossed: list[torch.fx.Node] = []
    started = False
    for n in insert_after.graph.nodes:
        if n in replaced:
            started = True
        elif started:
            if any(a in tainted for a in n.all_input_nodes):
                tainted.add(n)
                readers.append(n)
            else:
                crossed.append(n)
        if n is insert_after:
            break
    for inp in operands:
        if inp in tainted:
            # A path leaves the piece and comes back, e.g. through another
            # region's recompute.
            return [], f"its input {inp.name} depends on its own recompute"
    # Ordering against other nodes is carried by data edges (stream syncs are
    # wrapped in control_deps), except that a sync_dealloc must stay the last
    # use of its tensor and a side effect on an operand must stay after the
    # piece reads it.
    operand_set = set(operands)
    for n in crossed:
        if n.target is torch.ops.streams.sync_dealloc.default or (
            n.is_impure() and any(a in operand_set for a in n.all_input_nodes)
        ):
            return [], f"it would move work past {n.name}"
    return readers, None


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
) -> dict[str, torch.fx.Node]:
    """Insert ``invoke_subgraph(slice_gm, identifier, *operands)`` after
    ``insert_after``, swap each node in ``replace`` for the matching output, move
    the nodes in ``move_below`` (in order) right after the call's outputs, and
    erase the nodes named in ``erase_names``. Returns the output getitems by the
    name of the node they replaced."""
    from torch._higher_order_ops.invoke_subgraph import (
        invoke_subgraph,
        NestedCompileRegionOptions,
    )

    g = parent_gm.graph
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
    # heuristics instead, so both calls lower identically.
    config = NestedCompileRegionOptions(inductor_config_patches={"deterministic": True})
    call.meta.setdefault("custom", {})["nested_region_config"] = config
    _keep_budget(call, *replace.values())
    anchor = call
    getitems = {}
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
    # Erase in reverse topological order so no node is erased before its users.
    for n in reversed([n for n in g.nodes if n.name in erase_names]):
        g.erase_node(n)
    return getitems
