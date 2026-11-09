"""Share the backward's recompute with the forward.

The partitioner recomputes a checkpointed region by copying its forward ops
into the backward graph. The forward and backward are compiled separately, so
the two copies can lower to different kernels and the recompute can drift
numerically from the forward (gh-186572). With
torch._functorch.config.deterministic_recompute, each checkpointed region's
recompute is outlined into a subgraph that both graphs invoke, compiled with
Inductor deterministic=True so both lowerings match.

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
import logging
import operator
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._functorch import config
from torch._inductor.fx_utils import get_mutated_input_nodes, get_node_storage
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.fx.passes.reinplace import _is_view_op


log = logging.getLogger(__name__)


def deterministic_recompute(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_fwd_outputs: int,
) -> None:
    """Rewrite the partitioned fw_module and bw_module in place so each piece of
    the backward's recompute runs as a shared subgraph."""
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
                log.info("deterministic_recompute: %s is in a memory pool", n.name)
                return
    # Only the storage they write orders in-place writes (e.g. an input
    # mutation's copy_) against other nodes, which outlining could reorder; a
    # graph with any is left as partitioned for now.
    for gm in (fw_module, bw_module):
        if writer := next((n for n in gm.graph.nodes if _writes(n)), None):
            log.info("deterministic_recompute: %s writes in place", writer.name)
            return
    fw_outs = fw_module.graph.find_nodes(op="output")[0].args[0]
    saved = {n for n in fw_outs[num_fwd_outputs:] if isinstance(n, torch.fx.Node)}
    copies, regions = _recomputed_copies(fw_module, bw_module, saved)

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
                    "deterministic_recompute: %s left as partitioned, with what "
                    "is computed from it: it reads or computes a value carrying "
                    "an unbacked symbol",
                    n.name,
                )
        for nodes in regions.values():
            nodes[:] = [n for n in nodes if n not in dropped]

    # A constant is a value computed from no tensor graph input: an arange, an
    # attribute such as a tensor constant, math on those, or SymInt
    # arithmetic. An impure op isn't one, since copying it would run it again.
    # A piece copies the recomputed constants and the attributes it reads, so
    # they fuse into its kernels instead of being operands; a submodule (e.g.
    # flex_attention's score_mod) couldn't be an operand anyway.
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
    copied = copies & constants
    copied |= set(fw_module.graph.find_nodes(op="get_attr"))
    # A tensor constant the backward also reads directly (e.g. the gradient of
    # h * table reads table) is read from its own copy there, which could differ
    # from the one the forward's piece computes; say so.
    copy_names = {n.name for n in copies}
    bw_body = {n.name: n for n in bw_module.graph.nodes}
    for c in fw_module.graph.nodes:
        if c not in copies or c not in constants:
            continue
        direct = any(u.name not in copy_names for u in bw_body[c.name].users)
        if direct and isinstance(c.meta.get("val"), torch.Tensor):
            log.info(
                "deterministic_recompute: %s left as partitioned: the backward "
                "reads it directly",
                c.name,
            )

    # A region's recompute is one piece, with the copied constants it reads. If
    # a path leaves it and comes back (e.g. through a value a selective policy
    # saves), no single call can replace it, and it is left as partitioned (see
    # _readers_to_move).
    order = {n: i for i, n in enumerate(fw_module.graph.nodes)}

    def _with_copied(nodes: list[torch.fx.Node]) -> list[torch.fx.Node]:
        piece = set(nodes)
        stack = [a for n in nodes for a in n.all_input_nodes if a in copied]
        while stack:
            a = stack.pop()
            if a not in piece:
                piece.add(a)
                stack.extend(b for b in a.all_input_nodes if b in copied)
        return sorted(piece, key=order.__getitem__)

    pieces = []
    for nodes in regions.values():
        rest = [n for n in nodes if n not in copied]
        if rest:
            pieces.append(_with_copied(rest))

    changed = False
    # The backward node holding each value a shared piece returns, so later
    # pieces can read it.
    bw_of: dict[torch.fx.Node, torch.fx.Node] = {}
    # Name pieces by their order in this graph, not by ac_graph_id: that is a
    # process-global counter, so recompiling the same graph would get new names
    # and miss the FX graph cache.
    for i, piece in enumerate(pieces):
        identifier = f"recompute_{i}"
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


_SCALARS = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)


def _recomputed_copies(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    saved: set[torch.fx.Node],
) -> tuple[set[torch.fx.Node], dict[Any, list[torch.fx.Node]]]:
    """The forward nodes whose copies the backward recomputes, given the values
    the forward ``saved`` for it, and those of them in each checkpoint region
    (keyed by ac_graph_id, in graph order). Logs where the backward's recompute
    stops matching the forward."""
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

    def _names(n: torch.fx.Node) -> Any:
        return torch.fx.node.map_arg((n.args, n.kwargs), lambda a: a.name)

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
                    "deterministic_recompute: %s left as partitioned: the "
                    "backward recomputes it differently",
                    missing[0].name if missing else n.name,
                )
            not_copies.add(n)
            not_copies.update(missing)
            continue
        copies.add(n)
        # Recompute the partitioner chose outside checkpoint regions (e.g. under
        # activation_memory_budget) is a copy in no region; pieces recompute
        # what they read from it if it's cheap (see _share_region).
        if (key := n.meta.get("ac_graph_id")) is not None:
            regions[key].append(n)
    return copies, regions


@functools.cache
def _waits() -> tuple[Any, ...]:
    if not torch.distributed.is_available():
        return ()
    return (
        torch.ops._c10d_functional.wait_tensor.default,
        torch.ops._c10d_functional.wait_tensors.default,
    )


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
    # each graph computes separately (a copy no earlier piece shares, e.g.
    # recompute outside the region) could differ between them, so the piece
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
        cheap = a.op == "get_attr" or a in copied or _is_view(a.target)
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
        # Only views are recomputed: no kernels, nothing to drift.
        return _skip("it recomputes only views, so there is nothing to share")
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
    # placeholders), or outputs of pieces shared earlier.
    bw_operands: list[torch.fx.Node] = []
    for inp in s_inputs:
        if inp in bw_of:
            m = bw_of[inp]
        elif inp.op == "placeholder" or inp in saved:
            m = bw_by_name.get(inp.name)
            m = m if m is not None and m.op == "placeholder" else None
        else:
            m = None
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

    # Each call goes right before the first node that reads the piece (see
    # _place). The forward otherwise falls back to right after the last slice
    # input. (Every piece has one: a node computed from no tensor input is a
    # constant.)
    last_input = max(s_inputs, key=node_index.__getitem__)
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
    replaced: set[torch.fx.Node],
    operands: list[torch.fx.Node],
    fallback: torch.fx.Node,
    index: dict[torch.fx.Node, int],
) -> tuple[torch.fx.Node, list[torch.fx.Node], str | None]:
    """The node after which the call replacing ``replaced`` goes, and the
    readers that move below it, or the reason it can't go anywhere. The call
    goes right before the first node that reads one of ``replaced``, so it
    computes them no earlier than the first of them is needed. If an operand
    only comes then or later, it goes after ``fallback`` (at or after every
    operand) and the readers before that move below it. The replaced
    nodes move to the call and the readers below it with only data edges to
    order them, so the caller must make sure no in-place write (see _writes)
    is ordered against them."""
    readers = [u for n in replaced for u in n.users if u not in replaced]
    first = min(readers, key=index.__getitem__, default=None)
    if first is not None and all(index[o] < index[first] for o in operands):
        return first.prev, [], None
    moved, reason = _readers_to_move(replaced, operands, fallback)
    return fallback, moved, reason


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
        tgt.replace_all_uses_with(gi)
        getitems[name] = anchor = gi
    for n in move_below:
        anchor.append(n)
        anchor = n
    for n in reversed(erase):
        g.erase_node(n)
    return getitems
