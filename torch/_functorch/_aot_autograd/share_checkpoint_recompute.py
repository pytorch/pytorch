"""Share an activation-checkpointed region's recompute with its forward.

The partitioner recomputes a checkpointed region by copying the region's forward
ops into the backward graph. The forward and backward are compiled separately, so
the two copies can lower to different kernels and the recompute can drift
numerically from the forward (gh-186572). With
torch._functorch.config.share_checkpoint_recompute, each convex piece of a
region's recompute is outlined into a subgraph that both graphs invoke, compiled
with Inductor deterministic=True so both lowerings match.

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
import logging
import operator
from collections import defaultdict
from typing import Any

import torch
from torch.fx.passes.reinplace import _is_view_op
from torch.multiprocessing.reductions import StorageWeakRef


log = logging.getLogger(__name__)


def share_checkpoint_recompute(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_fwd_outputs: int,
) -> None:
    """Rewrite the partitioned fw_module and bw_module in place so each piece of
    a checkpointed region's recompute runs as a shared subgraph."""
    fw_outs = fw_module.graph.find_nodes(op="output")[0].args[0]
    saved = {n for n in fw_outs[num_fwd_outputs:] if isinstance(n, torch.fx.Node)}
    # Both graphs are extracted from the joint graph, so the backward's
    # recomputed copy of a forward node keeps its name; checkpointed nodes carry
    # their region's ac_graph_id. Partitioner passes can also create backward
    # nodes that reuse a forward name for a different computation (e.g. RNG
    # functionalization re-running dropout from the forward's saved state), so a
    # copy must also be the same op on the same inputs: primals, saved values,
    # or other copies.
    bw_body = {
        n.name: n
        for n in bw_module.graph.nodes
        if n.op not in ("placeholder", "output")
    }

    def _names(n: torch.fx.Node) -> Any:
        return torch.fx.node.map_arg((n.args, n.kwargs), lambda a: a.name)

    copies: set[torch.fx.Node] = set()
    regions: dict[Any, list[torch.fx.Node]] = defaultdict(list)
    for n in fw_module.graph.nodes:
        bw_n = bw_body.get(n.name)
        if (
            n.op == "call_function"
            and n not in saved
            and bw_n is not None
            and bw_n.target == n.target
            and _names(bw_n) == _names(n)
            and all(
                a.op == "placeholder" or a in saved or a in copies
                for a in n.all_input_nodes
            )
        ):
            copies.add(n)
            if (ac_graph_id := n.meta.get("ac_graph_id")) is not None:
                regions[ac_graph_id].append(n)

    # A constant depends on no tensor graph input (e.g. an arange, or math on
    # one); SymInt sizes don't count. A piece copies the recomputed
    # constants it reads instead of taking them as inputs, so they fuse into its
    # kernels. A constant the backward reads directly must be the value the
    # forward used, so it is shared as a piece of its own instead, which other
    # pieces read.
    recomputed = {n for nodes in regions.values() for n in nodes}
    scalar_types = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)
    constants: set[torch.fx.Node] = set()
    for n in fw_module.graph.nodes:
        if (
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
        if c.target is operator.getitem or _is_view_op(c.target):
            for a in c.all_input_nodes:
                if a in recomputed and a in constants and a not in direct:
                    direct.add(a)
                    stack.append(a)
    copied = (recomputed & constants) - direct

    # A piece must be convex: if a path left the recompute (e.g. through a saved
    # matmul) and came back into the same piece, no single call could replace
    # it. Count how often the path to each node passes from a recomputed node to
    # one that isn't, and only join nodes with the same count into a piece.
    # Constants never share a piece with other nodes, so paths out of them don't
    # count.
    stage: dict[torch.fx.Node, int] = {}
    for n in fw_module.graph.nodes:
        stage[n] = max(
            (
                stage[a]
                + (a in recomputed and a not in constants and n not in recomputed)
                for a in n.all_input_nodes
            ),
            default=0,
        )

    # Constant pieces go first, so the pieces that read them find their outputs
    # in bw_of.
    order = {n: i for i, n in enumerate(fw_module.graph.nodes)}
    pieces = _split_region(sorted(direct, key=order.__getitem__), stage, copied, order)
    for nodes in regions.values():
        rest = [n for n in nodes if n not in direct]
        pieces += _split_region(rest, stage, copied, order)

    changed = False
    # The backward node holding each value a shared piece returns, so later
    # pieces can read it.
    bw_of: dict[torch.fx.Node, torch.fx.Node] = {}
    # Name pieces by their order in this graph, not by ac_graph_id: that is a
    # process-global counter, so recompiling the same graph would get new names
    # and miss the FX graph cache.
    for i, piece in enumerate(pieces):
        identifier = f"checkpoint_recompute_{i}"
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
    stage: dict[torch.fx.Node, int],
    copied: set[torch.fx.Node],
    order: dict[torch.fx.Node, int],
) -> list[list[torch.fx.Node]]:
    """Split a region's recomputed nodes (in topological order) into convex
    pieces, e.g. two recomputed stretches separated by a saved matmul: nodes join
    a piece only through recomputed edges within one ``stage``. Each piece gets
    its own shared subgraph; a later piece may read an earlier one's outputs.
    Views of values outside the region and ``copied`` constants are free to
    replay, so they join every piece that reads them instead of joining pieces
    together. Pieces and their nodes are sorted by ``order``."""
    in_region = set(nodes)
    outside_views: set[torch.fx.Node] = set()
    for n in nodes:
        is_view = n.target is operator.getitem or _is_view_op(n.target)
        if is_view and all(
            a not in in_region or a in outside_views for a in n.all_input_nodes
        ):
            outside_views.add(n)
    replayed = outside_views | copied

    parent = {n: n for n in nodes if n not in replayed}

    def _find(n: torch.fx.Node) -> torch.fx.Node:
        while parent[n] is not n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for n in list(parent):
        for a in n.all_input_nodes:
            if a in parent and stage[a] == stage[n]:
                parent[_find(a)] = _find(n)

    groups: dict[torch.fx.Node, set[torch.fx.Node]] = defaultdict(set)
    for n in parent:
        groups[_find(n)].add(n)
    pieces = []
    for group in groups.values():
        stack = [a for n in group for a in n.all_input_nodes if a in replayed]
        while stack:
            v = stack.pop()
            if v not in group:
                group.add(v)
                stack.extend(a for a in v.all_input_nodes if a in replayed)
        pieces.append(sorted(group, key=order.__getitem__))
    return sorted(pieces, key=lambda piece: order[piece[0]])


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
            "share_checkpoint_recompute: %s left as partitioned: %s", identifier, reason
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
        if storage in seen_storages:
            if n.target is not operator.getitem and not _is_view_op(n.target):
                return _skip(f"{n.name} aliases a value but isn't a view op")
            inline.add(n)
        elif storage is not None:
            seen_storages.add(storage)
    # A multi-output view op (e.g. split) stays inline with its getitems.
    for n in s_nodes:
        if _is_tuple_val(n) and any(u in inline for u in n.users):
            if not _is_view_op(n.target):
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

    # The forward call goes right after the last slice input, or after the last
    # placeholder when the slice reads nothing (e.g. a recomputed factory).
    node_index = {n: i for i, n in enumerate(fw_graph.nodes)}
    if s_inputs:
        insert_after = max(s_inputs, key=lambda n: node_index[n])
    else:
        insert_after = fw_graph.find_nodes(op="placeholder")[-1]
    fw_replaced = {n for n in s_nodes if n not in inline} | set(fw_extra.values())
    fw_moved, reason = _readers_to_move(fw_replaced, s_inputs, insert_after)
    if reason is not None:
        return _skip(reason)

    # Run the backward call where the partitioner placed the piece's recompute,
    # so pieces still recompute one at a time instead of all up front, but not
    # before its operands.
    bw_index = {n: i for i, n in enumerate(bw_graph.nodes)}
    first_recompute = next(n for n in bw_graph.nodes if n.name in contracted_names)
    bw_insert_after = max(
        [first_recompute.prev, *bw_operands], key=lambda n: bw_index[n]
    )
    bw_replaced = contracted_names | bw_extra.keys()
    bw_moved, reason = _readers_to_move(
        {bw_by_name[name] for name in bw_replaced}, bw_operands, bw_insert_after
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
            env[n].meta = copy.copy(n.meta)
        outs = [env[n] for n in s_out_nodes]
        for n, i in extra_items.values():
            outs.append(s_graph.call_function(operator.getitem, (env[n], i)))
            outs[-1].meta["val"] = n.meta["val"][i]
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
    # use of its tensor.
    for n in crossed:
        if n.target is torch.ops.streams.sync_dealloc.default:
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
    call.meta["custom"] = {"nested_region_config": config}
    anchor = call
    getitems = {}
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
    # Erase in reverse topological order so no node is erased before its users.
    for n in reversed([n for n in g.nodes if n.name in erase_names]):
        g.erase_node(n)
    return getitems
