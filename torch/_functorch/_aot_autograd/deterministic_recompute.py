"""Share the backward's recompute with the forward.

The partitioner recomputes a checkpointed region by copying its forward ops
into the backward graph. The forward and backward are compiled separately, so
the two copies can lower to different kernels and the recompute can drift
numerically from the forward (gh-186572). The helpers here find that
recompute and outline it into a subgraph both graphs invoke.
"""

import copy
import logging
import operator
from collections import defaultdict
from typing import Any

import sympy

import torch
from torch._inductor.fx_utils import get_mutated_input_nodes, get_node_storage
from torch.fx.experimental.symbolic_shapes import free_symbols


log = logging.getLogger(__name__)


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
        # activation_memory_budget) is a copy in no region.
        if (key := n.meta.get("ac_graph_id")) is not None:
            regions[key].append(n)
    return copies, regions


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
) -> torch.fx.GraphModule:
    """A subgraph computing ``nodes`` (in topological order) from placeholders
    for ``inputs``, returning ``outputs``."""
    graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    for inp in inputs:
        env[inp] = graph.placeholder(inp.name)
        env[inp].meta = copy.copy(inp.meta)
    for n in nodes:
        env[n] = graph.node_copy(n, lambda x: env[x])
    graph.output(tuple(env[n] for n in outputs))
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
) -> dict[str, torch.fx.Node]:
    """Insert ``invoke_subgraph(slice_gm, identifier, *operands)`` after
    ``insert_after``, swap each node in ``replace`` for the matching output, move
    the nodes in ``move_below`` (in order) right after the call's outputs, and
    erase the nodes named in ``erase_names``. Returns the output getitems by the name
    of the node they replaced."""
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

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
    for n in reversed(erase):
        g.erase_node(n)
    return getitems
