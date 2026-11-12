"""Share the backward's recompute with the forward.

The partitioner recomputes a checkpointed region by copying its forward ops
into the backward graph. The forward and backward are compiled separately, so
the two copies can lower to different kernels and the recompute can drift
numerically from the forward (gh-186572). The helpers here find that
recompute.
"""

import logging
from collections import defaultdict
from typing import Any

import torch


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
