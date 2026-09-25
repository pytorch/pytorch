from __future__ import annotations

import torch
from torch.fx import Graph, Node

from .loop_body import LoopBody


def eliminate_redundant_lowp_round_trips(loop_body: LoopBody) -> bool:
    """Collapse repeated fp16/bf16 rounding boundaries in a LoopBody.

    Before::

        down1 = to_dtype(value, lowp, src_dtype=..., use_compute_types=False)
        up1 = to_dtype(down1, lowp)
        down2 = to_dtype(up1, lowp, use_compute_types=False)  # redundant!
        up2 = to_dtype(down2, lowp)  # redundant!
        use(up2)

    After::

        down1 = to_dtype(value, lowp, use_compute_types=False)
        up1 = to_dtype(down1, lowp)
        use(up1)

    ``up1`` contains the same value as the already-rounded ``down1``, represented
    in the type the backend uses for calculation. Casting it back to the same
    low-precision dtype therefore produces ``down1`` again, so ``down2`` and
    ``up2`` do not change the result. Running this pass again does nothing
    because only the first pair remains.
    """
    lowp_dtypes = (torch.float16, torch.bfloat16)

    def get_to_dtype(
        node: Node, *, allow_src_dtype: bool = False
    ) -> tuple[Node, torch.dtype, bool] | None:
        """Read ``to_dtype(value, dtype, ...)`` from one FX node.

        For example, ``to_dtype(x, torch.bfloat16, use_compute_types=False)``
        returns ``(x, torch.bfloat16, False)``. Other operations return None.
        """
        if node.op != "call_method" or node.target != "to_dtype":
            return None
        if not (3 <= len(node.args) <= 5):
            return None

        # (value, dtype, src_dtype=None, use_compute_types=True)
        value = node.args[1]
        dtype = node.args[2]
        src_dtype = (
            node.args[3] if len(node.args) >= 4 else node.kwargs.get("src_dtype")
        )
        use_compute_types = (
            node.args[4]
            if len(node.args) == 5
            else node.kwargs.get("use_compute_types", True)
        )
        # Repeated emulation casts omit src_dtype, for example:
        #   to_dtype(value, torch.bfloat16, use_compute_types=False)
        # The first downcast can be the real conversion that produced the
        # low-precision value, for example:
        #   to_dtype(value, torch.bfloat16, src_dtype=torch.int64,
        #            use_compute_types=False)
        if (
            not isinstance(value, Node)
            or not isinstance(dtype, torch.dtype)
            or dtype not in lowp_dtypes
            or (src_dtype is not None and not allow_src_dtype)
            or type(use_compute_types) is not bool
        ):
            return None
        return value, dtype, use_compute_types

    def eliminate_from_graph(graph: Graph) -> int:
        """Remove the second pair from ``down1 -> up1 -> down2 -> up2``.

        Match this exact chain::

            down1: ops.to_dtype(value, lowp, src_dtype=dtype, use_compute_types=False)
            up1: ops.to_dtype(down1, lowp)
            down2: ops.to_dtype(up1, lowp, use_compute_types=False)
            up2: ops.to_dtype(down2, lowp)

        Starting at ``up2``, walk backward to ``down1`` and verify that all
        four casts use the same low-precision dtype and alternate between the
        storage and compute representations.
        """
        erased = 0
        for up2_node in list(graph.nodes):
            if (up2 := get_to_dtype(up2_node)) is None:
                continue
            down2_node, dtype, use_compute_types = up2
            if not use_compute_types:
                continue

            if (down2 := get_to_dtype(down2_node)) is None:
                continue
            up1_node, down2_dtype, use_compute_types = down2
            if use_compute_types or down2_dtype != dtype:
                continue

            if (up1 := get_to_dtype(up1_node)) is None:
                continue
            down1_node, up1_dtype, use_compute_types = up1
            if not use_compute_types or up1_dtype != dtype:
                continue

            if (down1 := get_to_dtype(down1_node, allow_src_dtype=True)) is None:
                continue
            _, down1_dtype, use_compute_types = down1
            if use_compute_types or down1_dtype != dtype:
                continue

            # up2 and up1 contain the same value, so use up1 and remove up2.
            # Remove down2 too when up2 was its only user.
            up2_node.replace_all_uses_with(up1_node)
            graph.erase_node(up2_node)
            erased += 1
            if not down2_node.users:
                graph.erase_node(down2_node)
                erased += 1

        if erased and graph.owning_module is None:
            # Check that erased nodes are no longer referenced. Debug LoopBody
            # graphs contain function-valued call_module targets that this FX
            # check does not understand, so skip it for those graphs.
            graph.lint()
        return erased

    erased = sum(
        eliminate_from_graph(block.graph)
        for block in (loop_body.root_block, *loop_body.subblocks.values())
    )
    if not erased:
        return False

    # LoopBody copies intentionally share their FX graphs and op_counts counter.
    # Update the counter in place so scheduler cost checks see the rewrite.
    loop_body.op_counts["to_dtype"] -= erased
    if loop_body.op_counts["to_dtype"] == 0:
        del loop_body.op_counts["to_dtype"]

    # These cache_on_self analyses contain FX nodes and graph-derived bounds.
    # Clear them so their next users recompute from the rewritten graphs.
    LoopBody.get_nodes.clear_cache(loop_body)
    LoopBody.bounds.clear_cache(loop_body)
    return True
