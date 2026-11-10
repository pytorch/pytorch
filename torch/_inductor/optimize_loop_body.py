from __future__ import annotations

import dataclasses
import inspect
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.fx import Graph, Node

from .dtype_propagation import DtypePropagationOpsHandler
from .ops_handler import DefaultHandler, OpsHandler
from .virtualized import V


if TYPE_CHECKING:
    from .loop_body import LoopBody


_TO_DTYPE_SIGNATURE = inspect.signature(OpsHandler.to_dtype)


@dataclasses.dataclass
class _Dtype:
    dtype: torch.dtype

    def __getitem__(self, index: int) -> _Dtype:
        # Multi-output ops such as welford reductions get one dtype for all
        # outputs from dtype propagation.
        return self


class _RecordNodeDtypes(DefaultHandler):
    """Ops handler that records the dtype each FX node of a LoopBody produces."""

    def __init__(self) -> None:
        self.dtype_prop = DtypePropagationOpsHandler()
        self.dtypes: dict[Node, torch.dtype] = {}

    def indirect_indexing(self, *args: object, **kwargs: object) -> sympy.Expr:
        return sympy.S.Zero

    def _default(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> Any:
        out = getattr(self.dtype_prop, name)(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(_Dtype(dtype) for dtype in out)
        if not isinstance(out, torch.dtype):
            return out
        self.dtypes[V.interpreter.current_node] = out
        return _Dtype(out)


def form_explicit_fmas(loop_body: LoopBody) -> bool:
    """Rewrite each float ``add``/``sub`` of a ``mul`` into an ``fma``.

    Before::

        p = mul(a, b)
        y = add(p, c)  # sub(p, c), sub(c, p)

    After::

        y = fma(a, b, c)  # fma(a, b, neg(c)), fma(neg(a), b, c)

    Used when Triton's own FMA contraction is off for reproducibility: which
    multiply-adds Triton contracts depends on the block size, while this choice
    depends only on the graph, so it is the same for every config. ``p`` stays
    if something else uses it. Multiply-adds split across fused nodes, or inside
    ``masked`` subblocks, are not contracted.
    """
    counts = loop_body.op_counts
    if not counts["mul"] or not (counts["add"] or counts["sub"]):
        return False

    recorder = _RecordNodeDtypes()
    with V.set_ops_handler(recorder):
        loop_body(list(loop_body.var_ranges), allow_same_symbol_in_index=True)
    dtypes = recorder.dtypes

    def same_float_dtype(*nodes: object) -> torch.dtype | None:
        found = {dtypes.get(n) if isinstance(n, Node) else None for n in nodes}
        if found in ({torch.float32}, {torch.float64}):
            return found.pop()
        return None

    graph = loop_body.root_block.graph
    formed = 0
    candidates = [
        *graph.find_nodes(op="call_method", target="add"),
        *graph.find_nodes(op="call_method", target="sub"),
    ]
    for node in candidates:
        ops, lhs, rhs = node.args
        sub = node.target == "sub"
        for prod, addend, prod_is_lhs in ((lhs, rhs, True), (rhs, lhs, False)):
            if not (isinstance(prod, Node) and prod.target == "mul"):
                continue
            _, a, b = prod.args
            if (dtype := same_float_dtype(a, b, addend, prod)) is None:
                continue
            with graph.inserting_before(node):
                if sub and prod_is_lhs:
                    addend = graph.call_method("neg", (ops, addend))
                    dtypes[addend] = dtype
                elif sub:
                    a = graph.call_method("neg", (ops, a))
                    dtypes[a] = dtype
                fma = graph.call_method("fma", (ops, a, b, addend))
            dtypes[fma] = dtype
            node.replace_all_uses_with(fma)
            graph.erase_node(node)
            counts[node.target] -= 1
            counts["fma"] += 1
            counts["neg"] += sub
            if not prod.users:
                graph.erase_node(prod)
                counts["mul"] -= 1
            formed += 1
            break
    return formed > 0


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

        bound = _TO_DTYPE_SIGNATURE.bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        _, value, dtype, src_dtype, use_compute_types = bound.arguments.values()
        # Repeated emulation casts omit src_dtype, for example:
        #   to_dtype(value, torch.bfloat16, use_compute_types=False)
        # The first downcast can be the real conversion that produced the
        # low-precision value, for example:
        #   to_dtype(value, torch.bfloat16, src_dtype=torch.int64,
        #            use_compute_types=False)
        if (
            not isinstance(value, Node)
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
        for up2_node in graph.find_nodes(op="call_method", target="to_dtype"):
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
        return erased

    if loop_body.op_counts["to_dtype"] < 4:
        return False

    erased = sum(
        eliminate_from_graph(block.graph)
        for block in (loop_body.root_block, *loop_body.subblocks.values())
    )
    if not erased:
        return False

    # op_counts was filled in while tracing, so keep it in sync with the graphs.
    loop_body.op_counts["to_dtype"] -= erased
    return True
