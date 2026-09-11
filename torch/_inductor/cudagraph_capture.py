# mypy: allow-untyped-defs
"""Capture-safety analysis for AOTI cuda graph.

Answers one question: would capturing this node inside a cuda graph be wrong or
crash? That is deliberately narrower than "does this node need a partition
boundary". Boundary placement is a property of the REGIONAL path -- a partition
is limited to one dynamic symbol and cannot have a buffer's writers and readers
straddle its edge, which makes gathers, shared-storage concats and batch-dim
transitions boundaries there. None of those are capture-safety problems, so none
of them block whole-graph capture, which has no boundaries and keys on every
dynamic symbol at once.

The regional path builds on these same predicates rather than redefining them.
"""

from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from .scheduler import BaseSchedulerNode


# Examples listed per reason in the error message. A blocked model can have
# hundreds of offending nodes; listing them all buries the signal.
_MAX_EXAMPLES_PER_REASON = 5


@dataclasses.dataclass(frozen=True)
class CaptureBlocker:
    """One node that cannot be captured, and why."""

    node_name: str
    reason: str
    op: str


def _leaf_nodes(node: BaseSchedulerNode):
    """Flatten fused nodes so each check sees a single scheduler node."""
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        for snode in node.snodes:
            yield from _leaf_nodes(snode)
    else:
        yield node


def _origin_ops(node: BaseSchedulerNode) -> OrderedSet[Any]:
    """Aten/prim op packets this node was lowered from, via its FX origins."""
    ir_node = node.node
    origins = getattr(ir_node, "origins", None) if ir_node is not None else None
    packets: OrderedSet[Any] = OrderedSet()
    if not origins:
        return packets
    for fx_node in origins:
        target = getattr(fx_node, "target", None)
        if target is None:
            continue
        packets.add(
            target.overloadpacket
            if isinstance(target, torch._ops.OpOverload)
            else target
        )
    return packets


def _describe_op(node: BaseSchedulerNode) -> str:
    return ", ".join(sorted(str(op) for op in _origin_ops(node))) or "<unknown>"


def _sdpa_packets() -> OrderedSet[Any]:
    aten = torch.ops.aten
    names = (
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
    )
    return OrderedSet(
        op for op in (getattr(aten, n, None) for n in names) if op is not None
    )


def _is_rng_free_sdpa(fx_node: Any, target: Any) -> bool:
    """True for an SDPA call whose dropout is disabled.

    Attention carries Tag.nondeterministic_seeded because it *may* draw for
    dropout, but at inference dropout_p is 0 and it draws nothing -- so treating
    the tag alone as "uses RNG" needlessly blocks capture on any attention
    model. Deliberately narrow: only the SDPA family, and only when dropout_p is
    literally 0. Getting this wrong would silently freeze a real RNG draw, so it
    does not generalise to every op that happens to have a dropout argument.
    The argument is located by SCHEMA NAME rather than a hardcoded position,
    since it sits at a different index in each SDPA overload.
    """
    if not isinstance(target, torch._ops.OpOverload):
        return False
    if target.overloadpacket not in _sdpa_packets():
        return False
    for i, arg in enumerate(target._schema.arguments):
        if arg.name != "dropout_p":
            continue
        if "dropout_p" in fx_node.kwargs:
            value = fx_node.kwargs["dropout_p"]
        elif i < len(fx_node.args):
            value = fx_node.args[i]
        else:
            value = arg.default_value
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
    return False


def _node_uses_rng(node: BaseSchedulerNode) -> bool:
    """True if the node involves RNG (a random draw or seed generation).

    Unsafe to capture: capture freezes the philox seed+offset, so every replay
    reproduces identical draws instead of advancing, diverging from eager.
    (cuda graph trees handle this with capturable generator state registered on
    the graph; AOTI has no equivalent.) Detected through FX origins -- aten RNG
    ops carry Tag.nondeterministic_seeded, and Inductor lowers them to
    inductor_prims.{random,randint,seed,seeds,lookup_seed} (see
    fx_passes/replace_random.py). seed/seeds also carry the tag; random/randint/
    lookup_seed do not, so the op objects are matched directly as well.
    """
    from torch._inductor import inductor_prims

    rng_ops: OrderedSet[Any] = OrderedSet()
    for name in ("seed", "seeds", "lookup_seed", "random", "randint"):
        op = getattr(inductor_prims, name, None)
        if op is not None:
            rng_ops.add(op)

    ir_node = node.node
    origins = getattr(ir_node, "origins", None) if ir_node is not None else None
    for fx_node in origins or ():
        target = getattr(fx_node, "target", None)
        if target is None:
            continue
        if target in rng_ops:
            return True
        if isinstance(target, torch._ops.OpOverload):
            if (
                torch.Tag.nondeterministic_seeded in target.tags
                and not _is_rng_free_sdpa(fx_node, target)
            ):
                return True
            if target.overloadpacket in rng_ops:
                return True
    return False


def _fp8_dtypes() -> OrderedSet[Any]:
    dtypes: OrderedSet[Any] = OrderedSet()
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            dtypes.add(dt)
    return dtypes


def _node_uses_fp8_gemm(node: BaseSchedulerNode) -> bool:
    """True if the node is a matmul consuming float8 operands.

    Unsafe to capture: the fp8 cublasLt path allocates its workspace lazily on
    the first call, which is illegal during capture, so a captured fp8 gemm
    raises CUBLAS_STATUS_INTERNAL_ERROR / IMAs at any shape other than the one
    warmed up before capture. Matches the native fp8 gemm (aten._scaled_mm /
    _scaled_mm_v2) and the unfused fp8 fallback (aten.mm/addmm/bmm/baddbmm over
    float8 inputs, e.g. the addmm over a folded permuted fp8 weight).
    """
    aten = torch.ops.aten
    scaled = OrderedSet(
        op
        for op in (getattr(aten, n, None) for n in ("_scaled_mm", "_scaled_mm_v2"))
        if op is not None
    )
    gemms = OrderedSet(
        op
        for op in (getattr(aten, n, None) for n in ("mm", "addmm", "bmm", "baddbmm"))
        if op is not None
    )
    fp8 = _fp8_dtypes()

    ir_node = node.node
    origins = getattr(ir_node, "origins", None) if ir_node is not None else None
    for fx_node in origins or ():
        target = getattr(fx_node, "target", None)
        if target is None:
            continue
        packet = (
            target.overloadpacket
            if isinstance(target, torch._ops.OpOverload)
            else target
        )
        if packet in scaled:
            return True
        if packet in gemms and _has_fp8_input(fx_node, fp8):
            return True
    return False


def _has_fp8_input(fx_node: Any, fp8: OrderedSet[Any]) -> bool:
    for inp in getattr(fx_node, "all_input_nodes", ()):
        val = inp.meta.get("val") if getattr(inp, "meta", None) else None
        if getattr(val, "dtype", None) in fp8:
            return True
    return False


def _capture_blocker_reason(node: BaseSchedulerNode) -> str | None:
    """Why this single (non-fused) node cannot be captured, or None."""
    from . import ir
    from .utils import is_cudagraph_unsafe_op

    if node.node is None:
        return None
    if not node.is_gpu():
        return f"runs on {node.get_device()}, not the GPU"
    if isinstance(node.node, ir.DeviceCopy):
        return "device copy (host/device transfer cannot be captured)"
    if isinstance(node.node, ir.Switch):
        return "host-side control flow (torch.cond / torch.switch)"
    if getattr(node.node, "unbacked_bindings", None):
        return (
            "defines an unbacked symint (its size is only known at run time, so "
            "a captured graph would bake in one run's value)"
        )
    if is_cudagraph_unsafe_op(node.node):
        return "custom op marked cudagraph-unsafe"
    if _node_uses_rng(node):
        return "RNG (capture freezes the philox seed, so replays stop advancing)"
    if _node_uses_fp8_gemm(node):
        return (
            "fp8 gemm (cublasLt allocates its workspace lazily, illegal under capture)"
        )
    return None


def find_capture_blockers(
    nodes: list[BaseSchedulerNode],
) -> list[CaptureBlocker]:
    """Every node in `nodes` that cannot be captured inside a cuda graph."""
    blockers = []
    for node in nodes:
        for snode in _leaf_nodes(node):
            reason = _capture_blocker_reason(snode)
            if reason is not None:
                blockers.append(
                    CaptureBlocker(snode.get_name(), reason, _describe_op(snode))
                )
    return blockers


def _format_blockers(blockers: list[CaptureBlocker], total_nodes: int) -> str:
    by_reason: dict[str, list[CaptureBlocker]] = {}
    for blocker in blockers:
        by_reason.setdefault(blocker.reason, []).append(blocker)

    lines = [
        f'AOTI cuda graph: config.aot_inductor.cudagraph_mode is "whole", which '
        f"captures the entire lowered component as one cuda graph, but "
        f"{len(blockers)} of {total_nodes} scheduler nodes cannot be captured:",
        "",
    ]
    for reason, group in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {len(group)} node(s): {reason}")
        for blocker in group[:_MAX_EXAMPLES_PER_REASON]:
            lines.append(f"      {blocker.node_name}  (from {blocker.op})")
        if len(group) > _MAX_EXAMPLES_PER_REASON:
            lines.append(f"      ... and {len(group) - _MAX_EXAMPLES_PER_REASON} more")
    lines += [
        "",
        "Every blocker is listed above so the model can be fixed in one pass. "
        'Either remove them from the graph, or set cudagraph_mode back to "off".',
    ]
    return "\n".join(lines)


def check_whole_graph_capturable(nodes: list[BaseSchedulerNode]) -> None:
    """Raise unless every node in `nodes` is safe to capture.

    Whole-graph mode fails the lowering rather than silently falling back, so a
    model that was configured for cuda graph can never quietly ship without it.
    """
    blockers = find_capture_blockers(nodes)
    if blockers:
        raise RuntimeError(_format_blockers(blockers, len(nodes)))
