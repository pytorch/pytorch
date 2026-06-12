"""SPMD graph verification for overlap scheduling.

Verifies all ranks have identical FX graph structure before collective
reordering passes. Non-SPMD graphs cause NCCL collective ordering
mismatches and hangs.
"""

import hashlib
import logging
from collections import Counter

import torch
from torch._inductor import config
from torch._logging import trace_structured


log = logging.getLogger(__name__)


def _compute_hash(gm: torch.fx.GraphModule) -> int | None:
    """Compute a structural hash of the graph including tensor metadata.

    Uses FxGraphCachePickler(device_id_agnostic=True) to serialize
    (target, val) per call_function node, capturing op targets and
    FakeTensor metadata (dtype, shape, stride, etc.) with device indices
    normalized to 0.

    Returns None if the graph contains unpicklable objects.
    """
    from torch._inductor.codecache import BypassFxGraphCache, FxGraphCachePickler

    try:
        pickler = FxGraphCachePickler(gm, device_id_agnostic=True)
        data = pickler.dumps(
            tuple(
                (str(n.target), n.meta.get("val"))
                for n in gm.graph.nodes
                if n.op == "call_function"
            )
        )
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=True)
    except BypassFxGraphCache:
        # FxGraphCachePickler can't serialize certain objects:
        # mkldnn tensors, BackwardState, torchbind objects, or general
        # pickle failures. Skip the SPMD check gracefully.
        log.warning("SPMD check: skipping, unpicklable graph objects", exc_info=True)
        return None


def _build_diag_fingerprint(
    gm: torch.fx.GraphModule,
) -> tuple[tuple[str, str | None], ...]:
    """Build human-readable fingerprint for mismatch diagnostics.

    Only called on the rare mismatch path.
    """
    from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

    entries: list[tuple[str, str | None]] = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        target_str = str(n.target)
        val = n.meta.get("val")
        entries.append(
            (
                target_str,
                _format_val_metadata(val, extract_tensor_metadata_for_cache_key),
            )
        )
    return tuple(entries)


def _format_val_metadata(val: object, extract_fn: object) -> str | None:
    """Format node val metadata for human-readable diagnostics."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return str(extract_fn(val))  # type: ignore[operator]
    if isinstance(val, (tuple, list)):
        parts = []
        for v in val:
            if isinstance(v, torch.Tensor):
                parts.append(str(extract_fn(v)))  # type: ignore[operator]
            else:
                parts.append(str(type(v).__name__))
        return f"({', '.join(parts)})"
    return str(type(val).__name__)


def spmd_check(gm: torch.fx.GraphModule) -> bool:
    """Verify all ranks have identical FX graph structure (SPMD).

    Computes a structural hash (op targets + tensor metadata including
    shapes, dtypes, strides) and compares across ranks.
    On mismatch, emits a diagnostic report to stdout, logging, and
    trace_structured.

    Returns True if graphs match (SPMD), False on mismatch.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return True

    structure_hash = _compute_hash(gm)
    if structure_hash is None:
        return True

    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.distributed_c10d import _get_default_group

    pg = _get_default_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    with unset_fake_temporarily():
        all_hashes: list[int] = [0] * world_size
        dist.all_gather_object(all_hashes, structure_hash, pg)

    if all(h == all_hashes[0] for h in all_hashes):
        return True

    # Mismatch detected — build and gather diagnostic fingerprints
    fingerprint = _build_diag_fingerprint(gm)
    with unset_fake_temporarily():
        all_fingerprints: list[tuple[object, ...]] = [() for _ in range(world_size)]
        dist.all_gather_object(all_fingerprints, fingerprint, pg)

    report = _build_mismatch_report(all_fingerprints, rank, world_size)

    print(report, flush=True)
    log.warning("\n%s", report)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "inductor_spmd_graph_mismatch",
            "encoding": "string",
        },
        payload_fn=lambda: report,
    )

    if config.aten_distributed_optimizations.spmd_mismatch == "error":
        raise RuntimeError(
            "SPMD graph verification failed. "
            'Set aten_distributed_optimizations.spmd_mismatch="warn" '
            "to warn instead of fail.\n" + report
        )

    return False


def _entry_target(entry: object) -> str:
    """Extract the target string from a fingerprint entry."""
    if isinstance(entry, tuple):
        return str(entry[0])
    return str(entry)


def _entry_metadata(entry: object) -> str:
    """Format metadata from a fingerprint entry, if present."""
    if isinstance(entry, tuple) and len(entry) >= 2:
        meta = entry[1]
        if meta is not None:
            return f" meta={meta}"
    return ""


def _build_mismatch_report(
    all_fingerprints: list[tuple[object, ...]],
    rank: int,
    world_size: int,
) -> str:
    """Build diagnostic report for SPMD graph mismatch."""
    lines = [
        "=" * 80,
        f"SPMD GRAPH MISMATCH — rank {rank}, world_size={world_size}",
        "=" * 80,
    ]

    # Node count per rank
    counts = [len(t) for t in all_fingerprints]
    lines.append("NODE COUNTS PER RANK:")
    for r in range(world_size):
        marker = " <--" if counts[r] != counts[0] else ""
        lines.append(f"  rank {r}: {counts[r]} call_function nodes{marker}")
    lines.append("")

    # Find entries that differ
    ref = all_fingerprints[0]
    for r in range(1, world_size):
        other = all_fingerprints[r]
        if other == ref:
            continue
        lines.append(f"DIFFS rank 0 vs rank {r}:")

        # Show first few positional differences
        max_diffs = 10
        shown = 0
        for i, (a, b) in enumerate(zip(ref, other)):
            if a != b and shown < max_diffs:
                lines.append(f"  node {i}:")
                lines.append(f"    rank 0: {_entry_target(a)}{_entry_metadata(a)}")
                lines.append(f"    rank {r}: {_entry_target(b)}{_entry_metadata(b)}")
                shown += 1

        # Also show count-based diffs for op targets
        ref_targets = [_entry_target(e) for e in ref]
        other_targets = [_entry_target(e) for e in other]

        ref_counts = Counter(ref_targets)
        other_counts = Counter(other_targets)
        only_ref = ref_counts - other_counts
        only_other = other_counts - ref_counts
        if only_ref:
            lines.append("  Only on rank 0:")
            for op, cnt in only_ref.most_common(10):
                lines.append(f"    {op} (x{cnt})")
        if only_other:
            lines.append(f"  Only on rank {r}:")
            for op, cnt in only_other.most_common(10):
                lines.append(f"    {op} (x{cnt})")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)
