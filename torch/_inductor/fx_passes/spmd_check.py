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
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _get_subgraph_modules(
    gm: torch.fx.GraphModule,
) -> list[tuple[str, torch.fx.GraphModule]]:
    """Return (name, module) pairs for subgraph children referenced by get_attr nodes."""
    attr_names = OrderedSet([n.target for n in gm.graph.find_nodes(op="get_attr")])
    result = []
    for name, child in gm.named_children():
        if name in attr_names and isinstance(child, torch.fx.GraphModule):
            result.append((name, child))
    return result


def _compute_hash_bundle(
    gm: torch.fx.GraphModule, prefix: str = ""
) -> dict[str, int] | None:
    """Compute structural hashes for the graph and all subgraphs recursively.

    Returns a dict mapping graph path to hash, e.g.:
      {"": 123, "subgraph_0": 456, "subgraph_0.inner": 789}

    Returns None if any graph contains unpicklable objects.
    """
    from torch._inductor.codecache import BypassFxGraphCache, FxGraphCachePickler

    bundle: dict[str, int] = {}
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
        bundle[prefix] = int.from_bytes(digest, "big", signed=True)
    except BypassFxGraphCache:
        log.warning("SPMD check: skipping, unpicklable graph objects", exc_info=True)
        return None

    for name, child in _get_subgraph_modules(gm):
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_bundle = _compute_hash_bundle(child, child_prefix)
        if child_bundle is None:
            return None
        bundle.update(child_bundle)

    return bundle


def _build_diag_fingerprint_bundle(
    gm: torch.fx.GraphModule, prefix: str = ""
) -> dict[str, tuple[tuple[str, str | None], ...]]:
    """Build human-readable fingerprints for the graph and all subgraphs.

    Only called on the rare mismatch path.
    """
    from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

    bundle: dict[str, tuple[tuple[str, str | None], ...]] = {}

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
    bundle[prefix] = tuple(entries)

    for name, child in _get_subgraph_modules(gm):
        child_prefix = f"{prefix}.{name}" if prefix else name
        bundle.update(_build_diag_fingerprint_bundle(child, child_prefix))

    return bundle


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

    Computes structural hashes for the graph and all subgraphs, then
    compares across ranks via a single all_gather_object. On mismatch,
    emits a diagnostic report identifying which subgraph diverged.

    Returns True if graphs match (SPMD), False on mismatch.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return True

    hash_bundle = _compute_hash_bundle(gm)
    if hash_bundle is None:
        return True

    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.distributed_c10d import _get_default_group

    pg = _get_default_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    with unset_fake_temporarily():
        all_bundles: list[dict[str, int]] = [{} for _ in range(world_size)]
        dist.all_gather_object(all_bundles, hash_bundle, pg)

    if all(b == all_bundles[0] for b in all_bundles):
        return True

    # Mismatch detected — identify which subgraphs diverged
    mismatched_paths = _find_mismatched_paths(all_bundles)

    fingerprint_bundle = _build_diag_fingerprint_bundle(gm)
    with unset_fake_temporarily():
        all_fp_bundles: list[dict[str, tuple[object, ...]]] = [
            {} for _ in range(world_size)
        ]
        dist.all_gather_object(all_fp_bundles, fingerprint_bundle, pg)

    report = _build_mismatch_report(all_fp_bundles, mismatched_paths, rank, world_size)

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


def _find_mismatched_paths(
    all_bundles: list[dict[str, int]],
) -> list[str]:
    """Find graph paths where hashes differ across ranks."""
    all_paths: OrderedSet[str] = OrderedSet()
    for b in all_bundles:
        all_paths.update(b.keys())

    mismatched = []
    for path in sorted(all_paths):
        hashes = [b.get(path) for b in all_bundles]
        if any(h != hashes[0] for h in hashes):
            mismatched.append(path)
    return mismatched


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
    all_fp_bundles: list[dict[str, tuple[object, ...]]],
    mismatched_paths: list[str],
    rank: int,
    world_size: int,
) -> str:
    """Build diagnostic report for SPMD graph mismatch."""
    lines = [
        "=" * 80,
        f"SPMD GRAPH MISMATCH — rank {rank}, world_size={world_size}",
        "=" * 80,
    ]

    if not mismatched_paths:
        lines.append("Hash mismatch detected but no specific path identified.")
        lines.append("=" * 80)
        return "\n".join(lines)

    for path in mismatched_paths:
        graph_label = f"subgraph '{path}'" if path else "top-level graph"
        lines.append(f"MISMATCH IN {graph_label}:")

        fingerprints = [b.get(path, ()) for b in all_fp_bundles]

        counts = [len(fp) for fp in fingerprints]
        lines.append("  NODE COUNTS PER RANK:")
        for r in range(world_size):
            marker = " <--" if counts[r] != counts[0] else ""
            lines.append(f"    rank {r}: {counts[r]} call_function nodes{marker}")
        lines.append("")

        ref = fingerprints[0]
        for r in range(1, world_size):
            other = fingerprints[r]
            if other == ref:
                continue
            lines.append(f"  DIFFS rank 0 vs rank {r}:")

            max_diffs = 10
            shown = 0
            for i, (a, b) in enumerate(zip(ref, other)):
                if a != b and shown < max_diffs:
                    lines.append(f"    node {i}:")
                    lines.append(
                        f"      rank 0: {_entry_target(a)}{_entry_metadata(a)}"
                    )
                    lines.append(
                        f"      rank {r}: {_entry_target(b)}{_entry_metadata(b)}"
                    )
                    shown += 1

            ref_targets = [_entry_target(e) for e in ref]
            other_targets = [_entry_target(e) for e in other]

            ref_counts = Counter(ref_targets)
            other_counts = Counter(other_targets)
            only_ref = ref_counts - other_counts
            only_other = other_counts - ref_counts
            if only_ref:
                lines.append("    Only on rank 0:")
                for op, cnt in only_ref.most_common(10):
                    lines.append(f"      {op} (x{cnt})")
            if only_other:
                lines.append(f"    Only on rank {r}:")
                for op, cnt in only_other.most_common(10):
                    lines.append(f"      {op} (x{cnt})")
            lines.append("")

    # Also report subgraph count differences
    all_paths_per_rank = [OrderedSet(b.keys()) for b in all_fp_bundles]
    ref_paths = all_paths_per_rank[0]
    for r in range(1, world_size):
        only_on_0 = ref_paths - all_paths_per_rank[r]
        only_on_r = all_paths_per_rank[r] - ref_paths
        if only_on_0:
            lines.append(f"SUBGRAPHS only on rank 0 (not on rank {r}):")
            for p in sorted(only_on_0):
                lines.append(f"  {p}")
        if only_on_r:
            lines.append(f"SUBGRAPHS only on rank {r} (not on rank 0):")
            for p in sorted(only_on_r):
                lines.append(f"  {p}")

    lines.append("=" * 80)
    return "\n".join(lines)
