# Copyright (c) 2025-2026, Tri Dao.
"""Host-side arrival-count validation for pipeline construction.

Every mbarrier-based pipeline bakes an arrive count into its barrier init; a count that
disagrees with the set of agents that actually signal the barrier deadlocks (too high) or
races (too low) at runtime, usually only for the one config leg that got it wrong. These
helpers recompute the expected count from kernel facts (warp counts, multicast shape,
cluster routing, signaling mode) at trace time, so a construction-site mistake fails the
compile with the arithmetic spelled out instead of hanging the kernel.

Rules (protocol semantics; same derivation as the checks in CuTe-DSL 4.7's
task_scheduling.task_manager, reimplemented independently):

- TMA producer: ``arrive_and_expect_tx`` under ``elect_one`` -> 1 arrival per TMA-issuing
  warp. No cluster scaling: non-leader CTAs are credited by TMA multicast hardware
  (or gated by ``is_leader_cta``).
- cp.async producer: ``cp.async.mbarrier.arrive`` counts once per thread -> 32 per
  producing warp.
- tcgen05 MMA commit (producer or consumer side): hardware signal -> exactly 1 arrival
  per committing CTA, delivered to each barrier its multicast mask targets.
- Async-thread signaling: all-threads -> 32 per warp; elected-lane
  (``elect_one_commit/release``) -> 1 per warp; multiplied by the number of CTAs whose
  arrivals route to the same physical barrier (e.g. ``consumer_mask=0`` routes every
  CTA in the cluster to CTA 0's barrier).

Call sites pass only *facts* (warp counts, flags); the expected count is derived here.
Checks are skipped when any input is dynamic (an ``Int32`` leg, e.g. leader-dependent
producer counts) — only statically-known counts are validated.
"""

WARP_SIZE = 32


class PipelineArriveCountError(AssertionError):
    """Arrive count passed to a pipeline constructor contradicts the kernel facts."""


def _all_static(*vals) -> bool:
    return all(isinstance(v, (int, bool)) for v in vals)


def async_thread_arrives(num_warps, *, per_warp, ctas_routed=1):
    """Arrive count for async-thread signaling.

    per_warp=True models elected-lane arrives (elect_one_commit/release or an explicit
    lane guard); per_warp=False models all-thread arrives. ctas_routed is the number of
    CTAs whose arrivals land on the same physical barrier (cluster mask routing / DSMEM
    multicast release); 1 for a CTA-local barrier.
    """
    if not _all_static(num_warps, per_warp, ctas_routed):
        return None
    return num_warps * (1 if per_warp else WARP_SIZE) * ctas_routed


def tma_producer_arrives(*, num_tma_warps=1, cpasync_warps=0, extra_unit_arrives=0):
    """Arrive count for a TMA (+ optional cp.async) producer side.

    Each TMA-issuing warp contributes 1 (elect_one arrive_and_expect_tx); each cp.async
    warp contributes 32 (per-thread cp.async.mbarrier.arrive); extra_unit_arrives covers
    single-arrive side agents (e.g. a peer CTA's forwarding warp).
    """
    if not _all_static(num_tma_warps, cpasync_warps, extra_unit_arrives):
        return None
    return num_tma_warps + cpasync_warps * WARP_SIZE + extra_unit_arrives


def umma_producer_arrives():
    """tcgen05 MMA producer: hardware sends exactly 1 arrival per commit."""
    return 1


def mcast_peer_ctas(*, num_mcast_ctas_a, num_mcast_ctas_b):
    """Size of a CTA's combined A/B multicast peer set (self counted once).

    This is both the number of CTAs whose releases land on this CTA's empty barrier and
    the number of peer barriers each consumer signals: the CTAs in A's multicast group
    plus the CTAs in B's multicast group, minus the self double-count. Used directly as
    the UMMA-consumer arrive count on sm100 (one hardware tcgen05.commit per peer CTA)
    and as the ``ctas_routed`` factor for per-warp releases on sm90.
    """
    if not _all_static(num_mcast_ctas_a, num_mcast_ctas_b):
        return None
    return num_mcast_ctas_a + num_mcast_ctas_b - 1


def check_arrive_count(name, actual, expected, **facts):
    """Assert actual == expected; no-op when either side is dynamic (None/Int32)."""
    if expected is None or not _all_static(actual):
        return
    if actual != expected:
        fact_str = ", ".join(f"{k}={v}" for k, v in facts.items())
        raise PipelineArriveCountError(
            f"{name}: arrive count {actual} contradicts the kernel facts "
            f"(expected {expected} from {fact_str}). A too-high count deadlocks the "
            f"barrier; a too-low count releases/completes the stage early and races."
        )
