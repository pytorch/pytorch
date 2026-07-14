# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""Launch configuration for the RMSNorm forward and backward kernels.

Mirrors :mod:`quack.gemm_config`: frozen dataclasses that capture the launch
knobs, plus arch-specific factories that own the heuristics.
"""

import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class RmsNormFwdConfig:
    num_threads: int
    threads_per_row: int
    cluster_n: int
    # None = compute once into registers; "smem" / "gmem" = reload x (and
    # residual) before the post-reduction epilogue.
    reload_from: Optional[str]
    # Defer the weight/bias load until after the row reduction.
    delay_w_load: bool = False

    @classmethod
    def from_analytical_heuristic(
        cls,
        N: int,
        dtype_width: int,
        arch_major: Optional[int] = None,
        is_layernorm: bool = False,
    ) -> "RmsNormFwdConfig":
        """Pick a launch config from the hand-tuned analytical heuristic.

        ``arch_major`` defaults to the current device's capability. The same
        ladder is used for Hopper, Blackwell, and SM12x today; a future
        ``_for_blackwell_fwd`` factory can be added and dispatched on
        ``arch_major >= 10``. For autotuning, use :func:`get_all_fwd_configs`.
        """
        if arch_major is None:
            arch_major = _detect_arch_major()
        return _for_hopper_fwd(N, dtype_width, arch_major, is_layernorm)


def _for_hopper_fwd(
    N: int, dtype_width: int, arch_major: int, is_layernorm: bool
) -> RmsNormFwdConfig:
    num_threads = 128 if N <= 16 * 1024 else 256

    threads_per_row = 256
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            threads_per_row = threads
            break

    if arch_major < 9:
        cluster_n = 1
    else:
        max_cluster = 8 if arch_major == 12 else 16
        # cluster_n=4 is faster than cluster_n=2 for N=64k; cluster_n=8 is
        # faster for N=128k.
        if arch_major == 12 and dtype_width >= 32:
            # SM12x 99 KB SMEM: fp32 needs tighter clustering (conservative for residual case)
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        elif dtype_width == 16:
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        elif is_layernorm:
            # fp32 layernorm: bump cluster earlier than fp16/bf16. The 2-pass path's
            # single-CTA tile is bandwidth-limited at N=16k/32k; cluster_n=2 splits
            # the row across two CTAs and recovers ~3-14% at those sizes.
            thresholds = [(8 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        else:
            # fp32 rmsnorm (1-pass) is already saturated at cluster_n=1 for N<=32k;
            # bumping to cluster_n=2 there regresses ~3%.
            thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        cluster_n = max_cluster
        for limit, cluster in thresholds:
            if N <= limit:
                cluster_n = cluster
                break

    reload_threshold = 16 * 1024 if is_layernorm else 8 * 1024
    return RmsNormFwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_from=None if N <= reload_threshold else "smem",
        delay_w_load=False,
    )


@dataclass(frozen=True)
class RmsNormBwdConfig:
    num_threads: int
    threads_per_row: int
    cluster_n: int
    # None = recompute from registers; "smem" = reload from shared memory.
    reload_wdy: Optional[str]
    reload_x: Optional[str]
    use_tma: bool
    # Number of smem stages used by the prefetch pipeline. Drives both the
    # cp.async (use_tma=False) and TMA (use_tma=True) paths, which lead by
    # ``smem_stages - 1`` batches. Larger depths hide more latency at the cost
    # of smem footprint.
    smem_stages: int = 2

    @classmethod
    def from_analytical_heuristic(
        cls,
        N: int,
        dtype_width: int,
        dout_width: int,
        arch_major: Optional[int] = None,
        T_hint: int = 0,
    ) -> "RmsNormBwdConfig":
        """Pick a launch config from the hand-tuned analytical heuristic.

        ``arch_major`` defaults to the current device's capability.
        ``arch_major >= 10`` selects the Blackwell heuristic; anything else
        uses the legacy/default path tuned on Hopper. For autotuning, use
        :func:`get_all_bwd_configs`.
        """
        if arch_major is None:
            arch_major = _detect_arch_major()
        if arch_major >= 10:
            return _for_blackwell_bwd(N, dtype_width, dout_width, T_hint)
        return _for_hopper_bwd(N, dtype_width, arch_major)


def _for_hopper_bwd(N: int, dtype_width: int, arch_major: int) -> RmsNormBwdConfig:
    num_threads = 128 if N <= 4096 else 256
    for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
        if N <= limit:
            threads_per_row = threads
            break
    else:
        threads_per_row = 256

    if arch_major < 9:
        cluster_n = 1
    else:
        max_cluster = 8 if arch_major == 12 else 16
        if arch_major == 12 and dtype_width >= 32:
            thresholds = [(1024, 1), (8 * 1024, 2), (16 * 1024, 4), (32 * 1024, 8)]
        else:
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        cluster_n = max_cluster
        for limit, cluster in thresholds:
            if N <= limit:
                cluster_n = cluster
                break

    return RmsNormBwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_wdy=None if N <= 16 * 1024 else "smem",
        reload_x=None,
        use_tma=False,
    )


def _for_blackwell_bwd(
    N: int, dtype_width: int, dout_width: int, T_hint: int = 0
) -> RmsNormBwdConfig:
    """Pick a launch config for RMSNorm bwd on Blackwell.

    All thresholds are expressed in ``row_bytes = N * max(x, dout)`` so a
    single ladder handles bf16, fp32, and mixed-dtype combinations. ``x_bytes``
    governs the X tile (loads, smem footprint); ``max_bytes`` is the wider
    side which sets register pressure for the per-thread fragments.
    """
    # Safety floor for very narrow rows: keep tpr below 128 so we don't over-
    # parallelise tiny problems.
    if N <= 64:
        threads_per_row = 8
    elif N <= 128:
        threads_per_row = 16
    elif N <= 256:
        threads_per_row = 32
    elif N <= 512:
        threads_per_row = 64
    else:
        threads_per_row = None

    if threads_per_row is not None:
        return RmsNormBwdConfig(
            num_threads=128,
            threads_per_row=threads_per_row,
            cluster_n=1,
            reload_wdy=None,
            reload_x=None,
            use_tma=False,
        )

    max_bytes = max(dtype_width, dout_width) // 8
    row_bytes = N * max_bytes

    if row_bytes >= 48 * 1024:
        # Spread the row across a CTA cluster. Step back to cluster_n=4 only
        # when T is tiny AND the row isn't extreme — otherwise cn=8 keeps each
        # CTA's tile small enough to fit comfortably in registers.
        cluster_n = 4 if 0 < T_hint <= 1024 and row_bytes <= 64 * 1024 else 8
        num_threads, threads_per_row = 128, 128
        # Override if this cluster_n would overflow the device's smem budget.
        cluster_n = _bump_cluster_n_for_smem(
            cluster_n,
            N,
            smem_stages=2,
            sum_bytes=(dtype_width + dout_width) // 8,
            max_cluster=_max_cluster_for(10),  # Blackwell
        )
    elif row_bytes > 16 * 1024:
        # Wider than 128 threads can comfortably handle at cluster_n=1; bump
        # threads/row to keep per-thread fragments small.
        cluster_n = 1
        num_threads, threads_per_row = 256, 256
    else:
        cluster_n = 1
        num_threads, threads_per_row = 128, 128

    bytes_per_thread_frag = (N // cluster_n) // threads_per_row * max_bytes

    # TMA pays off when the cluster needs prefetch (cn>=4), and also for
    # fp32-class single-CTA wide rows where TMA's wider descriptors amortise
    # setup. Pure bf16 single-CTA cases mostly don't benefit and can lose ~5%.
    use_tma = cluster_n >= 4 or (max_bytes >= 4 and row_bytes >= 16 * 1024)
    # reload_x: wide end of the cluster ladder, plus fp32-class single-CTA
    # cases where the wider X fragments crowd registers across the row
    # reduction barrier.
    reload_x = (
        "smem"
        if (cluster_n >= 8 and N >= 32 * 1024)
        or (cluster_n == 1 and max_bytes >= 4 and bytes_per_thread_frag >= 64)
        else None
    )
    # reload_wdy: cluster cases get it for free, plus single-CTA cases where
    # each thread holds ≥64 bytes of fragment (the wdy register count is then
    # large enough to spill).
    reload_wdy = "smem" if cluster_n >= 4 or bytes_per_thread_frag >= 64 else None

    return RmsNormBwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_wdy=reload_wdy,
        reload_x=reload_x,
        use_tma=use_tma,
    )


def _get_sm_count_hopper(N: int, sm_count: int) -> int:
    # This should be tuned on how many CTAs can be launched on each SM.
    sm_count_multiple = (
        16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    # By right, if we're using cluster, this should be cluster_count not sm_count.
    # But for cluster >= 4, due to quantization we would need to query active max cluster.
    # Instead we just do sm_count * 2, which is reasonably larger than active_cluster_count to
    # avoid wave quantization.
    return (
        sm_count * sm_count_multiple if N <= 8192 else sm_count // 2 if N <= 16384 else sm_count * 2
    )


def _get_sm_count_blackwell(N: int, sm_count: int) -> int:
    if N <= 256:
        return sm_count * 16
    if N <= 1024:
        return sm_count * 8
    if N <= 2048:
        return sm_count * 4
    return sm_count * 2


def get_sm_count(N: int, device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    if props.major >= 10:
        return _get_sm_count_blackwell(N, props.multi_processor_count)
    return _get_sm_count_hopper(N, props.multi_processor_count)


_CTA_THREAD_SIZE = (128, 256)
# Full launch-knob menu lives here; the per-call pruner in
# ``prune_invalid_rmsnorm_{fwd,bwd}_configs`` drops layouts that don't fit the
# current row. The tiny widths (8, 16, 32) are kept for the analytical
# heuristic's narrow-row safety floor (N <= 512) but are dropped from the
# autotune search space below since they're only optimal for that floor.
_THREADS_PER_REDUCTION_DIM = (8, 16, 32, 64, 128, 256)
_AUTOTUNE_THREADS_PER_REDUCTION_DIM = (64, 128, 256)
# smem_stages=4 doubles the data-buffer footprint vs stages=2 and crashes at
# launch for fp32 N>=64K on Blackwell (227 KB opt-in smem). Stages=3 still
# offers a meaningful prefetch depth without that risk.
_AUTOTUNE_SMEM_STAGES = (2, 3)


def _max_dynamic_smem_bytes() -> int:
    """Per-CTA opt-in dynamic smem capacity for the current device.

    Returns 0 when CUDA is unavailable (callers should treat this as "no
    smem-budget guard"). Falls back to ``shared_memory_per_block`` on older
    PyTorch builds that lack the ``_optin`` field.
    """
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)


def _bump_cluster_n_for_smem(
    cluster_n: int,
    N: int,
    smem_stages: int,
    sum_bytes: int,
    max_cluster: int,
) -> int:
    """Raise ``cluster_n`` to the smallest power of 2 such that the bwd's
    sX+sdO data buffers fit under the device's opt-in dynamic smem.

    Footprint per CTA is ``(N / cluster_n) * smem_stages * sum_bytes`` where
    ``sum_bytes = x_bytes + dout_bytes``. Snaps up to the next power of 2,
    floors at the input ``cluster_n`` (never lowers the tuning's choice), and
    caps at ``max_cluster``. If the required cluster_n exceeds ``max_cluster``
    the runtime guard in ``RMSNormBackward.__init__`` raises a precise overflow
    error. Returns the input unchanged if CUDA is unavailable.
    """
    # Reserved for row-reduction buffer, mbars, and smem alignment overhead.
    _BWD_SMEM_RESERVED_BYTES = 4 * 1024
    smem_max = _max_dynamic_smem_bytes()
    if smem_max <= 0:
        return cluster_n
    budget = max(smem_max - _BWD_SMEM_RESERVED_BYTES, 1)
    needed = (N * smem_stages * sum_bytes + budget - 1) // budget  # ceil-div
    pow2 = 1
    while pow2 < needed:
        pow2 *= 2
    return min(max(pow2, cluster_n), max_cluster)


def _detect_arch_major() -> int:
    """Return the major device capability of the current CUDA device.

    Falls back to 0 (no-cluster, no-TMA) when CUDA is unavailable so the
    autotune search space stays well-defined for CPU-only imports.
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability(torch.cuda.current_device())[0]


def _max_cluster_for(arch_major: int) -> int:
    """Maximum cluster_n supported on this arch."""
    if arch_major < 9:
        return 1
    # SM12x (RTX 50) supports up to 8; Hopper/Blackwell up to 16.
    return 8 if arch_major == 12 else 16


def get_all_fwd_configs() -> List[RmsNormFwdConfig]:
    """Exhaustive search space of RMSNorm fwd configs for the current device.

    The search space is over launch knobs only — ``device_capacity`` is not a
    tunable parameter, so the current device's capability is queried once and
    used to bound ``cluster_n``.
    """
    arch_major = _detect_arch_major()
    max_cluster = _max_cluster_for(arch_major)
    cluster_vals = tuple(c for c in (1, 2, 4, 8, 16) if c <= max_cluster)
    reload_from_vals = (None, "smem", "gmem")
    delay_w_load_vals = (False,)

    configs: List[RmsNormFwdConfig] = []
    for num_threads, threads_per_row, cluster_n, reload_from, delay_w_load in itertools.product(
        _CTA_THREAD_SIZE,
        _AUTOTUNE_THREADS_PER_REDUCTION_DIM,
        cluster_vals,
        reload_from_vals,
        delay_w_load_vals,
    ):
        if threads_per_row > num_threads:
            continue
        if num_threads % threads_per_row != 0:
            continue
        configs.append(
            RmsNormFwdConfig(
                num_threads=num_threads,
                threads_per_row=threads_per_row,
                cluster_n=cluster_n,
                reload_from=reload_from,
                delay_w_load=delay_w_load,
            )
        )
    return configs


def get_all_bwd_configs() -> List[RmsNormBwdConfig]:
    """Exhaustive search space of RMSNorm bwd configs for the current device.

    Like :func:`get_all_fwd_configs`, the current device's capability bounds
    ``cluster_n`` and gates ``use_tma`` (TMA requires SM90+). ``smem_stages``
    sweeps the safe depths in :data:`_AUTOTUNE_SMEM_STAGES`.
    """
    arch_major = _detect_arch_major()
    max_cluster = _max_cluster_for(arch_major)
    cluster_vals = tuple(c for c in (1, 2, 4, 8, 16) if c <= max_cluster)
    use_tma_vals = (False, True) if arch_major >= 9 else (False,)
    reload_vals = (None, "smem")

    configs: List[RmsNormBwdConfig] = []
    for (
        num_threads,
        threads_per_row,
        cluster_n,
        reload_wdy,
        reload_x,
        use_tma,
        smem_stages,
    ) in itertools.product(
        _CTA_THREAD_SIZE,
        _AUTOTUNE_THREADS_PER_REDUCTION_DIM,
        cluster_vals,
        reload_vals,
        reload_vals,
        use_tma_vals,
        _AUTOTUNE_SMEM_STAGES,
    ):
        if threads_per_row > num_threads:
            continue
        if num_threads % threads_per_row != 0:
            continue
        configs.append(
            RmsNormBwdConfig(
                num_threads=num_threads,
                threads_per_row=threads_per_row,
                cluster_n=cluster_n,
                reload_wdy=reload_wdy,
                reload_x=reload_x,
                use_tma=use_tma,
                smem_stages=smem_stages,
            )
        )
    return configs


def prune_invalid_rmsnorm_fwd_configs(configs, named_args: dict, **kwargs):
    """Drop configs whose CTA layout doesn't fit the row width.

    The search space (see :func:`get_all_fwd_configs`) is already restricted to
    the current device's capability, so all that's left is a per-call shape
    check: ``threads_per_row * cluster_n > N`` would leave cluster CTAs with no
    work to do, so skip those.
    """
    kwargs = named_args | kwargs
    x = kwargs["x"]
    N = int(x.size(-1))
    pruned = []
    for ac in configs:
        c = ac.kwargs["config"]
        if c.threads_per_row * c.cluster_n > N:
            continue
        pruned.append(ac)
    return pruned


def prune_invalid_rmsnorm_bwd_configs(configs, named_args: dict, **kwargs):
    """Same per-call shape filter as the fwd, plus three ``use_tma`` drops
    that mirror the runtime ``USE_TMA`` guard in :class:`RMSNormBackward`
    (``USE_TMA = use_tma and not per_head and row_bytes_x % 16 == 0 and
    row_bytes_do % 16 == 0``). Configs that would silently fall back to the
    cp.async path are dropped here so the autotune bench doesn't time the
    same kernel twice and pick whichever happens to bench faster by noise.
    """
    kwargs = named_args | kwargs
    x = kwargs["x"]
    dout = kwargs.get("dout")
    N = int(x.size(-1))
    per_head = bool(kwargs.get("per_head", x.dim() == 3))
    x_bytes = x.element_size()
    dout_bytes = dout.element_size() if dout is not None else x_bytes
    pruned = []
    for ac in configs:
        c = ac.kwargs["config"]
        if c.threads_per_row * c.cluster_n > N:
            continue
        if c.use_tma:
            if per_head:
                continue
            tile_n = N // max(1, c.cluster_n)
            row_bytes_x = tile_n * x_bytes
            row_bytes_do = tile_n * dout_bytes
            if row_bytes_x % 16 != 0 or row_bytes_do % 16 != 0:
                continue
        pruned.append(ac)
    return pruned
