from __future__ import annotations

"""Hardware abstraction layer for pointwise kernel heuristics.

Queries actual GPU device properties via torch.cuda.get_device_properties()
and derives optimal heuristic parameters from first principles, rather than
hardcoding constants that would only be correct for one specific chip.

The key outputs are collected in ArchitectureConfig and consumed by the
scoring factors in pointwise.py and reduction.py.
"""

import math
import threading
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Per-architecture hardware constants
# ---------------------------------------------------------------------------
# Peak HBM bandwidth in GB/s, keyed by first 6 chars of gcnArchName.
# Used when the clock-formula fallback gives wrong results for HBM2e/HBM3/HBM3e.
# Add new entries as new AMD architectures ship.
_KNOWN_HBM_BANDWIDTH_GB_S: dict = {
    'gfx900': 484,    # Vega10 (MI25): HBM2
    'gfx906': 1024,   # Vega20 (MI50/MI60): HBM2
    'gfx908': 1229,   # CDNA1 (MI100): HBM2
    'gfx90a': 1638,   # CDNA2 (MI200/MI250X): HBM2e
    'gfx940': 1638,   # CDNA3 (MI300A): HBM3
    'gfx941': 1638,   # CDNA3
    'gfx942': 5300,   # CDNA3 (MI300X): HBM3
    'gfx950': 6000,   # CDNA4 (MI350X): HBM3e
}

# Per-architecture raw HBM round-trip latency in GPU clock cycles.
# Values for AMD from hardware documentation + memory-bound microbenchmarks.
# Conservative estimates (rounded up); actual latency is workload-dependent.
_KNOWN_HBM_LATENCY_CYCLES: dict = {
    'gfx900': 600,   # Vega10: HBM2 ~200 ns @ ~1.2 GHz
    'gfx906': 550,   # Vega20: HBM2 ~200 ns @ ~1.4 GHz
    'gfx908': 500,   # MI100: HBM2 ~200 ns @ ~1.5 GHz
    'gfx90a': 450,   # MI200: HBM2e ~200 ns @ ~1.7 GHz
    'gfx940': 420,   # MI300A: HBM3 ~180 ns @ ~2.0 GHz
    'gfx941': 420,   # MI300 variant
    'gfx942': 400,   # MI300X: HBM3 ~180 ns @ ~2.1 GHz
    'gfx950': 220,   # MI350X: HBM3e ~100 ns @ ~2.2 GHz
}

# Per-architecture L3 (Infinity Cache) size in bytes.
# Only AMD CDNA3+ and RDNA3+ have a discrete L3.
_KNOWN_L3_CACHE_BYTES: dict = {
    'gfx940': 256 * 1024 * 1024,   # MI300A: 256 MB
    'gfx941': 256 * 1024 * 1024,
    'gfx942': 256 * 1024 * 1024,   # MI300X: 256 MB
    'gfx950': 256 * 1024 * 1024,   # MI350X: 256 MB
}

def _gcn_arch_prefix(props) -> str:
    """Return the first 6 chars of gcnArchName, or '' if unavailable."""
    return getattr(props, 'gcnArchName', '')[:6]

@dataclass
class ArchitectureConfig:
    """Hardware-derived constants used by the scoring model.

    All fields marked "derived" are computed from the raw device properties
    using the mathematical analysis in ArchitectureConfig.from_device().
    """

    # Raw device properties
    device_name:           str
    num_cus:               int   # Compute units / multiprocessors
    warp_size:             int   # 64 on AMD (wave64), 32 on NVIDIA
    max_threads_per_block: int
    max_threads_per_cu:    int   # Maximum threads concurrently resident on one CU/SM
    max_wavefronts_per_cu: int
    regs_per_cu:           int   # Total VGPR file per CU (e.g. 65536 CDNA2, 131072 CDNA3)
    l1_cache_size:         int   # Per-CU L1 in bytes (32 KB for AMD CDNA)
    l2_cache_size:         int   # Total L2 in bytes (e.g. 4 MB on MI350X)
    l3_cache_size:         int   # Infinity Cache / L3 in bytes (0 if absent)
    cacheline_bytes:       int   # HBM/L2 cache-line width in bytes
                                  # 64 B on NVIDIA Ampere; 128 B on AMD CDNA2+
    memory_bandwidth_gb_s: float  # Peak HBM bandwidth (GB/s)

    # Derived optimal values
    optimal_threads_bandwidth:  int   # Thread count for best HBM utilization
    optimal_blocks_grid:        int   # Block count for full GPU saturation
    occupancy_sweetspot_min:    int   # Min wavefronts/block for latency hiding
    occupancy_sweetspot_max:    int   # Max wavefronts/block before VGPR pressure
    optimal_elements_per_block: int   # Elements/block to amortise launch overhead
    vgpr_budget_per_thread:     int   # VGPRs available per thread at full occupancy
                                       # = regs_per_cu // max_threads_per_cu
                                       # CDNA2: 65536//2048=32, CDNA3: 131072//2048=64

    # Raw latency / throughput constants — exposed so callers can recompute
    # optimal_threads with a kernel-specific instructions_per_load value.
    simd_units:                int    # SIMD units per CU (4 on AMD CDNA)
    effective_latency:         float  # Blended L2+HBM round-trip in cycles
    hbm_bytes_per_cycle_per_cu: float  # Peak HBM bytes / clock cycle / CU
                                        # Used in reduction sync-overhead scoring

    @classmethod
    def from_device(cls, device: torch.device | None = None) -> 'ArchitectureConfig':
        """Derive heuristic parameters from the actual GPU.

        Falls back to conservative defaults when no GPU is present.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not torch.cuda.is_available():
            return cls._get_default_config()

        props = torch.cuda.get_device_properties(device)
        is_hip = bool(getattr(torch.version, 'hip', None))

        device_name           = props.name
        num_cus               = props.multi_processor_count
        warp_size             = props.warp_size
        max_threads_per_block = props.max_threads_per_block
        max_threads_per_cu    = props.max_threads_per_multi_processor
        max_wavefronts_per_cu = max_threads_per_cu // warp_size

        # Total VGPR file size per CU/SM.
        # AMD:    65 536 on CDNA2 (gfx90a / MI250X),
        #        131 072 on CDNA3 (gfx942 / MI300X) and CDNA4.
        # NVIDIA: varies; Ampere SM = 65 536, Hopper = 65 536.
        # PyTorch exposes this as regs_per_multiprocessor; fall back to the
        # conservative CDNA2 value (65 536) when not available or reported as 0.
        regs_per_cu = int(getattr(props, 'regs_per_multiprocessor', 0)) or 65536

        # VGPRs available per thread when the CU/SM is fully occupied.
        # This is the per-thread VGPR budget at maximum theoretical occupancy.
        # A reduction kernel that uses more than this many VGPRs for accumulator
        # registers will prevent full-occupancy scheduling.
        vgpr_budget_per_thread = regs_per_cu // max(max_threads_per_cu, 1)

        # ROCm exposes 'L2_cache_size' (capital); CUDA uses 'l2_cache_size'.
        l2_cache_size = getattr(props, 'L2_cache_size',
                        getattr(props, 'l2_cache_size', 4 * 1024 * 1024))

        # ------------------------------------------------------------------
        # Cache-line width
        # ------------------------------------------------------------------
        # AMD CDNA2+ (gfx90a and later) use 128-byte cache lines.
        # NVIDIA Ampere and earlier use 64-byte cache lines.
        # Affects coalescing thresholds and 2-D/3-D tile scoring.
        arch_prefix = _gcn_arch_prefix(props)
        if is_hip:
            # All CDNA2+ and RDNA3+ use 128-byte cachelines.
            # CDNA1 (gfx908) and older used 64-byte cachelines.
            cacheline_bytes = 128 if arch_prefix >= 'gfx90a' else 64
        else:
            cacheline_bytes = 64  # NVIDIA Ampere / Ada / Hopper

        # ------------------------------------------------------------------
        # L3 cache (Infinity Cache — AMD CDNA3+ only)
        # ------------------------------------------------------------------
        l3_cache_size = _KNOWN_L3_CACHE_BYTES.get(arch_prefix, 0)

        # ------------------------------------------------------------------
        # Peak HBM bandwidth
        # ------------------------------------------------------------------
        # Primary: architecture lookup table (reliable for AMD HBM2e/3/3e).
        # Fallback: clock × bus-width formula (works for GDDR/HBM2 in most cases).
        if arch_prefix in _KNOWN_HBM_BANDWIDTH_GB_S:
            memory_bandwidth_gb_s = float(_KNOWN_HBM_BANDWIDTH_GB_S[arch_prefix])
        else:
            try:
                mem_clk_hz = props.memoryClockRate * 1e3   # kHz → Hz
                bus_bytes  = props.memoryBusWidth  / 8     # bits → bytes
                memory_bandwidth_gb_s = max(50.0, (mem_clk_hz * bus_bytes * 2) / 1e9)
            except Exception as e:  # noqa: BLE001
                memory_bandwidth_gb_s = 900.0  # safe AMD fallback

        # ------------------------------------------------------------------
        # Bandwidth: optimal threads per block (Little's Law)
        # ------------------------------------------------------------------
        # Self-consistent Little's Law for a CU with `simd_units` SIMD units:
        #
        #   N_min = sqrt(simd_units × effective_latency / I)
        #
        # where effective_latency blends L2 and HBM latencies by hit rate.
        #
        # HBM latency: use per-architecture lookup when known, otherwise
        # derive from (memory_bandwidth_gb_s / num_cus / clock_hz) assuming
        # ~100 ns for modern HBM3/HBM3e parts.
        simd_units = 4  # AMD CDNA: 4 SIMD units per CU; NVIDIA: treat as 4

        raw_hbm_latency = _KNOWN_HBM_LATENCY_CYCLES.get(arch_prefix, None)
        if raw_hbm_latency is None:
            # Estimate from clock rate: assume ~100 ns for HBM3+, ~200 ns otherwise.
            try:
                clock_hz = props.clockRate * 1e3
                latency_ns = 100.0 if memory_bandwidth_gb_s > 3000 else 200.0
                raw_hbm_latency = int(latency_ns * 1e-9 * clock_hz)
            except Exception as e:  # noqa: BLE001
                raw_hbm_latency = 400

        l2_hit_latency        = 50    # cycles: L2 cache hit (AMD CDNA / NVIDIA)
        l2_hit_rate           = 0.50  # conservative; streaming kernels can be lower
        instructions_per_load = 2     # arithmetic ops between consecutive memory ops

        effective_latency = l2_hit_rate * l2_hit_latency + (1.0 - l2_hit_rate) * raw_hbm_latency
        wavefronts_needed = math.sqrt(simd_units * effective_latency / instructions_per_load)
        wavefronts_needed = max(8, min(int(math.ceil(wavefronts_needed)), max_wavefronts_per_cu))
        optimal_threads_bandwidth = wavefronts_needed * warp_size

        # ------------------------------------------------------------------
        # HBM bytes / clock cycle / CU (used in reduction sync scoring)
        # ------------------------------------------------------------------
        # = peak_bandwidth / num_cus / clock_frequency
        try:
            clock_hz = props.clockRate * 1e3
            hbm_bytes_per_cycle_per_cu = (memory_bandwidth_gb_s * 1e9) / max(1, num_cus) / clock_hz
        except Exception as e:  # noqa: BLE001
            hbm_bytes_per_cycle_per_cu = 1.8  # original fallback

        # ------------------------------------------------------------------
        # Grid: optimal number of blocks
        # ------------------------------------------------------------------
        # Two blocks per CU: one fills the wave, the second provides
        # load-balancing for tail effects.
        optimal_blocks_grid = num_cus * 2

        # ------------------------------------------------------------------
        # Occupancy: wavefront sweet spot
        # ------------------------------------------------------------------
        # Assume a typical reduction/elementwise kernel uses ~50 VGPRs/thread
        # (indices + accumulator + temporaries).  Compute how many wavefronts
        # can be resident on one CU before the VGPR file is exhausted.
        # Uses the device-queried regs_per_cu so MI300X (131072) and future
        # parts with larger VGPR files are handled correctly.
        assumed_vgprs_per_thread = 50
        vgprs_per_wavefront      = assumed_vgprs_per_thread * warp_size
        max_wf_by_vgpr           = regs_per_cu // max(vgprs_per_wavefront, 1)

        occupancy_sweetspot_min = 4
        occupancy_sweetspot_max = min(8, max_wf_by_vgpr)

        # ------------------------------------------------------------------
        # Launch: optimal elements per block
        # ------------------------------------------------------------------
        # Kernel dispatch ~3 µs.  At ~0.05 µs/element, targeting < 5% overhead:
        #   min_elements = 3 / 0.05 = 60; optimal = 60 / 0.05 = 1200 → 2048.
        launch_overhead_us   = 3
        element_time_us      = 0.05
        target_overhead_frac = 0.05

        min_elements = launch_overhead_us / element_time_us
        optimal_elem = min_elements / target_overhead_frac
        optimal_elements_per_block = 2 ** math.ceil(math.log2(optimal_elem))
        optimal_elements_per_block = max(256, min(optimal_elements_per_block, 2048))

        # L1 cache size: 32 KB per CU on AMD CDNA/RDNA; 128 KB per SM on NVIDIA Ampere+.
        l1_cache_size = 32 * 1024 if is_hip else 128 * 1024

        return cls(
            device_name=device_name,
            num_cus=num_cus,
            warp_size=warp_size,
            max_threads_per_block=max_threads_per_block,
            max_threads_per_cu=max_threads_per_cu,
            max_wavefronts_per_cu=max_wavefronts_per_cu,
            regs_per_cu=regs_per_cu,
            l1_cache_size=l1_cache_size,
            l2_cache_size=l2_cache_size,
            l3_cache_size=l3_cache_size,
            cacheline_bytes=cacheline_bytes,
            memory_bandwidth_gb_s=memory_bandwidth_gb_s,
            optimal_threads_bandwidth=optimal_threads_bandwidth,
            optimal_blocks_grid=optimal_blocks_grid,
            occupancy_sweetspot_min=occupancy_sweetspot_min,
            occupancy_sweetspot_max=occupancy_sweetspot_max,
            optimal_elements_per_block=optimal_elements_per_block,
            vgpr_budget_per_thread=vgpr_budget_per_thread,
            simd_units=simd_units,
            effective_latency=effective_latency,
            hbm_bytes_per_cycle_per_cu=hbm_bytes_per_cycle_per_cu,
        )

    @classmethod
    def _get_default_config(cls) -> 'ArchitectureConfig':
        """Conservative fallback when no GPU is available.

        Values are conservatively chosen for AMD CDNA2 (MI250X):
          - regs_per_cu=65536, max_threads_per_cu=2048  →  vgpr_budget_per_thread=32
        This is the lower bound across all supported AMD CDNA generations, so
        heuristic pruning will be cautious rather than over-aggressive when no
        real device is accessible.
        """
        return cls(
            device_name='CPU (fallback)',
            num_cus=1,
            warp_size=32,
            max_threads_per_block=1024,
            max_threads_per_cu=2048,
            max_wavefronts_per_cu=32,
            regs_per_cu=65536,         # conservative CDNA2 value
            l1_cache_size=32 * 1024,
            l2_cache_size=0,
            l3_cache_size=0,
            cacheline_bytes=64,
            memory_bandwidth_gb_s=100.0,
            optimal_threads_bandwidth=256,
            optimal_blocks_grid=32,
            occupancy_sweetspot_min=4,
            occupancy_sweetspot_max=8,
            optimal_elements_per_block=1024,
            vgpr_budget_per_thread=32,  # 65536 // 2048
            simd_units=4,
            effective_latency=275.0,
            hbm_bytes_per_cycle_per_cu=1.8,
        )

    def __str__(self) -> str:
        wf = self.optimal_threads_bandwidth // self.warp_size
        cl = self.cacheline_bytes
        l3 = f"{self.l3_cache_size // (1024*1024)} MB" if self.l3_cache_size else "none"
        lines = [
            f"Device: {self.device_name}",
            f"  CUs / SMs              : {self.num_cus}",
            f"  Warp size              : {self.warp_size}",
            f"  Max threads/block      : {self.max_threads_per_block}",
            f"  Max threads/CU         : {self.max_threads_per_cu}",
            f"  Max wavefronts/CU      : {self.max_wavefronts_per_cu}",
            f"  VGPR file / CU         : {self.regs_per_cu} "
                f"({self.vgpr_budget_per_thread} VGPRs/thread at full occupancy)",
            f"  Cache-line             : {cl} B ({cl // 4} FP32 elements)",
            f"  L1 cache (per CU)      : {self.l1_cache_size // 1024} KB",
            f"  L2 cache               : {self.l2_cache_size // 1024} KB"
                if self.l2_cache_size else "  L2 cache               : unknown",
            f"  L3 / Infinity Cache    : {l3}",
            f"  Peak HBM bandwidth     : {self.memory_bandwidth_gb_s:.0f} GB/s",
            f"  HBM bytes/cycle/CU     : {self.hbm_bytes_per_cycle_per_cu:.2f}",
            f"Derived optimal values:",
            f"  Bandwidth (Little's Law): {self.optimal_threads_bandwidth} threads/block ({wf} wavefronts)",
            f"  Grid                   : {self.optimal_blocks_grid} blocks "
                f"({self.optimal_blocks_grid / max(self.num_cus, 1):.1f}× CUs)",
            f"  Occupancy range        : {self.occupancy_sweetspot_min}–{self.occupancy_sweetspot_max} wavefronts/block",
            f"  Launch amortise        : {self.optimal_elements_per_block} elements/block",
        ]
        return "\n".join(lines)

# Module-level singleton, lazily initialised on first call.
# _arch_config_lock serialises concurrent first-time initialisation so two
# threads cannot both pass the `is None` check and both call from_device().
# After initialisation the lock is never acquired again (fast path: just read
# the already-set global).
_arch_config: ArchitectureConfig | None = None
_arch_config_lock: threading.Lock = threading.Lock()

def get_architecture_config(device: torch.device | None = None) -> ArchitectureConfig:
    """Return the hardware configuration, querying the device on first call.

    Thread-safe: at most one thread will call ArchitectureConfig.from_device().
    Subsequent calls return the cached value without acquiring the lock.

    Note: ``device`` is only honoured on the very first call.  Later calls with
    a different device value are silently ignored — reset the singleton with
    reset_architecture_config() before calling again if a different device is
    needed.
    """
    global _arch_config
    # Fast path: already initialised (no lock needed for a plain read in CPython
    # because the GIL ensures reference reads are atomic).
    if _arch_config is not None:
        return _arch_config
    with _arch_config_lock:
        # Re-check inside the lock: another thread may have initialised while
        # we were waiting to acquire it.
        if _arch_config is None:
            _arch_config = ArchitectureConfig.from_device(device)
    return _arch_config

def reset_architecture_config() -> None:
    """Clear the cached configuration (useful in tests that mock device props)."""
    global _arch_config
    with _arch_config_lock:
        _arch_config = None
