from __future__ import annotations

"""Config generation, scoring, and top-N selection for AMD reduction kernels.

Provides ReductionHeuristics, which:
  1. Generates semi-exhaustive (XBLOCK, R0_BLOCK, num_warps) candidate sets
     for INNER and OUTER reduction kernels.
  2. Scores each candidate on four principled factors (reduction efficiency,
     grid coverage, sync overhead, warp parallelism) using adaptive weights
     derived from BottleneckAnalysis.
  3. Prunes the candidate set to a top-N shortlist before compilation,
     mirroring the pointwise TOP-N flow.

The CachingAutotuner benchmarks all top-N candidates at first execution and
caches the winner permanently.

This is the reduction analogue of pointwise.py.

------------------------------------------------------------------------------
AMD reduction execution model (from generated Triton IR):
------------------------------------------------------------------------------

  accumulator = tl.full([XBLOCK, R0_BLOCK], 0, dtype)
  for r0_offset in tl.range(0, r0_numel, R0_BLOCK):      # outer loop
      ... accumulate into accumulator[x_idx, r0_idx] ...
  result = tl.sum(accumulator, axis=1)[:, None]            # reduction call

The `tl.sum(accumulator, axis=1)` call reduces the R0 axis across all threads
in the block.  On AMD the compiler emits:

  1. Intra-wavefront DPP butterfly  (~24 cycles, purely register-based,
                                     no memory traffic at all).

  2. Cross-wavefront DS (LDS) sync  — only needed when multiple wavefronts
     share one output element (i.e. num_warps / XBLOCK > 1).
     DS bandwidth on MI300X is ~100 TB/s — cheap, rarely the bottleneck.

Key ratios:
  threads_per_output = (num_warps × warp_size) / XBLOCK

  • threads_per_output = warp_size (XBLOCK = num_warps):
        exactly 1 wavefront per output → DPP only, zero DS rounds.

  • threads_per_output > warp_size (XBLOCK < num_warps):
        multiple wavefronts per output → DPP + log₂(num_warps/XBLOCK) DS rounds.

  • threads_per_output < warp_size (XBLOCK > num_warps):
        sub-wavefront per output → partial-lane DPP, zero DS rounds.
        Useful for batching many output elements per block (large xnumel).

------------------------------------------------------------------------------
Tuning axes and their effects:
------------------------------------------------------------------------------

  XBLOCK   : outputs per block.  Controls grid = ⌈xnumel / XBLOCK⌉.
             1 → max blocks (best CU coverage), ≥ 2 → fewer but larger blocks.

  num_warps: wavefronts per block (× warp_size = thread count).
             Higher → more HBM bandwidth per block, more DS rounds.
             Lower  → better co-residency, fewer DS rounds.

  R0_BLOCK : r-elements processed per outer-loop iteration.
             Larger → fewer iterations (lower loop overhead), more registers.
             Smaller → fewer registers (avoids spill for Welford/var/std).

------------------------------------------------------------------------------
Filtering:
------------------------------------------------------------------------------

  1. Block-count guard:  ⌈xnumel / XBLOCK⌉ ≥ min_blocks(xnumel, num_cus)
     Ensures enough blocks to keep a meaningful fraction of CUs busy.
     min_blocks scales with xnumel relative to num_cus:
       x ≥ 8 × num_cus  →  num_cus // 2   (≥ 50% CU coverage)
       x ≥ 2 × num_cus  →  num_cus // 4   (≥ 25%)
       x < 2 × num_cus  →  num_cus // 8   (≥ 12.5% — best possible for small x)

  2. Work-per-thread guard:  rnumel / threads_per_output ≥ 1
     Ensures each thread reduces at least one r-element (otherwise most
     threads would be idle, wasting wavefront slots).

  3. Deduplication: configs that collapse to the same (XBLOCK, R0_BLOCK,
     num_warps, num_stages) after AMD's _num_warps clamping are removed.
"""

import math
from typing import Any, Callable

__all__ = ["ReductionHeuristics"]

# ---------------------------------------------------------------------------
# Architecture config (reuse the same singleton as pointwise)
# ---------------------------------------------------------------------------

try:
    from .arch import get_architecture_config
except ImportError:
    get_architecture_config = None  # type: ignore[assignment]

class ReductionHeuristics:
    """Semi-exhaustive config generation for AMD INNER and OUTER reductions.

    All public methods are @classmethods; the class is never instantiated.
    The _arch_config cache is a process-level singleton (one GPU per process).

    Enabled by default on ROCm.  To disable:
        TORCHINDUCTOR_REDUCTION_HEURISTICS=0
    """

    _arch_config = None

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_arch(cls):
        """Return the cached ArchitectureConfig, querying the device on first call."""
        if cls._arch_config is None:
            if get_architecture_config is not None:
                try:
                    cls._arch_config = get_architecture_config()
                except Exception as e:  # noqa: BLE001
                    pass
            if cls._arch_config is None:
                # Conservative defaults when no GPU context is available.
                # Use CDNA2 (MI250X) values: 65536 VGPRs/CU, 2048 threads/CU
                # → vgpr_budget_per_thread=32 → _max_r0_per_thread()=16.
                from types import SimpleNamespace
                cls._arch_config = SimpleNamespace(
                    num_cus=256,
                    warp_size=64,
                    occupancy_sweetspot_max=8,
                    regs_per_cu=65536,
                    max_threads_per_cu=2048,
                    vgpr_budget_per_thread=32,  # 65536 // 2048
                )
        return cls._arch_config

    @classmethod
    def _warp_size(cls) -> int:
        arch = cls._get_arch()
        return getattr(arch, "warp_size", 64)

    @classmethod
    def _num_cus(cls) -> int:
        arch = cls._get_arch()
        return getattr(arch, "num_cus", 256)

    @classmethod
    def _max_nw(cls) -> int:
        """Maximum num_warps the hardware / Triton allows for reduction kernels.

        AMD: warp_size=64, so 8 warps = 512 threads (hardware limit for r≤8192).
        Derived from arch.occupancy_sweetspot_max (typically 8 on CDNA3).
        """
        arch = cls._get_arch()
        return getattr(arch, "occupancy_sweetspot_max", 8)

    # ------------------------------------------------------------------
    # Derived limits (architecture-relative, not hardcoded)
    # ------------------------------------------------------------------

    @classmethod
    def min_blocks(cls, xnumel: int) -> int:
        """Minimum acceptable block count for the given output-element count.

        Scales relative to num_cus so the same logic works on MI200 (104 CUs),
        MI300X (256 CUs), or future devices without changing constants.

          x ≥ 8 × num_cus  →  num_cus // 2   (target ≥ 50% CU coverage)
          x ≥ 2 × num_cus  →  num_cus // 4   (target ≥ 25%)
          x < 2 × num_cus  →  num_cus // 8   (best possible for small x)

        Always clamped to xnumel — you cannot get more blocks than output
        elements (minimum XBLOCK=1 gives exactly xnumel blocks).  This prevents
        the filter from rejecting all configs for tiny-x problems (e.g. x=8),
        where even XBLOCK=1 gives fewer blocks than the coverage floor.
        """
        nc = cls._num_cus()
        if xnumel >= nc * 8:
            floor = nc // 2
        elif xnumel >= nc * 2:
            floor = nc // 4
        else:
            floor = max(8, nc // 8)
        # Can't demand more blocks than the problem has output elements.
        return min(floor, xnumel)

    @classmethod
    def _r0_tiers(
        cls,
        rnumel: int,
        max_r0_block: int,
        register_intensive: bool,
    ) -> dict[str, int]:
        """Return named R0_BLOCK values that cover the useful range.

          r_full    ≤ max_r0_block        primary tile; fewest loop iterations
          r_large   ≤ max_r0_block × 2   single-pass variant (fewer iters, more regs)
          r_xlarge  ≤ max_r0_block × 4   used when rnumel is very large
          r_half    ≥ 256                lower register pressure
          r_quarter ≥ 128                safety net for Welford / multi-accumulator

        All values are clamped to rnumel (no tile can exceed the problem size).
        r_quarter is only included when register_intensive=True.
        """
        r_full    = min(rnumel, max_r0_block)
        r_large   = min(rnumel, max_r0_block * 2)
        r_xlarge  = min(rnumel, max_r0_block * 4)
        r_half    = max(max_r0_block // 8, r_full // 2)   # ≥ 256 for max_r0=2048
        r_quarter = max(max_r0_block // 16, r_full // 4)  # ≥ 128

        tiers: dict[str, int] = {
            "r_full":   r_full,
            "r_large":  r_large,
            "r_xlarge": r_xlarge,
            "r_half":   r_half,
        }
        if register_intensive:
            tiers["r_quarter"] = r_quarter
        return tiers

    # Maximum R0 elements per thread before register spills are likely.
    # Derived empirically: on AMD (warp_size=64) reduction kernels that exceed
    # this ratio consistently produce n_spills > 32 (the AMD bench() threshold),
    # causing bench() to return float("inf") without executing on the GPU — i.e.
    # the config wastes a full Triton compile and returns nothing useful.
    #
    # From tuning log analysis across all reduction kernels:
    #   R0/thread ≤ 16 : 0 inf out of 256 benchmarks  (always safe)
    #   R0/thread = 32  : 14 inf, 85 OK               (kernel-dependent)
    #   R0/thread ≥ 64  : 76 inf, 7 OK                (almost always inf)
    #
    @classmethod
    def _max_r0_per_thread(cls) -> int:
        """Maximum reduction-domain elements per thread before register spilling.

        Derived from the device's actual VGPR file rather than a compile-time
        constant.  The formula is:

            vgpr_budget = regs_per_cu // max_threads_per_cu
            max_r0      = clamp(vgpr_budget // 2, lo=8, hi=32)

        The //2 factor reserves half the VGPR budget for non-accumulator
        registers (loop indices, address computation, compiler temporaries).

        Representative values:
          CDNA2 (MI250X): 65536 // 2048 = 32 VGPRs/thread → max_r0 = 16
          CDNA3 (MI300X): 131072 // 2048 = 64 VGPRs/thread → max_r0 = 32
          fallback (no GPU): vgpr_budget=32 → max_r0 = 16 (conservative)

        Clamped to [8, 32] so heuristics stay sane on exotic or future
        architectures where the device property may be unreliable.
        """
        arch = cls._get_arch()
        vgpr_budget = getattr(arch, 'vgpr_budget_per_thread', 32)
        return max(8, min(32, vgpr_budget // 2))

    @classmethod
    def _min_tpo_at_r0_limit(cls) -> int:
        """Minimum threads-per-output when r0_per_thread is at the VGPR limit.

        Derived from 2 × warp_size so this generalises across wave32 (RDNA) and
        wave64 (CDNA) architectures rather than being hardcoded to 128.
        """
        return 2 * cls._warp_size()

    @classmethod
    def _is_valid_inner(
        cls,
        xnumel: int,
        rnumel: int,
        xblock: int,
        num_warps: int,
        r0_block: int = 0,
        register_intensive: bool = False,
    ) -> bool:
        """Return True if this (XBLOCK, num_warps, R0_BLOCK) triple passes all guards.

        Guard 1 — Block-count:
          ⌈xnumel / XBLOCK⌉ ≥ min_blocks(xnumel)

        Guard 2 — Work-per-thread (lower bound):
          threads_per_output = (num_warps × warp_size) / XBLOCK
          Each thread must reduce ≥ 1 r-element:
            rnumel / threads_per_output ≥ 1
            ↔ threads_per_output ≤ rnumel

        Guard 3a — Register pressure hard ceiling:
          r0_per_thread = R0_BLOCK / threads_per_output
          r0_per_thread ≤ _max_r0_per_thread()   (device-derived, see method)

          Each thread holds r0_per_thread reduction-domain elements in VGPRs.
          Exceeding the threshold guarantees register spilling on AMD hardware.

        Guard 3b — Boundary unroll pressure (AMD-specific):
          When r0_per_thread == _max_r0_per_thread() (the boundary),
          enforce threads_per_output ≥ _min_tpo_at_r0_limit() (2 × warp_size).

          Hardware rationale (AMD CDNA, boundary occupancy):
          ──────────────────────────────────────────────────
          At r0/thread == _max_r0_per_thread(), kernel complexity determines
          whether the VGPR budget holds.  The Triton/AMDGPU compiler's inner-loop unroll factor U is
          inversely related to the number of threads contributing to each output
          element (threads_per_output, TPO):

            U ≈ max(1, ceil(warp_size / TPO))
               TPO=64  → U≈1   (but AMD LLVM still unrolls 4-8× for ILP)
               TPO=128 → U≈1   (sufficient thread-level parallelism to avoid it)

          At TPO < 128 the backend applies aggressive instruction-level unrolling
          (U ≈ 4–8) to hide HBM latency.  Each unrolled copy keeps a full set
          of live accumulator VGPRs in flight simultaneously:

            • Welford std/var: 3 accumulators × U copies = 3U VGPRs
            • Each transcendental op (cos, exp, gelu, …): ~8 VGPRs × U
            • Example (12 transcendentals, U=8): 12×8×8 = 768 VGPRs → spill

          At TPO ≥ 128 (2+ warps per output), thread-level parallelism provides
          sufficient HBM-latency hiding, allowing U=1–2.  VGPR usage stays below
          the per-thread budget for all observed kernel complexities.

          Empirical validation (tuning66 + tuning67 logs):
            • 55 spilling configs at r0pt=32, all with TPO < 128 (100% spill rate)
            • 140 valid configs at r0pt=32, all with TPO ≥ 128 (0% spill rate)
            • 0 winning configs have r0pt=32 AND TPO < 128 — no best-config loss
        """
        ws = cls._warp_size()
        if math.ceil(xnumel / xblock) < cls.min_blocks(xnumel):
            return False
        threads_per_output = (num_warps * ws) / xblock
        if rnumel / max(1.0, threads_per_output) < 1.0:
            return False
        if r0_block > 0:
            r0_per_thread = r0_block / max(1.0, threads_per_output)
            max_r0 = cls._max_r0_per_thread()
            # register_intensive kernels (fused transcendentals, many ops) use
            # half the budget because computational temps consume significant
            # portions of the VGPR file alongside the reduction accumulators.
            effective_max = max_r0 // 2 if register_intensive else max_r0
            if r0_per_thread > effective_max:
                return False
            # Guard 3b: at the VGPR-pressure boundary, require enough threads per
            # output so the compiler can use thread-level (not instruction-level)
            # parallelism to hide memory latency.  Uses effective_max (halved for
            # register-intensive kernels) so the boundary is consistent with 3a.
            if r0_per_thread >= effective_max:
                if threads_per_output < cls._min_tpo_at_r0_limit():
                    return False
        return True

    # ------------------------------------------------------------------
    # Public API: config generators
    # ------------------------------------------------------------------

    @classmethod
    def inner_configs(
        cls,
        *,
        xnumel: int,
        rnumel: int,
        max_r0_block: int,
        register_intensive: bool,
        make_config_fn: Callable,
    ) -> list[Any]:
        """Generate semi-exhaustive INNER reduction candidates.

        Sweeps (XBLOCK, num_warps, R0_BLOCK) over a principled grid, filtered
        by the block-count and work-per-thread guards.

        Parameters
        ----------
        xnumel            : number of output elements (problem x-dimension)
        rnumel            : total reduction elements across all r-dimensions
        max_r0_block      : hard cap on R0_BLOCK (typically 2048 on AMD)
        register_intensive: True for Welford/var/std — halves nw cap & adds r_quarter
        make_config_fn    : factory that creates a triton.Config
                            signature: (xblock, r0, num_warps, *, num_stages,
                                        register_intensive) → Config

        Returns
        -------
        list of Config objects (deduped, no guaranteed ordering).

        Note on num_stages
        ------------------
        On AMD/HIP, the Triton codegen already hardcodes ``tl.range(...,
        num_stages=2)`` directly in the kernel source for every non-persistent
        reduction loop (``triton.py`` lines ~4861-4866), so ``Config.num_stages``
        has no additional effect on reduction pipelining.  All configs here use
        ``num_stages=1`` (the default) intentionally.
        """
        nw_hw = cls._max_nw()

        # AMD halves effective num_warps when register_intensive (spill avoidance)
        nw_cap = nw_hw // 2 if register_intensive else nw_hw

        tiers = cls._r0_tiers(rnumel, max_r0_block, register_intensive)
        r_full    = tiers["r_full"]
        r_large   = tiers["r_large"]
        r_xlarge  = tiers["r_xlarge"]
        r_half    = tiers["r_half"]
        r_quarter = tiers.get("r_quarter")

        # ── XBLOCK sweep grid ─────────────────────────────────────────────────
        # Build a dynamic power-of-2 range from 1 up to max_xb, where max_xb
        # is the largest XBLOCK that still gives ≥ min_blocks blocks.
        #
        # max_xb = floor(xnumel / min_blocks), rounded DOWN to a power of 2,
        # then capped at 1024 to prevent tiles that exceed Triton's numel limit.
        #
        # Examples (MI300X, num_cus=256):
        #   xnumel=8      → min_blocks=1   → max_xb=8    → [1,2,4,8]
        #   xnumel=256    → min_blocks=32  → max_xb=8    → [1,2,4,8]
        #   xnumel=2048   → min_blocks=64  → max_xb=32   → [1,2,4,8,16,32]
        #   xnumel=65536  → min_blocks=128 → max_xb=512  → [1,2,4,…,512]
        #   xnumel=1M     → min_blocks=512 → max_xb=1024 → [1,2,4,…,1024]
        #
        # The _is_valid_inner filter still prunes any (xb, nw) pair that
        # violates block-count or work-per-thread guards, so extending the
        # grid is safe — it only adds candidates that the filter passes.
        _mb = cls.min_blocks(xnumel)
        _max_xb_raw = xnumel // max(1, _mb)          # exact quotient
        # Round down to power of 2 (bit_length()-1 for floor-power-of-2)
        _max_xb = 1 << (max(1, _max_xb_raw).bit_length() - 1)
        _max_xb = min(_max_xb, 1024)                 # hard cap
        xblock_grid = []
        _xb = 1
        while _xb <= _max_xb:
            xblock_grid.append(_xb)
            _xb *= 2

        # ── num_warps sweep grid ──────────────────────────────────────────────
        # nw=1: 0 DS barriers (1 wavefront/block), max occupancy potential
        # nw=2: 1 DS barrier
        # nw=4: 2 DS barriers
        # nw=8: 3 DS barriers, max per-block HBM bandwidth
        nw_grid = [nw for nw in [1, 2, 4, 8] if nw <= nw_cap]

        # ── Base R0_BLOCK candidates ──────────────────────────────────────────
        # Include r_full and r_half; skip duplicates.
        r0_base = sorted({r_full, r_half}, reverse=True)  # full first

        candidates: list[Any] = []
        seen: set = set()

        def _add(xb: int, r0: int, nw: int) -> None:
            """Append a config if it passes filters and is not a duplicate."""
            key = (xb, r0, nw)
            if key in seen:
                return
            if not cls._is_valid_inner(xnumel, rnumel, xb, nw, r0_block=r0,
                                       register_intensive=register_intensive):
                return
            seen.add(key)
            candidates.append(make_config_fn(
                xb, r0, num_warps=nw,
                num_stages=1,
                register_intensive=register_intensive,
            ))

        # ── Main sweep: XBLOCK × num_warps × R0_BLOCK ────────────────────────
        #
        #  r_full / r_half — the core candidates for every (xb, nw) pair
        for xb in xblock_grid:
            for nw in nw_grid:
                for r0 in r0_base:
                    _add(xb, r0, nw)
                if r_quarter is not None and r_quarter < r_half:
                    _add(xb, r_quarter, nw)

        # ── r_large / r_xlarge — fewer loop iterations, more registers ────────
        #  Only for large rnumel where the trade-off is worthwhile.
        #  Restrict to small XBLOCK (XBLOCK=1 and XBLOCK=2 when x is large enough)
        #  to avoid generating too many configs for already-large candidate sets.
        #  Guard: skip for register_intensive kernels — these already have reduced
        #  max_r0_block precisely to avoid register pressure; exceeding it by 2×/4×
        #  would risk spilling even with reduced num_warps.
        if not register_intensive and rnumel >= max_r0_block * 2 and r_large > r_full:
            for nw in nw_grid:
                _add(1, r_large, nw)
                # XBLOCK=2 only when there are enough x to maintain block count
                if xnumel >= cls.min_blocks(xnumel) * 2:
                    _add(2, r_large, nw)

        if not register_intensive and rnumel >= max_r0_block * 4 and r_xlarge > r_large:
            for nw in nw_grid:
                _add(1, r_xlarge, nw)

        return candidates

    @classmethod
    def outer_configs(
        cls,
        *,
        xnumel: int,
        rnumel: int,
        register_intensive: bool,
        make_config_fn: Callable,
    ) -> list[Any]:
        """Generate semi-exhaustive OUTER reduction candidates.

        For OUTER reductions the Triton layout is [XBLOCK, R0_BLOCK] where:
          - XBLOCK   controls grid = ⌈xnumel / XBLOCK⌉  (CU utilization)
          - R0_BLOCK controls the inner r-stride step     (HBM coalescing)

        Tuning axes:
          XBLOCK   : sweep to target 12% / 25% / 50% / 100% / 200% CU coverage
          num_warps: {1, 2, 4, 8}
          R0_BLOCK : {r_full, r_half} where r_full = min(rnumel, 8) for OUTER

        Parameters
        ----------
        xnumel             : number of output elements
        rnumel             : size of the reduction dimension
        register_intensive : currently unused for OUTER but kept for API symmetry
        make_config_fn     : same factory as for inner_configs

        Returns
        -------
        list of Config objects (deduped).
        """
        nc = cls._num_cus()

        # Adaptive minimum blocks (same logic as inner)
        mb = cls.min_blocks(xnumel)

        # OUTER R0_BLOCK: the kernel strides over r in steps of R0_BLOCK.
        # A small R0_BLOCK keeps the inner-loop tight and prefetch-friendly.
        r_o_full = min(rnumel, 8)       # OUTER default: tiny R0_BLOCK ≤ 8
        r_o_half = max(1, r_o_full // 2)
        r0_options = [r_o_full] if r_o_half == r_o_full else [r_o_full, r_o_half]

        seen: set = set()
        candidates: list[Any] = []

        def _add(xb: int, r0: int, nw: int) -> None:
            n_blocks = math.ceil(xnumel / xb)
            if n_blocks < mb:
                return
            key = (xb, r0, nw)
            if key in seen:
                return
            seen.add(key)
            candidates.append(make_config_fn(
                xb, r0, num_warps=nw,
                num_stages=1,
                register_intensive=register_intensive,
            ))

        # Target CU coverage levels: 200%, 100%, 50%, 25%, mb (floor)
        target_block_counts = [nc * 2, nc, nc // 2, nc // 4, mb]
        for target in target_block_counts:
            # xb = ⌈xnumel / target⌉ rounded up to next power of 2
            raw = max(1, xnumel // max(1, target))
            xb_t = 1 << (max(0, raw - 1)).bit_length()
            # nw=1 added: single-wavefront blocks have zero DS barriers and
            # maximum occupancy potential when r-loop is long.
            for nw in [8, 4, 2, 1]:
                for r0 in r0_options:
                    _add(xb_t, r0, nw)

        return candidates

    # ------------------------------------------------------------------
    # Scoring helpers — problem_metadata construction
    # ------------------------------------------------------------------

    @classmethod
    def problem_metadata_from_size_hints(
        cls,
        size_hints: dict,
        inductor_meta: dict | None = None,
    ) -> dict:
        """Build a problem_metadata dict for reduction scoring.

        Parameters
        ----------
        size_hints   : the size_hints dict passed to CachingAutotuner
                       (keys: 'x', 'r', optionally 'y', 'z')
        inductor_meta: the inductor_meta dict (ops_per_element, load_ops, …)

        Returns
        -------
        dict with keys understood by the scoring methods below and by
        BottleneckAnalysis.analyze_bottleneck().
        """
        im = inductor_meta or {}
        ws = cls._warp_size()

        xnumel  = size_hints.get("x", 1)
        # Accumulate ALL reduction dimensions.  size_hints may use the legacy
        # 'r' key or the modern 'r0_' / 'r1_' keys (both start with 'r').
        r_vals = [
            v for k, v in size_hints.items()
            if k.startswith("r") and k != "x" and isinstance(v, int) and v > 0
        ]
        rnumel = 1
        for rv in r_vals:
            rnumel *= rv
        if not r_vals:
            rnumel = 1  # fallback: no reduction dims found
        # For 3-D problems there may also be 'y'; multiply into rnumel.
        ynumel  = size_hints.get("y", 1)
        # reduction_hint can be a ReductionHint enum, a plain string like
        # "inner"/"INNER", or the full enum repr "ReductionHint.INNER".
        # Normalise everything to the bare name: "INNER" or "OUTER".
        _raw_hint = im.get("reduction_hint", "INNER")
        hint = str(_raw_hint).split(".")[-1].upper()
        reg_int = bool(im.get("register_intensive", False))

        MAX_R0 = 2048
        max_r0_block = MAX_R0 // 2 if reg_int else MAX_R0

        # Element size: try from inductor_meta, default to float32.
        elem_size = int(im.get("element_size", 4))

        ops_pe    = float(im.get("ops_per_element", 1))
        load_ops  = int(im.get("load_ops", 1))
        slow_ops  = int(im.get("slow_ops", 0))

        return {
            # Core dimensions
            "xnumel":            xnumel,
            "rnumel":            rnumel,
            "ynumel":            ynumel,
            "reduction_hint":    hint,          # 'INNER' or 'OUTER'
            "register_intensive": reg_int,
            "max_r0_block":      max_r0_block,
            # Architecture
            "warp_size":         ws,
            # Instruction mix / compute
            "ops_per_element":   ops_pe,
            "slow_ops":          slow_ops,
            "load_ops":          load_ops,
            "element_size":      elem_size,
            # ── BottleneckAnalysis-compatible fields ──────────────────────
            # Each output element requires loading rnumel inputs.
            "total_elements":    xnumel,           # output elements = "work items"
            "bytes_per_element": rnumel * elem_size,  # bytes per output (all inputs)
            "num_inputs":        max(1, load_ops),
            "num_outputs":       1,
            "num_tensors":       max(2, load_ops + 1),
        }

    # ------------------------------------------------------------------
    # Scoring factors (INNER and shared)
    # ------------------------------------------------------------------

    @classmethod
    def _score_r_efficiency(cls, config: dict, pm: dict) -> float:
        """Factor A — R0_BLOCK amortization of per-iteration DPP/DS overhead.

        Each outer-loop iteration pays a fixed overhead (DPP butterfly ≈ 24
        cycles, plus DS rounds × ~100 cycles).  A larger R0_BLOCK means fewer
        iterations → less total overhead per r-element.

        Peak at ``optimal_r0 = min(rnumel, max_r0_block)`` (halved for
        register-intensive kernels to avoid spill).  Score uses a log-scale
        so partial tiles (e.g. r0=1024 vs r0=2048) are not severely penalised.

        Above-optimal penalty for register-intensive / slow-op heavy kernels:
        Exceeding optimal_r0 increases the r0-elements processed per thread,
        which raises VGPR pressure:
          • Welford std/var: 3 accumulators (mean, M2, count) per r0-element.
            e.g. R0=8192, nw=8, x=4 → 16 r0/thread × 3 = 48 Welford VGPRs.
          • Transcendental fusions (tanh+sin+cos+exp): each SFU op needs temp
            registers; at high R0_BLOCK the unrolled loop holds many live temps.
        In both cases higher R0_BLOCK is empirically slower even with n_spills=0
        (sub-spill register pressure reduces occupancy / increases pipeline stalls).

        Penalty scale: at excess=2 → −5%; excess=4 → −10%; excess=8 → −15%.

        Returns a value in [0.65, 1.00].
        """
        r0_block           = config.get("R0_BLOCK", 1)
        max_r0_block       = pm.get("max_r0_block", 2048)
        rnumel             = pm.get("rnumel", 1)
        register_intensive = pm.get("register_intensive", False)
        slow_ops           = pm.get("slow_ops", 0)

        optimal_r0 = max_r0_block // 2 if register_intensive else max_r0_block
        optimal_r0 = min(optimal_r0, rnumel)

        if optimal_r0 <= 1:
            return 1.0

        r_ratio = r0_block / optimal_r0
        if r_ratio >= 1.0:
            # Above optimal: for register-intensive (Welford) kernels or kernels
            # with many transcendental ops (slow_ops ≥ 4), penalise excessive
            # R0_BLOCK proportionally to log2(excess).  This prevents the sync
            # factor (which rewards larger R0_BLOCK for fewer DPP passes) from
            # always pushing the highest R0_BLOCK to rank #1 even when it causes
            # meaningful VGPR-induced slowdowns empirically.
            if register_intensive or slow_ops >= 4:
                excess = r0_block / max(1, optimal_r0)
                # penalty: −5% per doubling of excess above optimal
                score = max(0.65, 1.0 - 0.05 * math.log2(max(1.0, excess)))
            elif slow_ops >= 2:
                excess = r0_block / max(1, optimal_r0)
                # moderate penalty: −3% per doubling
                score = max(0.70, 1.0 - 0.03 * math.log2(max(1.0, excess)))
            else:
                score = 1.0
        else:
            # Log-scale: log2(r0)/log2(optimal).  r0=optimal/2 → 1-1/log2(opt);
            # for optimal=2048 that's 1-1/11 ≈ 0.91.
            log_opt = math.log2(max(optimal_r0, 2))
            log_r0  = math.log2(max(r0_block, 1))
            score   = 0.65 + 0.35 * min(1.0, log_r0 / log_opt)

        return max(0.65, min(1.0, score))

    @classmethod
    def _score_grid_coverage(cls, config: dict, pm: dict) -> float:
        """Factor B — GPU wavefront coverage from the output-dimension grid.

        total_wavefronts = ⌈xnumel / XBLOCK⌉ × num_warps.

        The real measure of CU utilization for INNER reductions is total
        wavefronts in flight, not raw block count.  Each CU needs ~4
        wavefronts to fully hide HBM latency, so the target is:

            target_wf = num_cus × 4

        Example comparisons on MI300X (256 CUs, target = 1024 wavefronts):
          x=128, XBLOCK=1, nw=8  → 128×8 =1024 ≥ 1024 → 1.0   (100% GPU)
          x=128, XBLOCK=2, nw=8  → 64×8  = 512 < 1024 → ~0.90  (50%)
          x=128, XBLOCK=4, nw=4  → 32×4  = 128 < 1024 → ~0.80  (12.5%)

        Previously this used raw block count vs min_blocks, which gave
        XBLOCK=4/nw=4 a perfect score because 32 ≥ min_blocks=32 — even
        though only 128 wavefronts were created and 87.5% of the GPU sat
        idle.  XBLOCK=1/nw=8 (1024 wavefronts) also scored 1.0, so the
        CU starvation of large-XBLOCK configs was invisible to the scorer.

        Score is one-sided:
          total_wf ≥ target_wf  →  1.0   (sufficient wavefront coverage)
          total_wf < target_wf  →  falls smoothly from 1.0 toward 0.70
                                    using a left-tail Gaussian.

        Returns a value in [0.70, 1.00].
        """
        xnumel    = pm.get("xnumel", 1)
        xblock    = config.get("XBLOCK", 1)
        num_warps = config.get("num_warps", 4)

        num_blocks       = math.ceil(xnumel / max(1, xblock))
        total_wavefronts = num_blocks * num_warps

        # Target: num_cus × 4 wavefronts (4 per CU = AMD HBM-latency sweet spot)
        target_wf = cls._num_cus() * 4

        if total_wavefronts >= target_wf:
            return 1.0

        # Apply left-tail Gaussian penalty for insufficient wavefronts
        sigma = max(1.0, target_wf * 0.5)
        diff  = (total_wavefronts - target_wf) / sigma   # negative
        gauss = math.exp(-0.5 * diff * diff)
        return max(0.70, min(1.0, 0.75 + 0.25 * gauss))

    @classmethod
    def _score_sync_overhead(cls, config: dict, pm: dict) -> float:
        """Factor C — DS cross-wavefront synchronization cost (INNER only).

        When num_warps > XBLOCK, multiple wavefronts cooperate on a single
        output element.  After the intra-wavefront DPP butterfly, each warp
        must synchronize via DS (LDS):

            ds_rounds = log₂(num_warps / XBLOCK)   [when num_warps > XBLOCK]

        DS bandwidth on MI300X is ~100 TB/s — very fast, but there is still a
        fixed per-round overhead (~100 cycles for the barrier).  The penalty
        is most visible when R0_BLOCK is small (few HBM loads per iteration,
        so DS overhead is a large fraction of per-iteration cost).

        overhead_ratio = ds_cycles / (ds_cycles + hbm_cycles)
        score = 1.0 − 0.30 × overhead_ratio

        Returns a value in [0.70, 1.00].  Returns 1.0 for OUTER reductions
        (they do not use cross-wavefront DS for the x-dimension).
        """
        hint = pm.get("reduction_hint", "INNER")
        if hint != "INNER":
            return 1.0

        xblock    = config.get("XBLOCK", 1)
        num_warps = config.get("num_warps", 4)
        r0_block  = config.get("R0_BLOCK", 64)

        wavefronts_per_output = num_warps / max(1, xblock)
        if wavefronts_per_output <= 1.0:
            ds_rounds = 0.0
        else:
            ds_rounds = math.log2(wavefronts_per_output)

        ds_cost_per_iter  = ds_rounds * 100           # cycles, rough
        # HBM: r0_block FP32 elements at arch.hbm_bytes_per_cycle_per_cu.
        # Falls back to 1.8 bytes/cycle/CU if arch is unavailable.
        try:
            _sync_arch = cls._get_arch()
            _hbm_bpc   = getattr(_sync_arch, 'hbm_bytes_per_cycle_per_cu', 1.8)
        except Exception as e:  # noqa: BLE001
            _hbm_bpc = 1.8
        hbm_cost_per_iter = max(1.0, r0_block * 4 / _hbm_bpc)

        if ds_cost_per_iter <= 0:
            return 1.0

        overhead_ratio = ds_cost_per_iter / (ds_cost_per_iter + hbm_cost_per_iter)
        score = 1.0 - 0.30 * min(1.0, overhead_ratio)
        return max(0.70, min(1.0, score))

    @classmethod
    def _score_warp_parallelism(cls, config: dict, pm: dict) -> float:
        """Factor D — num_warps vs. sweet spot for HBM latency hiding.

        Mirrors the pointwise occupancy factor:
          sweet_min–sweet_max (4–8 on AMD) → score 1.00.
          Below sweet_min                  → penalty (under-subscribed HBM BW).
          Above sweet_max                  → slight penalty (VGPR pressure).

        Two reduction-specific corrections:
          1. register_intensive → halve sweet_max (tighter VGPR budget means
             more warps → more spills).
          2. Idle-thread penalty: if threads_per_output > rnumel, the extra
             threads sit idle inside each wavefront during the reduction;
             score is reduced proportionally.

        Returns a value in [0.70, 1.00].
        """
        num_warps  = config.get("num_warps", 4)
        xblock     = config.get("XBLOCK", 1)
        rnumel     = pm.get("rnumel", 1)
        ws         = pm.get("warp_size", cls._warp_size())
        reg_int    = pm.get("register_intensive", False)

        arch      = cls._get_arch()
        sweet_min = getattr(arch, "occupancy_sweetspot_min", 4)
        sweet_max = getattr(arch, "occupancy_sweetspot_max", 8)
        if reg_int:
            sweet_max = max(1, sweet_max // 2)

        # Idle-thread check
        threads_per_output = (num_warps * ws) / max(1, xblock)
        if threads_per_output > rnumel:
            idle_frac = min(1.0, (threads_per_output - rnumel) / threads_per_output)
            return max(0.70, 1.0 - 0.30 * idle_frac)

        if sweet_min <= num_warps <= sweet_max:
            score = 1.00
        elif num_warps < sweet_min:
            score = 0.82 + 0.18 * (num_warps / sweet_min)
        else:
            excess_frac = min(1.0, (num_warps - sweet_max) / max(1, sweet_max))
            score = 0.90 - 0.10 * excess_frac
            if reg_int:
                score -= 0.05 * excess_frac

        return max(0.70, min(1.0, score))

    @classmethod
    def _score_outer_coalescing(cls, config: dict, pm: dict) -> float:
        """Coalescing factor for OUTER reductions.

        OUTER kernels iterate over r in steps of R0_BLOCK while keeping XBLOCK
        output lanes active.  Good coalescing requires XBLOCK ≥ cache_line_elems
        (16 FP32 values = 64 bytes) so each wavefront's store to the output
        buffer is cache-line-aligned.

        Returns a value in [0.75, 1.00].
        """
        xblock = config.get("XBLOCK", 1)
        # Architecture-specific cache-line width in FP32 elements.
        # AMD CDNA2+ (gfx90a+): 128-byte cacheline = 32 FP32 elements.
        # NVIDIA / older AMD: 64-byte cacheline = 16 FP32 elements.
        try:
            _coal_arch       = cls._get_arch()
            cache_line_elems = getattr(_coal_arch, 'cacheline_bytes', 64) // 4
        except Exception as e:  # noqa: BLE001
            cache_line_elems = 16
        if xblock >= cache_line_elems:
            return 1.00
        coalescing = xblock / cache_line_elems
        return max(0.75, 0.75 + 0.25 * coalescing)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    @classmethod
    def score_config(cls, config: dict, problem_metadata: dict) -> float:
        """Return a composite score in [0.0, 1.0] for a single reduction config.

        Four factors are combined via a weighted geometric mean:

            score = (A^a × B^b × C^c × D^d)^(1/(a+b+c+d))

        INNER weights (adaptive, interpolated by BottleneckAnalysis):

            Factor                 overhead-bound  memory-bound  compute-bound
            ─────────────────────  ──────────────  ────────────  ─────────────
            A  r_efficiency        0.15            0.40          0.35
            B  grid_coverage       0.35            0.20          0.15
            C  sync_overhead       0.30            0.20          0.15
            D  warp_parallelism    0.20            0.20          0.35

        OUTER weights (fixed, grid_coverage dominated):

            A  r_efficiency     0.15
            B  grid_coverage    0.50
            C  coalescing       0.20
            D  warp_parallelism 0.15

        BottleneckAnalysis is invoked with total_elements=xnumel and
        bytes_per_element=rnumel×elem_size to correctly model the heavy
        per-output HBM load that distinguishes reductions from pointwise.
        """
        try:
            hint = problem_metadata.get("reduction_hint", "INNER")

            r_eff = cls._score_r_efficiency(config, problem_metadata)
            grid  = cls._score_grid_coverage(config, problem_metadata)
            par   = cls._score_warp_parallelism(config, problem_metadata)

            if hint == "INNER":
                sync = cls._score_sync_overhead(config, problem_metadata)

                # Run BottleneckAnalysis to get regime fractions.
                o_frac, m_frac, c_frac = 0.10, 0.80, 0.10  # default: memory-bound
                try:
                    from .triton_heuristics_bottleneck import BottleneckAnalysis
                    ba_config = {
                        "XBLOCK": config.get("XBLOCK", 1),
                        "num_warps": config.get("num_warps", 4),
                    }
                    analysis = BottleneckAnalysis.analyze_bottleneck(
                        ba_config, problem_metadata
                    )
                    o_frac = analysis["overhead_frac"]
                    m_frac = analysis["memory_frac"]
                    c_frac = analysis["compute_frac"]
                except Exception as e:  # noqa: BLE001
                    pass

                # Pure-regime weight tables (sum to 1.0 each)
                OVERHEAD_W = {"r_eff": 0.15, "grid": 0.35, "sync": 0.30, "par": 0.20}
                MEMORY_W   = {"r_eff": 0.40, "grid": 0.20, "sync": 0.20, "par": 0.20}
                COMPUTE_W  = {"r_eff": 0.35, "grid": 0.15, "sync": 0.15, "par": 0.35}

                weights = {
                    k: o_frac * OVERHEAD_W[k]
                       + m_frac * MEMORY_W[k]
                       + c_frac * COMPUTE_W[k]
                    for k in OVERHEAD_W
                }
                scores = {"r_eff": r_eff, "grid": grid, "sync": sync, "par": par}

            else:  # OUTER
                coale  = cls._score_outer_coalescing(config, problem_metadata)
                weights = {"r_eff": 0.15, "grid": 0.50, "coale": 0.20, "par": 0.15}
                scores  = {"r_eff": r_eff, "grid": grid, "coale": coale, "par": par}

            total_w = sum(weights.values())
            log_score = sum(
                weights[k] / total_w * math.log(max(scores[k], 1e-9))
                for k in scores
            )
            return max(0.0, min(1.0, math.exp(log_score)))

        except Exception as e:  # noqa: BLE001
            return 0.0

    @classmethod
    def get_detailed_scores(
        cls, config: dict, problem_metadata: dict
    ) -> dict[str, float]:
        """Return a per-factor breakdown dict for config introspection.

        Keys: r_efficiency, grid_coverage, sync_overhead (INNER) or
        outer_coalescing (OUTER), warp_parallelism, composite.
        All values in [0.0, 1.0].
        """
        hint = problem_metadata.get("reduction_hint", "INNER")
        try:
            r_eff  = cls._score_r_efficiency(config, problem_metadata)
            grid   = cls._score_grid_coverage(config, problem_metadata)
            par    = cls._score_warp_parallelism(config, problem_metadata)
            comp   = cls.score_config(config, problem_metadata)

            if hint == "INNER":
                return {
                    "r_efficiency":    r_eff,
                    "grid_coverage":   grid,
                    "sync_overhead":   cls._score_sync_overhead(config, problem_metadata),
                    "warp_parallelism": par,
                    "composite":       comp,
                }
            else:
                return {
                    "r_efficiency":      r_eff,
                    "grid_coverage":     grid,
                    "outer_coalescing":  cls._score_outer_coalescing(config, problem_metadata),
                    "warp_parallelism":  par,
                    "composite":         comp,
                }
        except Exception as e:  # noqa: BLE001
            return {"composite": 0.0}

    # ------------------------------------------------------------------
    # Pruning / top-N selection
    # ------------------------------------------------------------------

    @classmethod
    def prune_configs(
        cls,
        configs: list[Any],
        problem_metadata: dict,
        top_n: int = 5,
    ) -> tuple[list[Any], list[tuple[float, dict]]]:
        """Score all configs and return top-N plus the full ranked list.

        Parameters
        ----------
        configs          : Triton Config objects (with .kwargs and .num_warps)
        problem_metadata : dict from problem_metadata_from_size_hints()
        top_n            : maximum number of configs in the returned shortlist

        Returns
        -------
        (top_n_configs, all_scored)
          top_n_configs : list of Triton Config objects (≤ top_n)
          all_scored    : [(score, config_dict), …] for ALL valid configs,
                          sorted descending by score.  Used by the validation
                          summary to compare predicted vs. actual winners.
        """
        xnumel = problem_metadata.get("xnumel", 1)
        rnumel = problem_metadata.get("rnumel", 1)
        ws     = problem_metadata.get("warp_size", cls._warp_size())

        scored: list[tuple[float, Any, dict]] = []  # (score, triton_cfg, hdict)

        for cfg in configs:
            try:
                xb = cfg.kwargs.get("XBLOCK", 1)
                nw = cfg.num_warps

                # R0_BLOCK may be absent for persistent reductions (dynamic tile).
                # Treat the effective tile as rnumel so scoring reflects the full
                # reduction width (the whole row fits in one CTA).
                r0_raw = cfg.kwargs.get("R0_BLOCK", None)
                r0 = r0_raw if (isinstance(r0_raw, int) and r0_raw >= 1) else max(1, rnumel)

                # Basic sanity: threads must be ≥ warp_size and ≤ 1024.
                threads = nw * ws
                if not (ws <= threads <= 1024):
                    continue
                # R0_BLOCK must be ≥ 1 and ≤ rnumel (skip pathological configs).
                if not (1 <= r0 <= max(1, rnumel)):
                    continue
                # XBLOCK must be ≥ 1 and ≤ xnumel.
                if not (1 <= xb <= max(1, xnumel)):
                    continue

                hdict = {"XBLOCK": xb, "R0_BLOCK": r0, "num_warps": nw}
                s = cls.score_config(hdict, problem_metadata)
                if s > 0:
                    scored.append((s, cfg, hdict))
            except Exception as e:  # noqa: BLE001
                continue

        # Sort descending by score; tiebreaker: larger R0_BLOCK preferred when
        # scores tie (fewer reduction iterations → lower loop overhead).
        scored.sort(
            key=lambda x: (round(x[0], 4), x[2].get("R0_BLOCK", 0)),
            reverse=True,
        )

        # Diversity cap: at most 2 configs per XBLOCK bucket.
        # Prevents large-XBLOCK variants from monopolising the top-N when many
        # (XBLOCK, num_warps, R0_BLOCK) combos score nearly identically.
        xb_counts: dict[int, int] = {}
        primary: list[tuple[float, Any, dict]] = []
        overflow: list[tuple[float, Any, dict]] = []
        for entry in scored:
            xb = entry[2].get("XBLOCK", 1)
            if xb_counts.get(xb, 0) < 2:
                primary.append(entry)
                xb_counts[xb] = xb_counts.get(xb, 0) + 1
            else:
                overflow.append(entry)

        selected = (primary + overflow)[:top_n]

        top_n_configs  = [cfg  for _, cfg,  _     in selected]
        all_scored_out = [(s, hdict) for s, _, hdict in scored]

        return top_n_configs, all_scored_out

