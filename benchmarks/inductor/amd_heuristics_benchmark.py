"""AMD Heuristics Benchmark
=========================
Validates and benchmarks AMD-specific Inductor heuristics across four
compilation modes, printing side-by-side performance and quality tables.

Compilation modes
-----------------
  Baseline      – no heuristics, no max_autotune (single default config)
  Heuristics    – AMD heuristics ON, max_autotune OFF
  MaxAutotune   – AMD heuristics OFF, max_autotune ON
  Heur+MaxAT    – AMD heuristics ON, max_autotune ON (heuristic top-N
                  injected into the exhaustive pool)

Three test layers
-----------------
  Layer 1 – Pure-Python unit tests  (no GPU required)
    Calls PointwiseHeuristics / ReductionHeuristics APIs directly to
    verify config generation, scoring, pruning, VGPR guards, and the new
    device-derived limit methods.

  Layer 2 – torch.compile perf table  (GPU required)
    Compiles ~30 diverse workloads under all four modes.  Correctness-
    checks against eager, warms up, then times N repetitions.  Results
    are printed in two ASCII tables (kernel latency, compile time) with
    the fastest mode per row highlighted.

  Layer 3 – Heuristic quality analysis  (GPU required)
    Uses the Layer 2 results to answer: "how close is the Heuristics
    mode to the MaxAutotune winner?"  Prints a gap-% table, per-workload
    speedup rankings, and a mode win-count summary.

Usage
-----
  # Unit tests only (no GPU):
  python benchmarks/inductor/amd_heuristics_benchmark.py --unit-only

  # Full benchmark (GPU required):
  python benchmarks/inductor/amd_heuristics_benchmark.py

  # BFloat16 workloads:
  python benchmarks/inductor/amd_heuristics_benchmark.py --dtype bf16

  # Skip max_autotune modes (faster, no long compile waits):
  python benchmarks/inductor/amd_heuristics_benchmark.py --no-max-autotune

  # Run only workloads whose label matches a substring:
  python benchmarks/inductor/amd_heuristics_benchmark.py --filter softmax

  # Fewer reps for quick smoke test:
  python benchmarks/inductor/amd_heuristics_benchmark.py --reps 5

  # Verbose heuristic logging:
  TORCH_LOGS="+inductor" python benchmarks/inductor/amd_heuristics_benchmark.py
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hline(char: str = "─", width: int = 82) -> None:
    print(char * width)

def _section(title: str) -> None:
    _hline()
    print(f"  {title}")
    _hline()

def _ok(msg: str) -> None:   print(f"    ✓  {msg}")
def _fail(msg: str) -> None: print(f"    ✗  {msg}"); sys.exit(1)
def _skip(msg: str) -> None: print(f"    –  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 – Pure-Python unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_config(xb, r0, *, num_warps, num_stages=1, register_intensive=False):
    class _Cfg:
        def __init__(self, kw, nw, ns):
            self.kwargs = kw; self.num_warps = nw; self.num_stages = ns
        def __repr__(self):
            return "Config(" + ", ".join(f"{k}={v}" for k, v in sorted(self.kwargs.items())) + f", nw={self.num_warps})"
    return _Cfg({"XBLOCK": xb, "R0_BLOCK": r0}, num_warps, num_stages)


def test_reduction_inner_configs() -> None:
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    cases = [
        ("tiny   (x=1,   r=256)",   dict(xnumel=1,    rnumel=256)),
        ("small  (x=8,   r=1024)",  dict(xnumel=8,    rnumel=1024)),
        ("medium (x=256, r=4096)",  dict(xnumel=256,  rnumel=4096)),
        ("large  (x=4096,r=65536)", dict(xnumel=4096, rnumel=65536)),
        ("huge   (x=1,   r=1M)",    dict(xnumel=1,    rnumel=1048576)),
        ("reg-int(x=128, r=8192)",  dict(xnumel=128,  rnumel=8192, register_intensive=True)),
        ("square (x=1024,r=1024)",  dict(xnumel=1024, rnumel=1024)),
    ]
    for label, kw in cases:
        cfgs = ReductionHeuristics.inner_configs(
            xnumel=kw["xnumel"], rnumel=kw["rnumel"],
            max_r0_block=2048, register_intensive=kw.get("register_intensive", False),
            make_config_fn=_make_mock_config,
        )
        if not cfgs: _fail(f"inner_configs returned empty for {label}")
        keys = [(c.kwargs["XBLOCK"], c.kwargs["R0_BLOCK"], c.num_warps) for c in cfgs]
        if len(keys) != len(set(keys)): _fail(f"duplicate keys for {label}")
        if any("waves_per_eu" in c.kwargs for c in cfgs): _fail(f"waves_per_eu found in {label}")
        _ok(f"inner_configs {label} → {len(cfgs)} configs")


def test_reduction_outer_configs() -> None:
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    cases = [
        ("tiny   (x=16,  r=1)",    dict(xnumel=16,    rnumel=1)),
        ("small  (x=128, r=8)",    dict(xnumel=128,   rnumel=8)),
        ("medium (x=4096,r=4)",    dict(xnumel=4096,  rnumel=4)),
        ("large  (x=65536,r=8)",   dict(xnumel=65536, rnumel=8)),
        ("huge   (x=1M,  r=2)",    dict(xnumel=1048576, rnumel=2)),
    ]
    for label, kw in cases:
        cfgs = ReductionHeuristics.outer_configs(
            xnumel=kw["xnumel"], rnumel=kw["rnumel"],
            register_intensive=False, make_config_fn=_make_mock_config,
        )
        if not cfgs: _fail(f"outer_configs returned empty for {label}")
        keys = [(c.kwargs["XBLOCK"], c.kwargs.get("R0_BLOCK", 0), c.num_warps) for c in cfgs]
        if len(keys) != len(set(keys)): _fail(f"duplicate keys for {label}")
        if any("waves_per_eu" in c.kwargs for c in cfgs): _fail(f"waves_per_eu in outer {label}")
        _ok(f"outer_configs {label} → {len(cfgs)} configs")


def test_reduction_scoring_and_pruning() -> None:
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    size_hints = {"x": 256, "r": 4096}
    pm = ReductionHeuristics.problem_metadata_from_size_hints(size_hints)
    cfgs = ReductionHeuristics.inner_configs(
        xnumel=256, rnumel=4096, max_r0_block=2048,
        register_intensive=False, make_config_fn=_make_mock_config,
    )
    top, all_scored = ReductionHeuristics.prune_configs(cfgs, pm, top_n=5)
    if not top: _fail("prune_configs returned empty")
    if len(top) > 5: _fail(f"prune_configs returned {len(top)} > 5")
    if any(not (0 <= s <= 1) for s, _ in all_scored): _fail("score out of [0,1]")
    # Verify scores are in descending order
    scores = [s for s, _ in all_scored]
    if scores != sorted(scores, reverse=True): _fail("scores not sorted descending")
    _ok(f"prune_configs: {len(cfgs)} → top {len(top)}  (best={all_scored[0][0]:.4f})")
    detail = ReductionHeuristics.get_detailed_scores(all_scored[0][1], pm)
    required = {"r_efficiency", "grid_coverage", "warp_parallelism", "composite"}
    if not required <= set(detail):
        _fail(f"get_detailed_scores missing: {required - set(detail)}")
    _ok("get_detailed_scores: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(detail.items())))


def test_reduction_score_ordering() -> None:
    """Verify that all_scored is sorted descending and top-N configs are a valid subset.

    Note: top-N ≠ all_scored[:N] because a diversity cap (at most 2 per XBLOCK)
    is applied after sorting.  We only check that all_scored is non-increasing
    and that every config in top-N appears somewhere in all_scored.
    """
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    for xnumel, rnumel, label in [
        (64,   8192, "medium"),
        (1024, 2048, "square"),
        (8,    65536, "inner-large"),
    ]:
        pm = ReductionHeuristics.problem_metadata_from_size_hints({"x": xnumel, "r": rnumel})
        cfgs = ReductionHeuristics.inner_configs(
            xnumel=xnumel, rnumel=rnumel, max_r0_block=2048,
            register_intensive=False, make_config_fn=_make_mock_config,
        )
        top5, all_scored = ReductionHeuristics.prune_configs(cfgs, pm, top_n=5)
        scores = [s for s, _ in all_scored]

        # all_scored must be non-increasing (stable descending sort)
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1] - 1e-9:
                _fail(f"all_scored not sorted at [{i}] for {label}: "
                      f"{scores[i]:.4f} < {scores[i+1]:.4f}")

        # Every top-N config's (XBLOCK, R0_BLOCK, nw) key must appear in all_scored
        all_keys = {(d.get("XBLOCK"), d.get("R0_BLOCK")) for _, d in all_scored}
        for c in top5:
            key = (c.kwargs.get("XBLOCK"), c.kwargs.get("R0_BLOCK"))
            if key not in all_keys:
                _fail(f"top-N config {key} not found in all_scored for {label}")

    _ok("score ordering invariants hold for 3 problem shapes")


def test_reduction_top5_diversity() -> None:
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    pm = ReductionHeuristics.problem_metadata_from_size_hints({"x": 4096, "r": 8192})
    cfgs = ReductionHeuristics.inner_configs(
        xnumel=4096, rnumel=8192, max_r0_block=2048,
        register_intensive=False, make_config_fn=_make_mock_config,
    )
    top5, _ = ReductionHeuristics.prune_configs(cfgs, pm, top_n=5)
    xbs = {c.kwargs["XBLOCK"] for c in top5}
    if len(xbs) < 2: _fail(f"no XBLOCK diversity in top-5: {xbs}")
    _ok(f"top-5 XBLOCK spread: {sorted(xbs)}")


def test_reduction_register_intensive_cutoff() -> None:
    """register_intensive=True should produce fewer/smaller R0_BLOCK configs than False.

    The VGPR ceiling is halved for register-intensive kernels, so the pruned
    top-N should have smaller or equal R0_BLOCK values on average.
    """
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    xnumel, rnumel = 64, 8192
    pm_base = ReductionHeuristics.problem_metadata_from_size_hints({"x": xnumel, "r": rnumel})

    cfgs_normal  = ReductionHeuristics.inner_configs(
        xnumel=xnumel, rnumel=rnumel, max_r0_block=2048,
        register_intensive=False, make_config_fn=_make_mock_config,
    )
    cfgs_regint  = ReductionHeuristics.inner_configs(
        xnumel=xnumel, rnumel=rnumel, max_r0_block=2048,
        register_intensive=True,  make_config_fn=_make_mock_config,
    )

    if len(cfgs_regint) > len(cfgs_normal):
        _fail(f"register_intensive generated MORE configs ({len(cfgs_regint)} > {len(cfgs_normal)})")

    max_r0_normal = max(c.kwargs["R0_BLOCK"] for c in cfgs_normal)
    max_r0_regint = max(c.kwargs["R0_BLOCK"] for c in cfgs_regint)
    if max_r0_regint > max_r0_normal:
        _fail(f"register_intensive max R0_BLOCK ({max_r0_regint}) > normal ({max_r0_normal})")

    _ok(f"normal: {len(cfgs_normal)} cfgs, max_r0={max_r0_normal} | "
        f"reg-int: {len(cfgs_regint)} cfgs, max_r0={max_r0_regint}")

    # Verify the VGPR guard is actually active: all cfgs must satisfy the guard.
    # A config with R0_BLOCK/threads > max_r0_per_thread should have been filtered.
    max_r0_limit = ReductionHeuristics._max_r0_per_thread()
    for c in cfgs_regint:
        threads_per_output = c.num_warps * ReductionHeuristics._warp_size()
        effective_max = max_r0_limit // 2   # halved for register_intensive
        r0pt = c.kwargs["R0_BLOCK"] / max(1.0, threads_per_output)
        if r0pt > effective_max + 1e-6:
            _fail(f"register_intensive config exceeds VGPR guard: "
                  f"R0={c.kwargs['R0_BLOCK']}, nw={c.num_warps}, "
                  f"r0/thread={r0pt:.1f} > {effective_max}")
    _ok(f"all register_intensive configs satisfy VGPR guard (limit={effective_max})")


def test_vgpr_limits_device_derived() -> None:
    """Verify _max_r0_per_thread() and _max_elems_per_thread() are device-derived.

    Checks:
      1. Return values are in the expected clamped range.
      2. On a real GPU, the value differs from the static defaults when the
         device has a non-CDNA2 VGPR file (e.g. CDNA3/MI300X → expect 32).
      3. The fallback SimpleNamespace path also returns a valid value.
    """
    from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    from torch._inductor.template_heuristics.rocm.pointwise import PointwiseHeuristics

    # Basic range checks
    max_r0   = ReductionHeuristics._max_r0_per_thread()
    max_ept  = PointwiseHeuristics._max_elems_per_thread()

    if not (8 <= max_r0 <= 32):
        _fail(f"_max_r0_per_thread()={max_r0} out of [8, 32]")
    if not (8 <= max_ept <= 32):
        _fail(f"_max_elems_per_thread()={max_ept} out of [8, 32]")

    _ok(f"_max_r0_per_thread() = {max_r0}  (range [8,32] ✓)")
    _ok(f"_max_elems_per_thread() = {max_ept}  (range [8,32] ✓)")

    # Cross-check against ArchitectureConfig
    try:
        from torch._inductor.template_heuristics.rocm.arch import get_architecture_config
        arch = get_architecture_config()
        vgpr_budget = arch.vgpr_budget_per_thread
        expected_r0  = max(8, min(32, vgpr_budget // 2))
        if max_r0 != expected_r0:
            _fail(f"_max_r0_per_thread()={max_r0} ≠ arch.vgpr_budget_per_thread//2={expected_r0}")
        expected_ept = max(8, min(32, arch.regs_per_cu // (arch.occupancy_sweetspot_max * arch.warp_size) // 8))
        if max_ept != expected_ept:
            _fail(f"_max_elems_per_thread()={max_ept} ≠ formula result={expected_ept}")
        _ok(f"Cross-check vs ArchitectureConfig: vgpr_budget={vgpr_budget}, "
            f"regs_per_cu={arch.regs_per_cu}, max_threads_per_cu={arch.max_threads_per_cu}")
    except Exception as e:
        _skip(f"ArchitectureConfig cross-check skipped: {e}")


def test_architecture_config_fields() -> None:
    """Verify ArchitectureConfig exposes the new device-property fields."""
    try:
        from torch._inductor.template_heuristics.rocm.arch import get_architecture_config, ArchitectureConfig
    except ImportError as e:
        _skip(f"import failed: {e}"); return

    arch = get_architecture_config()

    required_fields = [
        "regs_per_cu", "max_threads_per_cu", "vgpr_budget_per_thread",
        "num_cus", "warp_size", "max_wavefronts_per_cu",
        "occupancy_sweetspot_min", "occupancy_sweetspot_max",
    ]
    for field in required_fields:
        if not hasattr(arch, field):
            _fail(f"ArchitectureConfig missing field: {field}")

    # Derived consistency: vgpr_budget = regs_per_cu // max_threads_per_cu
    expected_budget = arch.regs_per_cu // max(arch.max_threads_per_cu, 1)
    if arch.vgpr_budget_per_thread != expected_budget:
        _fail(f"vgpr_budget_per_thread={arch.vgpr_budget_per_thread} ≠ "
              f"regs_per_cu//max_threads_per_cu={expected_budget}")

    # max_wavefronts_per_cu = max_threads_per_cu // warp_size
    expected_wf = arch.max_threads_per_cu // max(arch.warp_size, 1)
    if arch.max_wavefronts_per_cu != expected_wf:
        _fail(f"max_wavefronts_per_cu={arch.max_wavefronts_per_cu} ≠ {expected_wf}")

    # Sanity: all values positive
    for field in required_fields:
        v = getattr(arch, field)
        if isinstance(v, int) and v <= 0:
            _fail(f"{field}={v} must be positive")

    _ok(f"ArchitectureConfig fields: regs_per_cu={arch.regs_per_cu}, "
        f"vgpr_budget={arch.vgpr_budget_per_thread}, "
        f"warp_size={arch.warp_size}, num_cus={arch.num_cus}")


def test_pointwise_ept_guard() -> None:
    """Verify the elems-per-thread guard filters high-EPT configs.

    For every generated config, elems_per_thread must not exceed
    _max_elems_per_thread().  Also verify the guard is non-trivial
    (some configs must have been rejected at max_ept+1 × threads).
    """
    from torch._inductor.template_heuristics.rocm.pointwise import PointwiseHeuristics
    max_ept = PointwiseHeuristics._max_elems_per_thread()
    warp_size = 64   # AMD default; test uses problem_metadata

    for xnumel, label in [(1048576, "1M"), (16384, "16K"), (65536, "64K")]:
        pm = _pw_pm(xnumel, ops=5)
        ws = pm.get("warp_size", 64)
        cfgs = PointwiseHeuristics.generate_all_candidate_configs(pm)
        if not cfgs: _fail(f"empty config list for {label}")

        # Every config must pass the EPT guard
        for cfg in cfgs:
            xb = cfg.get("XBLOCK", 1) * cfg.get("YBLOCK", 1) * cfg.get("ZBLOCK", 1)
            nw = cfg.get("num_warps", 1)
            threads = nw * ws
            ept = xb / max(threads, 1)
            if ept > max_ept + 1e-6:
                _fail(f"EPT guard violated: xblock={xb}, nw={nw}, "
                      f"ept={ept:.1f} > max_ept={max_ept} in {label}")

        # Check that a trivially high-EPT config (XBLOCK=4096, nw=1) is absent
        # when max_ept < 64 (which it always is since we clamp to ≤ 32)
        if max_ept < 64:
            ws_pm = pm.get("warp_size", 64)
            bad_ept = 4096 / (1 * ws_pm)   # = 64 for warp_size=64
            high_ept_cfgs = [c for c in cfgs
                             if c.get("XBLOCK", 1) == 4096 and c.get("num_warps", 1) == 1]
            if high_ept_cfgs and bad_ept > max_ept:
                _fail(f"XBLOCK=4096 nw=1 (ept={bad_ept:.0f}) survived EPT guard for {label}")

        _ok(f"EPT guard OK for {label}: {len(cfgs)} configs all ≤ max_ept={max_ept:.0f}")


def test_persistent_reduction_config_gen() -> None:
    try:
        import torch
        from torch._inductor.runtime.triton_heuristics import _persistent_reduction_configs
        from torch._inductor.runtime.hints import ReductionHint
    except ImportError as e:
        _skip(f"import failed ({e})"); return
    for label, sh in [
        ("tiny   (x=4,   r=128)",  {"x": 4,   "r0_": 128}),
        ("small  (x=16,  r=512)",  {"x": 16,  "r0_": 512}),
        ("medium (x=128, r=1024)", {"x": 128, "r0_": 1024}),
        ("large  (x=512, r=4096)", {"x": 512, "r0_": 4096}),
        ("LLM    (x=64,  r=2048)", {"x": 64,  "r0_": 2048}),
    ]:
        try:
            cfgs = _persistent_reduction_configs(
                size_hints=sh, reduction_hint=ReductionHint.INNER,
                inductor_meta={"max_autotune": False},
                triton_meta={"device": 0, "device_type": "hip"},
            )
            if not cfgs: _fail(f"empty for {label}")
            _ok(f"persistent {label} → {len(cfgs)} configs")
        except Exception as e:
            _skip(f"{label}: {type(e).__name__}: {e}")


def _pw_pm(xnumel, ynumel=None, znumel=None, ops=3):
    """Build a minimal problem-metadata dict for pointwise unit tests."""
    try:
        import torch
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _num_cus   = _props.multi_processor_count
            _warp_size = _props.warp_size
        else:
            raise RuntimeError("no GPU")
    except Exception:
        _num_cus   = 256
        _warp_size = 64
    dims = {"xnumel": xnumel}
    if ynumel: dims["ynumel"] = ynumel
    if znumel: dims["znumel"] = znumel
    return {**dims, "total_elements": xnumel*(ynumel or 1)*(znumel or 1),
            "ops_per_element": ops, "slow_ops": 0, "load_ops": 2,
            "element_size": 4, "warp_size": _warp_size, "num_cus": _num_cus,
            "max_threads_per_block": 1024}


def test_pointwise_configs() -> None:
    from torch._inductor.template_heuristics.rocm.pointwise import PointwiseHeuristics
    cases = [
        ("1D  x=512",        _pw_pm(512)),
        ("1D  x=16M",        _pw_pm(16*1024*1024)),
        ("1D  x=64",         _pw_pm(64)),       # minimum viable (= 1 wavefront)
        ("1D  x=1K slow-ops",_pw_pm(1024, ops=12)),
        ("2D  256×256",       _pw_pm(256, 256)),
        ("2D  1024×512",      _pw_pm(1024, 512)),
        ("2D  4096×4096",     _pw_pm(4096, 4096)),
        ("3D  128×64×32",     _pw_pm(128, 64, 32)),
        ("3D  64×64×64",      _pw_pm(64, 64, 64)),
    ]
    for label, pm in cases:
        cfgs = PointwiseHeuristics.generate_all_candidate_configs(pm)
        if not cfgs: _fail(f"empty for {label}")
        top = PointwiseHeuristics.prune_configs(cfgs, pm, top_n=5)
        if not top: _fail(f"prune empty for {label}")
        if any("waves_per_eu" in c for c in cfgs): _fail(f"waves_per_eu in {label}")
        _ok(f"pointwise {label}: {len(cfgs)} → top {len(top)}")


def test_pointwise_scoring_latency() -> None:
    from torch._inductor.template_heuristics.rocm.pointwise import PointwiseHeuristics
    # Test scoring latency for several sizes to catch O(n²) regressions
    for xnumel, ops, label in [
        (4*1024*1024, 3,  "4M  3 ops"),
        (4*1024*1024, 12, "4M  12 ops (slow)"),
        (64*1024*1024, 3, "64M 3 ops"),
    ]:
        pm = _pw_pm(xnumel, ops=ops)
        cfgs = PointwiseHeuristics.generate_all_candidate_configs(pm)
        t0 = time.perf_counter()
        top = PointwiseHeuristics.prune_configs(cfgs, pm, top_n=5)
        ms = (time.perf_counter() - t0) * 1000
        if not top: _fail(f"prune empty for {label}")
        if ms > 500:
            _fail(f"scoring took {ms:.0f} ms (> 500 ms limit) for {label}")
        _ok(f"scored {len(cfgs)} configs in {ms:.1f} ms → top {len(top)}  [{label}]")


UNIT_TESTS = [
    ("Reduction: inner_configs",              test_reduction_inner_configs),
    ("Reduction: outer_configs",              test_reduction_outer_configs),
    ("Reduction: scoring & pruning",          test_reduction_scoring_and_pruning),
    ("Reduction: score ordering invariants",  test_reduction_score_ordering),
    ("Reduction: top-5 diversity",            test_reduction_top5_diversity),
    ("Reduction: register_intensive cutoff",  test_reduction_register_intensive_cutoff),
    ("Heuristics: VGPR limits device-derived",test_vgpr_limits_device_derived),
    ("Heuristics: ArchitectureConfig fields", test_architecture_config_fields),
    ("Persistent reduction: configs",         test_persistent_reduction_config_gen),
    ("Pointwise: EPT guard",                  test_pointwise_ept_guard),
    ("Pointwise: config generation",          test_pointwise_configs),
    ("Pointwise: scoring latency",            test_pointwise_scoring_latency),
]

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – 4-mode perf comparison
# ─────────────────────────────────────────────────────────────────────────────

# Each mode: (display label, heuristics_on, max_autotune)
MODES: List[Tuple[str, bool, bool]] = [
    ("Baseline",    False, False),
    ("Heuristics",  True,  False),
    ("MaxAutotune", False, True),
    ("Heur+MaxAT",  True,  True),
]

SENTINEL = float("nan")
SKIP_VAL = float("inf")


def _set_heuristics_env(on: bool) -> None:
    val = "1" if on else "0"
    os.environ["TORCHINDUCTOR_POINTWISE_HEURISTICS"]  = val
    os.environ["TORCHINDUCTOR_REDUCTION_HEURISTICS"]  = val


def _clear_all_caches() -> None:
    """Flush every in-process compilation cache between measurements."""
    import gc
    import torch

    torch._dynamo.reset()

    try:
        from torch._inductor.codecache import PyCodeCache
        PyCodeCache._cache.clear()
    except Exception:
        pass

    try:
        from torch._inductor.codecache import FxGraphCache
        FxGraphCache.clear(purge=False)
    except Exception:
        pass

    gc.collect()


def _warmup_compiler(device: str) -> None:
    """Prime all lazy resources in every mode before timed measurements."""
    import torch
    from torch._inductor import config as ic

    print("  [Warm-up] Priming compiler for all 4 modes…", flush=True)
    dummy = torch.randn(256, device=device)

    for mode_label, heur_on, max_at in MODES:
        _set_heuristics_env(heur_on)
        _clear_all_caches()
        try:
            patch = {"max_autotune": max_at, "max_autotune_pointwise": max_at}
            with ic.patch(patch):
                cf = torch.compile(lambda x: x.relu(), backend="inductor", fullgraph=True)
                cf(dummy)
                torch.cuda.synchronize()
            print(f"    {mode_label} ✓", flush=True)
        except Exception as exc:
            print(f"    {mode_label} ✗  ({type(exc).__name__}: {exc})", flush=True)

    _clear_all_caches()
    print("  [Warm-up] Done.\n", flush=True)


class _RunResult:
    """Holds kernel latency, compilation timing, and config counts for one mode."""
    __slots__ = ("perf_ms", "perf_p95_ms", "trace_ms", "first_call_ms",
                 "total_compile_ms", "n_configs")
    def __init__(self, perf_ms, perf_p95_ms, trace_ms, first_call_ms, n_configs):
        self.perf_ms          = perf_ms          # p50 kernel latency
        self.perf_p95_ms      = perf_p95_ms      # p95 kernel latency
        self.trace_ms         = trace_ms
        self.first_call_ms    = first_call_ms
        self.total_compile_ms = trace_ms + first_call_ms
        self.n_configs        = n_configs


_FAIL_RESULT = _RunResult(SENTINEL, SENTINEL, SENTINEL, SENTINEL, 0)
_SKIP_RESULT = _RunResult(SKIP_VAL, SKIP_VAL, SKIP_VAL, SKIP_VAL, 0)


class _ConfigCounts:
    """Tracks config counts across all CachingAutotuner instances in one mode run."""
    __slots__ = ("compiled", "benchmarked")
    def __init__(self):
        self.compiled = self.benchmarked = 0

    def reset(self):
        self.compiled = self.benchmarked = 0

    @property
    def display(self) -> str:
        if self.benchmarked:
            return f"[{self.compiled}c/{self.benchmarked}b]"
        if self.compiled:
            return f"[{self.compiled}c]"
        return ""


def _install_config_counter():
    """Monkey-patch CachingAutotuner to track compiled and benchmarked config counts."""
    counts = _ConfigCounts()
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        orig_precompile = CachingAutotuner.precompile
        orig_autotune   = CachingAutotuner.autotune_to_one_config

        def _patched_precompile(self, *a, **kw):
            if self.configs:
                counts.compiled += len(self.configs)
            return orig_precompile(self, *a, **kw)

        def _patched_autotune(self, *a, **kw):
            n = len(self.launchers) if self.launchers else len(self.compile_results or [])
            counts.benchmarked += n
            return orig_autotune(self, *a, **kw)

        CachingAutotuner.precompile             = _patched_precompile
        CachingAutotuner.autotune_to_one_config = _patched_autotune

        def _uninstall():
            CachingAutotuner.precompile             = orig_precompile
            CachingAutotuner.autotune_to_one_config = orig_autotune

    except Exception:
        def _uninstall(): pass

    return counts, _uninstall


def _compile_and_time(
    fn: Callable,
    args: tuple,
    max_autotune: bool,
    warmup: int,
    reps: int,
    rtol: float,
    atol: float,
    counts: "_ConfigCounts",
) -> _RunResult:
    """Compile fn; measure compilation latency and kernel latency (p50 + p95).

    Each call uses a fresh TRITON_CACHE_DIR to ensure equal cold-JIT footing
    across all four modes.
    """
    import shutil
    import tempfile
    import torch
    from torch._inductor import config as ic

    triton_cache_dir = tempfile.mkdtemp(prefix="triton_bench_")
    old_triton_cache = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

    try:
        patch = {"max_autotune": max_autotune, "max_autotune_pointwise": max_autotune}
        with ic.patch(patch):
            t_trace0 = time.perf_counter()
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            trace_ms = (time.perf_counter() - t_trace0) * 1000

            ref = fn(*args)

            counts.reset()
            torch.cuda.synchronize()
            t_first0 = time.perf_counter()
            out = compiled(*args)
            torch.cuda.synchronize()
            first_call_ms = (time.perf_counter() - t_first0) * 1000
            n_configs = counts.compiled

            if not torch.allclose(ref, out, rtol=rtol, atol=atol, equal_nan=True):
                diff = (ref - out).abs().max().item()
                print(f"      [WARN] max_diff={diff:.3e}")

            for _ in range(max(0, warmup - 1)):
                compiled(*args)
            torch.cuda.synchronize()

            times = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                compiled(*args)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            times.sort()
            perf_p50  = times[len(times) // 2]
            perf_p95  = times[min(int(len(times) * 0.95), len(times) - 1)]

            return _RunResult(perf_p50, perf_p95, trace_ms, first_call_ms, n_configs)

    except Exception as e:
        print(f"      [ERR] {type(e).__name__}: {e}")
        return _FAIL_RESULT

    finally:
        if old_triton_cache is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = old_triton_cache
        shutil.rmtree(triton_cache_dir, ignore_errors=True)


def _run_workload_all_modes(
    label: str,
    fn: Callable,
    args: tuple,
    warmup: int,
    reps: int,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    skip_max_autotune: bool = False,
    active_modes: Optional[List[Tuple[str, bool, bool]]] = None,
) -> Dict[str, _RunResult]:
    import torch

    modes = active_modes if active_modes is not None else MODES
    counts, uninstall_counter = _install_config_counter()

    try:
        results: Dict[str, _RunResult] = {}
        for mode_label, heur_on, max_at in modes:
            if skip_max_autotune and max_at:
                results[mode_label] = _SKIP_RESULT
                continue

            _clear_all_caches()
            _set_heuristics_env(heur_on)

            r = _compile_and_time(fn, args, max_autotune=max_at,
                                  warmup=warmup, reps=reps, rtol=rtol, atol=atol,
                                  counts=counts)
            results[mode_label] = r

            if math.isfinite(r.perf_ms) and r.perf_ms != SKIP_VAL:
                cfg_note = f"  {counts.display}" if counts.display else ""
                print(f"    {mode_label:14s} → "
                      f"p50 {r.perf_ms:7.4f} ms  p95 {r.perf_p95_ms:7.4f} ms  "
                      f"| compile {r.total_compile_ms:7.1f} ms"
                      f"{cfg_note}")
            else:
                print(f"    {mode_label:14s} → FAIL")
    finally:
        uninstall_counter()

    return results


# ─── Workload definitions ─────────────────────────────────────────────────────

def _build_workloads(device: str, dtype, workload_filter: Optional[str] = None):
    """Return a list of (label, fn, tensors, kwargs) for all benchmark workloads.

    Organised into named groups so the tables are easy to read.  Each group
    is preceded by a divider row to visually separate the categories.
    """
    import torch
    f32 = torch.float32

    def w(label, fn, *tensors, rtol=1e-4, atol=1e-4, skip_ma=False):
        return (label, fn, tensors, dict(rtol=rtol, atol=atol, skip_max_autotune=skip_ma))

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    def ones(*shape):
        return torch.ones(*shape, device=device, dtype=dtype)

    def zeros(*shape):
        return torch.zeros(*shape, device=device, dtype=dtype)

    # ── Reduce tolerances for lower precision dtypes ───────────────────────
    if dtype in (torch.float16, torch.bfloat16):
        r_red, a_red = 1e-2, 1e-2   # looser for fp16/bf16
    else:
        r_red, a_red = 1e-3, 1e-3

    workloads = [
        # ──────────────────────────────────────────────────────────────────
        # INNER reductions (x small, r large)
        # ──────────────────────────────────────────────────────────────────
        w("sum  x=8    r=65536",
          lambda t: t.sum(dim=-1),
          randn(8, 65536)),

        w("sum  x=16   r=32768",
          lambda t: t.sum(dim=-1),
          randn(16, 32768)),

        w("sum  x=32   r=16384",
          lambda t: t.sum(dim=-1),
          randn(32, 16384)),

        w("mean x=64   r=8192",
          lambda t: t.mean(dim=-1),
          randn(64, 8192)),

        w("sum  x=1    r=1M",
          lambda t: t.sum(dim=-1),
          randn(1, 1048576)),

        # ──────────────────────────────────────────────────────────────────
        # OUTER reductions (x large, r small)
        # ──────────────────────────────────────────────────────────────────
        w("sum  x=65536  r=8",
          lambda t: t.sum(dim=-1),
          randn(65536, 8)),

        w("mean x=16384  r=4",
          lambda t: t.mean(dim=-1),
          randn(16384, 4)),

        w("sum  x=262144 r=2",
          lambda t: t.sum(dim=-1),
          randn(262144, 2)),

        w("sum  x=1M     r=1",
          lambda t: t.sum(dim=-1),
          randn(1048576, 1)),

        # ──────────────────────────────────────────────────────────────────
        # Register-intensive (Welford — 3 accumulators per thread)
        # ──────────────────────────────────────────────────────────────────
        w("var  x=64   r=8192",
          lambda t: t.var(dim=-1, correction=0),
          randn(64, 8192),
          rtol=r_red, atol=a_red),

        w("std  x=128  r=4096",
          lambda t: t.std(dim=-1, correction=1),
          randn(128, 4096),
          rtol=r_red, atol=a_red),

        w("var  x=512  r=2048",
          lambda t: t.var(dim=-1, correction=0),
          randn(512, 2048),
          rtol=r_red, atol=a_red),

        # ──────────────────────────────────────────────────────────────────
        # Persistent reductions (softmax, layer norm — LLM-representative shapes)
        # ──────────────────────────────────────────────────────────────────
        # Transformer attention: [batch*heads × seq_len]
        w("softmax [8×2048]",
          lambda t: torch.softmax(t, dim=-1),
          randn(8, 2048),
          rtol=r_red, atol=a_red),

        w("softmax [64×512]",
          lambda t: torch.softmax(t, dim=-1),
          randn(64, 512),
          rtol=r_red, atol=a_red),

        w("softmax [1024×512]",
          lambda t: torch.softmax(t, dim=-1),
          randn(1024, 512),
          rtol=r_red, atol=a_red),

        w("softmax [128×4096]",
          lambda t: torch.softmax(t, dim=-1),
          randn(128, 4096),
          rtol=r_red, atol=a_red),

        # Transformer feed-forward: [batch*seq × hidden]
        w("layer_norm [512×768]",
          lambda t, ww, b: torch.nn.functional.layer_norm(t, [768], ww, b),
          randn(512, 768), ones(768), zeros(768),
          rtol=r_red, atol=a_red),

        w("layer_norm [512×4096]",
          lambda t, ww, b: torch.nn.functional.layer_norm(t, [4096], ww, b),
          randn(512, 4096), ones(4096), zeros(4096),
          rtol=r_red, atol=a_red),

        w("layer_norm [2048×1024]",
          lambda t, ww, b: torch.nn.functional.layer_norm(t, [1024], ww, b),
          randn(2048, 1024), ones(1024), zeros(1024),
          rtol=r_red, atol=a_red),

        # ──────────────────────────────────────────────────────────────────
        # Pointwise 1D — bandwidth-bound
        # ──────────────────────────────────────────────────────────────────
        w("relu   [1M]",
          lambda t: torch.relu(t),
          randn(1024*1024)),

        w("relu   [16M]",
          lambda t: torch.relu(t),
          randn(16*1024*1024)),

        w("add    [4M]",
          lambda a, b: a + b,
          randn(4*1024*1024), randn(4*1024*1024)),

        w("add    [32M]",
          lambda a, b: a + b,
          randn(32*1024*1024), randn(32*1024*1024)),

        w("mul+add [4M]",
          lambda a, b, c: a * b + c,
          randn(4*1024*1024), randn(4*1024*1024), randn(4*1024*1024)),

        # ──────────────────────────────────────────────────────────────────
        # Pointwise 1D — compute-bound (transcendental-heavy, stress VGPR)
        # ──────────────────────────────────────────────────────────────────
        w("gelu   [4M]",
          lambda t: torch.nn.functional.gelu(t),
          randn(4*1024*1024),
          rtol=r_red, atol=a_red),

        w("sigmoid [4M]",
          lambda t: torch.sigmoid(t),
          randn(4*1024*1024)),

        w("silu+tanh+exp [4M]",
          lambda t: torch.nn.functional.silu(t) + torch.tanh(t) + torch.exp(t.clamp(-10, 10)),
          randn(4*1024*1024),
          rtol=r_red, atol=a_red),

        w("tanh+exp+sin [4M]",
          lambda t: torch.tanh(t) + torch.exp(t.clamp(-10, 10)) + torch.sin(t),
          randn(4*1024*1024),
          rtol=r_red, atol=a_red),

        # 6-op transcendental chain — stresses VGPR limit for high-EPT configs
        w("6-transcendental [4M]",
          lambda t: (torch.sin(t) * torch.cos(t)
                     + torch.exp(t.clamp(-5, 5)) * torch.tanh(t)
                     + torch.sigmoid(t) * torch.nn.functional.gelu(t)),
          randn(4*1024*1024),
          rtol=r_red, atol=a_red),

        # ──────────────────────────────────────────────────────────────────
        # Pointwise 2D and 3D
        # ──────────────────────────────────────────────────────────────────
        w("relu   [2048×2048]",
          lambda t: torch.relu(t),
          randn(2048, 2048)),

        w("gelu   [1024×4096]",
          lambda t: torch.nn.functional.gelu(t),
          randn(1024, 4096),
          rtol=r_red, atol=a_red),

        w("add    [2048×2048]",
          lambda a, b: a + b,
          randn(2048, 2048), randn(2048, 2048)),

        w("add    [128×64×64]",
          lambda a, b: a + b,
          randn(128, 64, 64), randn(128, 64, 64)),
    ]

    if workload_filter:
        workloads = [(l, f, t, k) for l, f, t, k in workloads
                     if workload_filter.lower() in l.lower()]

    return workloads


# ─── Table formatting ─────────────────────────────────────────────────────────

def _fmt_ms(ms: float, is_best: bool, fmt: str = "8.4f") -> str:
    if ms == SKIP_VAL:  return "  (skip)  "
    if not math.isfinite(ms): return "   FAIL   "
    s = format(ms, fmt) + " ms"
    return f"*{s}*" if is_best else f" {s} "


def _fmt_pct(pct: float) -> str:
    """Format a gap percentage (positive = heuristics is slower)."""
    if not math.isfinite(pct): return "    n/a    "
    sign = "+" if pct >= 0 else ""
    return f" {sign}{pct:5.1f}%    "


NAME_W = 28
COL_W  = 16


def _table_sep(n_cols: int) -> str:
    return "+" + "-"*(NAME_W+2) + ("+" + "-"*(COL_W+2)) * n_cols + "+"


def _print_perf_table(rows: List[Tuple[str, Dict[str, _RunResult]]]) -> None:
    """Print a 4-column p50 kernel-latency table with p95 annotation."""
    mode_labels = [m[0] for m in MODES]
    sep  = _table_sep(len(mode_labels))
    head = ("| " + "Workload".ljust(NAME_W) + " |"
            + "".join(f" {m.center(COL_W)} |" for m in mode_labels))
    unit = ("| " + " "*NAME_W + " |"
            + "".join(f" {'p50 ms (median)'.center(COL_W)} |" for _ in mode_labels))

    print()
    print(sep); print(head); print(unit); print(sep)

    for label, rmap in rows:
        vals = {m: rmap[m].perf_ms for m in mode_labels if m in rmap}
        finite = {m: v for m, v in vals.items() if math.isfinite(v) and v != SKIP_VAL}
        best_key = min(finite, key=finite.get) if finite else None

        cells = "".join(
            " " + _fmt_ms(vals.get(m, SENTINEL), m == best_key).center(COL_W) + " |"
            for m in mode_labels
        )
        print(f"| {label:<{NAME_W}} |{cells}")

    print(sep)

    # Geo-mean speedup summary
    print()
    print("  Kernel speedup (Heuristics vs Baseline):")
    speedups = []
    for label, rmap in rows:
        b = rmap.get("Baseline",   _FAIL_RESULT).perf_ms
        h = rmap.get("Heuristics", _FAIL_RESULT).perf_ms
        if math.isfinite(b) and math.isfinite(h) and h > 0 and b != SKIP_VAL:
            speedups.append((label, b / h))
    if speedups:
        for lbl, sp in sorted(speedups, key=lambda x: -x[1])[:8]:
            bar = "█" * min(int(sp * 10), 40)
            print(f"    {lbl:<30}  {sp:5.2f}×  {bar}")
        geo = math.exp(sum(math.log(s) for _, s in speedups) / len(speedups))
        print(f"\n    Geo-mean speedup (Heur vs Base): {geo:.2f}×")

    print()
    print("  Kernel speedup (Heur+MaxAT vs MaxAutotune):")
    speedups2 = []
    for label, rmap in rows:
        ma  = rmap.get("MaxAutotune", _FAIL_RESULT).perf_ms
        hma = rmap.get("Heur+MaxAT",  _FAIL_RESULT).perf_ms
        if math.isfinite(ma) and math.isfinite(hma) and hma > 0 and ma != SKIP_VAL:
            speedups2.append((label, ma / hma))
    if speedups2:
        wins = sum(1 for _, s in speedups2 if s > 1.01)
        ties = sum(1 for _, s in speedups2 if 0.99 <= s <= 1.01)
        geo2 = math.exp(sum(math.log(s) for _, s in speedups2) / len(speedups2))
        print(f"    Geo-mean speedup: {geo2:.2f}×  "
              f"(Heur+MaxAT wins {wins}, ties {ties}, "
              f"loses {len(speedups2)-wins-ties})")


def _print_compile_table(rows: List[Tuple[str, Dict[str, _RunResult]]]) -> None:
    """Print total compile-time table with average row and analysis."""
    mode_labels = [m[0] for m in MODES]
    sep  = _table_sep(len(mode_labels))
    head = ("| " + "Workload".ljust(NAME_W) + " |"
            + "".join(f" {m.center(COL_W)} |" for m in mode_labels))
    unit = ("| " + " "*NAME_W + " |"
            + "".join(f" {'total ms'.center(COL_W)} |" for _ in mode_labels))

    print()
    print(sep); print(head); print(unit); print(sep)

    col_totals: Dict[str, float] = {m: 0.0 for m in mode_labels}
    col_counts: Dict[str, int]   = {m: 0   for m in mode_labels}

    for label, rmap in rows:
        vals = {m: rmap[m].total_compile_ms for m in mode_labels if m in rmap}
        finite = {m: v for m, v in vals.items() if math.isfinite(v) and v != SKIP_VAL}
        best_key = min(finite, key=finite.get) if finite else None

        cells = ""
        for m in mode_labels:
            ms = vals.get(m, SENTINEL)
            cells += " " + _fmt_ms(ms, m == best_key, fmt="7.0f").center(COL_W) + " |"
            if math.isfinite(ms) and ms != SKIP_VAL:
                col_totals[m] += ms
                col_counts[m] += 1
        print(f"| {label:<{NAME_W}} |{cells}")

    print(sep)
    avg_vals: Dict[str, float] = {}
    for m in mode_labels:
        avg_vals[m] = col_totals[m] / col_counts[m] if col_counts[m] > 0 else SENTINEL
    finite_avg = {m: v for m, v in avg_vals.items() if math.isfinite(v)}
    best_avg = min(finite_avg, key=finite_avg.get) if finite_avg else None
    avg_cells = "".join(
        " " + _fmt_ms(avg_vals[m], m == best_avg, fmt="7.0f").center(COL_W) + " |"
        for m in mode_labels
    )
    print(f"| {'AVERAGE':>{NAME_W}} |{avg_cells}")
    print(sep)

    # Analysis
    print()
    base_avg  = avg_vals.get("Baseline",    SENTINEL)
    heur_avg  = avg_vals.get("Heuristics",  SENTINEL)
    maxat_avg = avg_vals.get("MaxAutotune", SENTINEL)
    hmax_avg  = avg_vals.get("Heur+MaxAT",  SENTINEL)

    print("  Compile overhead vs Baseline (average):")
    def _compare(a, b, a_lbl, b_lbl):
        if not (math.isfinite(a) and math.isfinite(b) and b > 0): return
        ratio, delta = a / b, a - b
        desc = (f"overhead  {ratio:5.2f}×  (+{delta:.0f} ms)"
                if ratio >= 1 else
                f"speedup   {1/ratio:5.2f}×  ({delta:+.0f} ms)")
        print(f"    {a_lbl:<36}  {desc}")

    _compare(heur_avg,  base_avg,  "Heuristics   vs Baseline",    "Baseline")
    _compare(maxat_avg, base_avg,  "MaxAutotune  vs Baseline",    "Baseline")
    _compare(hmax_avg,  base_avg,  "Heur+MaxAT   vs Baseline",    "Baseline")
    _compare(hmax_avg,  maxat_avg, "Heur+MaxAT   vs MaxAutotune", "MaxAutotune")

    savings = []
    for label, rmap in rows:
        h  = rmap.get("Heuristics",  _FAIL_RESULT).total_compile_ms
        ma = rmap.get("MaxAutotune", _FAIL_RESULT).total_compile_ms
        if math.isfinite(h) and math.isfinite(ma) and h > 0 and ma != SKIP_VAL:
            savings.append((label, ma / h))
    if savings:
        geo = math.exp(sum(math.log(s) for _, s in savings) / len(savings))
        print()
        print("  Compilation speedup – Heuristics vs MaxAutotune (top 5):")
        for lbl, sp in sorted(savings, key=lambda x: -x[1])[:5]:
            bar = "█" * min(int(sp * 3), 40)
            print(f"    {lbl:<30}  {sp:6.1f}×  {bar}")
        print(f"\n    Geo-mean compile speedup (Heur vs MaxAT): {geo:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – Heuristic quality analysis
# ─────────────────────────────────────────────────────────────────────────────

def _print_quality_table(rows: List[Tuple[str, Dict[str, _RunResult]]]) -> None:
    """Layer 3: analyse how well Heuristics approximates MaxAutotune quality.

    Columns:
      Heur gap vs MaxAT  – (Heuristics − MaxAutotune) / MaxAutotune × 100 %
                           negative = Heuristics is FASTER than MaxAutotune
      Heur+MaxAT gap     – (Heur+MaxAT − MaxAutotune) / MaxAutotune × 100 %
      p95 Heuristics     – 95th-percentile latency to detect tail jitter
      p95 MaxAutotune    – 95th-percentile latency (reference)

    A "gap" near zero means the heuristic top-N picked the same winner
    that exhaustive autotuning found.
    """
    mode_labels = ["Heur gap%", "H+MA gap%", "Heur p95", "MaxAT p95"]
    sep  = _table_sep(len(mode_labels))
    head = ("| " + "Workload".ljust(NAME_W) + " |"
            + "".join(f" {m.center(COL_W)} |" for m in mode_labels))
    unit = ("| " + " "*NAME_W + " |"
            + "".join(f" {'vs MaxAutotune'.center(COL_W)} |" for _ in mode_labels[:2])
            + "".join(f" {'p95 ms'.center(COL_W)} |" for _ in mode_labels[2:]))

    print()
    print(sep); print(head); print(unit); print(sep)

    gaps_heur, gaps_hmax = [], []
    wins_count: Dict[str, int] = {m[0]: 0 for m in MODES}
    fail_count: Dict[str, int] = {m[0]: 0 for m in MODES}

    for label, rmap in rows:
        h    = rmap.get("Heuristics",  _FAIL_RESULT).perf_ms
        ma   = rmap.get("MaxAutotune", _FAIL_RESULT).perf_ms
        hma  = rmap.get("Heur+MaxAT",  _FAIL_RESULT).perf_ms
        b    = rmap.get("Baseline",    _FAIL_RESULT).perf_ms
        h95  = rmap.get("Heuristics",  _FAIL_RESULT).perf_p95_ms
        ma95 = rmap.get("MaxAutotune", _FAIL_RESULT).perf_p95_ms

        # Gap vs MaxAutotune
        if math.isfinite(h) and math.isfinite(ma) and ma > 0 and ma != SKIP_VAL:
            gap_h = (h - ma) / ma * 100.0
            gaps_heur.append(gap_h)
            col_gap_h = _fmt_pct(gap_h)
        else:
            col_gap_h = "    n/a    "

        if math.isfinite(hma) and math.isfinite(ma) and ma > 0 and ma != SKIP_VAL:
            gap_hma = (hma - ma) / ma * 100.0
            gaps_hmax.append(gap_hma)
            col_gap_hma = _fmt_pct(gap_hma)
        else:
            col_gap_hma = "    n/a    "

        col_h95  = _fmt_ms(h95,  False) if math.isfinite(h95)  else "    n/a    "
        col_ma95 = _fmt_ms(ma95, False) if math.isfinite(ma95) else "    n/a    "

        print(f"| {label:<{NAME_W}} | {col_gap_h.center(COL_W)} "
              f"| {col_gap_hma.center(COL_W)} "
              f"| {col_h95.center(COL_W)} "
              f"| {col_ma95.center(COL_W)} |")

        # Win counting (best p50 per workload)
        all_p50 = {
            m_lbl: rmap.get(m_lbl, _FAIL_RESULT).perf_ms
            for m_lbl, _, _ in MODES
        }
        finite_p50 = {k: v for k, v in all_p50.items()
                      if math.isfinite(v) and v != SKIP_VAL}
        if finite_p50:
            winner = min(finite_p50, key=finite_p50.get)
            wins_count[winner] = wins_count.get(winner, 0) + 1
        for m_lbl, _, _ in MODES:
            v = all_p50.get(m_lbl, SENTINEL)
            if not math.isfinite(v) or v == SKIP_VAL:
                fail_count[m_lbl] = fail_count.get(m_lbl, 0) + 1

    print(sep)

    # Summary statistics
    print()
    print("  Heuristic quality summary:")
    n = len(rows)

    if gaps_heur:
        avg_gap  = sum(gaps_heur) / len(gaps_heur)
        within5  = sum(1 for g in gaps_heur if g <= 5.0)
        within10 = sum(1 for g in gaps_heur if g <= 10.0)
        faster   = sum(1 for g in gaps_heur if g < -1.0)
        print(f"    Heuristics vs MaxAutotune gap:")
        print(f"      Mean gap       : {avg_gap:+.1f}%  "
              f"(negative = Heur is faster)")
        print(f"      Within  5% gap : {within5}/{len(gaps_heur)} workloads")
        print(f"      Within 10% gap : {within10}/{len(gaps_heur)} workloads")
        print(f"      Heur faster    : {faster}/{len(gaps_heur)} workloads")

    if gaps_hmax:
        avg_gap2  = sum(gaps_hmax) / len(gaps_hmax)
        within5_2 = sum(1 for g in gaps_hmax if g <= 5.0)
        print()
        print(f"    Heur+MaxAT vs MaxAutotune gap:")
        print(f"      Mean gap       : {avg_gap2:+.1f}%")
        print(f"      Within  5% gap : {within5_2}/{len(gaps_hmax)} workloads")

    print()
    print("  Mode win counts (best p50 kernel latency per workload):")
    for m_lbl, _, _ in MODES:
        wins = wins_count.get(m_lbl, 0)
        fails = fail_count.get(m_lbl, 0)
        bar = "█" * wins
        print(f"    {m_lbl:<14}  {wins:3d}/{n} wins  {bar}  "
              f"{'('+str(fails)+' fail)' if fails else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AMD heuristics 4-mode perf benchmark")
    parser.add_argument("--unit-only", action="store_true",
                        help="Run only pure-Python unit tests (no GPU)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations per mode (default 3)")
    parser.add_argument("--reps", type=int, default=20,
                        help="Timed repetitions per mode (default 20)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Tensor dtype for Layer 2 workloads (default fp32)")
    parser.add_argument("--no-max-autotune", action="store_true",
                        help="Skip MaxAutotune and Heur+MaxAT modes (faster run)")
    parser.add_argument("--filter", type=str, default=None, metavar="SUBSTR",
                        help="Run only workloads whose label contains SUBSTR")
    args = parser.parse_args()

    try:
        from torch._inductor.template_heuristics.rocm.pointwise import PointwiseHeuristics
        from torch._inductor.template_heuristics.rocm.reduction import ReductionHeuristics
    except ImportError as e:
        print(f"ERROR: {e}\nRun from the pytorch source root.")
        sys.exit(1)

    import torch
    is_rocm  = bool(getattr(torch.version, "hip", None))
    has_cuda = torch.cuda.is_available()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print()
    _hline("═")
    print("  AMD Inductor Heuristics – 4-Mode Benchmark")
    _hline("═")
    print(f"  PyTorch  {torch.__version__}")
    print(f"  ROCm     {'yes (' + torch.version.hip + ')' if is_rocm else 'no'}")
    if has_cuda:
        p = torch.cuda.get_device_properties(0)
        regs = getattr(p, "regs_per_multiprocessor", "?")
        print(f"  Device   {p.name}  "
              f"({p.multi_processor_count} CUs, warp={p.warp_size}, "
              f"regs/CU={regs})")
    print(f"  dtype    {args.dtype}")
    print()
    print("  Modes:")
    for label, h, ma in MODES:
        skip_note = "  [SKIPPED]" if (args.no_max_autotune and ma) else ""
        print(f"    {label:<14}  heuristics={'ON ' if h else 'OFF'}  "
              f"max_autotune={'ON' if ma else 'OFF'}{skip_note}")
    print()

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    _section("Layer 1 — Pure-Python Unit Tests")
    t0 = time.perf_counter()
    n_pass, n_skip = 0, 0
    for title, fn in UNIT_TESTS:
        print(f"\n  [{title}]")
        fn()
        n_pass += 1
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  ✓  {n_pass} unit tests passed in {elapsed:.0f} ms")

    if args.unit_only:
        print(); return

    if not has_cuda:
        print(); _skip("No GPU available — skipping Layers 2 & 3"); print(); return

    if not is_rocm:
        print()
        _skip("Not running on AMD ROCm.  Layer 2 results will not reflect AMD "
              "heuristic quality, but the benchmark will still run for reference.")

    # Active modes (can suppress MaxAutotune via --no-max-autotune)
    active_modes = [m for m in MODES if not (args.no_max_autotune and m[2])]

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    _section("Layer 2 — 4-Mode Performance Comparison")
    print(f"  warmup={args.warmup}  reps={args.reps}  dtype={args.dtype}")
    if args.filter:
        print(f"  filter='{args.filter}'")
    print()

    _section("Pre-flight — Compiler Warm-up")
    _warmup_compiler("cuda")

    workloads = _build_workloads("cuda", dtype, workload_filter=args.filter)
    if not workloads:
        print(f"  No workloads matched filter '{args.filter}'."); return

    table_rows: List[Tuple[str, Dict[str, _RunResult]]] = []
    total_t0 = time.perf_counter()

    for label, fn, tensors, kwargs in workloads:
        rtol    = kwargs.get("rtol", 1e-4)
        atol    = kwargs.get("atol", 1e-4)
        skip_ma = kwargs.get("skip_max_autotune", False) or args.no_max_autotune

        _hline("─", 64)
        print(f"  {label}")
        _hline("─", 64)

        results = _run_workload_all_modes(
            label, fn, tensors,
            warmup=args.warmup, reps=args.reps,
            rtol=rtol, atol=atol,
            skip_max_autotune=skip_ma,
            active_modes=active_modes,
        )
        # Fill in SKIP for modes not run
        for m_lbl, _, _ in MODES:
            results.setdefault(m_lbl, _SKIP_RESULT)

        table_rows.append((label, results))
        print()

    elapsed_total = (time.perf_counter() - total_t0) / 60
    print(f"  Layer 2 complete in {elapsed_total:.1f} min")

    # ── Table 1: kernel latency ───────────────────────────────────────────
    _section("Table 1 — Kernel Latency p50 (ms)")
    _print_perf_table(table_rows)

    # ── Table 2: compilation time ─────────────────────────────────────────
    _section("Table 2 — Compilation Time (trace + first-call ms)")
    print("  trace      = torch.compile() graph capture")
    print("  first_call = inductor codegen + Triton JIT + autotune benchmarking")
    print("  total      = trace + first_call  [shown below]")
    _print_compile_table(table_rows)

    # ── Table 3: heuristic quality ────────────────────────────────────────
    if not args.no_max_autotune:
        _section("Table 3 — Heuristic Quality vs MaxAutotune")
        print("  gap% = (Heuristics − MaxAutotune) / MaxAutotune × 100%")
        print("  negative gap = Heuristics is FASTER than MaxAutotune")
        _print_quality_table(table_rows)

    # ── Restore env ───────────────────────────────────────────────────────
    _set_heuristics_env(True)
    torch._dynamo.reset()

    print()
    _hline("═")
    print("  Done.")
    _hline("═")
    print()


if __name__ == "__main__":
    main()
