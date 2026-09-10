"""MPS softmax before/after bench (G3) + memory-stability probe (G2).

Same-build comparison of the kept MPSGraph fallback (mode A, forced via
PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX=1) versus the native Metal softmax kernels
(mode B, env unset). The force toggle is read once per process (static const in
canUseMetalSoftmax), so each MODE MUST run in its own process. This script is a
single-mode worker: set BENCH_MODE={A,B} and BENCH_TASK={perf,mem,sanity}.
The interleaving/orchestration is done by the parent harness (bench_run.sh),
which alternates A and B per-shape in one session to neutralise thermal drift.

Methodology (perf): torch.utils.benchmark.Timer + blocked_autorange, warmup 30,
mean over N_TRIALS=5; amortised many-iter-one-sync to dodge the MPS sync floor.
"""

import json
import os
import sys

import torch


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available")

MODE = os.environ.get("BENCH_MODE", "B")
TASK = os.environ.get("BENCH_TASK", "perf")
DEV = torch.device("mps")

DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}

# (name, shape, dim)
SHAPES = [
    ("1024x1024_d-1", (1024, 1024), -1),
    ("1024x1024_d0", (1024, 1024), 0),
    ("4096x4096_d-1", (4096, 4096), -1),
    ("4096x4096_d0", (4096, 4096), 0),
    ("8x2048x4096_d-1", (8, 2048, 4096), -1),
    ("8x2048x4096_d0", (8, 2048, 4096), 0),
    ("8x2048x4096_d1", (8, 2048, 4096), 1),
    ("65536x128_d-1", (65536, 128), -1),
    ("65536x128_d0", (65536, 128), 0),
    ("128x65536_d-1", (128, 65536), -1),
    ("128x65536_d0", (128, 65536), 0),
    ("512x768_d-1", (512, 768), -1),
    ("2048x2048_d-1", (2048, 2048), -1),
]

# Focused regression-confirmation subset (BENCH_FOCUS=1).
FOCUS_SHAPES = [
    ("1024x1024_d0", (1024, 1024), 0),
    ("4096x4096_d0", (4096, 4096), 0),
    ("65536x128_d0", (65536, 128), 0),
    ("8x2048x4096_d0", (8, 2048, 4096), 0),  # the WIN (tiled path) - sanity
    ("1024x1024_d-1", (1024, 1024), -1),  # borderline last-dim
    ("2048x2048_d-1", (2048, 2048), -1),  # borderline last-dim
    ("8x2048x4096_d-1", (8, 2048, 4096), -1),  # +323us last-dim reg
]
if os.environ.get("BENCH_FOCUS") == "1":
    SHAPES = FOCUS_SHAPES


def amortized_timer(fn, sync, label):
    """Time fn with one sync amortised over the inner loop to dodge sync floor."""
    import torch.utils.benchmark as tb

    t = tb.Timer(
        stmt="for _ in range(I):\n    fn()\nsync()",
        globals={"fn": fn, "sync": sync, "I": 50},
        num_threads=1,
        label=label,
    )
    _mrt = float(os.environ.get("BENCH_MIN_RUN_TIME", "1.5"))
    m = t.blocked_autorange(min_run_time=_mrt)
    # per-call median in microseconds
    return (m.median / 50) * 1e6


def make_fwd(shape, dim, dt):
    x = torch.randn(*shape, device=DEV, dtype=dt)

    def fn():
        return torch.softmax(x, dim=dim)

    return fn


def make_fwd_bwd(shape, dim, dt):
    x = torch.randn(*shape, device=DEV, dtype=dt, requires_grad=True)
    go = torch.randn(*shape, device=DEV, dtype=dt)

    def fn():
        y = torch.softmax(x, dim=dim)
        y.backward(go)
        x.grad = None

    return fn


def sync():
    torch.mps.synchronize()


def run_perf():
    N_TRIALS = int(os.environ.get("N_TRIALS", "5"))
    out = []
    for sname, shape, dim in SHAPES:
        for dtkey, dt in DTYPES.items():
            for kind, maker in (("fwd", make_fwd), ("fwdbwd", make_fwd_bwd)):
                try:
                    fn = maker(shape, dim, dt)
                    # warmup
                    for _ in range(30):
                        fn()
                    sync()
                    trials = []
                    for _ in range(N_TRIALS):
                        us = amortized_timer(fn, sync, f"{sname}/{dtkey}/{kind}")
                        trials.append(us)
                    trials.sort()
                    med = trials[len(trials) // 2]
                    mn = sum(trials) / len(trials)
                    sd = (sum((t - mn) ** 2 for t in trials) / len(trials)) ** 0.5
                    out.append(
                        {
                            "shape": sname,
                            "dtype": dtkey,
                            "kind": kind,
                            "median_us": med,
                            "mean_us": mn,
                            "std_us": sd,
                            "trials": trials,
                        }
                    )
                except Exception as e:
                    out.append(
                        {"shape": sname, "dtype": dtkey, "kind": kind, "error": repr(e)}
                    )
    print(json.dumps({"mode": MODE, "results": out}))


def run_mem():
    """G2: loop over ~200 distinct shapes; isolate the per-shape GRAPH cache.

    The MPS *tensor* allocator pool is freed by empty_cache(); the MPSGraph
    executable cache (keyed per shape) is NOT. To isolate the cache growth from
    raw tensor data we (a) keep each shape TINY so tensor bytes are negligible,
    and (b) empty_cache()+synchronize before sampling so the residual driver
    memory reflects retained graph state, not live/pooled tensors.
    """
    torch.mps.empty_cache()
    torch.mps.synchronize()
    samples = []
    start_drv = torch.mps.driver_allocated_memory()
    start_cur = torch.mps.current_allocated_memory()
    count = 0
    # 100 distinct shapes x 2 dims = 200 distinct (shape,dim) softmax keys.
    # Tiny rows/cols so per-shape tensor bytes << any graph-cache footprint.
    for i in range(100):
        rows = 32 + i  # 32..131, distinct each iter
        cols = 48 + i  # 48..147, distinct each iter (keeps shape unique)
        for dim in (-1, 0):
            x = torch.randn(rows, cols, device=DEV, dtype=torch.float32)
            y = torch.softmax(x, dim=dim)
            # backward too (distinct bwd graph per shape under MPSGraph)
            xb = torch.randn(
                rows, cols, device=DEV, dtype=torch.float32, requires_grad=True
            )
            yb = torch.softmax(xb, dim=dim)
            yb.backward(torch.randn_like(yb))
            count += 1
            del x, y, xb, yb
        if i % 10 == 0:
            torch.mps.synchronize()
            # NO empty_cache here: with tiny tensors the live/pooled tensor bytes
            # are negligible, so growth in driver memory = per-shape graph cache.
            samples.append(
                {
                    "iter": i,
                    "distinct_shapes": count,
                    "driver_mb": torch.mps.driver_allocated_memory() / 1e6,
                    "current_mb": torch.mps.current_allocated_memory() / 1e6,
                }
            )
    torch.mps.synchronize()
    end_drv = torch.mps.driver_allocated_memory()
    end_cur = torch.mps.current_allocated_memory()
    # After draining the tensor pool, whatever driver memory REMAINS above the
    # start is retained graph/executable state (empty_cache does not rebuild it).
    torch.mps.empty_cache()
    torch.mps.synchronize()
    after_empty_drv = torch.mps.driver_allocated_memory()
    print(
        json.dumps(
            {
                "mode": MODE,
                "distinct_shapes": count,
                "start_driver_mb": start_drv / 1e6,
                "end_driver_mb": end_drv / 1e6,
                "delta_driver_mb": (end_drv - start_drv) / 1e6,
                "after_empty_cache_driver_mb": after_empty_drv / 1e6,
                "retained_after_empty_mb": (after_empty_drv - start_drv) / 1e6,
                "start_current_mb": start_cur / 1e6,
                "end_current_mb": end_cur / 1e6,
                "samples": samples,
            }
        )
    )


def run_sanity():
    """Confirm correctness vs CPU in this mode (so 'before' is a valid baseline)."""
    res = []
    for shape, dim in [
        ((1024, 1024), -1),
        ((1024, 1024), 0),
        ((8, 2048, 64), 1),
        ((128, 256), -1),
    ]:
        for dtkey, dt in DTYPES.items():
            x = torch.randn(*shape, dtype=torch.float32)
            xm = x.to(DEV).to(dt)
            ym = torch.softmax(xm, dim=dim).float().cpu()
            yc = torch.softmax(x, dim=dim)
            atol = 2e-2 if dtkey != "f32" else 1e-4
            ok = torch.allclose(ym, yc, atol=atol, rtol=1e-2)
            res.append(
                {
                    "shape": str(shape),
                    "dim": dim,
                    "dtype": dtkey,
                    "max_abs_err": (ym - yc).abs().max().item(),
                    "ok": bool(ok),
                }
            )
    allok = all(r["ok"] for r in res)
    print(json.dumps({"mode": MODE, "all_ok": allok, "checks": res}))


if __name__ == "__main__":
    forced = os.environ.get("PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX")
    sys.stderr.write(
        f"MODE={MODE} TASK={TASK} FORCE_ENV={forced!r} torch={torch.__file__}\n"
    )
    if TASK == "perf":
        run_perf()
    elif TASK == "mem":
        run_mem()
    elif TASK == "sanity":
        run_sanity()
