"""Benchmark native Metal var/std Welford reductions vs the MPSGraph path.

Run this script on the PR parent build and on the candidate build, then compare
the same rows. It reports:

  * steady-state: fixed shapes, warm mean of 5 blocked_autorange trials.
  * shape-varying: many distinct shapes, reduced once each.
  * RSS stability: peak RSS growth while visiting many distinct shapes.

MPS dispatch is asynchronous, so the timed statements synchronize inside the
statement. Without that, Timer mostly measures enqueue latency.
"""

import time

import torch
import torch.utils.benchmark as benchmark


STEADY_SHAPES = [
    ((1 << 24,), None),
    ((4096, 4096), 1),
    ((4096, 4096), 0),
    ((1024, 16384), 1),
    ((1_000_000, 8), 1),  # tall-thin: thread-per-row kernel
    ((8, 1_000_000), 1),  # short-wide: wide inner kernel
    ((8, 1_000_000), 0),  # dim=0 outer small-M kernel
]


def _fn(op, x, dim):
    f = torch.var if op == "var" else torch.std
    return f(x, dim=dim) if dim is not None else f(x)


def bench_us(stmt, globals_):
    for _ in range(30):
        exec(stmt, globals_)
    torch.mps.synchronize()
    trials = []
    for _ in range(5):
        t = benchmark.Timer(stmt=stmt, globals=globals_)
        trials.append(t.blocked_autorange(min_run_time=0.5).mean * 1e6)
    return sum(trials) / len(trials)


def steady_state(op="var"):
    print(f"# steady-state {op} (mean us, 5 warm blocked_autorange trials)")
    for shape, dim in STEADY_SHAPES:
        x = torch.randn(*shape, device="mps")
        torch.mps.synchronize()
        us = bench_us(
            "_fn(op, x, dim); torch.mps.synchronize()",
            {"_fn": _fn, "op": op, "x": x, "dim": dim, "torch": torch},
        )
        print(f"  {str(shape):20} dim={dim}: {us:8.1f} us")
        del x
        torch.mps.empty_cache()


def shape_varying(op="var", n=200):
    # n distinct shapes, each reduced once -> exercises per-shape compilation.
    buf = torch.randn(20_000_000, device="mps")
    _fn(op, torch.as_strided(buf, (64, 500), (500, 1)), 1)  # warm the machinery
    torch.mps.synchronize()
    t0 = time.time()
    for i in range(n):
        N = 999 + i * 101
        _fn(op, torch.as_strided(buf, (64, N), (N, 1)), 1)
    torch.mps.synchronize()
    dt = time.time() - t0
    print(
        f"# shape-varying {op}: {n} distinct shapes = {dt * 1e3:.0f} ms "
        f"({dt / n * 1e3:.2f} ms/shape)"
    )


def rss_stability(op="var", n=2000):
    """The migration removes MPSGraph graph-cache growth for shape-varying
    var/std. Reports peak-RSS growth across n distinct shapes."""
    import resource
    import sys

    def peak_rss_mb():
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1 << 20) if sys.platform == "darwin" else ru / (1 << 10)

    buf = torch.randn(20_000_000, device="mps")
    _fn(op, torch.as_strided(buf, (64, 500), (500, 1)), 1)
    torch.mps.synchronize()
    before = peak_rss_mb()
    for i in range(n):
        N = 999 + i * 101
        _fn(op, torch.as_strided(buf, (64, N), (N, 1)), 1)
    torch.mps.synchronize()
    after = peak_rss_mb()
    print(
        f"# RSS stability {op}: {n} distinct shapes, peak RSS {before:.0f} -> {after:.0f} MB "
        f"(+{after - before:.0f} MB; native has no per-shape graph cache)"
    )


if __name__ == "__main__":
    for op in ("var", "std"):
        steady_state(op)
        shape_varying(op)
        rss_stability(op)
