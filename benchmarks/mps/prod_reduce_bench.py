"""Benchmark for the native Metal prod reduction vs the MPSGraph implementation
it replaces.

Run on a build WITH the native kernel and on a build WITHOUT it (e.g. the PR
parent commit) and compare. Reports two regimes:

  * steady-state: a fixed shape benchmarked warm (mean of 5
    torch.utils.benchmark blocked_autorange trials).
  * shape-varying: many distinct shapes seen once each -- this is where MPSGraph
    pays a per-shape graph recompilation (~100ms/shape) the native kernel does
    not.

Methodology note: the per-shape MPS bench has high run-to-run variance; compare
the same script on the baseline and candidate builds, using the mean of 5 warm
blocked_autorange trials rather than a single sweep.
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
    ((2, 8_000_000), 1),  # short-wide: wide inner kernel
    ((1_000_000, 8), 0),  # dim=0 outer_bucketed: few columns, long reduced dim
    ((2, 8_000_000), 0),  # dim=0 outer_thin: short reduced dim, many columns
]


def bench_us(stmt, globals_):
    for _ in range(30):
        exec(stmt, globals_)
    torch.mps.synchronize()
    trials = []
    for _ in range(5):
        t = benchmark.Timer(stmt=stmt, globals=globals_)
        trials.append(t.blocked_autorange(min_run_time=0.5).mean * 1e6)
    return sum(trials) / len(trials)


def steady_state():
    print("# steady-state prod (mean us, 5 warm blocked_autorange trials)")
    for shape, dim in STEADY_SHAPES:
        x = torch.randn(*shape, device="mps")
        torch.mps.synchronize()
        stmt = (
            "torch.prod(x); torch.mps.synchronize()"
            if dim is None
            else "torch.prod(x, dim=dim); torch.mps.synchronize()"
        )
        us = bench_us(stmt, {"x": x, "dim": dim, "torch": torch})
        print(f"  {str(shape):20} dim={dim}: {us:8.1f} us")
        del x
        torch.mps.empty_cache()


def shape_varying(n=200):
    # n distinct shapes, each reduced once -> exercises per-shape compilation.
    buf = torch.randn(20_000_000, device="mps")
    torch.prod(torch.as_strided(buf, (64, 500), (500, 1)), dim=1)  # warm
    torch.mps.synchronize()
    t0 = time.time()
    for i in range(n):
        N = 999 + i * 101
        torch.prod(torch.as_strided(buf, (64, N), (N, 1)), dim=1)
    torch.mps.synchronize()
    dt = time.time() - t0
    print(
        f"# shape-varying prod: {n} distinct shapes = {dt * 1e3:.0f} ms "
        f"({dt / n * 1e3:.2f} ms/shape)"
    )


def rss_stability(n=2000):
    """The point of the migration: the native kernel has no per-shape graph
    cache, so process peak RSS stays bounded across distinct shapes, unlike the
    MPSGraph prod (which grows its graph cache per shape). Reports the peak-RSS
    growth across n distinct shapes -- a few MB on the native build, steadily
    climbing on the MPSGraph build."""
    import resource
    import sys

    def peak_rss_mb():
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kB on Linux
        return ru / (1 << 20) if sys.platform == "darwin" else ru / (1 << 10)

    buf = torch.randn(20_000_000, device="mps")
    torch.prod(torch.as_strided(buf, (64, 500), (500, 1)), dim=1)  # warm
    torch.mps.synchronize()
    before = peak_rss_mb()
    for i in range(n):
        N = 999 + i * 101
        torch.prod(torch.as_strided(buf, (64, N), (N, 1)), dim=1)
    torch.mps.synchronize()
    after = peak_rss_mb()
    print(
        f"# RSS stability: {n} distinct shapes, peak RSS {before:.0f} -> {after:.0f} MB "
        f"(+{after - before:.0f} MB; native has no per-shape graph cache)"
    )


if __name__ == "__main__":
    steady_state()
    shape_varying()
    rss_stability()
