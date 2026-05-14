# Owner(s): ["module: mps"]
"""Benchmarks for torch.mps.MPSGraph (MTLIndirectCommandBuffer capture/replay).

Measures CPU-side dispatch overhead eliminated by ICB graph mode vs eager,
across three axes:

  overhead   -- small kernel, N dispatches: eager O(N encode+submit) vs
                graph O(1 replay). Quantifies per-dispatch CPU savings.
  chains     -- multi-op chains via make_graphed_callables: realistic
                forward-pass pattern (capture once, replay every iteration).
                replay() encodes directly on the stream CB (non-blocking),
                so synchronize() amortizes the GPU wait as in eager mode.
  dtypes     -- overhead savings across float32/float16/bfloat16.
  scaling    -- per-dispatch savings vs chain length N (32→1024).

Usage:
  python bench_graph_overhead.py               # all benchmarks
  python bench_graph_overhead.py overhead
  python bench_graph_overhead.py chains scaling
"""

import sys
import timeit

import torch
from torch.utils.benchmark import Compare, Timer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync():
    torch.mps.synchronize()


def _warmup(fn, n=30):
    for _ in range(n):
        fn()
    _sync()


def _meas(stmt, globs, sub_label, description):
    t = Timer(
        stmt=stmt,
        globals=globs,
        language="python",
        timer=timeit.default_timer,
        sub_label=sub_label,
        description=description,
        env=torch.__version__,
    )
    return t.blocked_autorange()


# ---------------------------------------------------------------------------
# Layer A — CPU dispatch overhead (small kernel, N dispatches)
# ---------------------------------------------------------------------------

def _bench_overhead_n(n: int, dtype: torch.dtype = torch.float32):
    """Eager × N vs single graph replay (N commands) for relu on 4096 elems."""
    x = torch.randn(4096, device="mps", dtype=dtype)
    sub = f"relu-4096 ×{n} ({dtype})"

    # Eager warmup
    _warmup(lambda: x.relu())

    eager = _meas(
        stmt="for _ in range(n): x.relu()\n_sync()",
        globs={"x": x, "n": n, "_sync": _sync},
        sub_label=sub,
        description="eager",
    )

    # Build graph with n relu commands
    g = torch.mps.MPSGraph(max_commands=n + 16)
    with torch.mps.graph(g):
        for _ in range(n):
            x.relu()
    _sync()

    graph = _meas(
        stmt="g.replay(); _sync()",
        globs={"g": g, "_sync": _sync},
        sub_label=sub,
        description="graph-replay",
    )
    return eager, graph


def bench_overhead():
    rc = []
    for n in (32, 128, 256, 512, 1024):
        e, g = _bench_overhead_n(n)
        rc.extend([e, g])
        speedup = e.median / g.median
        print(f"  N={n:5d}: eager={e.median*1e6:.1f}µs  graph={g.median*1e6:.1f}µs  speedup={speedup:.2f}x")
    Compare(rc).print()


# ---------------------------------------------------------------------------
# Layer B — Multi-op chains via make_graphed_callables
# ---------------------------------------------------------------------------

def _bench_chain(fn, x, label: str):
    """Compare fn(x) eager vs make_graphed_callables(fn, (x,)) replay."""
    _warmup(lambda: fn(x))

    eager = _meas(
        stmt="fn(x); _sync()",
        globs={"fn": fn, "x": x, "_sync": _sync},
        sub_label=label,
        description="eager",
    )

    wrapped = torch.mps.make_graphed_callables(fn, (x,))
    _sync()
    _warmup(lambda: wrapped(x))

    graph = _meas(
        stmt="wrapped(x); _sync()",
        globs={"wrapped": wrapped, "x": x, "_sync": _sync},
        sub_label=label,
        description="graphed-callable",
    )
    return eager, graph


def bench_chains():
    rc = []

    chains = {
        "relu-sigmoid-tanh-exp (1024)": (
            lambda t: t.relu().sigmoid().tanh().exp(),
            torch.randn(1024, device="mps"),
        ),
        "mul-add-relu (2048)": (
            lambda t: (t * 2.0 + 1.0).relu(),
            torch.randn(2048, device="mps"),
        ),
        "relu-sigmoid-tanh-exp (16384)": (
            lambda t: t.relu().sigmoid().tanh().exp(),
            torch.randn(16384, device="mps"),
        ),
        "mul-add-relu (16384)": (
            lambda t: (t * 2.0 + 1.0).relu(),
            torch.randn(16384, device="mps"),
        ),
    }

    for label, (fn, x) in chains.items():
        e, g = _bench_chain(fn, x, label)
        rc.extend([e, g])
        speedup = e.median / g.median
        print(f"  {label}: eager={e.median*1e6:.1f}µs  graph={g.median*1e6:.1f}µs  speedup={speedup:.2f}x")

    Compare(rc).print()


# ---------------------------------------------------------------------------
# Layer C — dtype coverage
# ---------------------------------------------------------------------------

def bench_dtypes():
    rc = []
    n = 256
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        e, g = _bench_overhead_n(n, dtype=dtype)
        rc.extend([e, g])
        speedup = e.median / g.median
        print(f"  dtype={dtype} N={n}: eager={e.median*1e6:.1f}µs  graph={g.median*1e6:.1f}µs  speedup={speedup:.2f}x")
    Compare(rc).print()


# ---------------------------------------------------------------------------
# Layer D — per-dispatch savings vs N (scaling law)
# ---------------------------------------------------------------------------

def bench_scaling():
    """Shows dual-path replay: direct re-encode (N<16) and ICB (N>=16).

    Amortises sync overhead over K=10 reps so per-dispatch cost is
    visible even for N=1. See kDirectEncodeThreshold in MPSStreamGraph.h.
    """
    K = 10
    SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    print("  N    eager_N(us)  graph(us)  savings/dispatch(ns)  speedup  path")
    print("  " + "-" * 72)
    for n in SIZES:
        x = torch.randn(4096, device="mps")
        _warmup(lambda: x.relu())

        g = torch.mps.MPSGraph(max_commands=n + 16)
        with torch.mps.graph(g):
            for _ in range(n):
                x.relu()
        _sync()

        # K reps of N dispatches + sync -- amortises the fixed sync cost
        eager = _meas(
            stmt=f"for _ in range({K * n}): x.relu()\n_sync()",
            globs={"x": x, "_sync": _sync},
            sub_label=f"relu-4096 x{n}",
            description="eager",
        )
        graph = _meas(
            stmt=f"for _ in range({K}): g.replay()\n_sync()",
            globs={"g": g, "_sync": _sync},
            sub_label=f"relu-4096 x{n}",
            description="graph-replay",
        )
        eager_per = eager.median * 1e6 / K
        graph_per = graph.median * 1e6 / K
        savings_per = (eager_per - graph_per) / n * 1e3
        speedup = eager_per / graph_per
        replay_path = "direct" if n < 16 else "ICB"
        print(
            f"  {n:5d}  {eager_per:10.1f}  {graph_per:8.1f}  {savings_per:18.1f}  {speedup:6.2f}x  {replay_path}"
        )


BENCHMARKS = {
    "overhead": bench_overhead,
    "chains": bench_chains,
    "dtypes": bench_dtypes,
    "scaling": bench_scaling,
}


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; skipping bench_graph_overhead.py")
        return

    selected = sys.argv[1:]
    if not selected:
        selected = list(BENCHMARKS.keys())

    for name in selected:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}. Available: {', '.join(BENCHMARKS.keys())}")
            sys.exit(1)

    for name in selected:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        BENCHMARKS[name]()


if __name__ == "__main__":
    main()
