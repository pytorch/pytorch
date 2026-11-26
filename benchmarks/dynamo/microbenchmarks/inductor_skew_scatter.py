"""Benchmark Inductor's row scatter-adds with and without `skew_scatter_enabled`.

Sweeps index size n, destination rows r and width d with uniform, Zipf(s) and
hot-row (a fraction h of the indices on one row) indices. Reports median times
(both arms compiled on empty caches), eager time and errors vs float64.
`--startup` instead measures first-call time, cold and warm (caches populated).

    python benchmarks/dynamo/microbenchmarks/inductor_skew_scatter.py [--cases a,b] [--output-json out.json]
    python benchmarks/dynamo/microbenchmarks/inductor_skew_scatter.py --startup --cases idxadd_n8M_r50K_d64_bf16
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable

from triton.testing import do_bench

import torch
import torch._dynamo
from torch._dynamo.utils import compilation_time_metrics
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import fresh_cache


UNIFORM = ("uniform", None)
ZIPF08, ZIPF10, ZIPF12 = ("zipf", 0.8), ("zipf", 1.0), ("zipf", 1.2)
HOT05, HOT10, HOT50, HOT70 = ("hot", 0.05), ("hot", 0.1), ("hot", 0.5), ("hot", 0.7)
ALL = [UNIFORM, ZIPF08, ZIPF10, ZIPF12, HOT10, HOT50]

K, M = 1 << 10, 1 << 20
bf16, fp32 = torch.bfloat16, torch.float32
SHORT = {"index_add": "idxadd", "index_put": "idxput", "scatter_add": "scatadd"}


def case(op, n, r, d, width, src_dtype, out_dtype, dists=ALL) -> tuple:
    """``width`` > d: src is a strided slice."""

    def size(x: int) -> str:
        return f"{x // M}M" if x % M == 0 else f"{x // K}K" if x % K == 0 else str(x)

    name = f"{SHORT[op]}_n{size(n)}_r{size(r)}_d{d}" + (
        "_bf16" if src_dtype == bf16 else ""
    )
    return (name, op, n, r, d, width, src_dtype, out_dtype, dists)


CASES = [
    case("index_add", 2 * M, 256, 16, 16, fp32, fp32),
    case("index_put", 8 * M, 64, 64, 128, bf16, fp32),
    case("index_add", 512 * K, 50 * K, 128, 128, fp32, fp32),
    case("index_add", 2 * M, 50 * K, 64, 128, bf16, fp32, [*ALL, HOT05]),
    case("index_add", 8 * M, 50 * K, 64, 128, bf16, fp32),
    case("index_add", 32 * M, 50 * K, 64, 64, bf16, fp32),
    case("index_put", 2 * M, 2 * K, 16, 16, fp32, fp32),
    case("index_put", 8 * M, 2 * K, 64, 128, bf16, fp32),
    case("scatter_add", 2 * M, 2 * K, 256, 256, fp32, fp32, [*ALL, HOT70]),
    case("scatter_add", 8 * M, 50 * K, 256, 256, fp32, fp32),
    case("index_add", 8 * M, 1 * M, 128, 128, fp32, fp32),
    case("index_add", 2 * M, 1 * M, 256, 256, bf16, fp32),
    # bf16 output into a large destination: not rewritten (same code in both arms)
    case("scatter_add", 8 * M, 8 * M, 128, 128, bf16, bf16, [UNIFORM, ZIPF10, HOT50]),
]


def make_index(n: int, r: int, dist: tuple[str, float | None]) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    kind, p = dist
    idx = torch.randint(0, r, (n,), generator=g, device="cuda")
    if kind == "hot":
        hot = torch.rand(n, generator=g, device="cuda") < p
        idx = torch.where(hot, torch.full_like(idx, r - 1), idx)
    elif kind == "zipf":
        w = torch.arange(1, r + 1, device="cuda", dtype=torch.float64).pow(-p)
        cdf = torch.cumsum(w / w.sum(), 0)
        u = torch.rand(n, generator=g, device="cuda", dtype=torch.float64)
        rank = torch.searchsorted(cdf, u).clamp_(max=r - 1)
        idx = torch.randperm(r, generator=g, device="cuda")[rank]
    return idx


def make_fn(op: str) -> Callable[..., torch.Tensor]:
    def fn(src, idx, r: int, out_dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros(r, src.shape[1], device=src.device, dtype=out_dtype)
        src = src.to(out_dtype)
        if op == "index_add":
            return out.index_add(0, idx, src)
        if op == "index_put":
            return out.index_put((idx,), src, accumulate=True)
        return out.scatter_add(0, idx[:, None].expand(-1, src.shape[1]), src)

    return fn


NO_REMOTE_CACHES = {"fx_graph_remote_cache": False, "autotune_remote_cache": False}


def compile_arm(op: str, on: bool, **options) -> Callable:
    # A distinct code object per arm, so the arms do not share Dynamo's cache.
    fn = make_fn(op)
    code = fn.__code__.replace(co_name=f"fn_{'on' if on else 'off'}")
    fn = type(fn)(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)
    options = {"skew_scatter_enabled": on, **NO_REMOTE_CACHES, **options}
    return torch.compile(fn, dynamic=False, options=options)


def errors(x: torch.Tensor, src: torch.Tensor, idx: torch.Tensor, r: int) -> tuple:
    """max |err| / max |ref| and mean |err| / mean |ref| vs a float64 reference."""
    ref = torch.zeros(r, src.shape[1], device=src.device, dtype=torch.float64)
    step = 1 << 22
    for i in range(0, idx.numel(), step):
        ref.index_add_(0, idx[i : i + step], src[i : i + step].double())
    e, a = (x.double() - ref).abs(), ref.abs()
    floor = 1e-30
    return (e.max() / a.max().clamp_min(floor)).item(), (
        e.mean() / a.mean().clamp_min(floor)
    ).item()


def picked_configs() -> list[str]:
    """Config each first-call-tuned skew scatter kernel of the loaded modules runs."""
    seen = {}
    for mod in PyCodeCache.modules:
        for name, k in vars(mod).items():
            launchers = getattr(k, "launchers", None)
            if name.startswith("_skew_scatter_kernel") and launchers:
                if len(launchers) == 1:
                    c = launchers[0].config
                    kw = c.kwargs
                    seen[id(k)] = (
                        f"MODE={kw['MODE']} BLOCK={kw['BLOCK']} RB={kw['RB']} warps={c.num_warps}"
                    )
    return sorted(seen.values())


def dist_name(dist: tuple) -> str:
    return dist[0] if dist[1] is None else f"{dist[0]} {dist[1]}"


def run_case(case: tuple, rounds: int) -> list[dict]:
    name, op, n, r, d, width, src_dtype, out_dtype, dists = case
    torch.manual_seed(0)
    src = torch.randn(n, width, device="cuda", dtype=src_dtype)[:, :d]
    eager = make_fn(op)
    rows = []
    for dist in dists:
        torch._dynamo.reset()
        PyCodeCache.cache_clear()
        with fresh_cache():
            off, on = compile_arm(op, False), compile_arm(op, True)
            idx = make_index(n, r, dist)
            off(src, idx, r, out_dtype)
            on(src, idx, r, out_dtype)
            picked = picked_configs()
        args = (src, idx, r, out_dtype)
        off_err, on_err = (
            errors(off(*args), src, idx, r),
            errors(on(*args), src, idx, r),
        )
        t_off, t_on = [], []
        for _ in range(rounds):
            t_off.append(do_bench(lambda: off(*args), return_mode="median"))
            t_on.append(do_bench(lambda: on(*args), return_mode="median"))
        row = {
            "case": name,
            "op": op,
            "shape": f"n={n} r={r} d={d} {str(src_dtype)[6:]}->{str(out_dtype)[6:]}",
            "dist": dist_name(dist),
            "eager_ms": do_bench(lambda: eager(*args), return_mode="median"),
            "off_ms": statistics.median(t_off),
            "on_ms": statistics.median(t_on),
            "off_err": off_err,
            "on_err": on_err,
            "picked": picked,
        }
        row["speedup"] = row["off_ms"] / row["on_ms"]
        rows.append(row)
        print(
            f"| {name} | {row['dist']} | {row['eager_ms']:.3f} | {row['off_ms']:.3f} | {row['on_ms']:.3f} "
            f"| {row['speedup']:.2f}x | {off_err[0]:.2e} / {off_err[1]:.2e} | {on_err[0]:.2e} / {on_err[1]:.2e} | {', '.join(picked)} |",
            flush=True,
        )
    return rows


def first_call_s(fn: Callable, args: tuple) -> tuple[float, float]:
    """Wall time of the first call, and the part of it spent autotuning."""
    tuning = compilation_time_metrics.setdefault(
        "CachingAutotuner.benchmark_all_configs", []
    )
    before = sum(tuning)
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t, sum(tuning) - before


def run_startup(case: tuple) -> list[dict]:
    """First-call time of each arm, cold (fresh caches) and warm (a second compile
    on the same caches, as in a new process; the flag-on arm still autotunes)."""
    name, op, n, r, d, width, src_dtype, out_dtype, dists = case
    src = torch.randn(n, width, device="cuda", dtype=src_dtype)[:, :d]
    rows = []
    with fresh_cache():  # process warm-up
        compile_arm(op, False)(src, make_index(n, r, UNIFORM), r, out_dtype)
    for dist in dists:
        args = (src, make_index(n, r, dist), r, out_dtype)
        row = {"case": name, "dist": dist_name(dist)}
        for arm in ("off", "on"):
            with fresh_cache():
                for state in ("cold", "warm"):
                    torch._dynamo.reset()
                    PyCodeCache.cache_clear(purge=True)
                    caches = {"fx_graph_cache": True, "autotune_local_cache": True}
                    fn = compile_arm(op, arm == "on", **caches)
                    s, tune_s = first_call_s(fn, args)
                    row[f"{arm}_{state}_s"], row[f"{arm}_{state}_tune_s"] = s, tune_s
        rows.append(row)
        print(
            f"| {name} | {row['dist']} | {row['off_cold_s']:.2f} | {row['on_cold_s']:.2f} | {row['on_cold_s'] - row['off_cold_s']:+.2f} "
            f"| {row['off_warm_s']:.2f} | {row['on_warm_s']:.2f} | {row['on_warm_s'] - row['off_warm_s']:+.2f} "
            f"| {row['on_warm_tune_s'] - row['off_warm_tune_s']:+.2f} |",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    # optional leading action, for benchmark harnesses that pass one
    parser.add_argument("action", nargs="?", default="perf", choices=["perf"])
    parser.add_argument("--cases", default="", help="comma-separated case names")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--startup", action="store_true", help="first-call time")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    names = set(args.cases.split(",")) if args.cases else None
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    if args.startup:
        print(
            "| case | index dist | cold off s | cold on s | added | warm off s | warm on s | added | of which autotuning |"
        )
    else:
        print(
            "| case | index dist | eager ms | flag off ms | flag on ms | off/on | off rel err max / mean | on rel err max / mean | picked |"
        )
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in CASES:
        if names is None or c[0] in names:
            rows += run_startup(c) if args.startup else run_case(c, args.rounds)
    if args.output_json:
        out = {"args": vars(args), "device": torch.cuda.get_device_name(), "rows": rows}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
