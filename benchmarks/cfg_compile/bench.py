"""Benchmarks for torch.cfg_compile.

Compares eager execution with the tiers of the cfg VM on small workloads where
per-operation overhead matters, and prints the results as Markdown tables
(with Mermaid charts when --charts is given).

    python benchmarks/cfg_compile/bench.py
    python benchmarks/cfg_compile/bench.py --charts --json results.json > report.md

The modes are cumulative, each adding one mechanism to the previous one:

    eager          plain Python and the regular bindings
    generic        the VM running bytecode semantics (specialize=False)
    sites          + operations call the kernel the dispatcher resolved them to,
                   entering at the top of the chain (skip_autograd=False)
    skip_autograd  + enter below the Autograd kernel when no gradient is tracked
    unboxed        + kernel results get no Python object until Python needs one
                   (the default configuration)
"""

import argparse
import json
import platform
import time

import torch


MODES = ["eager", "generic", "sites", "skip_autograd", "unboxed"]
MODE_OPTIONS = {
    "generic": {"specialize": False},
    "sites": {"skip_autograd": False, "unboxed": False},
    "skip_autograd": {"unboxed": False},
    "unboxed": {},
}
COLOR = "#2a78d6"


def chain(x, y):
    z = x + y
    z = z * 2
    z = z - y
    z = torch.mul(z, x)
    z = z.relu()
    z = z + 1.0
    z = torch.tanh(z)
    z = z / 3
    return z.sum()


def for_loop(x, y, n):
    for _ in range(n):
        x = x * 0.5 + y
        if x.sum() > 100:
            x = x - 1
    return x


def while_loop(x):
    while x.sum() < 1000:
        x = x * 2 + 1
    return x


def mlp(x, w1, b1, w2, b2):
    h = torch.addmm(b1, x, w1).relu()
    return torch.addmm(b2, h, w2)


def shape_branch(x, y):
    if x.shape[0] > 1:
        x = x.mean(0, keepdim=True)
    return (x * y).relu()


def python_heavy(xs, scale):
    # Mostly Python: the tensor operations are a small part of the work.
    out = []
    for i, x in enumerate(xs):
        if i % 2 == 0:
            out.append(x * scale)
        else:
            out.append(x)
    return out


def workloads(size, device):
    def r(*shape):
        return torch.randn(*shape, device=device)

    return [
        ("chain of 9 ops", chain, (r(size), r(size))),
        ("for loop, 10 x (3 ops + branch)", for_loop, (r(size), r(size), 10)),
        (
            "while loop on a tensor value",
            while_loop,
            (torch.ones(size, device=device),),
        ),
        (
            "2-layer MLP (addmm)",
            mlp,
            (r(8, size), r(size, size), r(size), r(size, size), r(size)),
        ),
        (
            "Python loop over 8 tensors",
            python_heavy,
            ([r(size) for _ in range(8)], 2.0),
        ),
    ]


def single_ops(device):
    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    m = torch.randn(4, 4, device=device)

    def op(source):
        namespace = {"torch": torch}
        exec(f"def f(x, y, m):\n    return {source}\n", namespace)
        return source, namespace["f"], (x, y, m)

    return [
        op(source)
        for source in [
            "x",
            "x + y",
            "torch.add(x, y)",
            "x.add(y)",
            "x * 2",
            "x.relu()",
            "x.sum()",
            "x @ y",
            "torch.addmm(y, m, m)",
            "x.view(2, 2)",
            "x + y + y + y + y",
        ]
    ]


def sync(device):
    if device != "cpu":
        torch.accelerator.synchronize()


def time_calls(fn, args, number, device):
    sync(device)
    start = time.perf_counter_ns()
    for _ in range(number):
        fn(*args)
    sync(device)
    return (time.perf_counter_ns() - start) / number


def compare(fns, args, device, budget_ns=2e8, rounds=5):
    """ns per call of each callable: the best of several interleaved rounds."""
    for fn in fns.values():
        for _ in range(50):
            fn(*args)
    once = time_calls(next(iter(fns.values())), args, 20, device)
    number = max(20, min(50000, int(budget_ns / once)))
    best = dict.fromkeys(fns, float("inf"))
    for _ in range(rounds):
        for name, fn in fns.items():
            best[name] = min(best[name], time_calls(fn, args, number, device))
    return best


def compiled_modes(fn, modes=MODES):
    return {
        mode: fn
        if mode == "eager"
        else torch.cfg_compile(fn, fallback=False, **MODE_OPTIONS[mode])
        for mode in modes
    }


def bench_workloads(sizes, device):
    rows = []
    for size in sizes:
        for name, fn, args in workloads(size, device):
            rows.append(
                {
                    "name": name,
                    "size": size,
                    "ns": compare(compiled_modes(fn), args, device),
                }
            )
    return rows


def bench_ops(device):
    return [
        {
            "name": name,
            "ns": compare(
                compiled_modes(fn, ["eager", "generic", "unboxed"]), args, device
            ),
        }
        for name, fn, args in single_ops(device)
    ]


def bench_warmup(device):
    """What the first calls cost: building the graph, then profiling and
    specializing each block the first time it runs."""
    args = (torch.randn(64, device=device), torch.randn(64, device=device), 10)
    eager = compare({"eager": for_loop}, args, device)["eager"]
    build, calls = [], []
    for _ in range(20):
        start = time.perf_counter_ns()
        compiled = torch.cfg_compile(for_loop, fallback=False)
        build.append(time.perf_counter_ns() - start)
        times = []
        for _ in range(6):
            start = time.perf_counter_ns()
            compiled(*args)
            times.append(time.perf_counter_ns() - start)
        calls.append(times)
    steady = compare({"cfg": compiled}, args, device)["cfg"]
    per_call = [min(c[i] for c in calls) for i in range(6)]
    extra = min(build) + sum(t - steady for t in per_call)
    return {
        "eager": eager,
        "build": min(build),
        "calls": per_call,
        "steady": steady,
        "break_even_calls": extra / (eager - steady) if eager > steady else None,
    }


def bench_autograd(device):
    """With a gradient being tracked the sites enter at the Autograd kernel."""
    x = torch.randn(64, device=device)
    w = torch.randn(64, device=device, requires_grad=True)

    def forward_backward(fn):
        def run(x, w):
            w.grad = None
            fn(x, w).backward()

        return run

    fns = compiled_modes(chain, ["eager", "unboxed"])
    return {
        "forward": compare(fns, (x, w), device),
        "forward_backward": compare(
            {k: forward_backward(f) for k, f in fns.items()}, (x, w), device
        ),
    }


def bench_narrowing(device):
    """Shape-dependent control flow: everything dynamic versus a range guard
    on the dimension the branch looks at."""
    pipeline = torch.cfg_compile.Pipeline
    y = torch.randn(1, 16, device=device)
    out = {}
    for label, pipe in (
        ("default", pipeline.default()),
        ("narrowing", pipeline.narrowing()),
    ):
        compiled = torch.cfg_compile(shape_branch, pipeline=pipe, fallback=False)
        timings = {}
        for batch in (1, 8):
            x = torch.randn(batch, 16, device=device)
            timings[batch] = compare(
                {"eager": shape_branch, "cfg": compiled}, (x, y), device
            )
        for batch in (1, 8, 3, 1, 64, 1, 2):
            compiled(torch.randn(batch, 16, device=device), y)
        stats = compiled.stats()["blocks"]
        out[label] = {
            "ns": timings,
            "profile_runs": [b["profile_runs"] for b in stats],
            "versions": [b["versions"] for b in stats],
        }
    return out


def bench_unboxing(device):
    args = (torch.randn(64, device=device), torch.randn(64, device=device), 10)
    out = {}
    for name, fn, a in (
        ("chain of 9 ops", chain, args[:2]),
        ("for loop, 10 x (3 ops + branch)", for_loop, args),
    ):
        compiled = torch.cfg_compile(fn, fallback=False)
        for _ in range(3):
            compiled(*a)
        before = compiled.stats()
        for _ in range(100):
            compiled(*a)
        after = compiled.stats()
        out[name] = {
            "kernel_results_per_call": (
                after["unboxed_results"] - before["unboxed_results"]
            )
            / 100,
            "python_objects_per_call": (after["boxed_later"] - before["boxed_later"])
            / 100,
        }
    return out


def speedup(row, mode):
    return row["ns"]["eager"] / row["ns"][mode]


def chart(title, labels, values, axis, low=0.0, high=None):
    high = high if high is not None else max(values) * 1.15
    labels = ", ".join(f'"{label}"' for label in labels)
    values = ", ".join(f"{v:.2f}" for v in values)
    init = (
        '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "'
        + COLOR
        + '"}}}}%%'
    )
    return (
        f"```mermaid\n{init}\nxychart-beta horizontal\n"
        f'    title "{title}"\n    x-axis [{labels}]\n'
        f'    y-axis "{axis}" {low:.2f} --> {high:.2f}\n    bar [{values}]\n```\n'
    )


def report(results, charts):
    out = []
    meta = results["meta"]
    out.append(
        f"Device `{meta['device']}`, {meta['threads']} thread(s), torch {meta['torch']}, "
        f"Python {meta['python']}, {meta['machine']}. Times are ns per call, best of 5 rounds.\n"
    )

    out.append("### Workloads\n")
    out.append(
        "| workload | size | eager | generic | sites | skip autograd | unboxed | speedup |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in results["workloads"]:
        cells = " | ".join(f"{row['ns'][m]:,.0f}" for m in MODES)
        out.append(
            f"| {row['name']} | {row['size']} | {cells} | **{speedup(row, 'unboxed'):.2f}x** |"
        )
    out.append("")
    if charts:
        for size in sorted({row["size"] for row in results["workloads"]}):
            rows = [row for row in results["workloads"] if row["size"] == size]
            out.append(
                chart(
                    f"Speedup over eager, tensors of {size} elements",
                    [row["name"] for row in rows],
                    [speedup(row, "unboxed") for row in rows],
                    "speedup (x)",
                    high=2.0,
                )
            )
        first = results["workloads"][0]
        out.append(
            chart(
                f"What each mechanism adds: {first['name']}, {first['size']} elements",
                MODES[1:],
                [speedup(first, mode) for mode in MODES[1:]],
                "speedup over eager (x)",
                high=2.0,
            )
        )

    out.append("### Single operations (4-element tensors)\n")
    out.append("| function body | eager | generic | cfg | saved per call |")
    out.append("|---|---:|---:|---:|---:|")
    for row in results["ops"]:
        ns = row["ns"]
        out.append(
            f"| `{row['name']}` | {ns['eager']:,.0f} | {ns['generic']:,.0f} | {ns['unboxed']:,.0f} | "
            f"{ns['eager'] - ns['unboxed']:,.0f} |"
        )
    out.append("")
    if charts:
        rows = results["ops"][1:]
        out.append(
            chart(
                "Time saved per call, single operations",
                [row["name"] for row in rows],
                [row["ns"]["eager"] - row["ns"]["unboxed"] for row in rows],
                "ns saved (eager - cfg)",
                low=min(
                    0.0,
                    min(row["ns"]["eager"] - row["ns"]["unboxed"] for row in rows)
                    * 1.15,
                ),
            )
        )

    warm = results["warmup"]
    out.append("### Warm-up (for loop, 64 elements)\n")
    out.append("| step | time (us) |")
    out.append("|---|---:|")
    out.append(
        f"| `torch.cfg_compile(fn)`: bytecode to SSA | {warm['build'] / 1e3:,.1f} |"
    )
    for i, t in enumerate(warm["calls"]):
        out.append(f"| call {i + 1} | {t / 1e3:,.1f} |")
    out.append(f"| steady state | {warm['steady'] / 1e3:,.1f} |")
    out.append(f"| eager | {warm['eager'] / 1e3:,.1f} |")
    out.append("")
    if warm["break_even_calls"] is not None:
        out.append(
            f"The one-off cost is paid back after about {warm['break_even_calls']:.0f} calls.\n"
        )
    if charts:
        out.append(
            chart(
                "Cost of the first calls",
                [f"call {i + 1}" for i in range(len(warm["calls"]))]
                + ["steady state", "eager"],
                [t / 1e3 for t in warm["calls"]]
                + [warm["steady"] / 1e3, warm["eager"] / 1e3],
                "time per call (us)",
            )
        )

    grad = results["autograd"]
    out.append(
        "### With a gradient (chain of 9 ops, 64 elements, one input requires grad)\n"
    )
    out.append("| | eager | cfg | speedup |")
    out.append("|---|---:|---:|---:|")
    for key, label in (
        ("forward", "forward"),
        ("forward_backward", "forward + backward"),
    ):
        ns = grad[key]
        out.append(
            f"| {label} | {ns['eager']:,.0f} | {ns['unboxed']:,.0f} | {ns['eager'] / ns['unboxed']:.2f}x |"
        )
    out.append("")

    out.append("### Shape-dependent control flow\n")
    out.append(
        "| pipeline | batch 1: eager | cfg | batch 8: eager | cfg | profile runs per block | versions per block |"
    )
    out.append("|---|---:|---:|---:|---:|---|---|")
    for label, row in results["narrowing"].items():
        ns = row["ns"]
        one, eight = ns.get(1, ns.get("1")), ns.get(8, ns.get("8"))
        out.append(
            f"| {label} | {one['eager']:,.0f} | {one['cfg']:,.0f} | {eight['eager']:,.0f} | {eight['cfg']:,.0f} | "
            f"{row['profile_runs']} | {row['versions']} |"
        )
    out.append("")

    out.append("### Python objects created for kernel results\n")
    out.append("| workload | kernel results per call | of which got a Python object |")
    out.append("|---|---:|---:|")
    for name, row in results["unboxing"].items():
        out.append(
            f"| {name} | {row['kernel_results_per_call']:.0f} | {row['python_objects_per_call']:.0f} |"
        )
    out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 64, 1024])
    parser.add_argument("--json", help="also write the raw results to this file")
    parser.add_argument(
        "--charts", action="store_true", help="add Mermaid charts to the tables"
    )
    args = parser.parse_args()

    if not torch.cfg_compile.is_available():
        raise SystemExit(
            "this build has no cfg_compile runtime; build with USE_CFG_COMPILE=1"
        )
    torch.set_num_threads(args.threads)
    with torch.no_grad():
        results = {
            "meta": {
                "device": args.device,
                "threads": args.threads,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "machine": platform.processor() or platform.machine(),
            },
            "workloads": bench_workloads(args.sizes, args.device),
            "ops": bench_ops(args.device),
            "warmup": bench_warmup(args.device),
            "narrowing": bench_narrowing(args.device),
            "unboxing": bench_unboxing(args.device),
        }
    results["autograd"] = bench_autograd(args.device)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    print(report(results, args.charts))


if __name__ == "__main__":
    main()
