"""TorchBench under eager, torch.compile and torch.cfg_compile(backend="inductor").

Every model of https://github.com/pytorch/benchmark that loads is run for
inference and for training, as benchmarks/dynamo/torchbench.py runs it: the
module and its example inputs come from the model's ``get_module()``;
inference is ``model(*inputs)`` in eval mode under ``no_grad``; training is
``zero_grad``, forward, ``reduce_to_scalar_loss``, ``backward`` and an SGD
step in train mode. The same function is given to both compilers. Each
(model, test, compiler) runs in a process of its own and reports:

- whether it ran, and how far its results are from eager's on the same inputs
  (the worst relative difference over the outputs of the calls before the
  timed ones; over the losses of the first steps for training);
- what the calls up to the steady state cost over the steady state itself,
  which is compile time plus, for cfg_compile, the calls it spends measuring
  compiled code against the interpreter; and the compile time alone;
- the median time of a call in the steady state.

    python benchmarks/cfg_compile/torchbench.py --torchbench ~/benchmark resnet18 hf_Bert
    python benchmarks/cfg_compile/torchbench.py --torchbench ~/benchmark \\
        --cpus 8-11/12-15 --results results.jsonl > report.md

Without model names, all models are tried; the ones that do not load in the
environment (missing data, missing packages) are listed and left out.
``--cpus`` gives one group of cores per worker; the three compilers of a model
run one after the other on the same group. ``--results`` appends each result
as a line of JSON and skips what the file already has.
"""

import argparse
import collections
import copy
import gc
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import warnings


MARKER = "TORCHBENCH "
MODES = ("eager", "compile", "cfg")


# ----------------------------------------------------------------------
# One model, one test, one compiler: runs in its own process
# ----------------------------------------------------------------------


def infer(model, inputs):
    return model(*inputs)


def infer_kw(model, inputs):
    return model(**inputs)


def train(model, opt, inputs):
    from torch._dynamo.testing import reduce_to_scalar_loss

    opt.zero_grad(True)
    pred = model(*inputs)
    loss = reduce_to_scalar_loss(pred)
    loss.backward()
    opt.step()
    return loss


def train_kw(model, opt, inputs):
    from torch._dynamo.testing import reduce_to_scalar_loss

    opt.zero_grad(True)
    pred = model(**inputs)
    loss = reduce_to_scalar_loss(pred)
    loss.backward()
    opt.step()
    return loss


def tensors_of(x):
    """The tensors of a result, in a fixed order."""
    import torch

    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, dict):
        return [t for k in x for t in tensors_of(x[k])]
    if isinstance(x, (tuple, list)):
        return [t for item in x for t in tensors_of(item)]
    if hasattr(x, "__dict__") and not callable(x):
        return [t for v in vars(x).values() for t in tensors_of(v)]
    return []


def difference(result, expected):
    """The worst difference between two results, relative to the largest
    value of each tensor; None if they do not even have the same shape."""
    import torch

    a, b = tensors_of(result), tensors_of(expected)
    if len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype:
            return None
        if not x.is_floating_point() and not x.is_complex():
            worst = max(worst, 0.0 if torch.equal(x, y) else float("inf"))
            continue
        x, y = x.detach().double(), y.detach().double()
        finite = torch.isfinite(y)
        if not torch.equal(torch.isfinite(x), finite):
            return float("inf")
        x, y = x[finite], y[finite]
        if x.numel():
            worst = max(
                worst, (x - y).abs().max().item() / (y.abs().max().item() or 1.0)
            )
    return worst


def run_one(args):
    """Runs one configuration in this process and prints its result as JSON."""
    warnings.simplefilter("ignore")
    sys.path.insert(0, args.torchbench)
    import torch

    torch.set_num_threads(args.threads)
    # As the accuracy runs of benchmarks/dynamo: random operations in compiled
    # code draw from the generator eager uses, so results can be compared.
    torch._inductor.config.fallback_random = True
    name, test, mode = args.one
    out = {"name": name, "test": test, "mode": mode}

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    try:
        from torchbenchmark import load_model_by_name

        torch.manual_seed(1337)
        bench = load_model_by_name(name)(test=test, device=args.device)
        model, inputs = bench.get_module()
        out["batch_size"] = getattr(bench, "batch_size", None)
        training = test == "train"
        model.train(training)
        keywords = isinstance(inputs, dict)
        if not keywords:
            inputs = tuple(inputs)
        fn = (
            (train_kw if keywords else train)
            if training
            else (infer_kw if keywords else infer)
        )

        def optimizer(m):
            return torch.optim.SGD(
                [p for p in m.parameters() if p.requires_grad], lr=0.01
            )

        def call(f, m, opt):
            torch.manual_seed(1337)
            if training:
                return f(m, opt, inputs)
            with torch.no_grad():
                return f(m, inputs)

        # The reference: the same calls in the interpreter, on a copy of the
        # model when the calls change it.
        reference = []
        if mode != "eager":
            ref_model = copy.deepcopy(model) if training else model
            ref_opt = optimizer(ref_model) if training else None
            start = time.perf_counter()
            for _ in range(args.compared if training else 1):
                r = call(fn, ref_model, ref_opt)
                reference.append(r.detach().clone() if training else r)
                if (
                    len(reference) >= 3
                    and time.perf_counter() - start > args.warmup_seconds
                ):
                    break
            del ref_model, ref_opt
            gc.collect()

        if mode == "eager":
            compiled = fn
        elif mode == "compile":
            compiled = torch.compile(fn)
        else:
            compiled = torch.cfg_compile(fn, backend="inductor", fallback=False)

        opt = optimizer(model) if training else None
        warm, worst, quiet = [], 0.0, 0.0
        for i in range(args.max_warmup):
            sync()
            start = time.perf_counter()
            r = call(compiled, model, opt)
            sync()
            warm.append(time.perf_counter() - start)
            # Seconds since the last call that stood out.
            quiet = quiet + warm[-1] if warm[-1] < 2.5 * min(warm) else 0.0
            # Inference gives the same result on every call, so every call is
            # compared: under cfg_compile some of them run in the interpreter.
            if reference and (i < len(reference) if training else True):
                d = difference(r, reference[i] if training else reference[0])
                worst = float("inf") if d is None else max(worst, d)
            # A JIT is warm when it has stopped changing its mind: cfg_compile
            # goes on measuring compiled code against the interpreter for a
            # while after it has compiled.
            measuring = mode == "cfg" and compiled.stats()["measuring"]
            enough = i + 1 >= args.warmup or (
                i + 1 >= 6 and sum(warm) > args.warmup_seconds
            )
            if enough and not (measuring and quiet < args.settle):
                break
        del r
        if reference:
            out["difference"] = worst
        times = []
        start = time.perf_counter()
        while len(times) < args.calls and (
            len(times) < 5 or time.perf_counter() - start < args.seconds
        ):
            sync()
            t = time.perf_counter()
            call(compiled, model, opt)
            sync()
            times.append(time.perf_counter() - t)
        times.sort()
        out["call_s"] = times[len(times) // 2]
        out["best_s"] = times[0]
        out["warmup_calls"] = len(warm)
        out["measuring"] = mode == "cfg" and bool(compiled.stats()["measuring"])
        out["warmup_s"] = max(0.0, sum(warm) - len(warm) * out["call_s"])
        if mode == "compile":
            from torch._dynamo.utils import compile_times, counters

            out["graphs"] = counters["stats"]["unique_graphs"]
            out["ops"] = counters["stats"]["calls_captured"]
            out["graph_breaks"] = sum(counters["graph_break"].values())
            out["why"] = [
                str(reason)[:160] for reason in list(counters["graph_break"])[:6]
            ]
            totals = compile_times(repr="csv", aggregate=True)
            names = totals[0]
            if "_compile.compile_inner" in names:
                out["compile_s"] = float(
                    totals[1][names.index("_compile.compile_inner")]
                )
        elif mode == "cfg":
            regions = torch.cfg_compile.regions(compiled)
            hot = [r for r in regions if r["state"] == "compiled"]
            out["graphs"] = sum(r["variants"] for r in hot)
            out["ops"] = sum(r["ops"] for r in hot)
            out["compile_s"] = sum(r["seconds"] for r in hot)
            # Why the VM kept an instruction, where that says something about
            # the model rather than about plain Python.
            dull = ("no tensor or module involved", "not worth a graph")
            out["why"] = [
                f"{r['where']}: {r['reason']}"[:160]
                for r in regions
                if r["state"] == "vm" and not any(d in r["reason"] for d in dull)
            ][:6]
            out["demoted"] = bool(compiled.stats()["demoted"])
        out["ok"] = True
    except BaseException as e:
        text = str(e).strip().splitlines()
        out["ok"] = False
        out["error"] = (
            f"{type(e).__name__}: {text[0][:300]}" if text else type(e).__name__
        )
    print(MARKER + json.dumps(out))


# ----------------------------------------------------------------------
# All of them
# ----------------------------------------------------------------------


def run_config(args, name, test, mode, cpus):
    cmd = [sys.executable, os.path.abspath(__file__), "--torchbench", args.torchbench]
    cmd += [
        "--one",
        name,
        test,
        mode,
        "--device",
        args.device,
        "--threads",
        str(args.threads),
    ]
    for option in (
        "warmup",
        "warmup_seconds",
        "max_warmup",
        "settle",
        "calls",
        "seconds",
        "compared",
    ):
        cmd += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = dict(
        os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(args.threads)
    )
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "test": test,
            "mode": mode,
            "ok": False,
            "error": "timed out",
        }
    for line in reversed(r.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    tail = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    error = f"crashed (exit {r.returncode}): {tail}"
    return {"name": name, "test": test, "mode": mode, "ok": False, "error": error}


def run_all(args, names):
    results = {}
    if args.results and os.path.exists(args.results):
        with open(args.results) as f:
            for line in f:
                r = json.loads(line)
                results[(r["name"], r["test"], r["mode"])] = r
    jobs = queue.Queue()
    for test in args.tests.split(","):
        for name in names:
            jobs.put((name, test))
    lock = threading.Lock()

    def worker(cpus):
        while True:
            try:
                name, test = jobs.get_nowait()
            except queue.Empty:
                return
            for mode in MODES:
                if (name, test, mode) not in results:
                    r = run_config(args, name, test, mode, cpus)
                    with lock:
                        results[(name, test, mode)] = r
                        if args.results:
                            with open(args.results, "a") as f:
                                f.write(json.dumps(r) + "\n")
                        state = (
                            f"{r['call_s'] * 1e3:.1f} ms"
                            if r["ok"]
                            else r["error"][:100]
                        )
                        print(
                            f"{name} {test} {mode}: {state}",
                            file=sys.stderr,
                            flush=True,
                        )
                if mode == "eager" and not results[(name, test, mode)]["ok"]:
                    break  # nothing to compare with

    groups = args.cpus.split("/") if args.cpus else [None]
    threads = [threading.Thread(target=worker, args=(g,)) for g in groups]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def geomean(values):
    return (
        math.exp(sum(math.log(v) for v in values) / len(values))
        if values
        else float("nan")
    )


def report(args, results):
    by = collections.defaultdict(dict)
    for (name, test, mode), r in results.items():
        by[(test, name)][mode] = r
    for test in args.tests.split(","):
        rows = sorted((name, d) for (t, name), d in by.items() if t == test)
        ran = [(n, d) for n, d in rows if d["eager"]["ok"]]
        print(f"\n### {test}: {len(ran)} models\n")
        print(
            "| model | eager ms | torch.compile ms | cfg_compile ms | speedup: torch.compile | cfg_compile "
            "| compile s: torch.compile | cfg_compile | to steady state s: torch.compile | cfg_compile "
            "| graphs: torch.compile | cfg_compile | difference: torch.compile | cfg_compile |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        speed = {"compile": [], "cfg": []}
        compile_s = {"compile": [], "cfg": []}
        warmup_s = {"compile": [], "cfg": []}
        passed = {"compile": 0, "cfg": 0}
        accurate = {"compile": 0, "cfg": 0}
        notes = []
        for name, d in ran:
            eager = d["eager"]["call_s"]
            cells = {}
            for mode in ("compile", "cfg"):
                r = d.get(mode)
                if r is None or not r["ok"]:
                    cells[mode] = None
                    notes.append(
                        f"- `{name}` {mode}: {(r or {}).get('error', 'not run')}"
                    )
                    continue
                passed[mode] += 1
                accurate[mode] += r.get("difference", 0.0) <= args.tolerance
                cells[mode] = r
                for why in r.get("why", [])[:2]:
                    notes.append(f"- `{name}` {mode}: {why}")
            if all(cells.values()):
                for mode in ("compile", "cfg"):
                    speed[mode].append(eager / cells[mode]["call_s"])
                    compile_s[mode].append(max(cells[mode].get("compile_s", 0.0), 0.05))
                    warmup_s[mode].append(max(cells[mode]["warmup_s"], 0.05))

            def column(text):
                return [
                    "-" if cells[m] is None else text(cells[m])
                    for m in ("compile", "cfg")
                ]

            line = [
                name,
                f"{eager * 1e3:.1f}",
                *column(lambda r: f"{r['call_s'] * 1e3:.1f}"),
                *column(
                    lambda r: f"{eager / r['call_s']:.2f}x"
                    + (" (demoted)" if r.get("demoted") else "")
                ),
                *column(lambda r: f"{r.get('compile_s', 0.0):.1f}"),
                *column(lambda r: f"{r['warmup_s']:.1f}"),
                *column(lambda r: str(r.get("graphs", "-"))),
                *column(lambda r: f"{r.get('difference', 0.0):.0e}"),
            ]
            print("| " + " | ".join(line) + " |")
        n = len(speed["cfg"])
        print(
            f"\nRan and gave a result: torch.compile {passed['compile']}, cfg_compile {passed['cfg']} of {len(ran)}; "
            f"within {args.tolerance:g} of eager: {accurate['compile']} and {accurate['cfg']}. "
            f"Over the {n} models both ran: geometric mean speedup over eager {geomean(speed['compile']):.2f}x "
            f"and {geomean(speed['cfg']):.2f}x; compile time {geomean(compile_s['compile']):.1f} s and "
            f"{geomean(compile_s['cfg']):.1f} s (sum {sum(compile_s['compile']):.0f} s and {sum(compile_s['cfg']):.0f} s); "
            f"time to the steady state over the steady state itself {geomean(warmup_s['compile']):.1f} s and "
            f"{geomean(warmup_s['cfg']):.1f} s."
        )
        failed = [(n, d["eager"]["error"]) for n, d in rows if not d["eager"]["ok"]]
        if failed:
            print(
                f"\nNot run, eager fails in this environment ({len(failed)}): "
                + ", ".join(n for n, _ in failed)
                + "."
            )
        if notes:
            print("\nWhat was not one graph, and why (first reasons per model):\n")
            print("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--torchbench", required=True, help="a checkout of pytorch/benchmark"
    )
    parser.add_argument("--tests", default="eval,train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--cpus", help="core groups for taskset, one worker each: 8-11/12-15"
    )
    parser.add_argument(
        "--timeout", type=int, default=2400, help="seconds per configuration"
    )
    parser.add_argument("--warmup", type=int, default=12, help="calls before timing")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=120.0,
        help="or this long, and six calls",
    )
    parser.add_argument("--max-warmup", type=int, default=80)
    parser.add_argument(
        "--settle",
        type=float,
        default=300.0,
        help="seconds to wait for cfg_compile to stop measuring",
    )
    parser.add_argument("--calls", type=int, default=20, help="timed calls, at most")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="of timed calls, at most"
    )
    parser.add_argument(
        "--compared", type=int, default=8, help="training steps compared with eager"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="relative difference that counts as the same",
    )
    parser.add_argument(
        "--results", help="file of JSON lines to append to and resume from"
    )
    parser.add_argument("--one", nargs=3, help=argparse.SUPPRESS)
    parser.add_argument("models", nargs="*")
    args = parser.parse_args()
    args.torchbench = os.path.abspath(os.path.expanduser(args.torchbench))
    if args.one:
        run_one(args)
        return
    names = args.models
    if not names:
        models = os.path.join(args.torchbench, "torchbenchmark", "models")
        names = sorted(
            n
            for n in os.listdir(models)
            if os.path.isdir(os.path.join(models, n)) and n[0] != "_"
        )
    report(args, run_all(args, names))


if __name__ == "__main__":
    main()
