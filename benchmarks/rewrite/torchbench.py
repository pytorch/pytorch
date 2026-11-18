"""TorchBench under eager, torch.compile and torch.rewrite(backend="inductor").

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
  which is compile time plus, for torch.rewrite, the calls it spends measuring
  compiled code against the interpreter; and the compile time alone;
- the median time of a call in the steady state;
- for torch.rewrite, the same with torch.compile in the same process, the calls
  of the two taking turns (``paired``): on a machine that is not idle, two
  processes one after the other can differ by more than the compilers do.

    python benchmarks/rewrite/torchbench.py --torchbench ~/benchmark resnet18 hf_Bert
    python benchmarks/rewrite/torchbench.py --torchbench ~/benchmark \\
        --cpus 8-11/12-15 --results results.jsonl > report.md

Without model names, all models are tried; the ones that do not load in the
environment (missing data, missing packages) are listed and left out.
``--cpus`` gives one group of cores per worker; the three compilers of a model
run one after the other on the same group. ``--results`` appends each result
as a line of JSON and skips what the file already has. Every process gets an
empty directory for Inductor's caches: with a shared one, the compiler that
runs second finds the kernels the first one built, and both find those of
earlier runs.

The workers together must fit the memory of the machine, or of the container
the script runs in: a training step of a large model takes tens of gigabytes,
and a container that goes over its limit is killed as a whole. No process is
started while more than ``--memory-start`` of that memory is in use, and
above ``--memory-stop`` the process that holds the most is stopped and run
again later, alone. Every result has the most memory its process held
(``peak_rss_mb``; for torch.rewrite before torch.compile joins it for the
paired calls, and ``process_peak_rss_mb`` with them).
"""

import argparse
import collections
import copy
import gc
import json
import math
import os
import queue
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings


MARKER = "TORCHBENCH "
MODES = ("eager", "compile", "cfg")


# ----------------------------------------------------------------------
# One model, one test, one compiler: runs in its own process
# ----------------------------------------------------------------------


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


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
        # A model that was built without giving its weights values
        # (cm3leon_generate) has them in memory that was never written. To
        # the kernel that is one page of zeros, however large the weights
        # are, and the model computes faster than one with weights can. So
        # every tensor is written once, with what it holds.
        with torch.no_grad():
            for tensor in [*model.parameters(), *model.buffers()]:
                try:
                    tensor.copy_(tensor.clone())
                except RuntimeError:
                    # One whose elements share memory: it was written.
                    pass
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
            compiled = torch.rewrite(fn, backend="inductor", fallback=False)

        def compiled_so_far():
            # Changes whenever the compiler compiled something.
            if mode == "compile":
                from torch._dynamo.utils import counters

                return sum(counters["frames"].values())
            if mode == "cfg":
                regions = torch.rewrite.regions(compiled)
                return len(regions), sum(r.get("variants", 0) for r in regions)
            return None

        opt = optimizer(model) if training else None
        warm, worst, quiet, calm, before = [], 0.0, 0.0, 0, None
        for i in range(args.max_warmup):
            sync()
            start = time.perf_counter()
            r = call(compiled, model, opt)
            sync()
            warm.append(time.perf_counter() - start)
            # Seconds since the last call that stood out.
            quiet = quiet + warm[-1] if warm[-1] < 2.5 * min(warm) else 0.0
            # Inference gives the same result on every call, so every call is
            # compared: under torch.rewrite some of them run in the interpreter.
            if reference and (i < len(reference) if training else True):
                d = difference(r, reference[i] if training else reference[0])
                worst = float("inf") if d is None else max(worst, d)
            # A JIT is warm when it has stopped compiling and changing its
            # mind: a region gets another graph for inputs it has not seen,
            # torch.compile recompiles, and torch.rewrite goes on measuring
            # compiled code against the interpreter for a while after it has
            # compiled, the function and what it calls.
            measuring = mode == "cfg" and compiled.stats()["measuring"]
            now = compiled_so_far()
            calm = 0 if measuring or now != before else calm + 1
            before = now
            enough = i + 1 >= args.warmup or (
                i + 1 >= 6 and sum(warm) > args.warmup_seconds
            )
            if enough and (calm >= 3 or quiet >= args.settle):
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
            regions = torch.rewrite.regions(compiled)
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
            # Before torch.compile and its copy of the model share the process.
            out["peak_rss_mb"] = peak_rss_mb()
            if args.paired and out["call_s"] <= args.paired_limit:
                try:
                    # A copy of its own, also where calls do not change the
                    # model: what Inductor makes asks for writable pointers
                    # to the weights, which moves the ones torch.rewrite has
                    # built into a graph and watches (``freeze=``).
                    other_model = copy.deepcopy(model)
                    other_opt = optimizer(other_model) if training else None
                    other = torch.compile(fn)
                    settle = []
                    while len(settle) < 12 and (
                        len(settle) < 3 or max(settle[-2:]) > 1.5 * min(settle)
                    ):
                        t = time.perf_counter()
                        call(other, other_model, other_opt)
                        settle.append(time.perf_counter() - t)
                    turns = {"compile": [], "cfg": []}
                    sides = {
                        "compile": (other, other_model, other_opt),
                        "cfg": (compiled, model, opt),
                    }
                    start = time.perf_counter()
                    while len(turns["cfg"]) < args.paired and (
                        len(turns["cfg"]) < 3
                        or time.perf_counter() - start < 2 * args.seconds
                    ):
                        order = ("compile", "cfg", "compile")
                        for side in order[len(turns["cfg"]) % 2 :][:2]:
                            sync()
                            t = time.perf_counter()
                            call(*sides[side])
                            sync()
                            turns[side].append(time.perf_counter() - t)
                    for side, times in turns.items():
                        times.sort()
                        out[f"paired_{side}_s"] = times[len(times) // 2]
                        out[f"paired_{side}_best_s"] = times[0]
                    out["paired_turns"] = len(turns["cfg"])
                except Exception as e:
                    out["paired_error"] = f"{type(e).__name__}: {e}"[:200]
        out["ok"] = True
    except BaseException as e:
        text = str(e).strip().splitlines()
        out["ok"] = False
        out["error"] = (
            f"{type(e).__name__}: {text[0][:300]}" if text else type(e).__name__
        )
    out["process_peak_rss_mb"] = peak_rss_mb()
    out.setdefault("peak_rss_mb", out["process_peak_rss_mb"])
    print(MARKER + json.dumps(out))


# ----------------------------------------------------------------------
# All of them
# ----------------------------------------------------------------------


def memory_in_use():
    """The part of its memory that the container, or the machine, has in use
    and cannot drop: what counts against its limit, without the page cache."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                used = int(f.read())
            with open("/sys/fs/cgroup/memory.stat") as f:
                stat = dict(line.split() for line in f)
            used -= int(stat.get("inactive_file", 0)) + int(stat.get("active_file", 0))
            return used / int(limit)
    except (OSError, ValueError):
        pass
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return 1 - info["MemAvailable"] / info["MemTotal"]


def resident(group):
    """What the processes of a process group hold, in bytes."""
    total = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                # After the name, which may have spaces: state, parent, group
                # and, 21 fields on, the resident pages.
                fields = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == group:
            total += int(fields[21]) * resource.getpagesize()
    return total


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Memory:
    """Keeps the processes of a run within the memory there is."""

    def __init__(self, start, stop):
        self.start, self.stop = start, stop
        self.lock = threading.Lock()
        self.running = {}
        self.starting = 0
        self.stopped = set()
        # The process that is being stopped, until it is gone.
        self.leaving = None
        self.last = 0.0

    def admit(self):
        """Waits until a process may start: not while much is in use, and
        not right after another, which has yet to take what it takes; unless
        nothing else runs."""
        while True:
            with self.lock:
                settled = time.time() - self.last > 30
                idle = not self.running and not self.starting
                if idle or (settled and memory_in_use() < self.start):
                    self.last = time.time()
                    self.starting += 1
                    return
            time.sleep(2)

    def watch(self, process):
        """Waits for a process; True if it had to be stopped for memory. It
        is the process that holds the most that is stopped, whichever worker
        notices, and one at a time."""
        with self.lock:
            self.starting -= 1
            self.running[process.pid] = process
        try:
            while process.poll() is None:
                time.sleep(1)
                with self.lock:
                    if self.leaving is not None or memory_in_use() < self.stop:
                        continue
                    largest = max(self.running.values(), key=lambda p: resident(p.pid))
                    self.stopped.add(largest.pid)
                    self.leaving = largest.pid
                    stop(largest)
            return process.pid in self.stopped
        finally:
            with self.lock:
                self.running.pop(process.pid, None)
                if self.leaving == process.pid:
                    self.leaving = None


def run_config(args, name, test, mode, cpus, memory):
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
        "paired",
        "paired_limit",
    ):
        cmd += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = dict(
        os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(args.threads)
    )
    failed = {"name": name, "test": test, "mode": mode, "ok": False}
    memory.admit()
    with tempfile.TemporaryDirectory(prefix="cfg_torchbench_") as caches:
        env["TORCHINDUCTOR_CACHE_DIR"] = caches
        # Output goes to files: nothing reads a pipe while the process runs.
        with (
            open(os.path.join(caches, "out"), "w+") as out,
            open(os.path.join(caches, "err"), "w+") as err,
        ):
            process = subprocess.Popen(
                cmd, stdout=out, stderr=err, env=env, start_new_session=True
            )
            timer = threading.Timer(args.timeout, stop, [process])
            timer.start()
            try:
                if memory.watch(process):
                    return None
            finally:
                expired = not timer.is_alive()
                timer.cancel()
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(), err.read()
    if expired:
        return dict(failed, error="timed out")
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    tail = (stderr.strip().splitlines() or ["?"])[-1][:200]
    return dict(failed, error=f"crashed (exit {process.returncode}): {tail}")


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
    memory = Memory(args.memory_start, args.memory_stop)
    # What was stopped for memory: run again when the workers are through.
    later = queue.Queue()

    def worker(cpus, jobs=jobs, alone=False):
        while True:
            try:
                name, test = jobs.get_nowait()
            except queue.Empty:
                return
            for mode in MODES:
                if (name, test, mode) not in results:
                    r = run_config(args, name, test, mode, cpus, memory)
                    if r is None and not alone:
                        print(
                            f"{name} {test} {mode}: stopped for memory, runs later",
                            file=sys.stderr,
                            flush=True,
                        )
                        later.put((name, test))
                        break
                    if r is None:
                        r = {"name": name, "test": test, "mode": mode, "ok": False}
                        r["error"] = "takes more memory than there is"
                    with lock:
                        results[(name, test, mode)] = r
                        if args.results:
                            with open(args.results, "a") as f:
                                f.write(json.dumps(r) + "\n")
                        state = (
                            f"{r['call_s'] * 1e3:.1f} ms, {r.get('peak_rss_mb', 0)} MB"
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
    worker(groups[0], later, alone=True)
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
            "| model | eager ms | torch.compile ms | torch.rewrite ms | speedup: torch.compile | torch.rewrite "
            "| torch.rewrite over torch.compile, taking turns "
            "| compile s: torch.compile | torch.rewrite | to steady state s: torch.compile | torch.rewrite "
            "| graphs: torch.compile | torch.rewrite | difference: torch.compile | torch.rewrite |"
        )
        print(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        speed = {"compile": [], "cfg": []}
        paired = []
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

            turns = "-"
            if cells["cfg"] and "paired_cfg_s" in cells["cfg"]:
                ratio = cells["cfg"]["paired_cfg_s"] / cells["cfg"]["paired_compile_s"]
                paired.append(ratio)
                turns = f"{ratio:.2f}"
            line = [
                name,
                f"{eager * 1e3:.1f}",
                *column(lambda r: f"{r['call_s'] * 1e3:.1f}"),
                *column(
                    lambda r: f"{eager / r['call_s']:.2f}x"
                    + (" (demoted)" if r.get("demoted") else "")
                ),
                turns,
                *column(lambda r: f"{r.get('compile_s', 0.0):.1f}"),
                *column(lambda r: f"{r['warmup_s']:.1f}"),
                *column(lambda r: str(r.get("graphs", "-"))),
                *column(lambda r: f"{r.get('difference', 0.0):.0e}"),
            ]
            print("| " + " | ".join(line) + " |")
        n = len(speed["cfg"])
        print(
            f"\nRan and gave a result: torch.compile {passed['compile']}, torch.rewrite {passed['cfg']} of {len(ran)}; "
            f"within {args.tolerance:g} of eager: {accurate['compile']} and {accurate['cfg']}. "
            f"Over the {n} models both ran: geometric mean speedup over eager {geomean(speed['compile']):.2f}x "
            f"and {geomean(speed['cfg']):.2f}x; compile time {geomean(compile_s['compile']):.1f} s and "
            f"{geomean(compile_s['cfg']):.1f} s (sum {sum(compile_s['compile']):.0f} s and {sum(compile_s['cfg']):.0f} s); "
            f"time to the steady state over the steady state itself {geomean(warmup_s['compile']):.1f} s and "
            f"{geomean(warmup_s['cfg']):.1f} s. With the calls of the two taking turns in one process, "
            f"on {len(paired)} models: the time of torch.rewrite over that of torch.compile is {geomean(paired):.3f} "
            f"(geometric mean), above 1.03 on {sum(r > 1.03 for r in paired)} and below 0.97 on "
            f"{sum(r < 0.97 for r in paired)}."
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
    parser.add_argument("--max-warmup", type=int, default=400)
    parser.add_argument(
        "--settle",
        type=float,
        default=1500.0,
        help="seconds to wait for torch.rewrite to stop measuring",
    )
    parser.add_argument("--calls", type=int, default=20, help="timed calls, at most")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="of timed calls, at most"
    )
    parser.add_argument(
        "--compared", type=int, default=8, help="training steps compared with eager"
    )
    parser.add_argument(
        "--paired",
        type=int,
        default=10,
        help="calls of each compiler when the two take turns in one process; 0 for none",
    )
    parser.add_argument(
        "--paired-limit",
        type=float,
        default=40.0,
        help="seconds a call may take for the two to take turns",
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
    parser.add_argument(
        "--memory-start",
        type=float,
        default=0.5,
        help="start no process while more than this part of memory is in use",
    )
    parser.add_argument(
        "--memory-stop",
        type=float,
        default=0.75,
        help="above this part, stop the largest process and run it later, alone",
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
