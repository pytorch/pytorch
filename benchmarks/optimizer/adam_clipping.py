"""Compare fused CUDA Adam/AdamW clipping against clip_grad_norm_ followed by Adam/AdamW.

Run with a locally built PyTorch:
    python benchmarks/optimizer/adam_clipping.py --output results.json
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def measure(fn, iterations, restore):
    events = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        restore()
        start.record()
        fn()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    return statistics.mean(start.elapsed_time(end) for start, end in events)


def benchmark(total_numel, tensor_count, optimizer_name, args):
    started = datetime.now(timezone.utc).isoformat()
    per_tensor = total_numel // tensor_count
    if total_numel % tensor_count:
        raise ValueError("numel must be divisible by tensor_count")
    params = [
        torch.nn.Parameter(torch.zeros(per_tensor, device="cuda"))
        for _ in range(tensor_count)
    ]
    grads = [torch.randn_like(p) for p in params]
    for p, g in zip(params, grads):
        p.grad = g
    optimizer_cls = getattr(torch.optim, optimizer_name)
    kwargs = dict(
        lr=1e-3, fused=True, capturable=True, weight_decay=0.01, amsgrad=args.amsgrad
    )
    clipped = optimizer_cls(params, max_grad_norm=args.max_norm, **kwargs)
    stock = optimizer_cls(params, **kwargs)
    saved_grads = [g.clone() for g in grads]
    saved_params = [p.detach().clone() for p in params]
    # Initialize optimizer state before capture and timing.
    stock.step()
    clipped.step()

    @torch.no_grad()
    def restore(optimizer):
        for p, g, saved_p, saved_g in zip(params, grads, saved_params, saved_grads):
            p.copy_(saved_p)
            g.copy_(saved_g)
            state = optimizer.state[p]
            state["exp_avg"].fill_(0.01)
            state["exp_avg_sq"].fill_(0.001)
            state["step"].fill_(100)
            if args.amsgrad:
                state["max_exp_avg_sq"].fill_(0.001)

    def stock_step():
        torch.nn.utils.clip_grad_norm_(params, args.max_norm)
        stock.step()

    def clipped_step():
        clipped.step()

    with torch.no_grad():
        restore(stock)
        stock_step()
        expected_params = [p.detach().clone() for p in params]
        expected_state = [
            {k: v.clone() for k, v in stock.state[p].items()} for p in params
        ]
        restore(clipped)
        clipped_step()
        for p, ref, g, saved, state in zip(
            params, expected_params, grads, saved_grads, expected_state
        ):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
            torch.testing.assert_close(g, saved, rtol=0, atol=0)
            for key, value in state.items():
                torch.testing.assert_close(clipped.state[p][key], value, rtol=0, atol=0)
    del expected_params, expected_state

    paths = {"stock": (stock_step, stock), "clipped": (clipped_step, clipped)}
    for fn, optimizer in paths.values():
        for _ in range(args.warmup):
            restore(optimizer)
            fn()
    graphs = {}
    for name, (fn, optimizer) in paths.items():
        restore(optimizer)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    timings = {name: [] for name in graphs}
    for rep in range(args.repeats):
        names = ["stock", "clipped"] if rep % 2 == 0 else ["clipped", "stock"]
        for name in names:
            optimizer = paths[name][1]
            timings[name].append(
                measure(
                    graphs[name].replay, args.iterations, lambda: restore(optimizer)
                )
            )
    stock_ms = statistics.median(timings["stock"])
    clipped_ms = statistics.median(timings["clipped"])
    row = {
        "started_at": started,
        "optimizer": optimizer_name,
        "amsgrad": args.amsgrad,
        "numel": total_numel,
        "tensor_count": tensor_count,
        "shape_per_tensor": [per_tensor],
        "dtype": "float32",
        "stock_ms": stock_ms,
        "clipped_ms": clipped_ms,
        "speedup": stock_ms / clipped_ms,
        "samples_ms": timings,
        "logical_gradient_bytes_saved_per_step": total_numel * 8,
        "timing": "CUDA events around each graph replay, parameter/gradient/state resets excluded; median of repeated means",
        "correctness": "bitwise parameter and optimizer-state equality; clipped path leaves gradients unchanged",
    }
    if args.trace:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for name, (fn, optimizer) in paths.items():
                restore(optimizer)
                torch.cuda.synchronize()
                with torch.profiler.record_function(name):
                    fn()
                    torch.cuda.synchronize()
        trace = (
            Path(args.trace)
            / f"{optimizer_name.lower()}_clip_{total_numel}_{tensor_count}.json"
        )
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        row["trace"] = str(trace.resolve())
    row["finished_at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, nargs="+", default=[2**20, 2**24, 2**28])
    parser.add_argument("--tensor-count", type=int, nargs="+", default=[1, 64])
    parser.add_argument(
        "--optimizer", nargs="+", choices=["Adam", "AdamW"], default=["Adam", "AdamW"]
    )
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("adam_clipping_results.json")
    )
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled local PyTorch build is required")
    if args.max_norm <= 0 or min(args.numel + args.tensor_count) <= 0:
        raise ValueError("max_norm, numel and tensor_count must be positive")
    if args.warmup < 0 or min(args.iterations, args.repeats) <= 0:
        raise ValueError(
            "warmup must be nonnegative; iterations and repeats must be positive"
        )
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    torch.manual_seed(0)
    started = datetime.now(timezone.utc).isoformat()
    results = [
        benchmark(n, count, optimizer, args)
        for optimizer in args.optimizer
        for n in args.numel
        for count in args.tensor_count
    ]
    report = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "repeats": args.repeats,
        "max_norm": args.max_norm,
        "optimizer_settings": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "fused": True,
            "capturable": True,
            "amsgrad": args.amsgrad,
        },
        "initial_state_per_replay": {
            "step": 100,
            "exp_avg": 0.01,
            "exp_avg_sq": 0.001,
            "max_exp_avg_sq_if_amsgrad": 0.001,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".patch").write_bytes(patch)


if __name__ == "__main__":
    main()
