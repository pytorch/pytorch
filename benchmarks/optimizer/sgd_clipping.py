"""Compare fused CUDA SGD clipping against clip_grad_norm_ followed by SGD.

Run with a locally built PyTorch:
    python benchmarks/optimizer/sgd_clipping.py --output results.json
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


def benchmark(total_numel, tensor_count, args):
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
    clipped = torch.optim.SGD(params, lr=1e-3, fused=True, max_grad_norm=args.max_norm)
    stock = torch.optim.SGD(params, lr=1e-3, fused=True)
    # Reset gradient values outside timing so both paths see the same input norm.
    saved_grads = [g.clone() for g in grads]

    def restore():
        for g, saved in zip(grads, saved_grads):
            g.copy_(saved)

    def stock_step():
        torch.nn.utils.clip_grad_norm_(params, args.max_norm)
        stock.step()

    def clipped_step():
        clipped.step()

    with torch.no_grad():
        stock_step()
        expected = [p.detach().clone() for p in params]
        for p in params:
            p.zero_()
        restore()
        clipped_step()
        for p, ref, g, saved in zip(params, expected, grads, saved_grads):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
            torch.testing.assert_close(g, saved, rtol=0, atol=0)
    del expected

    # Warmup scales the stock gradients once; restore before graph capture.
    for fn in (stock_step, clipped_step):
        restore()
        for _ in range(args.warmup):
            fn()
    graphs = {}
    for name, fn in (("stock", stock_step), ("clipped", clipped_step)):
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    timings = {name: [] for name in graphs}
    for rep in range(args.repeats):
        names = ["stock", "clipped"] if rep % 2 == 0 else ["clipped", "stock"]
        for name in names:
            restore()
            timings[name].append(measure(graphs[name].replay, args.iterations, restore))
    stock_ms = statistics.median(timings["stock"])
    clipped_ms = statistics.median(timings["clipped"])
    row = {
        "started_at": started,
        "numel": total_numel,
        "tensor_count": tensor_count,
        "shape_per_tensor": [per_tensor],
        "dtype": "float32",
        "stock_ms": stock_ms,
        "clipped_ms": clipped_ms,
        "speedup": stock_ms / clipped_ms,
        "samples_ms": timings,
        "logical_gradient_bytes_saved_per_step": total_numel * 8,
        "timing": "CUDA events around each graph replay, resets excluded; median of repeated means",
        "correctness": "bitwise parameter equality; clipped path leaves gradients unchanged",
    }
    if args.trace:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for name, fn in (("stock", stock_step), ("clipped", clipped_step)):
                restore()
                with torch.profiler.record_function(name):
                    fn()
                    torch.cuda.synchronize()
        trace = Path(args.trace) / f"sgd_clip_{total_numel}_{tensor_count}.json"
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
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("sgd_clipping_results.json")
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
        benchmark(n, count, args) for n in args.numel for count in args.tensor_count
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
        "optimizer": "SGD, lr=0.001, momentum=0, weight_decay=0, fused=True",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".patch").write_bytes(patch)


if __name__ == "__main__":
    main()
