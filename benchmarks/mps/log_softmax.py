"""Run from the repo root with ./venv/bin/python benchmarks/mps/log_softmax.py."""

import argparse
import gc
import math
import statistics
import time

from softmax_common import add_arguments, operators, save_results, selected_cases

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Measure MPS softmax and log_softmax latency."
    )
    add_arguments(parser)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Calls per trial; 0 chooses 1-300 calls from a pilot.",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=200,
        help="Target duration for an adaptive trial.",
    )
    args = parser.parse_args()
    if args.trials < 2 or args.warmup < 1 or args.steps < 0 or args.target_ms <= 0:
        parser.error("Need trials >= 2, warmup >= 1, steps >= 0, and target-ms > 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"Build: {torch.version.git_version}; dtype: {args.dtype}", flush=True)
    print(
        "MPS event intervals include queue gaps; wall time includes batch submission and synchronization."
    )
    print("Inputs and outputs are preallocated. Trial 1 is excluded from summaries.")
    records = []
    for case, shape, dim in selected_cases(args):
        torch.manual_seed(case)
        x = torch.randn(shape, device="mps", dtype=getattr(torch, args.dtype))
        out = torch.empty_like(x)
        steps = {}
        samples = {name: [] for name in operators(args)}
        for name in samples:
            op = getattr(torch, name)
            for _ in range(args.warmup):
                op(x, dim=dim, out=out)
            start = torch.mps.Event(enable_timing=True)
            end = torch.mps.Event(enable_timing=True)
            start.record()
            op(x, dim=dim, out=out)
            end.record()
            torch.mps.synchronize()
            pilot_ms = max(start.elapsed_time(end), 0.001)
            steps[name] = args.steps or max(
                1, min(300, math.ceil(args.target_ms / pilot_ms))
            )

        for trial in range(args.trials):
            names = list(samples)
            if trial % 2:
                names.reverse()
            for name in names:
                op = getattr(torch, name)
                start = torch.mps.Event(enable_timing=True)
                end = torch.mps.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(steps[name]):
                    op(x, dim=dim, out=out)
                end.record()
                torch.mps.synchronize()
                wall_us = (time.perf_counter() - wall_start) * 1e6 / steps[name]
                event_us = start.elapsed_time(end) * 1000 / steps[name]
                samples[name].append({"event_us": event_us, "wall_us": wall_us})

        for name, trials in samples.items():
            event_us = [sample["event_us"] for sample in trials[1:]]
            record = {
                "case": case,
                "op": name,
                "shape": shape,
                "dim": dim,
                "steps": steps[name],
                "trials": trials,
                "median_us": statistics.median(event_us),
                "min_us": min(event_us),
                "max_us": max(event_us),
            }
            records.append(record)
            print(
                f"{case:2d} {name:11s} {str(shape):24s} dim={dim} median={record['median_us']:.3f} us",
                flush=True,
            )
        del x, out
        gc.collect()
        torch.mps.empty_cache()
        save_results(args, records)


if __name__ == "__main__":
    main()
