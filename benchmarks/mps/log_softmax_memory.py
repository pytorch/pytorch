"""Run from the repo root with ./venv/bin/python benchmarks/mps/log_softmax_memory.py."""

import argparse
import gc
import math

from softmax_common import add_arguments, operators, save_results, selected_cases

import torch


MIB = 2**20


def snapshot():
    stats = torch.accelerator.memory_stats("mps")
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    driver = torch.mps.driver_allocated_memory()
    return {
        "allocated": allocated,
        "reserved": reserved,
        "driver": driver,
        "other": driver - reserved,
        "peak_allocated": stats.get("allocated_bytes.all.peak", 0),
        "peak_reserved": stats.get("reserved_bytes.all.peak", 0),
    }


def probe(op, shape, dim, dtype):
    before = snapshot()
    x = torch.randn(shape, device="mps", dtype=dtype)
    torch.mps.synchronize()
    torch.accelerator.reset_peak_memory_stats("mps")
    y = op(x, dim=dim)
    torch.mps.synchronize()
    live = snapshot()
    del x, y
    gc.collect()
    torch.mps.synchronize()
    deleted = snapshot()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return {"before": before, "live": live, "deleted": deleted, "clean": snapshot()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Measure peak allocator and retained MPS softmax memory."
    )
    add_arguments(parser)
    parser.add_argument("--suite", choices=("models", "sweep", "all"), default="all")
    parser.add_argument("--show-stages", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"Build: {torch.version.git_version}; dtype: {args.dtype}", flush=True)
    print(
        "Peak bytes cover the PyTorch allocator. Driver bytes are snapshots, not driver peaks."
    )
    print(
        "For isolated operator comparisons, run --op softmax and --op log_softmax in fresh processes."
    )
    cases = []
    if args.suite in ("models", "all"):
        cases.extend(
            ("models", case, shape, dim) for case, shape, dim in selected_cases(args)
        )
    if args.suite in ("sweep", "all"):
        cases.append(("warmup", 0, (1024, 8192), -1))
        for phase in ("first", "repeated"):
            cases.extend(
                (phase, i, (tokens, 8192), -1)
                for i, tokens in enumerate(range(1024, 3325, 100))
            )
        cases.append(("large", 0, (3812, 32000), -1))

    records = []
    dtype = getattr(torch, args.dtype)
    for name in operators(args):
        for phase, case, shape, dim in cases:
            torch.manual_seed(case)
            stages = probe(getattr(torch, name), shape, dim, dtype)
            record = {
                "op": name,
                "phase": phase,
                "case": case,
                "shape": shape,
                "dim": dim,
                **stages,
            }
            record["tensor_bytes"] = (
                math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            )
            records.append(record)
            peak = stages["live"]["peak_allocated"] / MIB
            retained = stages["clean"]["driver"] / MIB
            print(
                f"{name:11s} {phase:8s} {str(shape):24s} peak={peak:9.1f} retained={retained:8.1f} MiB",
                flush=True,
            )
            if args.show_stages:
                print(stages, flush=True)
            save_results(args, records)


if __name__ == "__main__":
    main()
