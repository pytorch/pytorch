#!/usr/bin/env python3
"""
Benchmark compute/comm overlap: CUDA matmul overlapped with NCCL all-gather.

Run with torchrun, e.g.:

  torchrun --nproc_per_node=8 test/distributed/bench_overlapped_matmul_allgather.py \
    --m 8192 --n 8192 --k 8192 --ag-mb 64 --dtype fp16 --iters 200 --warmup 50

This measures *total* per-iteration GPU time for:
  - sequential: matmul then all-gather (same stream)
  - overlap:    all-gather on a dedicated comm stream overlapped with matmul on default stream
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class Result:
    mode: str
    avg_ms: float
    p50_ms: float
    p90_ms: float
    max_ms: float


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024 or unit == "GiB":
            return f"{num_bytes:.2f}{unit}" if unit != "B" else f"{num_bytes}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}GiB"


def _make_allgather_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    world_size: int,
    ag_mb: int,
    use_symmetric_memory: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bytes_per_elem = torch.empty((), device="cpu", dtype=dtype).element_size()
    num_bytes = int(ag_mb) * 1024 * 1024
    numel = max(1, num_bytes // bytes_per_elem)
    if use_symmetric_memory:
        # Allocate on symmetric memory for NCCL Copy Engine collectives / registrations.
        x = symm_mem.empty(numel, dtype=dtype, device=device)
        x.normal_()
        out = symm_mem.empty(world_size * numel, dtype=dtype, device=device)
    else:
        x = torch.randn((numel,), device=device, dtype=dtype)
        out = torch.empty((world_size * numel,), device=device, dtype=dtype)
    return x, out


def _step_sequential(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    y = a @ b
    dist.all_gather_into_tensor(out, x)
    return y


def _step_overlap(
    *,
    comm_stream: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    # Enqueue collective on comm_stream and compute on default stream.
    with torch.cuda.stream(comm_stream):
        dist.all_gather_into_tensor(out, x)
    y = a @ b
    # Ensure the default stream waits for the comm stream before timing end event.
    torch.cuda.current_stream().wait_stream(comm_stream)
    return y


def _time_mode(
    *,
    mode: str,
    iters: int,
    warmup: int,
    comm_stream: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
) -> Result:
    if warmup > 0:
        for _ in range(warmup):
            if mode == "sequential":
                _step_sequential(a=a, b=b, x=x, out=out)
            elif mode == "overlap":
                _step_overlap(comm_stream=comm_stream, a=a, b=b, x=x, out=out)
            else:
                raise AssertionError(f"unknown mode: {mode}")
        torch.cuda.synchronize()

    times_ms: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        if mode == "sequential":
            _step_sequential(a=a, b=b, x=x, out=out)
        elif mode == "overlap":
            _step_overlap(comm_stream=comm_stream, a=a, b=b, x=x, out=out)
        else:
            raise AssertionError(f"unknown mode: {mode}")
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    avg_ms = sum(times_ms) / len(times_ms)
    return Result(
        mode=mode,
        avg_ms=avg_ms,
        p50_ms=_percentile(times_ms, 50),
        p90_ms=_percentile(times_ms, 90),
        max_ms=times_ms[-1],
    )


def _reduce_max_ms(ms: float, *, device: torch.device) -> float:
    t = torch.tensor([ms], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark overlapped matmul and all-gather (CUDA/NCCL)."
    )
    parser.add_argument("--backend", default="nccl", choices=["nccl"])
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--ag-mb", type=int, default=64, help="All-gather input size in MiB")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sequential", "overlap"],
        choices=["sequential", "overlap"],
    )
    parser.add_argument(
        "--nccl-cta-policy-zero",
        action="store_true",
        help=(
            "Initialize NCCL with NCCL_CTA_POLICY_ZERO and allocate all-gather buffers via "
            "torch.distributed._symmetric_memory."
        ),
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul (fp32 only)")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires CUDA + NCCL.", file=sys.stderr)
        return 0

    # torchrun env
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.nccl_cta_policy_zero:
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dist.init_process_group(backend=args.backend, init_method="env://", pg_options=opts)
        symm_mem.set_backend("NCCL")
    else:
        dist.init_process_group(backend=args.backend, init_method="env://")

    # Try to reduce accidental desync from first-touch allocations.
    torch.manual_seed(1337 + rank)
    dtype = DTYPE_MAP[args.dtype]

    if args.tf32:
        # TF32 affects fp32 matmul. Safe to set either way.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Independent compute + comm buffers to measure overlap (no artificial deps).
    a = torch.randn((args.m, args.k), device=device, dtype=dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=dtype)
    x, out = _make_allgather_tensors(
        device=device,
        dtype=dtype,
        world_size=world_size,
        ag_mb=args.ag_mb,
        use_symmetric_memory=args.nccl_cta_policy_zero,
    )
    if args.nccl_cta_policy_zero:
        # Mirror test/distributed/test_ce_colls.py:
        # - symmetric memory backend
        # - ensure NCCL communicator is initialized to avoid hangs
        # - rendezvous/register symmetric buffers
        dist.all_reduce(torch.ones(1, device=device))
        group_name = dist.group.WORLD.group_name
        symm_mem.rendezvous(x, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

    # Create a dedicated stream for collectives.
    comm_stream = torch.cuda.Stream(device=device)

    dist.barrier()
    t0 = time.time()
    results: List[Result] = []
    for mode in args.modes:
        results.append(
            _time_mode(
                mode=mode,
                iters=args.iters,
                warmup=args.warmup,
                comm_stream=comm_stream,
                a=a,
                b=b,
                x=x,
                out=out,
            )
        )
        dist.barrier()
    t1 = time.time()

    # Report: per-rank numbers + global max(average) for each mode.
    bytes_per_elem = torch.empty((), device="cpu", dtype=dtype).element_size()
    ag_in_bytes = int(args.ag_mb) * 1024 * 1024
    ag_in_numel = max(1, ag_in_bytes // bytes_per_elem)
    if rank == 0:
        print(
            f"world_size={world_size} dtype={args.dtype} matmul=({args.m},{args.k})@({args.k},{args.n}) "
            f"all_gather_in={_format_bytes(ag_in_numel * bytes_per_elem)} iters={args.iters} warmup={args.warmup}"
        )
    dist.barrier()

    for r in results:
        max_avg = _reduce_max_ms(r.avg_ms, device=device)
        # Print each rank's stats (useful for diagnosing imbalance), and rank0 prints global max avg.
        print(
            f"[rank {rank:03d}] mode={r.mode:10s} avg={r.avg_ms:8.3f}ms p50={r.p50_ms:8.3f}ms "
            f"p90={r.p90_ms:8.3f}ms max={r.max_ms:8.3f}ms"
        )
        if rank == 0:
            print(f"[global max] mode={r.mode:10s} avg={max_avg:8.3f}ms")
        dist.barrier()

    if rank == 0:
        print(f"total_wall_time={t1 - t0:.2f}s")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

