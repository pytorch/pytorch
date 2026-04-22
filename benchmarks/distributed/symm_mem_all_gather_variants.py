#!/usr/bin/env python3
"""
Benchmark for low-contention all-gather variants in torch.ops.symm_mem.

Compares v1 (baseline), v2 (stream_wait_value32), v3 (P2P push/put),
v4 (NVLS multicast + CE), and v5 (in-kernel multimem.st) across
a sweep of shard sizes at a fixed world size. v4 and v5 require multicast
support (NVSwitch / NVLink SHARP) and are skipped automatically on systems
without it.

Launch with torchrun on a single node:

    torchrun --nproc_per_node=8 \
        benchmarks/distributed/symm_mem_all_gather_variants.py

The script prints a markdown table of per-variant TPS (GB/s) for each
shard size. All numbers are the mean over ``--iters`` runs after
``--warmup`` warmup iterations.
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory


DEFAULT_SHARD_BYTES = (
    64 * 1024,
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    256 * 1024 * 1024,
)


def _build_variant(
    op_name: str,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    shard_numel: int,
    group_name: str,
    symm_mem_input: bool,
) -> Optional[Callable[[], torch.Tensor]]:
    """Return a zero-arg callable that runs the given AG variant, or None if
    the variant is not applicable on this device.
    """
    if op_name in (
        "_low_contention_all_gather_v4",
        "_low_contention_all_gather_v5",
    ):
        if not _SymmetricMemory.has_multicast_support(
            DeviceType.CUDA, torch.cuda.current_device()
        ):
            return None

    if symm_mem_input:
        t = _SymmetricMemory.empty_strided_p2p(
            size=(shard_numel,),
            stride=(1,),
            dtype=dtype,
            device=torch.device("cuda", rank),
            group_name=group_name,
        ).fill_(rank)
    else:
        t = torch.full(
            (shard_numel,),
            rank,
            dtype=dtype,
            device=torch.device("cuda", rank),
        )

    op = getattr(torch.ops.symm_mem, op_name)

    def run() -> torch.Tensor:
        res = op(t, group_name)
        return torch.ops._c10d_functional.wait_tensor(res)

    return run


def _bench(
    fn: Callable[[], torch.Tensor], warmup: int, iters: int
) -> tuple[float, float]:
    """Return (mean_seconds, stdev_seconds) averaged over ``iters`` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times_s = [s.elapsed_time(e) / 1000.0 for s, e in zip(start_events, end_events)]
    return statistics.mean(times_s), statistics.stdev(times_s) if iters > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--shard-bytes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHARD_BYTES),
        help="Per-rank shard sizes in bytes to benchmark.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["v1", "v2", "v3", "v4", "v5"],
        choices=["v1", "v2", "v3", "v4", "v5"],
    )
    parser.add_argument(
        "--symm-mem-input",
        action="store_true",
        help="Allocate the input tensor via empty_strided_p2p().",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group_name = dist.group.WORLD.group_name

    dtype = getattr(torch, args.dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()

    op_map = {
        "v1": "_low_contention_all_gather",
        "v2": "_low_contention_all_gather_v2",
        "v3": "_low_contention_all_gather_v3",
        "v4": "_low_contention_all_gather_v4",
        "v5": "_low_contention_all_gather_v5",
    }

    rows: list[tuple[int, dict[str, float]]] = []
    for shard_bytes in args.shard_bytes:
        shard_numel = shard_bytes // element_size
        per_variant: dict[str, float] = {}
        for v in args.variants:
            fn = _build_variant(
                op_map[v],
                rank,
                world_size,
                dtype,
                shard_numel,
                group_name,
                args.symm_mem_input,
            )
            if fn is None:
                per_variant[v] = float("nan")
                continue
            mean_s, _stdev = _bench(fn, args.warmup, args.iters)
            # Algorithm bandwidth: per-rank output bytes / time. AG output is
            # world_size * shard_bytes; the per-rank useful bandwidth is
            # reported as total output bytes / time (algo bw).
            per_variant[v] = (world_size * shard_bytes) / mean_s / 1e9
        rows.append((shard_bytes, per_variant))

    if rank == 0:
        print(
            f"\n# symm_mem all-gather variants, world_size={world_size}, "
            f"dtype={args.dtype}, symm_mem_input={args.symm_mem_input}"
        )
        header = ["shard_bytes", *args.variants]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for shard_bytes, per_variant in rows:
            cells = [f"{shard_bytes}"] + [
                "n/a" if per_variant[v] != per_variant[v] else f"{per_variant[v]:.1f}"
                for v in args.variants
            ]
            print("| " + " | ".join(cells) + " |")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
