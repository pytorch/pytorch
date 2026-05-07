"""Minimal all-reduce example that works with any backend.

Launch two workers on the same machine::

    # NCCL (requires 2 GPUs):
    torchrun --nproc-per-node 2 example_allreduce.py --backend nccl

    # TorchMux:
    torchrun --nproc-per-node 2 example_allreduce.py --backend torchmux
"""

import argparse

import torch
import torch.distributed as dist


ap = argparse.ArgumentParser()
ap.add_argument(
    "--backend",
    default="nccl",
    help="Distributed backend: nccl, gloo, torchmux, etc.",
)
args = ap.parse_args()

if args.backend == "torchmux":
    import torch.distributed._experimental.torchmux

dist.init_process_group(backend=args.backend)
rank = dist.get_rank()
world = dist.get_world_size()

if args.backend == "nccl":
    device = f"cuda:{rank}"
else:
    device = "cpu"
t = torch.full((4,), float(rank), device=device)
print(f"[rank {rank}] before: {t.tolist()}", flush=True)

dist.all_reduce(t)

expected = sum(range(world))
print(
    f"[rank {rank}] after:  {t.tolist()} (expected all {expected:.0f}s)",
    flush=True,
)

dist.destroy_process_group()
