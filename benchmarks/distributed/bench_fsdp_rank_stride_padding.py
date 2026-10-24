"""Benchmark FSDP2's 16-byte rank-stride padding on a misaligned parameter group.

    torchrun --standalone --nproc_per_node=8 \
        benchmarks/distributed/bench_fsdp_rank_stride_padding.py [--exact] [--no-scale]

Each layer is a Linear(dim, dim) and a one-element scale in its own FSDP group,
without resharding after forward. With dim=16384 and 8 ranks, the scale makes
each rank's chunk 2048 * 16385 + 1 elements, 2 bytes off a 16-byte boundary in
bf16 (all-gather) and 4 bytes off in fp32 (reduce-scatter). --exact keeps the
chunks exact, as before padding, and --no-scale drops the scale, which aligns
them without padding.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import profiler as torch_profiler
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    DefaultAllGather,
    DefaultReduceScatter,
)


class RecordingAllGather(DefaultAllGather):
    def __init__(self, exact: bool) -> None:
        self.exact = exact
        self.chunk_bytes = 0

    def rank_stride_alignment(self, dtype, group):
        return 1 if self.exact else super().rank_stride_alignment(dtype, group)

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        self.chunk_bytes = input_tensor.nbytes
        return super().__call__(output_tensor, input_tensor, group, async_op)


class RecordingReduceScatter(DefaultReduceScatter):
    def __init__(self, exact: bool) -> None:
        self.exact = exact
        self.chunk_bytes = 0

    def rank_stride_alignment(self, dtype, group):
        return 1 if self.exact else super().rank_stride_alignment(dtype, group)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        self.chunk_bytes = output_tensor.nbytes
        return super().__call__(output_tensor, input_tensor, group, op, async_op)


class Layer(nn.Module):
    def __init__(self, dim: int, scale: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.ones(1)) if scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y if self.scale is None else y * self.scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=16384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(0)
    layers = [Layer(args.dim, not args.no_scale) for _ in range(args.layers)]
    model = nn.Sequential(*layers)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    all_gather = RecordingAllGather(args.exact)
    reduce_scatter = RecordingReduceScatter(args.exact)
    for layer in layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=False)
        layer.set_custom_all_gather(all_gather)
        layer.set_custom_reduce_scatter(reduce_scatter)
    fully_shard(model, mp_policy=mp_policy)
    model.to(device)
    inp = torch.randn(args.tokens, args.dim, device=device)

    def unshard() -> None:
        for layer in layers:
            layer.unshard()
            layer.reshard()

    def step() -> None:
        model(inp).float().pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
        model.reshard()

    def timed(fn, windows: int = 10) -> float:
        # Median over windows of the slowest rank's mean
        for _ in range(5):
            fn()
        means = []
        for _ in range(windows):
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters // windows):
                fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / (args.iters // windows)
            mean = torch.tensor(ms, device=device)
            dist.all_reduce(mean, op=dist.ReduceOp.MAX)
            means.append(mean.item())
        return sorted(means)[windows // 2]

    unshard_ms, step_ms = timed(unshard), timed(step)

    prof = torch_profiler.profile(
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch_profiler.schedule(
            wait=1, warmup=2, active=1, repeat=1, skip_first=1
        ),
    )
    prof.start()
    for _ in range(6):
        torch.cuda.synchronize()
        dist.barrier()
        step()
        torch.cuda.synchronize()
        dist.barrier()
        prof.step()
    prof.stop()
    # NCCL kernels also spin while waiting for peers, so take the fastest rank
    kernel_ms = []
    for name in ("AllGather", "ReduceScatter"):
        evts = [e for e in prof.key_averages() if "nccl" in e.key and name in e.key]
        total_us = sum(e.device_time_total for e in evts)
        kernel_ms.append(total_us / max(sum(e.count for e in evts), 1) / 1e3)
    kernels = torch.tensor(kernel_ms, device=device)
    dist.all_reduce(kernels, op=dist.ReduceOp.MIN)

    if dist.get_rank() == 0:
        mode = "exact" if args.exact else "padded"
        ag_bytes, rs_bytes = all_gather.chunk_bytes, reduce_scatter.chunk_bytes
        print(
            f"{mode}{'' if args.no_scale else ' +scale'}: "
            f"chunk bytes mod 16: all-gather {ag_bytes % 16}, reduce-scatter {rs_bytes % 16} | "
            f"unshard {unshard_ms:.2f} ms | fwd+bwd {step_ms:.2f} ms | "
            f"NCCL kernel per call: all-gather {kernels[0].item():.3f} ms, "
            f"reduce-scatter {kernels[1].item():.3f} ms"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
