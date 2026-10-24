#!/usr/bin/env python3
"""Benchmark symmetric-memory unicast and multicast all-gather variants.

Run on an 8-GPU NVLink node:

    torchrun --standalone --nproc-per-node=8 \
        benchmarks/distributed/bench_symmetric_memory_all_gather.py
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch.cuda._utils import _check_cuda_bindings

try:
    import cuda.bindings.driver as cuda_driver
except ImportError as exc:
    raise SystemExit(
        "This benchmark requires cuda-python. Install it with `pip install cuda-python`."
    ) from exc


def _ensure_cuda_context() -> None:
    current_ctx = _check_cuda_bindings(cuda_driver.cuCtxGetCurrent())
    if int(current_ctx) == 0:
        _check_cuda_bindings(cuda_driver.cuInit(0))


def _copy_async(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    _ensure_cuda_context()
    stream = torch.cuda.current_stream().cuda_stream
    _check_cuda_bindings(cuda_driver.cuMemcpyAsync(dst_ptr, src_ptr, nbytes, stream))


def all_gather_sm_free(
    input_tensor: torch.Tensor,
    dst_handle: _SymmetricMemory,
    dst_buffer_ptrs: Sequence[int],
    *,
    rank: int,
    world_size: int,
) -> None:
    assert input_tensor.is_contiguous()
    assert len(dst_buffer_ptrs) == world_size

    shard_bytes = input_tensor.numel() * input_tensor.element_size()
    src = input_tensor.data_ptr()

    dst_handle.barrier()
    for step in range(world_size):
        # For world_size=4, step 0 sends 0->0, 1->1, 2->2, 3->3; step 1 sends
        # 0->1, 1->2, 2->3, 3->0; and so on. Each step forms a permutation, so
        # no rank receives more than one peer write at the same time.
        peer = (rank + step) % world_size
        dst = dst_buffer_ptrs[peer] + rank * shard_bytes
        _copy_async(dst, src, shard_bytes)
    dst_handle.barrier()


def all_gather_sm_free_multicast(
    input_tensor: torch.Tensor,
    dst_handle: _SymmetricMemory,
    *,
    rank: int,
    world_size: int,
) -> None:
    assert input_tensor.is_contiguous()
    assert dst_handle.rank == rank
    assert dst_handle.world_size == world_size
    assert len(dst_handle.buffer_ptrs) == world_size

    shard_bytes = input_tensor.numel() * input_tensor.element_size()
    assert dst_handle.buffer_size >= shard_bytes * world_size
    assert _SymmetricMemory.has_multicast_support(
        DeviceType.CUDA, input_tensor.device.index
    ), "symmetric-memory multicast is not supported on this device"
    assert dst_handle.multicast_ptr != 0, "destination handle has no multicast VA"

    dst_handle.barrier()
    dst = dst_handle.multicast_ptr + rank * shard_bytes
    _copy_async(dst, input_tensor.data_ptr(), shard_bytes)
    dst_handle.barrier()


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:5.0f} KB"
    return f"{num_bytes / (1024 * 1024):5.1f} MB"


def _format_perf(us: float, payload_bytes: int) -> str:
    gbps = payload_bytes / us / 1000
    return f"{us:7.1f}us {gbps:6.1f}GB/s"


def _bench_cuda_graph(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    return start.elapsed_time(end) * 1000 / iters


def _check_correctness(
    input_tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> None:
    expected = torch.empty(
        input_tensor.shape[0] * world_size,
        input_tensor.shape[1],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    dist.all_gather_into_tensor(expected, input_tensor)

    for use_multicast in (False, True):
        output = symm_mem.empty(
            *expected.shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        handle = symm_mem.rendezvous(output, group=dist.group.WORLD.group_name)
        assert handle is not None
        if use_multicast:
            all_gather_sm_free_multicast(
                input_tensor, handle, rank=rank, world_size=world_size
            )
        else:
            all_gather_sm_free(
                input_tensor,
                handle,
                handle.buffer_ptrs,
                rank=rank,
                world_size=world_size,
            )
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def _print_table(
    rows: list[tuple[int, int, int, float, float, float]],
    *,
    dim: int,
    warmup: int,
    iters: int,
    world_size: int,
) -> None:
    title = (
        f"all_gather modes - world_size={world_size}, D={dim}, dtype=bf16, "
        f"warmup={warmup}, iters={iters}, cuda-graph"
    )
    width = 134
    print()
    print("=" * width)
    print(title)
    print("=" * width)
    print(
        f"{'per-peer token':>14} | {'output tensor size':>18} | "
        f"{'nccl-ag':>21} | {'ag-sm-free':>21} | {'speedup':>9} | "
        f"{'ag-multicast':>21} | {'speedup':>9}"
    )
    print("-" * width)
    for tokens, output_bytes, payload_bytes, nccl_us, unicast_us, multicast_us in rows:
        print(
            f"{tokens:>14} | {_format_size(output_bytes):>18} | "
            f"{_format_perf(nccl_us, payload_bytes):>21} | "
            f"{_format_perf(unicast_us, payload_bytes):>21} | "
            f"{nccl_us / unicast_us:>7.3f}x | "
            f"{_format_perf(multicast_us, payload_bytes):>21} | "
            f"{nccl_us / multicast_us:>7.3f}x"
        )
    print("-" * width)
    print("=" * width)


def run(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    symm_mem.set_backend(args.symm_mem_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 8 and rank == 0:
        print(f"warning: intended for 8 GPUs, running with world_size={world_size}")

    if args.correctness:
        torch.manual_seed(1234 + rank)
        tokens = min(args.tokens)
        sample = torch.randn(tokens, args.dim, device=device, dtype=torch.bfloat16)
        _check_correctness(sample, rank=rank, world_size=world_size)

    dtype_size = torch.empty((), dtype=torch.bfloat16).element_size()
    rows: list[tuple[int, int, int, float, float, float]] = []
    for tokens in args.tokens:
        input_shape = (tokens, args.dim)
        output_shape = (tokens * world_size, args.dim)
        input_bytes = tokens * args.dim * dtype_size
        output_bytes = input_bytes * world_size
        payload_bytes = input_bytes * (world_size - 1)

        nccl_input = torch.zeros(input_shape, device=device, dtype=torch.bfloat16)
        nccl_output = torch.empty(output_shape, device=device, dtype=torch.bfloat16)
        nccl_us = _bench_cuda_graph(
            lambda: dist.all_gather_into_tensor(nccl_output, nccl_input),
            warmup=args.warmup,
            iters=args.iters,
        )

        unicast_input = torch.zeros(input_shape, device=device, dtype=torch.bfloat16)
        unicast_output = symm_mem.empty(
            *output_shape, dtype=torch.bfloat16, device=device
        )
        unicast_handle = symm_mem.rendezvous(
            unicast_output, group=dist.group.WORLD.group_name
        )
        assert unicast_handle is not None
        unicast_us = _bench_cuda_graph(
            lambda: all_gather_sm_free(
                unicast_input,
                unicast_handle,
                unicast_handle.buffer_ptrs,
                rank=rank,
                world_size=world_size,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        multicast_input = torch.zeros(input_shape, device=device, dtype=torch.bfloat16)
        multicast_output = symm_mem.empty(
            *output_shape, dtype=torch.bfloat16, device=device
        )
        multicast_handle = symm_mem.rendezvous(
            multicast_output, group=dist.group.WORLD.group_name
        )
        assert multicast_handle is not None
        multicast_us = _bench_cuda_graph(
            lambda: all_gather_sm_free_multicast(
                multicast_input,
                multicast_handle,
                rank=rank,
                world_size=world_size,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        rows.append((tokens, output_bytes, payload_bytes, nccl_us, unicast_us, multicast_us))
        torch.cuda.empty_cache()

    if rank == 0:
        _print_table(
            rows,
            dim=args.dim,
            warmup=args.warmup,
            iters=args.iters,
            world_size=world_size,
        )

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--symm-mem-backend", default="NCCL")
    parser.add_argument("--no-correctness", dest="correctness", action="store_false")
    parser.set_defaults(correctness=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
