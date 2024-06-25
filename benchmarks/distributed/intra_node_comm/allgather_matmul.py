#!/usr/bin/env python3
# This file contains an example for using cuda_p2p backend to implement efficient fused
# allgather_matmul (inspired by https://dl.acm.org/doi/pdf/10.1145/3567955.3567959 and
# @lw's efficient GPU implementation in xformers). Its purpose to help guide the
# development of relevant primitives and serve as an example for interested users.
#
# The benchmark can be executed as follows:
#   torchrun --nproc-per-node 8 allgather_matmul.py
import os

import torch
import torch.distributed as dist
from torch.distributed._cuda_p2p import ProcessGroupCudaP2P

M = 16384
N = 8192
K = 2752

WARMUP_ITERS = 200
BENCH_ITERS = 50


def allgather_matmul(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    group = dist.group.WORLD
    group_size = group.size()
    A = torch.ops._c10d_functional.all_gather_into_tensor(A_shard, group_size, "0")
    A = torch.ops._c10d_functional.wait_tensor(A)
    return A @ B


def allgather_matmul_p2p(A_shard: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to `torch.matmul(dist.all_gather(A_shard), B)`.
    """
    group = dist.group.WORLD
    group_size = group.size()
    rank = group.rank()
    backend = group._get_backend(torch.device("cuda"))

    out = torch.empty(
        (A_shard.shape[0] * group.size(), B.shape[1]),
        dtype=A_shard.dtype,
        device="cuda",
    )
    out_shards = out.chunk(group_size)
    local_p2p_buf = backend.get_p2p_buffer(rank, A_shard.shape, A_shard.dtype)

    # Perform matmul with the local input shard
    torch.matmul(A_shard, B, out=out_shards[rank])

    with torch.cuda.stream(backend.stream()):
        local_p2p_buf.copy_(A_shard)
        work = backend.intra_node_barrier()
    work.wait()

    buf_0 = torch.empty_like(A_shard)
    buf_1 = torch.empty_like(A_shard)
    for i in range(1, group_size):
        if i % 2 == 0:
            buf = buf_0
            stream = torch.cuda.current_stream()
        else:
            buf = buf_1
            stream = backend.stream()
        remote_rank = (i + rank) % group_size
        remote_p2p_buf = backend.get_p2p_buffer(
            remote_rank, A_shard.shape, A_shard.dtype
        )
        with torch.cuda.stream(stream):
            buf.copy_(remote_p2p_buf)
            torch.matmul(buf, B, out=out_shards[remote_rank])

    with torch.cuda.stream(backend.stream()):
        work = backend.intra_node_barrier()
    work.wait()
    return out


def do_bench(fn):
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    begin_evts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    for _ in range(WARMUP_ITERS):
        fn()
    for i in range(BENCH_ITERS):
        cache.zero_()
        begin_evts[i].record()
        fn()
        end_evts[i].record()

    torch.cuda.synchronize()
    return sum(b.elapsed_time(e) for b, e in zip(begin_evts, end_evts)) / BENCH_ITERS


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert M % world_size == 0

    torch.cuda.set_device(local_rank)

    options = ProcessGroupCudaP2P.Options()
    options.buffer_size = M * N * 2 // world_size
    dist.init_process_group("cuda_p2p", pg_options=options)

    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")

    stride = M // world_size
    A_shard = A[rank * stride : (rank + 1) * stride]

    assert torch.allclose(
        allgather_matmul(A_shard, B),
        allgather_matmul_p2p(A_shard, B),
    )

    dist.barrier()
    torch.cuda.synchronize()
    allgather_matmul_ms = do_bench(lambda: allgather_matmul(A_shard, B))

    dist.barrier()
    torch.cuda.synchronize()
    allgather_matmul_p2p_ms = do_bench(lambda: allgather_matmul_p2p(A_shard, B))

    dist.barrier()
    torch.cuda.synchronize()
    matmul_ms = do_bench(lambda: torch.matmul(A, B))

    if rank == 0:
        print(
            "allgather_matmul "
            f"(M={M // world_size}, N={N}, K={K}, world_size={world_size}): "
            f"{allgather_matmul_ms:.4} ms/iter"
        )
        print(
            "allgather_matmul_p2p "
            f"(M={M // world_size}, N={N}, K={K}, world_size={world_size}): "
            f"{allgather_matmul_p2p_ms:.4} ms/iter"
        )
        print(f"matmul (M={M}, N={N}, K={K}): {matmul_ms:.4} ms/iter")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
