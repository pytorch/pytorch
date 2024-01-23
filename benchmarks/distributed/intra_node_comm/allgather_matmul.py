#!/usr/bin/env python3
# This benchmark contains an example for using IntraNodeComm to implement efficient
# fused allgather_matmul (inspired by https://dl.acm.org/doi/pdf/10.1145/3567955.3567959 and
# @lw's efficient GPU implementation in xformers). The purpose of the example is to drive the
# development of relevant primitives and serve as a recipe for interested users.
#
# The benchmark can be executed as follows:
#   RDZV_ID=$(uuidgen) torchrun --nproc-per-node 8 allgather_matmul.py
import os

import torch
import torch._C._distributed_c10d as c10d

M = 16384
N = 8192
K = 2752

WARMUP_ITERS = 200
BENCH_ITERS = 50


comm = None
internal_stream = None
internal_event = None


def allgather_matmul(A_slice, B, out, rank, world_size):
    """
    Equivalent to `torch.matmul(dist.all_gather(A_slice), B)`.
    """
    buf_0 = torch.empty_like(A_slice)
    buf_1 = torch.empty_like(A_slice)
    out_slices = [
        out[i : i + A_slice.shape[0]]
        for i in range(0, world_size * A_slice.shape[0], A_slice.shape[0])
    ]

    # Perform matmul with the local input slice
    torch.matmul(A_slice, B, out=out_slices[rank])

    # In another stream, copy the local input slice into the intra-node
    # buffer. After the barrier, all peers' input slices are accessible
    # via their intra-node buffer without requiring synchronization.
    with torch.cuda.stream(internal_stream):
        comm.put(A_slice)
        comm.barrier()
        internal_event.record()
    internal_event.wait()

    # Copy input slice from remote buffer and perform matmul.
    # Alternate between two streams to offset the wave quantization
    # effect of smaller matmuls.
    for i in range(1, world_size):
        if i % 2 == 0:
            buf = buf_0
            stream = torch.cuda.current_stream()
        else:
            buf = buf_1
            stream = internal_stream
        remote = (i + rank) % world_size
        with torch.cuda.stream(stream):
            comm.get(remote, buf)
            torch.matmul(buf, B, out=out_slices[remote])

    # Perform another barrier to ensure all peers have completed consuming the
    # intra-node buffer so it can be reused.
    with torch.cuda.stream(internal_stream):
        comm.barrier()
        internal_event.record()
    internal_event.wait()


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
    rdzv_id = os.environ["RDZV_ID"]
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert M % world_size == 0

    torch.cuda.set_device(local_rank)
    store = c10d.FileStore(f"/tmp/{rdzv_id}")
    os.environ["ENABLE_INTRA_NODE_COMM"] = "1"

    global comm, internal_stream, internal_event
    comm = c10d._IntraNodeComm.rendezvous(store, "", rank, world_size)
    internal_stream = torch.cuda.Stream()
    internal_event = torch.cuda.Event()

    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    stride = M // world_size
    A_slice = A[rank * stride : (rank + 1) * stride]

    allgather_matmul_ms = do_bench(
        lambda: allgather_matmul(A_slice, B, out, rank, world_size)
    )
    matmul_ms = do_bench(lambda: torch.matmul(A, B))
    if rank == 0:
        print(
            "allgather_matmul "
            f"(M={M // world_size}, N={N}, K={K}, world_size={world_size}): "
            f"{allgather_matmul_ms:.4} ms/iter"
        )
        print("matmul " f"(M={M}, N={N}, K={K}): " f"{matmul_ms:.4} ms/iter")


if __name__ == "__main__":
    main()
