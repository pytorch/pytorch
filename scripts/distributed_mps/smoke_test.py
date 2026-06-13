"""Two-rank smoke test for the MPS distributed backend (TB RDMA / JACCL).

Covers every ProcessGroupMPS collective with correctness asserts: allreduce
SUM/MIN/MAX, broadcast, send/recv, barrier, and a 16 MB bf16 allreduce.

Needs two Apple-Silicon Macs cabled with Thunderbolt 5 support and `rdma_ctl enable`
run in Recovery on each.

To run it (rank-0 IP from configure_tb5_cluster.sh):

    # rank 0, backgrounded so its TCPStore binds before rank 1 attaches
    MASTER_ADDR=<rank-0-tb-ip> MASTER_PORT=29501 RANK=0 WORLD_SIZE=2 \\
        python scripts/distributed_mps/smoke_test.py

    # rank 1
    MASTER_ADDR=<rank-0-tb-ip> MASTER_PORT=29501 RANK=1 WORLD_SIZE=2 \\
        python scripts/distributed_mps/smoke_test.py"
"""

import os
import sys

import torch
import torch.distributed as dist


def log(rank: int, msg: str) -> None:
    print(f"[rank{rank}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world < 2:
        print("smoke_test.py needs WORLD_SIZE >= 2", file=sys.stderr)
        return 1

    log(
        rank, f"torch={torch.__version__} mps_avail={torch.backends.mps.is_available()}"
    )
    log(
        rank,
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
    )

    dist.init_process_group(backend="mps", rank=rank, world_size=world)
    log(rank, f"init_process_group done (backend={dist.get_backend()})")

    dev = torch.device("mps")

    # 1. allreduce SUM (each rank contributes rank+1).
    t = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    log(rank, f"allreduce SUM result={t.tolist()} (expected {expected})")
    assert torch.allclose(t, torch.full_like(t, expected)), "allreduce SUM mismatch"

    # 2. broadcast from rank 0.
    if rank == 0:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
    else:
        t = torch.zeros(4, device=dev)
    dist.broadcast(t, src=0)
    log(rank, f"broadcast from rank 0 result={t.tolist()}")
    assert torch.allclose(t.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0])), (
        "broadcast mismatch"
    )

    # 3. send/recv between rank 0 and rank 1.
    if rank == 0:
        x = torch.tensor([10.0, 20.0, 30.0, 40.0], device=dev)
        dist.send(x, dst=1)
        log(rank, f"sent {x.tolist()} to rank 1")
    elif rank == 1:
        x = torch.zeros(4, device=dev)
        dist.recv(x, src=0)
        log(rank, f"recvd {x.tolist()} from rank 0")
        assert torch.allclose(x.cpu(), torch.tensor([10.0, 20.0, 30.0, 40.0])), (
            "send/recv mismatch"
        )

    # 4. allreduce MIN / MAX.
    t = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    log(rank, f"allreduce MIN result={t.tolist()}")
    assert torch.allclose(t, torch.full_like(t, 1.0)), "allreduce MIN mismatch"

    t = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    log(rank, f"allreduce MAX result={t.tolist()}")
    assert torch.allclose(t, torch.full_like(t, float(world))), "allreduce MAX mismatch"

    # 5. barrier.
    dist.barrier()
    log(rank, "barrier OK")

    # 6. Larger payload, bf16 (~16 MB).
    n = 4 * 1024 * 1024
    t = torch.full((n,), float(rank + 1), device=dev, dtype=torch.bfloat16)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world + 1))
    err = (t.float() - expected).abs().max().item()
    log(rank, f"allreduce bf16 16MB max_abs_err={err}")
    assert err < 1e-2, f"bf16 allreduce error too large: {err}"

    dist.destroy_process_group()
    log(rank, "ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
