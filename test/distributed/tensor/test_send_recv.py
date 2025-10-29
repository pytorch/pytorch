import torch
import torch.distributed as dist
from torch.distributed.tensor import (
    DTensor,
    DeviceMesh,
    distribute_tensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalTensor,
    maybe_run_for_local_tensor,
    _map_to_rank_local_val,
    LocalIntNode,
    LocalTensorMode,
)
from typing import Callable


@maybe_run_for_local_tensor
def get_peer(rank, world_size, dir) -> int:
    return (rank + dir) % world_size


def get_rank(world_size) -> torch.SymInt:
    return torch.SymInt(LocalIntNode({r: r for r in range(world_size)}))


def maybe_wait(work: dist.Work | list[dist.Work]) -> None:
    if isinstance(work, dist.Work):
        work = [work]
    for w in work:
        w.wait()


def work_or_default(work: dist.Work | None) -> dist.Work:
    return work if work is not None else dist.Work()


def attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    setattr(tensor, "__rank__", rank)
    return tensor


def run_peer_op(
    tensor: torch.Tensor,
    peer: int | torch.SymInt,
    op: Callable[[torch.Tensor, int], dist.Work | None],
) -> dist.Work | list[dist.Work]:
    if isinstance(peer, torch.SymInt) and isinstance(peer.node, LocalIntNode):
        return [
            work_or_default(op(attach_rank(tensor, r), p))
            for r, p in peer.node._local_ints.items()
        ]
    if isinstance(peer, int):
        return work_or_default(op(tensor, peer))
    else:
        raise AssertionError(f"Unsupported peer type {type(peer)}")


if __name__ == "__main__":
    world_size = 3
    dist.init_process_group("fake", rank=0, world_size=world_size)

    mesh = init_device_mesh("cpu", (world_size,))
    rank = get_rank(world_size)

    with LocalTensorMode(world_size):
        x = torch.arange(world_size)
        xd = DTensor.from_local(x, mesh)
        xd = xd.redistribute(placements=[Shard(0)])

        y = xd.to_local()
        z = torch.zeros_like(y)

        n = get_peer(rank, world_size, +1)
        p = get_peer(rank, world_size, -1)

        run_peer_op(y, n, lambda tensor, dst: dist.isend(tensor, dst))

        rw = run_peer_op(z, p, lambda tensor, src: dist.irecv(tensor, src))

        maybe_wait(rw)

        print(f"{y=}\n{z=}")
