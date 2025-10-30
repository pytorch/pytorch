from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    LocalIntNode,
    LocalRunnerMode,
    LocalTensorMode,
    maybe_run_for_local_tensor,
)
from torch.distributed.tensor import DTensor, init_device_mesh, Shard


@maybe_run_for_local_tensor
def get_peer(rank, world_size, dir) -> int:
    return (rank + dir) % world_size


def get_rank(world_size) -> torch.SymInt:
    return torch.SymInt(LocalIntNode({r: r for r in range(world_size)}))


def maybe_wait(work: dist.Work | None | list[dist.Work | None]) -> None:
    if work is None:
        return
    if isinstance(work, dist.Work):
        work = [work]
    for w in work:
        if w is None:
            continue
        w.wait()


def attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    tensor.__rank__ = rank
    return tensor


def run_peer_op(
    tensor: torch.Tensor,
    peer: int | torch.SymInt,
    op: Callable[[torch.Tensor, int], dist.Work | None],
) -> dist.Work | None | list[dist.Work | None]:
    if isinstance(peer, torch.SymInt) and isinstance(peer.node, LocalIntNode):
        return [op(attach_rank(tensor, r), p) for r, p in peer.node._local_ints.items()]
    if isinstance(peer, int):
        return op(tensor, peer)
    else:
        raise AssertionError(f"Unsupported peer type {type(peer)}")


def _run(world_size: int) -> None:
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


if __name__ == "__main__":
    world_size = 3
    dist.init_process_group("fake", rank=0, world_size=world_size)

    mesh = init_device_mesh("cpu", (world_size,))
    rank = get_rank(world_size)

    with LocalRunnerMode(world_size, 1, lambda: _run(world_size)):
        pass
