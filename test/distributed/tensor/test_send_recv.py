from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import (
    LocalIntNode,
    LocalRunnerMode,
    LocalTensorMode,
    maybe_run_for_local_tensor,
)
from torch.distributed.tensor import init_device_mesh


@maybe_run_for_local_tensor
def _get_peer(rank, world_size, dir) -> int:
    return (rank + dir) % world_size


def get_rank(world_size) -> torch.SymInt:
    return torch.SymInt(LocalIntNode({r: r for r in range(world_size)}))


def _maybe_wait(work: dist.Work | None | list[dist.Work | None]) -> None:
    if work is None:
        return
    if isinstance(work, dist.Work):
        work = [work]
    for w in work:
        if w is None:
            continue
        w.wait()


def _attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    tensor.__rank__ = rank
    return tensor


def _unpack_ranks(ranks: int | torch.SymInt) -> list[int]:
    if isinstance(ranks, torch.SymInt) and isinstance(ranks.node, LocalIntNode):
        return list(ranks.node._local_ints.keys())
    if isinstance(ranks, int):
        return [ranks]
    raise AssertionError(f"Unsupported ranks type {type(ranks)}")


def _run_peer_op(
    rank: int | torch.SymInt,
    peer: int | torch.SymInt,
    tensor: torch.Tensor,
    op: Callable[[torch.Tensor, int], dist.Work | None],
) -> dist.Work | None | list[dist.Work | None]:
    w = []
    for r in _unpack_ranks(rank):
        for p in _unpack_ranks(peer):
            tensor = _attach_rank(tensor, r)
            w.append(op(tensor, p))
    return w


def _run(world_size: int, rank: int) -> None:
    ltm = LocalTensorMode(world_size)
    with ltm:
        x = torch.ones(1) * rank
        y = torch.zeros_like(x)

        next_rank = _get_peer(rank, world_size, +1)
        prev_rank = _get_peer(rank, world_size, -1)

        _run_peer_op(rank, next_rank, x, lambda tensor, dst: dist.isend(tensor, dst))

        rw = _run_peer_op(
            rank, prev_rank, y, lambda tensor, src: dist.irecv(tensor, src)
        )

        _maybe_wait(rw)
        print(f"{rank=}\n{x=}\n{y=}")


if __name__ == "__main__":
    world_size = 4
    dist.init_process_group("fake", rank=0, world_size=world_size)

    mesh = init_device_mesh("cpu", (world_size,))

    with LocalRunnerMode(world_size, world_size, lambda rank: _run(world_size, rank)):
        pass
