import contextlib
import socket
import unittest
from datetime import timedelta
from multiprocessing.pool import ThreadPool

import torch.distributed as dist

from ._scalable_coordinator_collectives import (
    _ScalableCoordinator,
    _ScalableCoordinatorCollectives,
)


@contextlib.contextmanager
def with_sockets(world_size):
    sockets = []
    try:
        sockets = [
            socket.create_server(
                ("::", 0),
                family=socket.AF_INET6,
            )
            for _ in range(world_size)
        ]
        yield sockets
    finally:
        for sock in sockets:
            sock.close()


@contextlib.contextmanager
def create_collectives(world_size, **kwargs):
    with with_sockets(world_size) as sockets:
        addrs = [("::1", sock.getsockname()[1]) for num, sock in enumerate(sockets)]
        collectives = []
        for rank in range(world_size):
            coordinator = _ScalableCoordinator(
                rank=rank,
                socket=sockets[rank].detach(),
                addrs=addrs,
                **kwargs,
            )
            coordinator.create_sub_world("all", list(range(world_size)))
            collective = _ScalableCoordinatorCollectives(
                coordinator=coordinator,
                rank=rank,
                sub_world_id="all",
            )
            collectives.append(collective)
        yield collectives


def simple_user_func(collectives: dist._ControlCollectives, rank: int) -> int:
    timeout = timedelta(seconds=10)
    # first a barrier
    collectives.barrier("1", timeout, True)
    # then an all_sum
    out = collectives.all_sum("2", rank, timeout)
    return out


class ScalableCoordinatorCollectives(unittest.TestCase):

    def test_barrier(self) -> None:
        world_size = 2

        with create_collectives(world_size) as collectives:

            def f(rank: int) -> None:
                collectives[rank].barrier("foo", timedelta(seconds=10), True)

            with ThreadPool(world_size) as pool:
                pool.map(f, range(world_size))

    def test_simple_user_func(self) -> None:
        world_size = 4

        with create_collectives(world_size) as sc_collectives:

            def f(rank: int) -> None:
                # user need to create child collectives
                # but simple_user_func do not need to be changed for different child collectives
                out = simple_user_func(sc_collectives[rank], rank)
                self.assertEqual(out, sum(range(world_size)))

            with ThreadPool(world_size) as pool:
                pool.map(f, range(world_size))
