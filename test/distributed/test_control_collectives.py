# Owner(s): ["oncall: distributed"]

from datetime import timedelta
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests, TestCase


# simple example of user code that takes the base class ControlCollectives
# and executes multiple different collectives
def simple_user_func(collectives: dist._ControlCollectives, rank: int) -> int:
    timeout = timedelta(seconds=10)
    # first a barrier
    collectives.barrier("1", timeout, True)
    # then an all_sum
    out = collectives.all_sum("2", rank, timeout)
    return out


class TestCollectives(TestCase):
    def test_barrier(self) -> None:
        store = dist.HashStore()

        world_size = 2

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            collectives.barrier("foo", timedelta(seconds=10), True)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                collectives.broadcast_send("foo", b"data", timeout)
            else:
                out = collectives.broadcast_recv("foo", timeout)
                self.assertEqual(out, b"data")

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_gather(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.gather_recv("foo", str(rank), timeout)
                self.assertEqual(out, [b"0", b"1", b"2", b"3"])
            else:
                collectives.gather_send("foo", str(rank), timeout)

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_scatter(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            if rank == 2:
                out = collectives.scatter_send(
                    "foo", [str(i) for i in range(world_size)], timeout
                )
            else:
                out = collectives.scatter_recv("foo", timeout)
            self.assertEqual(out, str(rank).encode())

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_all_sum(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(seconds=10)

        def f(rank: int) -> None:
            collectives = dist._StoreCollectives(store, rank, world_size)
            out = collectives.all_sum("foo", rank, timeout)
            self.assertEqual(out, sum(range(world_size)))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))

    def test_broadcast_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.broadcast_recv("foo", timeout)

    def test_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.gather_recv("foo", "data", timeout)

    def test_scatter_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(Exception, "Wait timeout"):
            collectives.scatter_recv("foo", timeout)

    def test_all_gather_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "all_gather failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_gather("foo", "data", timeout)

    def test_barrier_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.barrier("foo", timeout, True)

    def test_all_sum_timeout(self) -> None:
        store = dist.HashStore()

        world_size = 4
        timeout = timedelta(milliseconds=1)
        collectives = dist._StoreCollectives(store, 1, world_size)
        with self.assertRaisesRegex(
            Exception, "barrier failed -- missing ranks: 0, 2, 3"
        ):
            collectives.all_sum("foo", 1, timeout)

    def test_unique(self) -> None:
        store = dist.HashStore()

        collectives = dist._StoreCollectives(store, 1, 1)
        collectives.broadcast_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.broadcast_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.broadcast_recv("foo")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.gather_send("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.gather_recv("foo", "asdf")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.scatter_send("foo", ["asdf"])

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.scatter_recv("foo")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.all_gather("foo", "bar")

        with self.assertRaisesRegex(Exception, "Key foo has already been used"):
            collectives.all_sum("foo", 2)

    def test_simple_user_func(self) -> None:
        store = dist.HashStore()
        world_size = 4

        def f(rank: int) -> None:
            # user need to create child collectives
            # but simple_user_func do not need to be changed for different child collectives
            store_collectives = dist._StoreCollectives(store, rank, world_size)
            out = simple_user_func(store_collectives, rank)
            self.assertEqual(out, sum(range(world_size)))

        with ThreadPool(world_size) as pool:
            pool.map(f, range(world_size))


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
