# Owner(s): ["oncall: distributed"]

import os
import sys
import threading
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.distributed.hooks as dhooks

if not dist.is_available():
    print("torch.distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.common_utils import run_tests


class PgHooks(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def test_pg_hook(self):
        pgs = []

        def pg_hook(pg, pg_name):
            pgs.append((pg, pg_name))

        dhooks.register_process_group_hook(pg_hook)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )
        self.assertEqual(len(pgs), 1)
        self.assertEqual(pgs[0][0], dist.group.WORLD)

        # create two partial world PGs
        pg0 = dist.new_group(ranks=[0, 1])
        pg1 = dist.new_group(ranks=[2, 3])

        self.assertEqual(len(pgs), 2)
        self.assertEqual(pgs[1][0], pg0 if self.rank < 2 else pg1)


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.init_comms()
        func(self, *args, **kwargs)
        self.destroy_comms()

    return wrapper


class CollectiveHooks:
    @property
    def world_size(self) -> int:
        return 4

    def _collective_hooks(self):
        # it's ok to access them directly since there's a single bg thread poking at them.
        starts = []
        ends = []
        cv = threading.Condition()

        def coll_start(status):
            starts.append(status)
            print(f"col_start {len(starts)} rank{self.rank}")

        def coll_end(status):
            ends.append(status)
            print(f"col_end {len(ends)} rank{self.rank}")
            if len(ends) == 2:
                with cv:
                    cv.notify()

        dhooks.register_collective_start_hook(coll_start)
        dhooks.register_collective_end_hook(coll_end)

        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]

        dist.all_gather(tensor_list, tensor)

        tensor2 = torch.ones([2, 3]).cuda(self.rank) * self.rank
        dist.all_reduce(tensor2)

        with cv:
            cv.wait(1)

        default_pg_name = dist.group.WORLD.group_name
        self.assertEqual(2, len(starts))
        self.assertEqual(2, len(ends))


        def check_op(idx, coll_name):
            self.assertEqual(default_pg_name, starts[idx].pg_name)
            self.assertEqual(self.backend_name, starts[idx].backend)
            self.assertGreaterEqual(starts[idx].sequence_number, 0)
            self.assertGreaterEqual(starts[idx].timestamp, 0)
            self.assertEqual(coll_name, starts[idx].operation)

            self.assertEqual(default_pg_name, ends[idx].pg_name)
            self.assertEqual(self.backend_name, ends[idx].backend)

            self.assertEqual(starts[idx].sequence_number, ends[idx].sequence_number)
            self.assertLessEqual(starts[idx].timestamp, ends[idx].timestamp)
            self.assertEqual(coll_name, ends[idx].operation)

        check_op(0, "ALLGATHER")
        check_op(1, "ALLREDUCE")

class GlooHooks(MultiProcessTestCase, CollectiveHooks):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def init_comms(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def destroy_comms(self):
        dist.destroy_process_group()

    @property
    def backend_name(self):
        return "gloo"

    @with_comms
    def test_collective_hooks(self):
        self._collective_hooks()


class NcclHooks(MultiProcessTestCase, CollectiveHooks):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def init_comms(self):
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def destroy_comms(self):
        dist.destroy_process_group()

    @property
    def backend_name(self):
        return "nccl"

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_collective_hooks(self):
        self._collective_hooks()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
