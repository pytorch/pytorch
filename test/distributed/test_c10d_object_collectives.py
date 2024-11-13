# Owner(s): ["oncall: distributed"]

import os
import sys
from functools import partial, wraps

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.dist_init()
        func(self)
        self.destroy_comms()

    return wrapper


class TestObjectCollectives(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        self._spawn_processes()

    @property
    def device(self):
        return (
            torch.device("cuda", self.rank % torch.cuda.device_count())
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        if BACKEND == dist.Backend.NCCL:
            return torch.cuda.device_count()
        return super().world_size

    @property
    def process_group(self):
        return dist.group.WORLD

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def dist_init(self):
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    @with_comms()
    def test_all_gather_object(self):
        output = [None] * dist.get_world_size()
        dist.all_gather_object(object_list=output, obj=self.rank)

        for i, v in enumerate(output):
            self.assertEqual(i, v, f"rank: {self.rank}")

    @with_comms()
    def test_gather_object(self):
        output = [None] * dist.get_world_size() if self.rank == 0 else None
        dist.gather_object(obj=self.rank, object_gather_list=output)

        if self.rank == 0:
            for i, v in enumerate(output):
                self.assertEqual(i, v, f"rank: {self.rank}")

    @with_comms()
    def test_send_recv_object_list(self):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        if self.rank == 0:
            dist.send_object_list(object_list, 1)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0)

        if self.rank < 2:
            self.assertEqual(99, object_list[0])
        else:
            self.assertEqual(None, object_list[0])

    @with_comms()
    def test_broadcast_object_list(self):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        # TODO test with broadcast_object_list's device argument
        dist.broadcast_object_list(object_list=object_list)

        self.assertEqual(99, object_list[0])

    @with_comms()
    def test_scatter_object_list(self):
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        output_list = [None]
        dist.scatter_object_list(
            scatter_object_output_list=output_list, scatter_object_input_list=input_list
        )

        self.assertEqual(self.rank, output_list[0])

    # Test Object Collectives With Sub Pg

    def setup_sub_pg(self):
        rank = dist.get_rank()
        base_rank = rank - (rank % 2)
        ranks = [base_rank, base_rank + 1]
        my_pg = dist.new_group(ranks, use_local_synchronization=True)
        return rank, ranks, my_pg

    @with_comms()
    def test_subpg_scatter_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        dist.scatter_object_list(out_list, ranks, src=ranks[0], group=my_pg)
        self.assertEqual(rank, out_list[0])

    @with_comms()
    def test_subpg_all_gather_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks)
        dist.all_gather_object(out_list, rank, group=my_pg)
        self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_gather_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None] * len(ranks) if rank == ranks[0] else None
        dist.gather_object(rank, out_list, dst=ranks[0], group=my_pg)
        if rank == ranks[0]:
            self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_broadcast_object(self):
        rank, ranks, my_pg = self.setup_sub_pg()
        out_list = [None]
        if rank == ranks[0]:
            out_list[0] = rank
        dist.broadcast_object_list(out_list, src=ranks[0], group=my_pg)
        self.assertEqual(ranks[0], out_list[0])


if __name__ == "__main__":
    run_tests()
