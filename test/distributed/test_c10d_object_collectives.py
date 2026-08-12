# Owner(s): ["oncall: distributed"]

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistributedTestBase, TEST_SKIPS
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    TEST_WITH_DEV_DBG_ASAN,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)
device_count = torch.accelerator.device_count()


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if device_type != "cpu" and device_count < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=device_type)
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


class TestObjectCollectives(DistributedTestBase):
    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_all_gather_object(self, weights_only):
        output = [None] * dist.get_world_size()
        dist.all_gather_object(
            object_list=output, obj=self.rank, weights_only=weights_only
        )

        for i, v in enumerate(output):
            self.assertEqual(i, v, lambda msg: f"{msg}\nrank: {self.rank}")

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_gather_object(self, weights_only):
        output = [None] * dist.get_world_size() if self.rank == 0 else None
        dist.gather_object(
            obj=self.rank, object_gather_list=output, weights_only=weights_only
        )

        if self.rank == 0:
            for i, v in enumerate(output):
                self.assertEqual(i, v, lambda msg: f"{msg}\nrank: {self.rank}")

    @skipIfHpu
    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_send_recv_object_list(self, weights_only):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        if self.rank == 0:
            dist.send_object_list(object_list, 1, weights_only=weights_only)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, weights_only=weights_only)

        if self.rank < 2:
            self.assertEqual(99, object_list[0])
        else:
            self.assertEqual(None, object_list[0])

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_broadcast_object_list(self, weights_only):
        val = 99 if self.rank == 0 else None
        object_list = [val] * dist.get_world_size()
        # TODO test with broadcast_object_list's device argument
        dist.broadcast_object_list(object_list=object_list, weights_only=weights_only)

        self.assertEqual(99, object_list[0])

    @parametrize("weights_only", [True, False])
    @with_comms()
    def test_scatter_object_list(self, weights_only):
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        output_list = [None]
        dist.scatter_object_list(
            scatter_object_output_list=output_list,
            scatter_object_input_list=input_list,
            weights_only=weights_only,
        )

        self.assertEqual(self.rank, output_list[0])

    @with_comms()
    def test_weights_only_rejects_unsafe_object(self):
        # An arbitrary function is not deserializable under weights_only=True
        output = [None] * dist.get_world_size()
        with self.assertRaises(Exception):
            dist.all_gather_object(
                object_list=output, obj=with_comms, weights_only=True
            )

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


instantiate_parametrized_tests(TestObjectCollectives)


if __name__ == "__main__":
    run_tests()
