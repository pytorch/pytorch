# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests


class AbstractProcessGroupTest(C10dBackendTest):
    def test_sequence_numbers(self):
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(default_pg._get_sequence_number_for_group(), 0)
        for sequence_number in range(1, 4):
            dist.all_reduce(torch.ones(1, device=self.device))
            self.assertEqual(
                default_pg._get_sequence_number_for_group(), sequence_number
            )

        subgroup = dist.new_group(list(range(self.world_size)))
        self.assertEqual(subgroup._get_sequence_number_for_group(), 0)
        dist.all_reduce(torch.ones(1, device=self.device), group=subgroup)
        self.assertEqual(subgroup._get_sequence_number_for_group(), 1)
        self.assertEqual(default_pg._get_sequence_number_for_group(), 3)

    def test_different_group_initialization_order(self):
        self._init_pg()
        if self.device_type == "cuda":
            torch.cuda.set_device(0)
        full_group = dist.new_group(list(range(self.world_size)))
        singleton = dist.new_group([0])
        if self.rank == 0:
            tensor = torch.ones(1, device=self.device)
            dist.all_reduce(tensor, group=singleton)
            self.assertEqual(tensor, torch.ones_like(tensor))
        dist.barrier(group=full_group)

    def test_wait_unregisters_work(self):
        self._init_pg()
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            tensor = torch.ones(1, device=self.device)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(tensor, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

    def test_async_work_lifetime(self):
        self._init_pg()
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del work
        dist.barrier()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del tensor
        del work
        dist.barrier()

    def test_work_future(self):
        self._init_pg()
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        future = work.get_future()
        future.wait()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)


instantiate_backend_tests(
    globals(), "ProcessGroup", AbstractProcessGroupTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()
