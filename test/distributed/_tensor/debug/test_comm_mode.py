# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor

from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed._tensor.placement_types import Shard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore

c10d_functional = torch.ops.c10d_functional


class TestCommMode(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        self.world_size = 2
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=1, world_size=self.world_size, store=store
        )
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.world_pg = dist.distributed_c10d._get_default_group()

    def test_comm_mode(self):
        world_pg = self.world_pg

        class WrapperModel(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.model = MLPModule(device=device)

            def forward(self, x):
                x = funcol.all_gather_tensor(x, 0, world_pg)
                x = funcol.reduce_scatter_tensor(x, "sum", 0, world_pg)
                out = self.model(x)
                return funcol.all_reduce(out, "sum", world_pg)

        model = WrapperModel(self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            model(torch.randn(20, 10, device=self.device_type))

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 3)
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 1)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 1)

    def test_comm_mode_with_dtensor(self):
        world_pg = self.world_pg
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        def f(x, y):
            return torch.mm(x, y)

        comm_mode = CommDebugMode()
        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 32, requires_grad=True)
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        with comm_mode:
            f(x_dtensor, y_dtensor)

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 0)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 0)


if __name__ == "__main__":
    run_tests()
