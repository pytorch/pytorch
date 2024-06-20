# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor

from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed._tensor.placement_types import Shard
from torch.testing._internal.common_distributed import requires_nccl
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore

c10d_functional = torch.ops.c10d_functional
c10d_ops = torch.ops.c10d


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

    def checksAssert(self, comm_mode, key, expected_value, expected_total_value):
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), expected_total_value)
        self.assertEqual(comm_counts[key], expected_value)

        return

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

    def test_comm_mode_coalesced(self):
        world_pg = self.world_pg

        class WrapperModelCoalesced(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.model = MLPModule(device=device)

            def forward(self, x):
                x = funcol.all_gather_tensor(x, 0, world_pg)
                x = funcol.reduce_scatter_tensor(x, "sum", 0, world_pg)
                out = self.model(x)
                return funcol.all_reduce_coalesced([out], "sum", world_pg)

        model = WrapperModelCoalesced(self.device_type)

        comm_mode = CommDebugMode()
        with comm_mode:
            model(torch.randn(20, 10, device=self.device_type))

        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 3)
        self.assertEqual(comm_counts[c10d_functional.all_reduce_coalesced], 1)
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

    @requires_nccl()
    def test_comm_mode_with_c10d(self):
        world_pg = self.world_pg

        inp = torch.rand(2, 8, 16).cuda()
        all_gather_out = inp.new_empty(self.world_size * 2, 8, 16)

        comm_mode = CommDebugMode()

        # tests c10d all_reduce tracing
        with comm_mode:
            dist.all_reduce(inp)

        self.checksAssert(comm_mode, c10d_ops.allreduce_, 1, 1)

        # tests c10d all_gather_into_tensor tracing
        with comm_mode:
            dist.all_gather_into_tensor(all_gather_out, inp)

        self.checksAssert(comm_mode, c10d_ops._allgather_base_, 1, 1)

        # tests c10d reduce_scatter tracing
        with comm_mode:
            dist.reduce_scatter_tensor(inp, all_gather_out)

        self.checksAssert(comm_mode, c10d_ops._reduce_scatter_base_, 1, 1)

        # tests c10d broadcast tracing
        with comm_mode:
            dist.broadcast(inp, 0)

        self.checksAssert(comm_mode, c10d_ops.broadcast_, 1, 1)

        # tests c10d gather tracing
        with comm_mode:
            dist.gather(inp, None, 0)

        self.checksAssert(comm_mode, c10d_ops.gather_, 1, 1)

        # tests c10d reduce tracing
        with comm_mode:
            dist.reduce(inp, 0)

        self.checksAssert(comm_mode, c10d_ops.reduce_, 1, 1)

        # tests c10d scatter tracing
        with comm_mode:
            dist.scatter(inp, None, 0)

        self.checksAssert(comm_mode, c10d_ops.scatter_, 1, 1)

        # tests c10d all_gather tracing
        output_list = []

        with comm_mode:
            dist.all_gather(output_list, inp, None)

        self.checksAssert(comm_mode, c10d_ops.allgather_, 1, 1)

        # tests c10d allgather_coalesced_ tracing
        output_list = []

        with comm_mode:
            dist.all_gather_coalesced(output_list, [inp], None)

        self.checksAssert(comm_mode, c10d_ops.allgather_coalesced_, 1, 1)

        # tests c10d allgather_into_tensor_coalesced_ tracing
        with comm_mode, dist._coalescing_manager():
            dist.all_gather_into_tensor(all_gather_out, inp)

        self.checksAssert(comm_mode, c10d_ops.allgather_into_tensor_coalesced_, 1, 1)

        # tests c10d allreduce_coalesced
        with comm_mode:
            dist.all_reduce_coalesced(inp)

        self.checksAssert(comm_mode, c10d_ops.allreduce_coalesced_, 1, 1)

        # tests c10d reduce_scatter_
        with comm_mode:
            dist.reduce_scatter(all_gather_out, [inp])

        self.checksAssert(comm_mode, c10d_ops.reduce_scatter_, 1, 1)

        # tests c10d reduce_scatter_tensor_coalesced
        with comm_mode as A, dist._coalescing_manager() as B:
            dist.reduce_scatter_tensor(all_gather_out, inp)

        self.checksAssert(comm_mode, c10d_ops.reduce_scatter_tensor_coalesced_, 1, 1)

        # tests c10d alltoall_
        with comm_mode:
            dist.all_to_all([inp], [inp])

        self.checksAssert(comm_mode, c10d_ops.alltoall_, 1, 1)

        # tests c10d alltoall_base_
        with comm_mode:
            dist.all_to_all_single(inp, inp)

        self.checksAssert(comm_mode, c10d_ops.alltoall_base_, 1, 1)


if __name__ == "__main__":
    run_tests()
