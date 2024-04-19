# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, init_device_mesh, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

HAS_CUDA = torch.cuda.is_available()


class TestFakePG(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_all_reduce(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_allgather(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for _, out_tensor in enumerate(output_tensors):
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_reduce_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device="cuda"))

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_fsdp_fake_e2e(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        my_module = nn.Sequential(
            nn.Linear(2, 3, device="cuda"),
            nn.ReLU(),
            nn.Linear(3, 2, device="cuda"),
        )
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_fake_pg_tracing(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            return funcol.all_gather_tensor(tensor, 0, default_pg)

        gm = make_fx(allgather_fn)(torch.randn(2, 2, device="cuda"))
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # src == rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=0)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # src == rank
        output = torch.ones(3, 3)
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        dist.scatter(output, to_scatter)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.scatter(output, None, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(3, 3)
        in_tensor = torch.ones(3, 3)
        output_split = [1, 1]
        input_split = [1, 1]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_send(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        tensor = torch.ones(3, 3)
        dist.send(tensor, 1)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    def test_recv(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.ones(3, 3)
        dist.recv(output, 1)
        self.assertEqual(tuple(output.shape), (3, 3))

    @unittest.skipIf(not HAS_CUDA, "No CUDA or TP+FSDP")
    def test_fsdp_tp_fake_e2e(self):
        world_size = 4
        tp_size = 2

        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        device_mesh = DeviceMesh("cuda", torch.arange(0, world_size).view(-1, tp_size))
        device_mesh = init_device_mesh(
            "cuda", (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
        )

        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        pairwise_parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        for parallel_plan in [sequence_parallelize_plan, pairwise_parallelize_plan]:
            my_module = parallelize_module(
                MLPModule(device="cuda"),
                device_mesh["tp"],
                parallel_plan,
            )

            sharded_module = FSDP(
                my_module, use_orig_params=True, device_mesh=device_mesh["dp"]
            )
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

            for i in range(10):
                dp_rank = dist.get_rank()
                torch.manual_seed(i + dp_rank)
                input = torch.randn(20, 10).cuda(dist.get_rank())
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()


if __name__ == "__main__":
    run_tests()
