# Owner(s): ["oncall: distributed"]

import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C._distributed_c10d import FakeProcessGroup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import HAS_ACCELERATOR
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfHpu,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

device_type = get_devtype().type


class TestFakePG(TestCase):
    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def test_invalid_rank(self):
        with self.assertRaisesRegex(
            RuntimeError, "Cannot init process group where rank .* >= world_size"
        ):
            dist.init_process_group(backend="fake", rank=3, world_size=2)

    def test_all_reduce(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_set_timeout(self):
        # FakeProcessGroup used to inherit Backend's default set_timeout, which
        # raised "does not support setting timeout". Setting a timeout is a
        # no-op for a group that does no real communication, but it must not
        # raise.
        backend = FakeProcessGroup._create_internal(0, world_size=2)
        backend.set_timeout(timedelta(seconds=42))

    def test_set_timeout_via_dist(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        # Forwards through ProcessGroup to the fake backend; must not raise (the
        # default init path leaves the backend options null).
        dist.set_timeout(timedelta(seconds=30))

    def test_add_ephemeral_timeout_is_noop(self):
        backend = FakeProcessGroup._create_internal(0, world_size=2)
        backend._add_ephemeral_timeout(timedelta(seconds=42))

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=42))

    def test_allgather(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out_tensor in output_tensors:
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_allgather_into_tensor_requires_grad(self):
        # Regression test: fake PG's _allgather_base uses chunk() which
        # produces multi-output views. Writing into those views inplace when
        # the input requires grad used to trip an autograd safety check.
        dist.init_process_group(backend="fake", rank=0, world_size=4)
        world_size = dist.get_world_size()

        input_tensor = torch.randn(4, requires_grad=True)
        output_tensor = torch.empty(4 * world_size)
        dist.all_gather_single(output_tensor, input_tensor)
        self.assertEqual(output_tensor.shape, (4 * world_size,))

    def test_reduce_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device=device_type))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_fake_e2e(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        my_module = nn.Sequential(
            nn.Linear(2, 3, device=device_type),
            nn.ReLU(),
            nn.Linear(3, 2, device=device_type),
        )
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fake_pg_tracing(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            return funcol.all_gather_single(tensor, 0, default_pg)

        gm = make_fx(allgather_fn)(torch.randn(2, 2, device=device_type))
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        dist.init_process_group(backend="fake", rank=0, world_size=2)

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

        out_tensor = torch.ones(4, 3)
        in_tensor = torch.ones(4, 3)
        output_split = [2, 2]
        input_split = [2, 2]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (4, 3))

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

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_tp_fake_e2e(self):
        world_size = 4
        tp_size = 2

        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        device_mesh = init_device_mesh(
            device_type, (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
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
                MLPModule(device=device_type),
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
                input = torch.randn(20, 10, device=f"{device_type}:{dp_rank}")
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * r for r in range(2)]
        output = torch.empty(3, 3)
        dist.reduce_scatter(output, to_reduce_scatter)
        self.assertEqual(output, to_reduce_scatter[rank])

    def test_reduce_scatter_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output = torch.empty(3, 3)
        dist.reduce_scatter(output, inputs)
        self.assertEqual(output, inputs[1])
        self.assertFalse(output.requires_grad)

    @parametrize("rank", [0, 1])
    def test_scatter_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        if rank == 0:
            to_scatter = [torch.ones(3, 3) * r for r in range(2)]
            output = torch.empty(3, 3)
            dist.scatter(output, to_scatter)
            self.assertEqual(output, to_scatter[0])
        else:
            output = torch.ones(3, 3) * 5
            dist.scatter(output, None, src=0)
            self.assertEqual(output, torch.ones(3, 3) * 5)

    def test_scatter_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output = torch.empty(3, 3)
        dist.scatter(output, inputs)
        self.assertEqual(output, inputs[0])
        self.assertFalse(output.requires_grad)

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_base_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_buf = torch.arange(12.0).reshape(6, 2)
        out_buf = torch.empty(3, 2)
        dist._reduce_scatter_base(out_buf, in_buf)
        self.assertEqual(out_buf, in_buf.chunk(2)[rank])

    @parametrize("rank", [0, 1])
    def test_reduce_scatter_tensor_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(2, 2)
        dist.reduce_scatter_tensor(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[rank])

    def test_reduce_scatter_base_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(2, 1)
        dist._reduce_scatter_base(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[1])

    def test_reduce_scatter_tensor_coalesced_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(2, 1)
        with dist._coalescing_manager():
            dist.reduce_scatter_tensor(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.chunk(2)[1])

    @parametrize("rank", [0, 1])
    def test_allgather_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_tensor = torch.ones(3, 3) * 42
        output_tensors = [torch.empty(3, 3) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out in output_tensors:
            self.assertEqual(out, input_tensor)

    def test_allgather_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        input_tensor = torch.ones(3, 3).requires_grad_(True)
        output_tensors = [torch.empty(3, 3) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out in output_tensors:
            self.assertEqual(out, input_tensor)
            self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_gather_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_tensor = torch.ones(3, 3) * 42
        if rank == 0:
            gather_list = [torch.empty(3, 3) for _ in range(2)]
            dist.gather(input_tensor, gather_list)
            for out in gather_list:
                self.assertEqual(out, input_tensor)
        else:
            dist.gather(input_tensor, None, dst=0)

    def test_gather_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        input_tensor = torch.ones(3, 3).requires_grad_(True)
        gather_list = [torch.empty(3, 3) for _ in range(2)]
        dist.gather(input_tensor, gather_list)
        for out in gather_list:
            self.assertEqual(out, input_tensor)
            self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_allgather_coalesced_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        inputs = [torch.ones(3, 3) * i for i in range(3)]
        output_lists = [[torch.empty(3, 3) for _ in inputs] for _ in range(2)]
        dist.all_gather_coalesced(output_lists, inputs)
        for output_list in output_lists:
            for i, out in enumerate(output_list):
                self.assertEqual(out, inputs[i])

    def test_allgather_coalesced_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        inputs = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output_lists = [[torch.empty(3, 3) for _ in inputs] for _ in range(2)]
        dist.all_gather_coalesced(output_lists, inputs)
        for output_list in output_lists:
            for i, out in enumerate(output_list):
                self.assertEqual(out, inputs[i])
                self.assertFalse(out.requires_grad)

    @parametrize("rank", [0, 1])
    def test_alltoall_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        input_list = [torch.ones(3, 3) * i for i in range(2)]
        output_list = [torch.empty(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        for i in range(2):
            self.assertEqual(output_list[i], input_list[i])

    def test_alltoall_requires_grad(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        input_list = [
            torch.ones(3, 3).requires_grad_(True),
            (torch.ones(3, 3) * 2).requires_grad_(True),
        ]
        output_list = [torch.empty(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        for i in range(2):
            self.assertEqual(output_list[i], input_list[i])

    @parametrize("rank", [0, 1])
    def test_alltoall_base_copy_semantics(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(4, 2)
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor)

    @parametrize("rank", [0, 1])
    def test_alltoall_base_output_larger_than_input(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        # out_tensor is larger than in_tensor, values are first
        # filled with in_tensor then zeros
        in_tensor = torch.arange(4.0).reshape(4, 1)
        out_tensor = torch.full((8, 1), -1.0)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[2, 6],
            input_split_sizes=[1, 3],
        )
        expected = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0], [0.0], [0.0], [0.0], [0.0]]
        )
        self.assertEqual(out_tensor, expected)

    @parametrize("rank", [0, 1])
    def test_alltoall_base_output_smaller_than_input(self, rank):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=rank, world_size=2, store=store)

        # out_tensor is smaller than in_tensor, values are first
        # filled with values of in_tensor.
        in_tensor = torch.arange(8.0).reshape(8, 1)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[1, 3],
            input_split_sizes=[4, 4],
        )
        expected = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.assertEqual(out_tensor, expected)

    def test_alltoall_base_empty_output_split_sizes(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[],
            input_split_sizes=[2, 2],
        )
        self.assertEqual(out_tensor, in_tensor)

    def test_alltoall_base_empty_input_split_sizes(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(2, 1)
        dist.all_to_all_single(
            out_tensor,
            torch.empty(0, 1),
            output_split_sizes=[0, 2],
            input_split_sizes=[],
        )
        self.assertEqual(out_tensor, torch.zeros_like(out_tensor))

    def test_alltoall_base_equal_split_numel_mismatch(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertEqual(out_tensor, torch.arange(4.0).reshape(4, 1))

    def test_alltoall_base_flat_copy_equal_splits(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1)
        out_tensor = torch.empty(2, 2)
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor.view_as(out_tensor))

    def test_alltoall_base_flat_copy_explicit_splits(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(2, 2)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(
            out_tensor,
            in_tensor,
            output_split_sizes=[2, 2],
            input_split_sizes=[1, 1],
        )
        self.assertEqual(out_tensor, in_tensor.view_as(out_tensor))

    def test_alltoall_base_split_size_validation(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        with self.assertRaisesRegex(
            RuntimeError, "does not divide equally across group size"
        ):
            dist.all_to_all_single(torch.empty(3, 2), torch.ones(3, 2))

        with self.assertRaisesRegex(
            RuntimeError, "Number of tensor splits not equal to group size"
        ):
            dist.all_to_all_single(
                torch.empty(4, 2),
                torch.ones(4, 2),
                output_split_sizes=[4],
                input_split_sizes=[4],
            )

        with self.assertRaisesRegex(
            RuntimeError, "Split sizes doesn't match total dim 0 size"
        ):
            dist.all_to_all_single(
                torch.empty(4, 2),
                torch.ones(4, 2),
                output_split_sizes=[1, 1],
                input_split_sizes=[2, 2],
            )

    def test_alltoall_base_zero_sized_input(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(2, 1)
        dist.all_to_all_single(
            out_tensor,
            torch.empty(0, 1),
            output_split_sizes=[0, 2],
            input_split_sizes=[0, 0],
        )
        self.assertEqual(out_tensor, torch.zeros_like(out_tensor))

    def test_alltoall_base_zero_sized_output(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.empty(0, 1)
        dist.all_to_all_single(
            out_tensor,
            torch.arange(4.0).reshape(4, 1),
            output_split_sizes=[0, 0],
            input_split_sizes=[0, 4],
        )
        self.assertEqual(out_tensor, torch.empty_like(out_tensor))

    @parametrize("noncontiguous_buffer", ["input", "output"])
    def test_alltoall_base_requires_contiguous(self, noncontiguous_buffer):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(8.0).reshape(4, 2)
        out_tensor = torch.empty(4, 2)
        if noncontiguous_buffer == "input":
            in_tensor = torch.arange(8.0).reshape(2, 4).t()
        else:
            out_tensor = torch.empty(2, 4).t()

        with self.assertRaisesRegex(RuntimeError, "tensor must be contiguous"):
            dist.all_to_all_single(out_tensor, in_tensor)

    def test_alltoall_base_channels_last(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(32.0).reshape(2, 2, 2, 4)
        in_tensor = in_tensor.contiguous(memory_format=torch.channels_last)
        out_tensor = torch.empty_like(in_tensor)
        dist.all_to_all_single(out_tensor, in_tensor)
        self.assertEqual(out_tensor, in_tensor)

    def test_alltoall_list_size_validation(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        with self.assertRaisesRegex(RuntimeError, "does not match world size"):
            dist.all_to_all(
                [torch.empty(3, 3)],
                [torch.ones(3, 3), torch.ones(3, 3)],
            )

    def test_alltoall_base_requires_grad(self):
        # Real backends write into output from C++ kernels autograd never sees.
        # Without AutoDispatchBelowAutograd, copy_ into a narrow()/chunk() view
        # would fail when input requires grad.
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        in_tensor = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor = torch.empty(4, 1)
        dist.all_to_all_single(out_tensor, in_tensor)

        in_tensor2 = torch.arange(4.0).reshape(4, 1).requires_grad_(True)
        out_tensor2 = torch.empty(8, 1)
        dist.all_to_all_single(
            out_tensor2,
            in_tensor2,
            output_split_sizes=[2, 6],
            input_split_sizes=[1, 3],
        )

    def test_error_on_collective(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        # Test with error_on_collective=False (default behavior)
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # These should work normally
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor)
        self.assertEqual(tuple(tensor.shape), (3, 3))

        dist.destroy_process_group()

        # Test with error_on_collective=True
        from torch._C._distributed_c10d import FakeProcessGroup

        options = FakeProcessGroup.Options()
        options.error_on_collective = True

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=2, store=store, pg_options=options
        )

        # These should now raise errors
        tensor = torch.ones(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.all_reduce(tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            output_tensors = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(output_tensors, tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.broadcast(tensor, src=0)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.barrier()

    def test_fake_process_group_direct_usage_error(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with self.assertRaisesRegex(TypeError, r"No constructor defined"):
            fake_pg = FakeProcessGroup(rank=0, world_size=3)

            with SimpleTensorMode():
                tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                dist.all_reduce(tensor, group=fake_pg)

    def test_fake_process_group_proper_usage_dispatch(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __init__(self):
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.append(str(func))
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=3)

        with SimpleTensorMode() as mode:
            tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            dist.all_reduce(tensor)

        op_names = [str(op) for op in mode.ops]
        self.assertIn("aten.lift_fresh.default", op_names)
        self.assertIn("c10d.allreduce_.default", op_names)

    def test_reduce_scatter_wrong_input_list_size(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        output = torch.empty(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "invalid input tensor list size, must be world size"
        ):
            dist.reduce_scatter(output, [torch.ones(3, 3)])

    def test_scatter_wrong_input_list_size(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.empty(3, 3)
        with self.assertRaisesRegex(RuntimeError, "Incorrect input list size"):
            dist.scatter(output, [torch.ones(3, 3)])

    def test_scatter_non_root_rejects_input_list(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.ScatterOptions()
        opts.rootRank = 0
        output = torch.empty(3, 3)
        inputs = [[torch.ones(3, 3), torch.ones(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "requires empty input on non-root"):
            pg.scatter([output], inputs, opts)

    def test_gather_non_root_rejects_output_list(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.GatherOptions()
        opts.rootRank = 0
        output = [[torch.empty(3, 3), torch.empty(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "requires empty output on non-root"):
            pg.gather(output, [torch.ones(3, 3)], opts)

    def test_gather_invalid_root_rank(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.GatherOptions()
        opts.rootRank = 3
        output = [[torch.empty(3, 3), torch.empty(3, 3)]]
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            pg.gather(output, [torch.ones(3, 3)], opts)

    def test_reduce_scatter_wrong_output_list_size(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        pg = dist.distributed_c10d._get_default_group()

        opts = dist.distributed_c10d.ReduceScatterOptions()
        outputs = [torch.empty(3, 3), torch.empty(3, 3)]
        inputs = [[torch.ones(3, 3), torch.ones(3, 3)]]
        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.reduce_scatter(outputs, inputs, opts)

    def test_reduce_scatter_base_wrong_input_size(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_buf = torch.empty(3, 2)
        in_buf = torch.ones(3, 2)
        with self.assertRaisesRegex(
            RuntimeError,
            "input tensor must be the same size as output size times world size",
        ):
            dist._reduce_scatter_base(out_buf, in_buf)

    def test_allgather_coalesced_wrong_inner_list_size(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = [[torch.empty(3, 3)], []]
        inputs = [torch.ones(3, 3)]
        with self.assertRaisesRegex(RuntimeError, "invalid output size"):
            dist.all_gather_coalesced(output, inputs)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    @parametrize("rank", [0, 1, 2, 3])
    def test_split_group(self, rank):
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=rank,
            world_size=world_size,
            store=store,
            device_id=torch.device(device_type, 0),
        )

        parent_pg = dist.distributed_c10d._get_default_group()
        self.assertTrue(
            parent_pg._get_backend(torch.device(device_type)).supports_splitting
        )

        # Interleaved, unsorted subgroups: each rank shares a group with the
        # rank two away, and is not always listed first. split_group preserves
        # the order ranks are listed in, so the child rank is the position
        # within the subgroup as given -- which here differs from the parent
        # rank, making this a meaningful check of rank assignment.
        split_ranks = [[2, 0], [3, 1]]
        new_pg = dist.split_group(split_ranks=split_ranks)
        self.assertIsNotNone(new_pg)

        my_group = next(g for g in split_ranks if rank in g)
        self.assertEqual(new_pg.size(), len(my_group))
        self.assertEqual(new_pg.rank(), my_group.index(rank))
        # Independent cross-check: the child rank must map back to this
        # process's original global rank via the world's rank mapping, which is
        # built separately from the C++ backend's rank.
        self.assertEqual(
            dist.distributed_c10d.get_global_rank(new_pg, new_pg.rank()), rank
        )

        # Collectives on the split group should still work.
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor, group=new_pg)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_backend_filter(self):
        # A fake world's backend string is the bare name "fake", which
        # BackendConfig expands to every device fake supports. split_group's
        # filter validation used to re-expand it through
        # Backend.default_device_backend_map -- where fake is only the default
        # for hpu -- so every device-qualified filter was rejected as "not
        # present in the parent" and the bare "fake" filter selected hpu alone.
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=2,
            store=store,
            device_id=torch.device(device_type, 0),
        )
        parent_pg = dist.distributed_c10d._get_default_group()
        parent_devices = {d.type for d in parent_pg._device_types}
        self.assertIn(device_type, parent_devices)

        # A bare filter naming the parent's backend keeps every parent device.
        full = dist.split_group(split_ranks=[[0, 1]], backend="fake")
        self.assertEqual({d.type for d in full._device_types}, parent_devices)

        # A device-qualified filter keeps exactly the named devices. The C++
        # split additionally requires the filter to keep the parent's default
        # backend device, which for an all-fake parent is cpu.
        cpu_only = dist.split_group(split_ranks=[[0, 1]], backend="cpu:fake")
        self.assertEqual({d.type for d in cpu_only._device_types}, {"cpu"})

        pair = dist.split_group(
            split_ranks=[[0, 1]], backend=f"cpu:fake,{device_type}:fake"
        )
        self.assertEqual({d.type for d in pair._device_types}, {"cpu", device_type})

        # Filters that genuinely do not match the parent are still rejected.
        with self.assertRaisesRegex(ValueError, "is not present in the parent"):
            dist.split_group(split_ranks=[[0, 1]], backend="mps:fake")
        with self.assertRaisesRegex(ValueError, "Backend mismatch"):
            dist.split_group(split_ranks=[[0, 1]], backend="cpu:gloo")
        with self.assertRaisesRegex(ValueError, "is not present in the parent"):
            dist.split_group(split_ranks=[[0, 1]], backend="gloo")

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_non_member(self):
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=4,
            store=store,
            device_id=torch.device(device_type, 0),
        )

        # Rank 0 is in none of the splits, so it gets the non-member sentinel.
        new_pg = dist.split_group(split_ranks=[[1, 2, 3]])
        self.assertIs(new_pg, dist.GroupMember.NON_GROUP_MEMBER)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_split_group_store_not_retained(self):
        # split_group clones the parent's store, and the process group holds the
        # store only at the C++ level. FakeStore.clone() must therefore be a
        # real C++ method: a pure-Python Store.clone() override would be garbage
        # collected once the caller drops its reference, and the split would
        # fail with "pure virtual function Store::clone".
        def setup():
            dist.init_process_group(
                backend="fake",
                rank=0,
                world_size=2,
                store=FakeStore(),
                device_id=torch.device(device_type, 0),
            )

        setup()  # the only Python reference to the store is dropped here
        new_pg = dist.split_group(split_ranks=[[0, 1]])
        self.assertIsNotNone(new_pg)
        self.assertEqual(new_pg.size(), 2)

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    @parametrize("rank", [0, 1, 2, 3])
    def test_split_group_consistent_naming_after_partial_split(self, rank):
        # Regression test for https://github.com/pytorch/pytorch/issues/190396.
        #
        # _hash_ranks_to_str previously used len(_world.pg_names) as the
        # uniqueness suffix. Non-member ranks of a partial split don't register
        # the PG, so their counter stayed lower than member ranks. Subsequent
        # splits then computed different names for the same communicator on
        # different ranks, causing inconsistent teardown ordering and (for NCCL)
        # circular ncclCommFinalize waits that deadlock destroy_process_group.
        #
        # The fix uses _world.group_count as the salt. _process_group_name now
        # increments group_count on BOTH paths so it advances on every rank that
        # reaches it, including non-members, keeping it collective-consistent.
        # This test verifies group_count advances consistently even for non-member
        # ranks and that PG names are computed from it correctly.
        world_size = 4
        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            rank=rank,
            world_size=world_size,
            store=store,
            device_id=torch.device(device_type, 0),
        )
        import hashlib as _hashlib

        from torch.distributed import distributed_c10d

        # group_count starts at 1 (the default PG consumed count 0).
        count_after_init = distributed_c10d._world.group_count

        # Partial split: ranks 0,1,2 are members; rank 3 is not.
        partial = dist.split_group(split_ranks=[[0, 1, 2]])

        # group_count must advance on ALL ranks, including non-member rank 3,
        # because _process_group_name is called before the member check.
        self.assertEqual(distributed_c10d._world.group_count, count_after_init + 1)
        if rank == 3:
            self.assertNotIsInstance(partial, dist.ProcessGroup)
        else:
            self.assertIsInstance(partial, dist.ProcessGroup)

        # Full split: all ranks are members.
        full = dist.split_group(split_ranks=[[0, 2], [1, 3]])
        self.assertEqual(distributed_c10d._world.group_count, count_after_init + 2)
        self.assertIsInstance(full, dist.ProcessGroup)

        # PG name must use count_after_init+1 as the group_count salt
        # (the value at the time of the second split_group call, before it
        # was incremented). Co-participants of [0,2] and [1,3] each compute
        # the same name because group_count is consistent across all ranks.
        pg_name = distributed_c10d._world.pg_names[full]
        my_group = [0, 2] if rank in [0, 2] else [1, 3]
        rank_join = "_".join(map(str, my_group))
        expected = _hashlib.sha1(
            f"{rank_join}_{count_after_init + 1}".encode(), usedforsecurity=False
        ).hexdigest()
        self.assertEqual(pg_name, expected)


class TestFakePGUniformRanks(TestCase):
    """Tests for Options.simulate_uniform_ranks.

    See NOTE [FakeProcessGroup uniform-rank simulation]. The contract is that
    the group behaves as if every rank held data identical to this one, which
    makes every collective well defined from local inputs alone.
    """

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _init(self, rank, world_size):
        opts = FakeProcessGroup.Options()
        opts.simulate_uniform_ranks = True
        dist.init_process_group(
            backend="fake",
            rank=rank,
            world_size=world_size,
            store=FakeStore(),
            pg_options=opts,
        )
        return dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cpu")
        )

    def test_option_defaults_to_off(self):
        """Existing behavior must be untouched unless the flag is set."""
        self.assertFalse(FakeProcessGroup.Options().simulate_uniform_ranks)

    def test_create_internal_uses_fresh_default_options(self):
        first = FakeProcessGroup._create_internal(0, world_size=2)
        second = FakeProcessGroup._create_internal(0, world_size=2)

        self.assertIsNot(first.options, second.options)
        first.options.simulate_uniform_ranks = True
        self.assertFalse(second.options.simulate_uniform_ranks)

    def test_create_internal_preserves_explicit_none_options(self):
        backend = FakeProcessGroup._create_internal(0, world_size=2, options=None)

        self.assertIsNone(backend.options)

    def test_new_group_inherits_uniform_rank_simulation(self):
        opts = FakeProcessGroup.Options()
        opts.fake_option = 17
        opts.simulate_uniform_ranks = True
        dist.init_process_group(
            backend="cuda:fake",
            rank=0,
            world_size=2,
            store=FakeStore(),
            pg_options=opts,
        )

        process_group = dist.new_group(ranks=[0, 1], backend="fake")
        self.assertIsInstance(process_group, dist.ProcessGroup)
        backend = process_group._get_backend(torch.device("cpu"))
        self.assertIsInstance(backend, FakeProcessGroup)
        self.assertTrue(backend.options.simulate_uniform_ranks)
        self.assertEqual(backend.options.fake_option, 0)

        backend.options.simulate_uniform_ranks = False
        self.assertTrue(opts.simulate_uniform_ranks)

    def test_split_group_inherits_uniform_rank_simulation(self):
        self._init(rank=0, world_size=2)
        parent = dist.distributed_c10d._get_default_group()

        process_group = parent.split_group([0, 1], device_types=[torch.device("cpu")])

        self.assertIsInstance(process_group, dist.ProcessGroup)
        backend = process_group._get_backend(torch.device("cpu"))
        self.assertIsInstance(backend, FakeProcessGroup)
        self.assertTrue(backend.options.simulate_uniform_ranks)

    def test_allreduce_sum_scales_by_world_size(self):
        """Summing world_size identical tensors multiplies by world_size."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.ones(3)

        pg.allreduce([tensor]).wait()

        self.assertEqual(tensor, torch.full((3,), 4.0))

    def test_allreduce_avg_is_identity(self):
        """Averaging identical values leaves them unchanged."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.full((3,), 7.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.AVG

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.full((3,), 7.0))

    def test_allreduce_product_raises_to_the_world_size(self):
        """Multiplying world_size identical tensors is exponentiation."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.full((3,), 2.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.PRODUCT

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.full((3,), 16.0))

    def test_sparse_allreduce_sum_scales_values(self):
        """Sparse SUM scales the stored values without changing sparsity."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[0, 2]]), torch.tensor([2.0, 3.0]), size=(4,)
        ).coalesce()

        work = pg.allreduce_sparse([tensor])
        work.wait()

        expected = torch.tensor([8.0, 0.0, 12.0, 0.0])
        self.assertEqual(tensor.to_dense(), expected)
        self.assertEqual(work.result()[0].to_dense(), expected)

    def test_sparse_allreduce_product_raises(self):
        """Sparse PRODUCT remains unsupported under uniform-rank simulation."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.sparse_coo_tensor(
            torch.tensor([[0, 2]]), torch.tensor([2.0, 3.0]), size=(4,)
        ).coalesce()
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.PRODUCT

        with self.assertRaisesRegex(
            RuntimeError, "allreduce_sparse does not support PRODUCT"
        ):
            pg.allreduce_sparse([tensor], opts)

    def test_allreduce_premul_sum_applies_factor_then_scales(self):
        """PREMUL_SUM sums factor * x over every rank."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.full((3,), 3.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.PREMUL_SUM(0.5)

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.full((3,), 6.0))

    def test_allreduce_premul_sum_accepts_a_tensor_factor(self):
        pg = self._init(rank=0, world_size=4)
        tensor = torch.full((3,), 3.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.PREMUL_SUM(torch.tensor(0.5))

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.full((3,), 6.0))

    def test_allreduce_premul_sum_rejects_integral_tensors(self):
        pg = self._init(rank=0, world_size=4)
        tensor = torch.ones(3, dtype=torch.int64)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.PREMUL_SUM(0.5)

        with self.assertRaisesRegex(
            RuntimeError, "requires a floating point or complex tensor"
        ):
            pg.allreduce([tensor], opts)

    def test_allreduce_premul_sum_requires_a_factor(self):
        pg = self._init(rank=0, world_size=4)
        tensor = torch.ones(3)
        opts = dist.AllreduceOptions()
        reduce_op = dist.ReduceOp(dist.ReduceOp.SUM)
        reduce_op.op = dist.ReduceOp.PREMUL_SUM
        opts.reduceOp = reduce_op

        with self.assertRaisesRegex(RuntimeError, "without its scaling factor"):
            pg.allreduce([tensor], opts)

    @parametrize("op", [dist.ReduceOp.BAND, dist.ReduceOp.BOR])
    def test_allreduce_bitwise_and_or_are_identity(self, op):
        """AND and OR are idempotent, so equal operands reduce to themselves."""
        pg = self._init(rank=0, world_size=4)
        tensor = torch.tensor([0b1100, 0b1010], dtype=torch.int32)
        opts = dist.AllreduceOptions()
        opts.reduceOp = op

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.tensor([0b1100, 0b1010], dtype=torch.int32))

    @parametrize("world_size,expected", [(4, [0, 0]), (3, [0b1100, 0b1010])])
    def test_allreduce_bitwise_xor_cancels_in_pairs(self, world_size, expected):
        """XOR over equal operands depends only on the parity of world_size."""
        pg = self._init(rank=0, world_size=world_size)
        tensor = torch.tensor([0b1100, 0b1010], dtype=torch.int32)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.BXOR

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.tensor(expected, dtype=torch.int32))

    def test_reduce_scatter_sum_scales_the_local_chunk(self):
        """Each rank keeps its own chunk, scaled by the number of ranks."""
        pg = self._init(rank=2, world_size=4)
        output = torch.empty(1)

        pg.reduce_scatter_single(output, torch.tensor([1.0, 2.0, 3.0, 4.0])).wait()

        # chunk[2] is 3.0, summed across 4 identical ranks.
        self.assertEqual(output.item(), 12.0)

    def test_all_gather_replicates_local_input(self):
        """Unchanged by the flag: every slot already held the local value."""
        pg = self._init(rank=1, world_size=3)
        output = torch.empty(6)

        pg.all_gather_single(output, torch.tensor([5.0, 6.0])).wait()

        for chunk in output.chunk(3):
            self.assertEqual(chunk, torch.tensor([5.0, 6.0]))

    def test_all_to_all_single_preserves_split_structure(self):
        """The regression that motivated the flag.

        A caller that reshapes all-to-all output by [world_size, per_rank]
        needs every slot filled with a full-size segment. The default
        approximation copies a prefix and zeroes the rest, which yields the
        wrong element count once splits are uneven.
        """
        world_size = 4
        pg = self._init(rank=1, world_size=world_size)
        # Each rank sends 2 elements to every peer, so it receives 2 from each.
        send = torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0])
        recv = torch.empty(8)

        pg.all_to_all_single(recv, send, [2] * world_size, [2] * world_size).wait()

        # Every peer sends us the segment it would send to rank 1, and under
        # the uniform contract that is our own split at index 1.
        for slot in recv.chunk(world_size):
            self.assertEqual(slot, torch.tensor([20.0, 21.0]))

    def test_all_to_all_single_uses_equal_default_splits(self):
        self._init(rank=1, world_size=2)
        send = torch.tensor([10.0, 11.0, 20.0, 21.0])
        recv = torch.empty_like(send)

        dist.all_to_all_single(recv, send)

        self.assertEqual(recv, torch.tensor([20.0, 21.0, 20.0, 21.0]))

    def test_all_to_all_single_fills_a_2d_buffer(self):
        """Split sizes count rows along dim 0, not elements.

        A 2-D buffer with explicit splits has to be written in full: narrowing
        the flat view by the raw split value would write one element per slot
        instead of one row, leaving the rest of the buffer untouched.
        """
        pg = self._init(rank=1, world_size=4)
        output = torch.empty(4, 8)
        input_tensor = torch.arange(32.0).reshape(4, 8)

        pg.all_to_all_single(output, input_tensor, [1] * 4, [1] * 4).wait()

        # Row 1 is what this rank would send to a peer in its own position, so
        # under the uniform contract every slot receives it.
        expected = input_tensor[1].expand(4, 8)
        self.assertEqual(output, expected)

    def test_all_to_all_single_reshapes_like_torchrec(self):
        """Output must survive a [world_size, per_rank] view."""
        world_size = 16
        per_rank = 8
        pg = self._init(rank=0, world_size=world_size)
        send = torch.arange(world_size * per_rank, dtype=torch.float)
        recv = torch.empty(world_size * per_rank)

        pg.all_to_all_single(
            recv, send, [per_rank] * world_size, [per_rank] * world_size
        ).wait()

        # This view is what KeyedJaggedTensor.dist_init performs; under the
        # default approximation the element count does not line up.
        self.assertEqual(recv.view(world_size, per_rank).shape, (world_size, per_rank))

    def test_alltoall_list_form_uses_our_own_entry(self):
        """Every peer sends what it would send to our position."""
        pg = self._init(rank=2, world_size=3)
        inputs = [torch.full((2,), float(i)) for i in range(3)]
        outputs = [torch.empty(2) for _ in range(3)]

        pg.alltoall(outputs, inputs).wait()

        for output in outputs:
            self.assertEqual(output, torch.full((2,), 2.0))

    def test_alltoall_tiles_and_truncates_mismatched_slots(self):
        """The list form lets each slot have its own shape.

        Unlike all_to_all_single there is no size equality to lean on, so a
        slot wider than our contribution is tiled and a narrower one is cut
        short. Both must be written in full; a partial write would leave the
        tail holding whatever the caller's buffer already contained.
        """
        pg = self._init(rank=0, world_size=2)
        inputs = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([7.0, 8.0, 9.0])]
        wide = torch.full((7,), -1.0)
        narrow = torch.full((2,), -1.0)

        pg.alltoall([wide, narrow], inputs).wait()

        self.assertEqual(wide, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]))
        self.assertEqual(narrow, torch.tensor([1.0, 2.0]))

    def test_all_to_all_single_fills_asymmetric_slots(self):
        """The output shape the caller declared is the one it gets.

        Strict uniformity would force every output split to equal the input
        split at our own index, and rejecting anything else was tried: it
        breaks callers that declare asymmetric splits or a receive-only rank.
        Each slot is tiled or truncated to the width it asked for instead.
        """
        pg = self._init(rank=1, world_size=2)
        # Input split 1 is our segment, so every slot is filled from [1, 2, 3].
        send = torch.arange(4.0)
        recv = torch.empty(8)

        pg.all_to_all_single(recv, send, [2, 6], [1, 3]).wait()

        self.assertEqual(recv, torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))

    def test_all_to_all_single_zero_fills_a_receive_only_rank(self):
        self._init(rank=0, world_size=2)
        send = torch.tensor([7.0])
        recv = torch.full((2,), -1.0)

        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=[1, 1],
            input_split_sizes=[0, 1],
        )

        self.assertEqual(recv, torch.zeros(2))

    @parametrize("noncontiguous_buffer", ["input", "output"])
    def test_alltoall_rejects_noncontiguous_tensors(self, noncontiguous_buffer):
        pg = self._init(rank=0, world_size=1)
        inputs = [torch.arange(4.0).reshape(2, 2)]
        outputs = [torch.empty(2, 2)]
        if noncontiguous_buffer == "input":
            inputs[0] = inputs[0].t()
        else:
            outputs[0] = outputs[0].t()

        with self.assertRaisesRegex(
            RuntimeError, f"{noncontiguous_buffer} tensor must be contiguous"
        ):
            pg.alltoall(outputs, inputs)

    @parametrize("rank,expected", [(0, 4.0), (1, 1.0)])
    def test_reduce_writes_only_the_root(self, rank, expected):
        """Real backends leave every non-root tensor unspecified."""
        pg = self._init(rank=rank, world_size=4)
        tensor = torch.ones(3)
        opts = dist.ReduceOptions()
        opts.rootRank = 0

        pg.reduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.full((3,), expected))

    def test_allreduce_coalesced_scales_every_tensor(self):
        """Each tensor in the batch is reduced independently."""
        pg = self._init(rank=0, world_size=3)
        tensors = [torch.ones(2), torch.full((3,), 2.0)]

        pg.allreduce_coalesced(tensors).wait()

        self.assertEqual(tensors[0], torch.full((2,), 3.0))
        self.assertEqual(tensors[1], torch.full((3,), 6.0))

    def test_reduce_scatter_list_form_scales_our_entry(self):
        """We keep the input at our own index, summed over the world."""
        pg = self._init(rank=2, world_size=4)
        output = torch.empty(2)
        inputs = [[torch.full((2,), float(r)) for r in range(4)]]

        pg.reduce_scatter([output], inputs).wait()

        self.assertEqual(output, torch.full((2,), 8.0))

    def test_reduce_scatter_single_coalesced_scales_each_output(self):
        """Every buffer in the batch keeps its own chunk, scaled."""
        pg = self._init(rank=1, world_size=2)
        outputs = [torch.empty(2), torch.empty(1)]
        inputs = [torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 6.0])]

        pg.reduce_scatter_single_coalesced(outputs, inputs).wait()

        self.assertEqual(outputs[0], torch.tensor([6.0, 8.0]))
        self.assertEqual(outputs[1], torch.tensor([12.0]))

    @parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.PRODUCT])
    def test_bool_sum_and_product_are_identity(self, op):
        """Bool has no in-place form of the scaling the other dtypes use.

        Under c10d's nonzero-is-true convention both reductions leave equal
        bools alone, so take that branch rather than raise where a real
        backend succeeds.
        """
        pg = self._init(rank=0, world_size=4)
        tensor = torch.tensor([True, False])
        opts = dist.AllreduceOptions()
        opts.reduceOp = op

        pg.allreduce([tensor], opts).wait()

        self.assertEqual(tensor, torch.tensor([True, False]))

    def test_result_carries_the_reduced_tensor(self):
        """async_op callers read the output through Work.result()."""
        pg = self._init(rank=0, world_size=2)
        tensor = torch.ones(2)

        work = pg.allreduce([tensor])
        work.wait()

        self.assertEqual(work.result()[0], torch.full((2,), 2.0))

    def test_get_future_carries_collective_output(self):
        pg = self._init(rank=0, world_size=2)
        output = torch.empty(4)

        work = pg.all_gather_single(output, torch.tensor([3.0, 4.0]))

        self.assertEqual(
            work.get_future().value()[0], torch.tensor([3.0, 4.0, 3.0, 4.0])
        )

    def test_result_still_raises_without_the_flag(self):
        """Work::result() errors by default and must keep doing so.

        Nothing outside simulate_uniform_ranks records an output, so returning
        an empty list here would turn a diagnosable failure into a silent one.
        """
        pg = FakeProcessGroup._create_internal(0, world_size=2)

        work = pg.allreduce([torch.ones(2)])

        with self.assertRaisesRegex(RuntimeError, "recorded no output"):
            work.result()


instantiate_parametrized_tests(TestFakePG)
instantiate_parametrized_tests(TestFakePGUniformRanks)

if __name__ == "__main__":
    run_tests()
