# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest
from itertools import product
from math import prod
from unittest.mock import patch

import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.config as dist_config
import torch.distributed.tensor as dtensor
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    linspace,
    logspace,
    Replicate,
    Shard,
    zeros,
)
from torch.distributed.tensor._random import manual_seed
from torch.distributed.tensor._symmetric_memory import (
    _compute_max_local_numel,
    copy_to_symmetric_memory,
)
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.experimental import use_symmetric_memory
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import PLATFORM_SUPPORTS_SYMM_MEM
from torch.testing._internal.common_utils import (
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorContinuousTestBase,
    LocalDTensorContinuousTestBase,
    NUM_DEVICES,
    with_comms,
)


class DTensorInitOpsTest(DTensorContinuousTestBase):
    world_size = NUM_DEVICES

    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        local_tensor_clone = torch.clone(input_tensor)
        torch.manual_seed(self.rank)
        local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
        torch.manual_seed(self.rank)
        dtensor = init_op(dtensor, *args, **kwargs)
        self.assertEqual(local_tensor_clone, dtensor.to_local())

    @with_comms
    def test_init_ops(self):
        # NOTE: random init tests are moved to test_random_ops.py
        self._run_init_op(torch.nn.init.constant_, 2.4)

    @with_comms
    def test_eye_init(self):
        # Test nn.init.eye_() with DTensor (issue #173357)
        device_mesh = self.build_device_mesh()
        shard_spec = [Replicate()]
        input_size = (8, 8)

        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        torch.nn.init.eye_(dtensor)

        expected = torch.eye(*input_size, device=self.device_type)
        self.assertEqual(expected, dtensor.to_local())


class DTensorConstructorTest(DTensorContinuousTestBase):
    world_size = 4

    def _run_init_op(self, init_op, dist_init_op, eq_op, *args, **kwargs):
        # 1d mesh test
        device_mesh = self.build_device_mesh()
        placements_list = [[Shard(0)], [Shard(1)], [Shard(2)], [Replicate()]]

        # even sharding
        tensor_size = [4, 8, 12]
        for placements in placements_list:
            local_tensor_size = tensor_size.copy()
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                local_tensor_size[shard_dim] //= self.world_size

            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            ones_expected = init_op(local_tensor_size, *args, **kwargs)
            eq_op(ones_expected, dist_tensor.to_local())

        # uneven sharding
        tensor_size = [5, 10, 15]
        for placements in placements_list:
            dist_tensor = dist_init_op(
                tensor_size,
                *args,
                **kwargs,
                device_mesh=device_mesh,
                placements=placements,
            )
            if isinstance(placements[0], Shard):
                shard_dim = placements[0].dim
                exp_tensor_list = list(
                    torch.chunk(
                        init_op(tensor_size, *args, **kwargs),
                        self.world_size,
                        dim=shard_dim,
                    )
                )

                @maybe_run_for_local_tensor
                def check_per_rank_chunk(rank, local_tensor):
                    if rank < len(exp_tensor_list):
                        eq_op(exp_tensor_list[rank], local_tensor)

                check_per_rank_chunk(self.rank, dist_tensor.to_local())
            else:
                exp_tensor = init_op(tensor_size, *args, **kwargs)
                eq_op(exp_tensor, dist_tensor.to_local())

        # empty shape
        local_tensor = dist_init_op(
            [], *args, **kwargs, device_mesh=device_mesh, placements=[Replicate()]
        ).to_local()
        expected_tensor = init_op([], *args, **kwargs)
        eq_op(expected_tensor, local_tensor)

    @with_comms
    def test_ones(self):
        self._run_init_op(
            torch.ones,
            torch.distributed.tensor.ones,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_empty(self):
        self._run_init_op(
            torch.empty,
            torch.distributed.tensor.empty,
            lambda x, y: (x.shape == y.shape)
            and (x.dtype == y.dtype)
            and (x.layout == y.layout),
            requires_grad=True,
        )

    @with_comms
    def test_full(self):
        self._run_init_op(
            torch.full,
            torch.distributed.tensor.full,
            self.assertEqual,
            123.4,
            requires_grad=True,
        )

    @with_comms
    def test_linspace(self):
        mesh = self.build_device_mesh()
        steps = 8

        for placements in ([Replicate()], [Shard(0)]):
            dist_tensor = linspace(
                1.0,
                2.0,
                steps,
                dtype=torch.float64,
                device_mesh=mesh,
                placements=placements,
            )
            self.assertEqual(dist_tensor.size(), torch.Size([steps]))
            self.assertEqual(
                dist_tensor.full_tensor(),
                torch.linspace(1.0, 2.0, steps, dtype=torch.float64),
            )

            dist_tensor = linspace(
                0.0, 1.0, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(0.0, 1.0, steps))

            dist_tensor = linspace(1.0, 2.0, 1, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([1]))
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(1.0, 2.0, 1))

            dist_tensor = linspace(1.0, 2.0, 0, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([0]))

            dist_tensor = linspace(
                2.0, -2.0, 5, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(2.0, -2.0, 5))

            start = DTensor.from_local(
                torch.tensor(1.0, device=self.device_type), mesh, [Replicate()]
            )
            end = DTensor.from_local(
                torch.tensor(2.0, device=self.device_type), mesh, [Replicate()]
            )
            dist_tensor = linspace(
                start, end, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.linspace(1.0, 2.0, steps))

            with self.assertRaisesRegex(
                RuntimeError,
                "linspace only supports 0-dimensional start and end tensors",
            ):
                linspace(
                    torch.tensor([1.0, 2.0], device=self.device_type),
                    2.0,
                    steps,
                    device_mesh=mesh,
                    placements=placements,
                )

    @with_comms
    def test_logspace(self):
        mesh = self.build_device_mesh()
        steps = 8

        for placements in ([Replicate()], [Shard(0)]):
            dist_tensor = logspace(
                1.0,
                2.0,
                steps,
                dtype=torch.float64,
                device_mesh=mesh,
                placements=placements,
            )
            self.assertEqual(dist_tensor.size(), torch.Size([steps]))
            self.assertEqual(
                dist_tensor.full_tensor(),
                torch.logspace(1.0, 2.0, steps, dtype=torch.float64),
            )

            dist_tensor = logspace(
                0.0, 1.0, steps, base=2.0, device_mesh=mesh, placements=placements
            )
            self.assertEqual(
                dist_tensor.full_tensor(), torch.logspace(0.0, 1.0, steps, base=2.0)
            )

            dist_tensor = logspace(1.0, 2.0, 1, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([1]))
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(1.0, 2.0, 1))

            dist_tensor = logspace(1.0, 2.0, 0, device_mesh=mesh, placements=placements)
            self.assertEqual(dist_tensor.size(), torch.Size([0]))

            dist_tensor = logspace(
                2.0, -2.0, 5, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(2.0, -2.0, 5))

            start = DTensor.from_local(
                torch.tensor(1.0, device=self.device_type), mesh, [Replicate()]
            )
            end = DTensor.from_local(
                torch.tensor(2.0, device=self.device_type), mesh, [Replicate()]
            )
            dist_tensor = logspace(
                start, end, steps, device_mesh=mesh, placements=placements
            )
            self.assertEqual(dist_tensor.full_tensor(), torch.logspace(1.0, 2.0, steps))

            with self.assertRaisesRegex(
                RuntimeError,
                "logspace only supports 0-dimensional start and end tensors",
            ):
                logspace(
                    torch.tensor([1.0, 2.0], device=self.device_type),
                    2.0,
                    steps,
                    device_mesh=mesh,
                    placements=placements,
                )

    @with_comms
    def test_zeros(self):
        self._run_init_op(
            torch.zeros,
            torch.distributed.tensor.zeros,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_zeros_full_mesh(self):
        # construct a gpu device 1d mesh
        mesh = self.build_device_mesh()
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([8, 3]))

        local_tensor = torch.zeros(8, 3)
        self.assertEqual(dist_tensor.to_local(), local_tensor)

        self.assertEqual(dist_tensor.device.type, self.device_type)

        # 1d sharded unevenly
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        @maybe_run_for_local_tensor
        def check_per_rank_tensors(rank, local_tensor):
            if rank <= 2:
                self.assertEqual(local_tensor.size(), torch.Size([8, 3]))
                self.assertEqual(torch.zeros(8, 3), local_tensor)
            else:
                self.assertEqual(local_tensor.size(), torch.Size([7, 3]))
                self.assertEqual(torch.zeros(7, 3), local_tensor)

        check_per_rank_tensors(self.rank, local_tensor)

        # construct a gpu device mesh with 2d: shard, replicate
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        placements = [Shard(0), Replicate()]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 4]))
        self.assertEqual(local_tensor, torch.zeros([16, 4]))

        # construct a gpu device mesh with 2d: shard, shard
        placements = [Shard(0), Shard(1)]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 2]))
        self.assertEqual(local_tensor, torch.zeros([16, 2]))

        # 2d sharded unevenly
        placements = [Shard(0), Shard(1)]
        size = [31, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        if self.rank == 0:
            self.assertEqual(local_tensor, torch.zeros([16, 2]))
        elif self.rank == 1:
            self.assertEqual(local_tensor, torch.zeros([16, 1]))
        elif self.rank == 2:
            self.assertEqual(local_tensor, torch.zeros([15, 2]))
        elif self.rank == 3:
            self.assertEqual(local_tensor, torch.zeros([15, 1]))

    @with_comms
    def test_zeros_submesh(self):
        # default world_size is 4
        # construct a gpu device 1d mesh, with no sub pg initialized
        sub_mesh_list = [0, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.zeros(0))

        # construct a gpu device 1d mesh: unevenly, with subpg initialized
        sub_mesh_list = [0, 1, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in sub_mesh_list:
            if self.rank != 3:
                self.assertEqual(local_tensor.size(), torch.Size([11, 3]))
                self.assertEqual(local_tensor, torch.zeros([11, 3]))
            else:
                self.assertEqual(local_tensor.size(), torch.Size([10, 3]))
                self.assertEqual(local_tensor, torch.zeros([10, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))

        # construct a gpu device 2d mesh, with no subpg initialized
        sub_mesh_list = [[0], [3]]
        mesh = DeviceMesh(self.device_type, sub_mesh_list)
        placements = [Shard(0), Shard(1)]
        size = [32, 3]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)
        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()

        if self.rank in [0, 3]:
            self.assertEqual(local_tensor.size(), torch.Size([16, 3]))
            self.assertEqual(local_tensor, torch.zeros([16, 3]))
        else:
            self.assertEqual(local_tensor.size(), torch.Size([0]))
            self.assertEqual(local_tensor, torch.tensor([]))


@unittest.skipIf(
    not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this platform"
)
@requires_cuda_p2p_access()
class DTensorSymmetricMemoryTest(DTensorContinuousTestBase):
    world_size = 2

    def _assert_is_symmetric_dtensor(self, dist_tensor):
        local_tensor = dist_tensor.to_local()
        self.assertTrue(symm_mem.is_symm_mem_tensor(local_tensor))
        self.assertTrue(local_tensor.is_contiguous())

    @with_comms
    def test_symmetric_memory_context_manager(self):
        mesh = self.build_device_mesh()
        placements = [Shard(1)]

        dist_tensor = dtensor.zeros((5, 7), device_mesh=mesh, placements=placements)
        self.assertFalse(symm_mem.is_symm_mem_tensor(dist_tensor.to_local()))

        with use_symmetric_memory():
            dist_tensor = dtensor.zeros((5, 7), device_mesh=mesh, placements=placements)
            self._assert_is_symmetric_dtensor(dist_tensor)
            torch.testing.assert_close(
                dist_tensor.full_tensor(),
                torch.zeros(5, 7, device=self.device_type),
            )

        self.assertFalse(dist_config.dtensor_use_symmetric_memory)

    @with_comms
    def test_symmetric_memory_config_assignment(self):
        mesh = self.build_device_mesh()
        old_value = dist_config.dtensor_use_symmetric_memory
        try:
            dist_config.dtensor_use_symmetric_memory = True
            dist_tensor = dtensor.ones((5, 7), device_mesh=mesh, placements=[Shard(1)])
            self._assert_is_symmetric_dtensor(dist_tensor)
            torch.testing.assert_close(
                dist_tensor.full_tensor(),
                torch.ones(5, 7, device=self.device_type),
            )
        finally:
            dist_config.dtensor_use_symmetric_memory = old_value

    @with_comms
    @parametrize(
        "init_op",
        [
            dtensor.empty,
            dtensor.zeros,
            dtensor.ones,
            dtensor.full,
            dtensor.rand,
            dtensor.randn,
            dtensor.linspace,
            dtensor.logspace,
        ],
    )
    @parametrize("requires_grad", [False, True])
    @parametrize("placement", [Shard(0), Replicate()])
    def test_symmetric_memory_factory_values(
        self, device, init_op, requires_grad, placement
    ):
        mesh = self.build_device_mesh()
        if init_op in (dtensor.linspace, dtensor.logspace):
            args = (1.0, 2.0, 7)
        elif init_op is dtensor.full:
            args = ((5, 7), 2.5)
        else:
            args = ((5, 7),)
        kwargs = dict(
            device_mesh=mesh, placements=[placement], requires_grad=requires_grad
        )
        manual_seed(1234, mesh)
        expected = init_op(*args, **kwargs)
        manual_seed(1234, mesh)
        with use_symmetric_memory():
            actual = init_op(*args, **kwargs)
        self._assert_is_symmetric_dtensor(actual)
        self.assertTrue(actual._local_tensor.is_leaf)
        self.assertIsNone(actual._local_tensor.grad_fn)
        self.assertEqual(actual._local_tensor.requires_grad, requires_grad)
        self.assertEqual(actual.device.type, torch.device(device).type)
        if init_op is not dtensor.empty:
            self.assertEqual(actual.to_local(), expected.to_local())
        if requires_grad:
            actual.sum().backward()
            expected.sum().backward()
            self.assertEqual(actual.grad, expected.grad)

    @with_comms
    @parametrize("fill_value", [True, 2, 2.5, 1 + 2j])
    def test_symmetric_memory_inferred_dtype(self, device, fill_value):
        mesh = self.build_device_mesh()
        expected = dtensor.full((5, 7), fill_value, device_mesh=mesh)
        with use_symmetric_memory():
            actual = dtensor.full((5, 7), fill_value, device_mesh=mesh)
        self._assert_is_symmetric_dtensor(actual)
        self.assertEqual(actual.to_local(), expected.to_local())
        self.assertEqual(actual.dtype, expected.dtype)

    @with_comms
    def test_distribute_tensor_copies_to_symmetric_memory(self):
        mesh = self.build_device_mesh()
        full_tensor = (
            torch.arange(35, device=self.device_type, dtype=torch.float32)
            .view(5, 7)
            .requires_grad_()
        )

        with use_symmetric_memory():
            dist_tensor = distribute_tensor(full_tensor, mesh, [Shard(1)])

        self._assert_is_symmetric_dtensor(dist_tensor)
        self.assertTrue(dist_tensor._local_tensor.is_leaf)
        self.assertTrue(dist_tensor._local_tensor.requires_grad)
        expected_chunks = list(torch.chunk(full_tensor, self.world_size, dim=1))
        torch.testing.assert_close(dist_tensor.to_local(), expected_chunks[self.rank])
        torch.testing.assert_close(dist_tensor.full_tensor(), full_tensor)

        with use_symmetric_memory():
            dist_tensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        self._assert_is_symmetric_dtensor(dist_tensor)
        torch.testing.assert_close(dist_tensor.to_local(), full_tensor)

    @with_comms
    def test_from_local_preserves_input_tensor(self):
        mesh = self.build_device_mesh()
        local_tensor = torch.ones(5, 7, device=self.device_type)

        with use_symmetric_memory():
            dist_tensor = DTensor.from_local(local_tensor, mesh, [Replicate()])

        self.assertEqual(dist_tensor.to_local().data_ptr(), local_tensor.data_ptr())
        self.assertFalse(symm_mem.is_symm_mem_tensor(dist_tensor.to_local()))

    @with_comms
    @parametrize("factory", [True, False])
    def test_symmetric_memory_rejects_nvshmem_submesh(self, factory):
        mesh = DeviceMesh(self.device_type, [0])
        tensor = torch.ones(5, device=self.device_type)
        with (
            use_symmetric_memory(),
            patch.object(symm_mem, "get_backend", return_value="NVSHMEM"),
            patch.object(symm_mem, "empty") as allocate,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "NVSHMEM requires a mesh containing every rank"
            ):
                if factory:
                    dtensor.zeros(5, device_mesh=mesh)
                else:
                    distribute_tensor(tensor, mesh, [Replicate()])
            allocate.assert_not_called()

    @with_comms
    @parametrize("factory", [True, False])
    def test_symmetric_memory_rejects_fake_mode(self, factory):
        mesh = self.build_device_mesh()
        with (
            use_symmetric_memory(),
            FakeTensorMode(allow_non_fake_inputs=True),
            patch.object(symm_mem, "empty") as allocate,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "not supported under torch.compile or FakeTensorMode"
            ):
                if factory:
                    dtensor.zeros(5, device_mesh=mesh)
                else:
                    tensor = torch.ones(5, device=self.device_type)
                    distribute_tensor(tensor, mesh, [Replicate()])
            allocate.assert_not_called()

    @with_comms
    def test_symmetric_memory_rejects_fake_input(self):
        mesh = self.build_device_mesh()
        with FakeTensorMode():
            tensor = torch.ones(5, device=self.device_type)
        with use_symmetric_memory(), patch.object(symm_mem, "empty") as allocate:
            with self.assertRaisesRegex(
                RuntimeError, "not supported under torch.compile or FakeTensorMode"
            ):
                copy_to_symmetric_memory(tensor, tensor.shape, mesh, [Replicate()])
            allocate.assert_not_called()

    @with_comms
    @parametrize("factory", [True, False])
    @parametrize("fullgraph", [False, True])
    def test_symmetric_memory_compile(self, device, factory, fullgraph):
        torch._dynamo.reset()
        mesh = self.build_device_mesh()

        def fn(tensor):
            if factory:
                return dtensor.zeros(5, device_mesh=mesh)
            return distribute_tensor(tensor, mesh, [Replicate()], src_data_rank=None)

        tensor = torch.ones(5, device=self.device_type)
        expected = fn(tensor)
        empty = symm_mem.empty

        def allocate_eagerly(*args, **kwargs):
            self.assertFalse(torch.compiler.is_compiling())
            self.assertIsNone(detect_fake_mode())
            return empty(*args, **kwargs)

        with (
            use_symmetric_memory(),
            patch.object(symm_mem, "empty", side_effect=allocate_eagerly) as allocate,
        ):
            compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph)
            if fullgraph:
                with self.assertRaisesRegex(
                    RuntimeError, "not supported under torch.compile or FakeTensorMode"
                ):
                    compiled(tensor)
                allocate.assert_not_called()
            else:
                actual = compiled(tensor)
                self._assert_is_symmetric_dtensor(actual)
                self.assertEqual(actual.to_local(), expected.to_local())
                again = compiled(tensor)
                self._assert_is_symmetric_dtensor(again)
                self.assertEqual(again.to_local(), expected.to_local())
                self.assertNotEqual(
                    actual.to_local().data_ptr(), again.to_local().data_ptr()
                )
                self.assertEqual(allocate.call_count, 2)
        torch._dynamo.reset()

    @with_comms
    def test_symmetric_memory_submesh(self, device):
        mesh = DeviceMesh(self.device_type, [0])
        with use_symmetric_memory():
            if symm_mem.get_backend(self.device_type) == "NVSHMEM":
                with self.assertRaisesRegex(
                    RuntimeError, "submeshes are not supported"
                ):
                    dtensor.ones(5, device_mesh=mesh)
                return
            tensor = dtensor.ones(5, device_mesh=mesh)
        if self.rank == 0:
            self._assert_is_symmetric_dtensor(tensor)
            self.assertEqual(tensor.to_local(), torch.ones(5, device=self.device_type))
        else:
            self.assertEqual(tensor.to_local().numel(), 0)

    @with_comms
    @parametrize("placement", [Shard(0), Replicate()])
    def test_symmetric_memory_noncontiguous_input(self, device, placement):
        mesh = self.build_device_mesh()
        tensor = torch.arange(35, device=self.device_type).view(5, 7).t()
        self.assertFalse(tensor.is_contiguous())
        with use_symmetric_memory():
            actual = distribute_tensor(tensor, mesh, [placement])
        self._assert_is_symmetric_dtensor(actual)
        self.assertEqual(actual.full_tensor(), tensor)

    @with_comms
    def test_symmetric_memory_cuda_graph_reuse(self, device):
        mesh = self.build_device_mesh()
        with use_symmetric_memory():
            tensor = dtensor.zeros(5, device_mesh=mesh, placements=[Shard(0)])
        local = tensor.to_local()
        handle = symm_mem.rendezvous(local, mesh.get_group())
        self.assertIsNotNone(handle)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            local.add_(1)
        for _ in range(3):
            graph.replay()
        self.assertEqual(local, torch.full_like(local, 3))


@unittest.skipIf(
    not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this platform"
)
@requires_cuda_p2p_access()
class DTensorSymmetricMemoryMeshTest(DTensorContinuousTestBase):
    world_size = 4
    device_mesh: DeviceMesh

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        super()._init_pg(rank, world_size, rdvz_file)
        ranks = torch.arange(world_size).view(2, 2)
        cls.device_mesh = DeviceMesh(cls.device_type, ranks)

    @with_comms
    @parametrize("init_op", [dtensor.ones, dtensor.full])
    @parametrize("size", [0, 1, 5, 7, 19])
    @parametrize(
        "placements",
        [
            [Shard(0), Shard(0)],
            [_StridedShard(0, split_factor=2), Shard(0)],
            [Shard(0), Shard(1)],
        ],
    )
    def test_symmetric_memory_multidimensional_mesh(
        self, device, init_op, size, placements
    ):
        mesh = self.device_mesh
        shape = torch.Size((size, 7))
        local_sizes = [
            prod(
                _compute_local_shape_and_global_offset(
                    shape, mesh.shape, coord, placements, skip_offset=True
                )[0]
            )
            for coord in product(range(2), repeat=2)
        ]
        max_numel = _compute_max_local_numel(shape, mesh, placements)
        self.assertGreaterEqual(max_numel, max(local_sizes))
        args = (shape, 1.0) if init_op is dtensor.full else (shape,)
        expected = init_op(*args, device_mesh=mesh, placements=placements)
        with use_symmetric_memory():
            actual = init_op(*args, device_mesh=mesh, placements=placements)
        local = actual.to_local()
        self.assertTrue(symm_mem.is_symm_mem_tensor(local))
        self.assertEqual(local, expected.to_local())
        handle = symm_mem.rendezvous(local, torch.distributed.group.WORLD)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.buffer_size, max(1, max_numel) * local.element_size())


DTensorConstructorTestWithLocalTensor = create_local_tensor_test_class(
    DTensorConstructorTest,
    base_class=LocalDTensorContinuousTestBase,
    skipped_tests=[
        # Non-contigous sub-meshes are not supported
        "test_zeros_submesh",
    ],
)

instantiate_device_type_tests(DTensorSymmetricMemoryTest, globals(), only_for="cuda")
instantiate_device_type_tests(
    DTensorSymmetricMemoryMeshTest, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
