# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed.config as dist_config
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.tensor as dtensor
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
    zeros,
)
from torch.distributed.tensor.experimental import use_symmetric_memory
from torch.testing._internal.common_distributed import PLATFORM_SUPPORTS_SYMM_MEM
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    with_comms,
)


class DTensorInitOpsTest(DTensorTestBase):
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


class DTensorConstructorTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

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
class DTensorSymmetricMemoryTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _assert_is_symmetric_dtensor(self, dist_tensor):
        local_tensor = dist_tensor.to_local()
        self.assertTrue(symm_mem.is_symm_mem_tensor(local_tensor))
        self.assertTrue(local_tensor.is_contiguous())

    @with_comms
    def test_symmetric_memory_context_manager(self):
        mesh = self.build_device_mesh()
        placements = [Shard(1)]

        dist_tensor = dtensor.zeros(
            (5, 7), device_mesh=mesh, placements=placements
        )
        self.assertFalse(symm_mem.is_symm_mem_tensor(dist_tensor.to_local()))

        with use_symmetric_memory():
            dist_tensor = dtensor.zeros(
                (5, 7), device_mesh=mesh, placements=placements
            )
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
            dist_tensor = dtensor.ones(
                (5, 7), device_mesh=mesh, placements=[Shard(1)]
            )
            self._assert_is_symmetric_dtensor(dist_tensor)
            torch.testing.assert_close(
                dist_tensor.full_tensor(),
                torch.ones(5, 7, device=self.device_type),
            )
        finally:
            dist_config.dtensor_use_symmetric_memory = old_value

    @with_comms
    def test_symmetric_memory_factory_values(self):
        mesh = self.build_device_mesh()
        tests = [
            (dtensor.empty, (), None),
            (dtensor.zeros, (), torch.zeros),
            (dtensor.ones, (), torch.ones),
            (
                dtensor.full,
                (2.5,),
                lambda size, **kwargs: torch.full(size, 2.5, **kwargs),
            ),
            (dtensor.rand, (), None),
            (dtensor.randn, (), None),
        ]

        with use_symmetric_memory():
            for init_op, args, expected_op in tests:
                dist_tensor = init_op(
                    (5, 7),
                    *args,
                    device_mesh=mesh,
                    placements=[Shard(1)],
                )
                self._assert_is_symmetric_dtensor(dist_tensor)
                if expected_op is not None:
                    expected = expected_op((5, 7), device=self.device_type)
                    torch.testing.assert_close(dist_tensor.full_tensor(), expected)

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


DTensorConstructorTestWithLocalTensor = create_local_tensor_test_class(
    DTensorConstructorTest,
    skipped_tests=[
        # Non-contigous sub-meshes are not supported
        "test_zeros_submesh",
    ],
)

if __name__ == "__main__":
    run_tests()
