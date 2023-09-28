# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard, zeros
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
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


class DTensorConstructorTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _run_init_op(self, init_op, dist_init_op, eq_op, *args, **kwargs):
        # 1d mesh test
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
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
                if self.rank < len(exp_tensor_list):
                    eq_op(exp_tensor_list[self.rank], dist_tensor.to_local())
            else:
                exp_tensor = init_op(tensor_size, *args, **kwargs)
                eq_op(exp_tensor, dist_tensor.to_local())

    @with_comms
    def test_ones(self):
        self._run_init_op(
            torch.ones,
            torch.distributed._tensor.ones,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_empty(self):
        self._run_init_op(
            torch.empty,
            torch.distributed._tensor.empty,
            lambda x, y: (x.shape == y.shape)
            and (x.dtype == y.dtype)
            and (x.layout == y.layout),
            requires_grad=True,
        )

    @with_comms
    def test_full(self):
        self._run_init_op(
            torch.full,
            torch.distributed._tensor.full,
            self.assertEqual,
            123.4,
            requires_grad=True,
        )

    @with_comms
    def test_zeros(self):
        self._run_init_op(
            torch.zeros,
            torch.distributed._tensor.zeros,
            self.assertEqual,
            requires_grad=True,
        )

    @with_comms
    def test_zeros_full_mesh(self):
        # construct a cuda device 1d mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
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
        if self.rank <= 2:
            self.assertEqual(local_tensor.size(), torch.Size([8, 3]))
            self.assertEqual(torch.zeros(8, 3), local_tensor)
        else:
            self.assertEqual(local_tensor.size(), torch.Size([7, 3]))
            self.assertEqual(torch.zeros(7, 3), local_tensor)

        # construct a cuda device mesh with 2d: shard, replicate
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        placements = [Shard(0), Replicate()]
        size = [32, 4]
        dist_tensor = zeros(size, device_mesh=mesh, placements=placements)

        self.assertEqual(dist_tensor.size(), torch.Size(size))
        local_tensor = dist_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([16, 4]))
        self.assertEqual(local_tensor, torch.zeros([16, 4]))

        # construct a cuda device mesh with 2d: shard, shard
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
        # construct a cuda device 1d mesh, with no sub pg initialized
        sub_mesh_list = [0, 3]
        mesh = DeviceMesh(self.device_type, sub_mesh_list, _init_process_groups=False)
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
            self.assertEqual(local_tensor, torch.tensor([]))

        # construct a cuda device 1d mesh: unevenly, with subpg initialized
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

        # construct a cuda device 2d mesh, with no subpg initialized
        sub_mesh_list = [[0], [3]]
        mesh = DeviceMesh(self.device_type, sub_mesh_list, _init_process_groups=False)
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


if __name__ == "__main__":
    run_tests()
