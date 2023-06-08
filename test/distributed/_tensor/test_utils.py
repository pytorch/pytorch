# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor._utils import compute_local_offset, compute_local_shape
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Replicate, Shard

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class UtilTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_compute_local_offset_1d(self):
        # mesh: 8 * 1
        mesh_tensor = torch.arange(self.world_size)
        device_mesh = DeviceMesh(self.device_type, mesh_tensor)
        my_rank = device_mesh.get_rank()
        size = torch.Size([10])

        placement = [Shard(0)]
        local_size = compute_local_shape(size, device_mesh, placement)
        local_offset = compute_local_offset(size, device_mesh, placement)

        tensor = torch.ones([10])
        tensor_lists = list(torch.chunk(tensor, self.world_size, dim=0))
        # chunk_sizes = [2, 2, 2, 2, 2, 0, 0, 0]
        chunk_sizes = [
            tensor_lists[idx].size(dim=0) if idx < len(tensor_lists) else 0
            for idx, tensor in enumerate(range(self.world_size))
        ]

        self.assertEqual(local_size[0], chunk_sizes[my_rank])
        # Offset for empty shard on the current dimension is equal to
        # global tensor dim size on the current dimension.
        self.assertEqual(local_offset[0], sum(chunk_sizes[:my_rank]))

    @with_comms
    def test_compute_local_shape_2d(self):
        # mesh: 4 * 2
        mesh_tensor = torch.arange(self.world_size).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        size = torch.Size([8, 6])

        # replicate, replicate
        placements1 = [Replicate(), Replicate()]
        local_size1 = compute_local_shape(size, mesh, placements1)
        self.assertEqual(local_size1, torch.Size([8, 6]))

        # replicate, shard
        placements2 = [Replicate(), Shard(0)]
        local_size2 = compute_local_shape(size, mesh, placements2)
        self.assertEqual(local_size2, torch.Size([4, 6]))

        # shard, shard
        placements3 = [Shard(0), Shard(1)]
        local_size3 = compute_local_shape(size, mesh, placements3)
        self.assertEqual(local_size3, torch.Size([2, 3]))

    @with_comms
    def test_compute_local_shape_2d_uneven(self):
        # mesh: 4 * 2
        mesh_tensor = torch.arange(self.world_size).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        size = torch.Size([7, 7])
        rank_coordinates = mesh.get_coordinate()

        # replicate, shard
        placements2 = [Replicate(), Shard(0)]
        local_size2 = compute_local_shape(size, mesh, placements2)
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size2, torch.Size([4, 7]))
        else:
            self.assertEqual(local_size2, torch.Size([3, 7]))

        # shard, shard
        placements3 = [Shard(0), Shard(1)]
        local_size3 = compute_local_shape(size, mesh, placements3)
        # first dim
        if rank_coordinates[0] < 3:
            self.assertEqual(local_size3[0], 2)
        else:
            self.assertEqual(local_size3[0], 1)
        # second dim
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size3[1], 4)
        else:
            self.assertEqual(local_size3[1], 3)


if __name__ == "__main__":
    run_tests()
