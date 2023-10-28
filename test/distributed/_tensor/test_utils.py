# Owner(s): ["oncall: distributed"]

import itertools

import torch
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor._utils import (
    compute_local_shape,
    compute_local_shape_and_global_offset,
)
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

    @with_comms
    def test_compute_local_shape_and_global_offset_1D(self):
        one_d_placements = [[Shard(0)], [Replicate()]]

        for placements in one_d_placements:
            mesh_tensor = torch.arange(self.world_size)
            device_mesh = DeviceMesh(self.device_type, mesh_tensor)
            global_tensor = torch.arange(64).view(8, 8)
            global_shape = global_tensor.size()

            dtensor = distribute_tensor(global_tensor, device_mesh, placements)
            local_size, global_offset = compute_local_shape_and_global_offset(
                global_shape, device_mesh, placements
            )

            # TODO: make this test cleaner and work for nD
            dim0_start = global_offset[0]
            dim0_end = global_offset[0] + local_size[0]

            # Check the local tensor of dtensor is exactly the same
            # if we slice the global_tensor with local_size and global_offset
            self.assertEqual(
                dtensor.to_local(),
                global_tensor[dim0_start:dim0_end],
            )

    @with_comms
    def test_compute_local_shape_and_global_offset_2D(self):
        two_d_placements_options = [Shard(0), Shard(1), Replicate()]
        # Generating 6 two-d placements combinations
        two_d_placements = list(
            itertools.combinations_with_replacement(two_d_placements_options, 2)
        )

        for placements in two_d_placements:
            # mesh: 2 * 4
            mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
            device_mesh = DeviceMesh(self.device_type, mesh_tensor)
            global_tensor = torch.arange(64).view(8, 8)
            global_shape = global_tensor.size()

            dtensor = distribute_tensor(global_tensor, device_mesh, placements)
            local_size, global_offset = compute_local_shape_and_global_offset(
                global_shape, device_mesh, placements
            )

            # TODO: make this test cleaner and work for nD
            dim0_start = global_offset[0]
            dim0_end = global_offset[0] + local_size[0]
            dim1_start = global_offset[1]
            dim1_end = global_offset[1] + local_size[1]

            # Check the local tensor of dtensor is exactly the same
            # if we slice the global_tensor with local_size and global_offset
            self.assertEqual(
                dtensor.to_local(),
                global_tensor[dim0_start:dim0_end, dim1_start:dim1_end],
            )


if __name__ == "__main__":
    run_tests()
