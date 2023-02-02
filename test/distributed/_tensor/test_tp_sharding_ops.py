# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import _Partial
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TPShardingOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_sharded_view(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(16, 35, 26)
        sharding = [Shard(0)]
        st = distribute_tensor(tensor, device_mesh, sharding).view(8, 4, 35, 13)
        st_new = distribute_tensor(tensor.view(8, 4, 35, 13), device_mesh, sharding)
        self.assertEqual(st.to_local(), st_new.to_local())
        self.assertEqual(st.placements[0], st_new.placements[0])

    @with_comms
    def test_sharded_transpose(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.transpose(0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=2))
        self.assertEqual(new_dt.to_local(), tensor.transpose(0, 2))
        new_dt = dist_tensor.transpose(1, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.to_local(), tensor.transpose(1, 2))

    @with_comms
    def test_sharded_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=1))
        self.assertEqual(new_dt.to_local(), tensor.permute(1, 0, 2))

    @with_comms
    def test_replicated_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Replicate()]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_replicate())
        self.assertEqual(new_dt.to_local(), tensor.permute(1, 0, 2))
        self.assertEqual(new_dt.stride(), tensor.permute(1, 0, 2).stride())

    @with_comms
    def test_split_partial_tensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        dist_tensor = DTensor.from_local(tensor, device_mesh, [_Partial()])
        with self.assertRaisesRegex(
            RuntimeError,
            "_Partial placement is not implemented",
        ):
            dist_tensor = dist_tensor.split(3)


if __name__ == "__main__":
    run_tests()
