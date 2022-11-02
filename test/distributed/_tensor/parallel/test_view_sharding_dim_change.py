# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed._tensor.parallel._view_with_dim_change import (
    _view_with_sharding_dim_change,
)


class TPViewShardingDimChangeTest(DTensorTestBase):
    @with_comms
    def test_view_with_sharding_dim_change(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(2)]
        dt = DTensor.from_local(tensor, device_mesh, sharding)
        dt = _view_with_sharding_dim_change(dt, 1, (3, -1, 6))
        self.assertTrue(dt.placements[0].is_shard(dim=1))
        self.assertEqual(dt.to_local(), tensor.view(3, -1, 6))


if __name__ == "__main__":
    run_tests()
