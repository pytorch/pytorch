# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor._collective_utils import shard_dim_alltoall

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestCollectiveUtils(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_shard_dim_alltoall(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.randn(12, self.world_size, device=self.device_type)
        new_tensor = shard_dim_alltoall(tensor, 0, 1, mesh, 0)

        meta_tensor = torch.randn(12, self.world_size, device="meta")
        new_meta_tensor = shard_dim_alltoall(meta_tensor, 0, 1, mesh, 0)

        self.assertEqual(new_tensor.shape, new_meta_tensor.shape)


if __name__ == "__main__":
    run_tests()
