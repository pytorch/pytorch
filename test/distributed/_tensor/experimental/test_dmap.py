# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from functools import partial

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor.experimental.dmap import dmap
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestDMap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_dmap(self) -> None:
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        colwise_shard = [Shard(1)]
        rowwise_shard = [Shard(0)]

        lhs_gloabl_tensor = torch.randn(3, 3 * self.world_size)
        rhs_global_tensor = torch.randn(3 * self.world_size, 3)

        lhs_dtensor = distribute_tensor(lhs_gloabl_tensor, mesh, colwise_shard)
        rhs_dtensor = distribute_tensor(rhs_global_tensor, mesh, rowwise_shard)

        @partial(dmap, mesh=mesh, out_placements=[Replicate()])
        def mm_manual_reduction(lhs, rhs):
            partial_sum_tensor = torch.mm(lhs, rhs)
            reduced_tensor = mesh.all_reduce(partial_sum_tensor)
            return reduced_tensor

        # dmap style computation
        dt_dmap_out = mm_manual_reduction(lhs_dtensor, rhs_dtensor)

        # dtensor style global computation
        dt_out = torch.mm(lhs_dtensor, rhs_dtensor).redistribute(mesh, [Replicate()])
        self.assertEqual(dt_dmap_out, dt_out)
