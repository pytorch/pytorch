# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed._tensor.experimental import dmap
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

DEVICE_TYPE = "cpu"


def mm_allreduce_forward(device_mesh, W, X):
    partial_sum_tensor = torch.mm(W, X)
    reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh)
    return reduced_tensor


class TestDmap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_dmap(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )

        # Y = W @ X
        W = torch.randn(12, 8, requires_grad=False)
        X = torch.randn(8, 16, requires_grad=False)
        Y = torch.mm(W, X)

        row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
        dW = distribute_tensor(W, device_mesh, col_wise)  # col-wisely sharded W tensor
        dX = distribute_tensor(X, device_mesh, row_wise)  # row-wisely sharded X tensor
        # get the function wrapped with DTensor/Tensor convertion
        # mm_allreduce_forward is a function that applies to Tensors with manual collective
        # d_mm_allreduce_forward is the function that does the same but applies to DTensors
        d_mm_allreduce_forward = dmap(
            mm_allreduce_forward,
            out_placements=[Replicate()],
            device_mesh=device_mesh,
            in_placements=[col_wise, row_wise],
        )
        dY = d_mm_allreduce_forward(dW, dX)
        for placement in dY.placements:
            self.assertTrue(placement.is_replicate())
        self.assertEqual(dY.to_local(), Y)


if __name__ == "__main__":
    run_tests()
