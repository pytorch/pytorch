# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.ops.decomposed_ops import register_op_decomposition
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

aten = torch.ops.aten


# a utility to remove op from the op_strategy_funcs dict and
# register decomposition for testing purposes
def _register_op_decomposition(op):
    DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(op, None)
    register_op_decomposition(op)


class DistDecomposedOpsTest(DTensorTestBase):
    # test op decomposition with multiple outputs
    @with_comms
    def test_decomposed_aminmax(self):
        device_mesh = self.build_device_mesh()

        x = torch.rand(4, 8, device=self.device_type, dtype=torch.float32)
        y1, y2 = torch.aminmax(x)

        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y1_dt, y2_dt = torch.aminmax(x_dt)

        self.assertTrue(y1_dt.placements[0].is_replicate())
        self.assertTrue(y2_dt.placements[0].is_replicate())
        self.assertEqual(y1_dt.to_local(), y1)
        self.assertEqual(y2_dt.to_local(), y2)

        for dim in range(len(x.shape)):
            dist_x = distribute_tensor(x, device_mesh, [Shard(dim)])
            y1_dt, y2_dt = torch.aminmax(dist_x)
            self.assertEqual(y1_dt.placements[0], Partial("min"))
            self.assertEqual(y2_dt.placements[0], Partial("max"))
            self.assertEqual(y1_dt.full_tensor(), y1)
            self.assertEqual(y2_dt.full_tensor(), y2)

    # test op decomposition with possibly infeasible sharding input
    # i.e. when the input is sharded on the softmax_dim
    @with_comms
    def test_decompose_log_softmax_fwd(self):
        _register_op_decomposition(aten._log_softmax.default)
        assert (
            aten._log_softmax.default
            not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 8, device=self.device_type)
        softmax_dim = 1
        y = torch.nn.functional.softmax(x, dim=softmax_dim)

        # replicated input
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)

        self.assertTrue(y_dt.placements[0].is_replicate())
        # TODO(lty): numerical test doesn't work -- similar to the complex mul bug
        # self.assertEqual(y_dt.to_local(), y)

        # sharded input
        for shard_dim in range(len(x.shape)):
            x_dt = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)
            if shard_dim == softmax_dim:
                self.assertTrue(y_dt.placements[0].is_replicate())
            else:
                self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))

            # self.assertEqual(y_dt.full_tensor(), y)


if __name__ == "__main__":
    run_tests()
