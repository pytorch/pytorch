# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.tensor._ops._decomposed_ops import register_op_decomposition
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _register_custom_decomposition(op, fn):
    DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(op, None)
    DTensor._op_dispatcher.sharding_propagator.register_op_decomposition(op, fn)


# where(c, x, y) = c * x + (~c) * y
def _where_decomp(condition, input, other):
    cond = condition
    x = input
    y = other
    if cond.dtype is not torch.bool:
        cond = aten.ne.Scalar(cond, 0)  # cond != 0
    not_cond = aten.logical_not.default(cond)
    return aten.add.Tensor(aten.mul.Tensor(cond, x), aten.mul.Tensor(not_cond, y))


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
        y = torch.nn.functional.log_softmax(x, dim=softmax_dim)

        # replicated input
        x_dt = distribute_tensor(x, device_mesh, [Replicate()])
        y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)

        self.assertTrue(y_dt.placements[0].is_replicate())
        self.assertEqual(y_dt.to_local(), y)

        # sharded input
        for shard_dim in range(len(x.shape)):
            x_dt = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            y_dt = torch.nn.functional.log_softmax(x_dt, dim=softmax_dim)

            if shard_dim == softmax_dim:
                self.assertTrue(y_dt.placements[0].is_replicate())
            else:
                self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))

            self.assertEqual(y_dt.full_tensor(), y)

    @with_comms
    def test_decompose_where_with_tensor_kwargs_basic(self):
        _register_custom_decomposition(aten.where.self, _where_decomp)

        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0

        ref = torch.where(cond, x, y)

        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Replicate()])
        cond_dt = distribute_tensor(cond, mesh, [Replicate()])

        out_dt = aten.where.self(condition=cond_dt, self=x_dt, other=y_dt)

        self.assertTrue(out_dt.placements[0].is_shard(dim=0))
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_decompose_where_with_tensor_kwargs_mixed(self):
        _register_custom_decomposition(aten.where.self, _where_decomp)

        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0

        ref = torch.where(cond, x, y)

        x_dt = distribute_tensor(x, mesh, [Replicate()])
        y_dt = distribute_tensor(y, mesh, [Shard(1)])
        cond_dt = distribute_tensor(cond, mesh, [Replicate()])

        out_dt = aten.where.self(cond_dt, self=x_dt, other=y_dt)

        self.assertTrue(out_dt.placements[0].is_shard(dim=1))
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_decompose_where_with_tensor_kwargs_both_sharded_same_dim(self):
        _register_custom_decomposition(aten.where.self, _where_decomp)

        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0

        ref = torch.where(cond, x, y)

        x_dt = distribute_tensor(x, mesh, [Shard(1)])
        y_dt = distribute_tensor(y, mesh, [Shard(1)])
        cond_dt = distribute_tensor(cond, mesh, [Replicate()])

        out_dt = aten.where.self(condition=cond_dt, self=x_dt, other=y_dt)

        self.assertTrue(out_dt.placements[0].is_shard(dim=1))
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_where_cond_sharded_preserve_shard(self):
        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0
        ref = torch.where(cond, x, y)

        cond_dt = distribute_tensor(cond, mesh, [Shard(1)])
        x_dt = distribute_tensor(x, mesh, [Replicate()])
        y_dt = distribute_tensor(y, mesh, [Replicate()])

        out_dt = torch.where(cond_dt, x_dt, y_dt)
        self.assertTrue(out_dt.placements[0].is_shard(dim=1))
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_where_cond_sharded_and_data_sharded_same_dim(self):
        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0
        ref = torch.where(cond, x, y)

        cond_dt = distribute_tensor(cond, mesh, [Shard(0)])
        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Replicate()])

        out_dt = torch.where(cond_dt, x_dt, y_dt)
        self.assertTrue(out_dt.placements[0].is_shard(dim=0))
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_where_cond_sharded_conflict_dims(self):
        mesh = self.build_device_mesh()
        x = torch.randn(8, 8, device=self.device_type)
        y = torch.randn(8, 8, device=self.device_type)
        cond = torch.randn(8, 8, device=self.device_type) > 0
        ref = torch.where(cond, x, y)

        cond_dt = distribute_tensor(cond, mesh, [Shard(0)])
        x_dt = distribute_tensor(x, mesh, [Shard(1)])
        y_dt = distribute_tensor(y, mesh, [Replicate()])

        out_dt = torch.where(cond_dt, x_dt, y_dt)
        self.assertEqual(out_dt.full_tensor(), ref)

    @with_comms
    def test_decompose__softmax_out(self):
        _register_op_decomposition(aten._softmax.out)
        assert (
            aten._softmax.out
            not in DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        )

        mesh = self.build_device_mesh()
        x = torch.rand(8, 8, device=self.device_type)
        softmax_dim = 1
        ref = torch.nn.functional.softmax(x, dim=softmax_dim)

        # replicated
        x_dt = distribute_tensor(x, mesh, [Replicate()])
        out_dt = distribute_tensor(torch.empty_like(x), mesh, [Replicate()])
        y_dt = aten._softmax.out(x_dt, softmax_dim, False, out=out_dt)
        self.assertTrue(y_dt.placements[0].is_replicate())
        self.assertEqual(y_dt.to_local(), ref)

        # sharded
        for shard_dim in range(x.ndim):
            x_dt = distribute_tensor(x, mesh, [Shard(shard_dim)])

            # choose the correct expected placement for the OUT buffer
            if shard_dim == softmax_dim:
                out_placement = [Replicate()]
            else:
                out_placement = [Shard(shard_dim)]

            out_dt = distribute_tensor(torch.empty_like(x), mesh, out_placement)
            y_dt = aten._softmax.out(x_dt, softmax_dim, False, out=out_dt)

            if shard_dim == softmax_dim:
                self.assertTrue(y_dt.placements[0].is_replicate())
            else:
                self.assertTrue(y_dt.placements[0].is_shard(dim=shard_dim))

            self.assertEqual(y_dt.full_tensor(), ref)


if __name__ == "__main__":
    run_tests()
