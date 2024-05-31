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
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental import local_map
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def equal_forward(device_mesh, X, Y):
    eq = torch.tensor([torch.equal(X, Y)], device=X.device)
    eq_gather = funcol.all_gather_tensor(eq, 0, device_mesh)
    return torch.all(eq_gather).item()


def mm_forward(device_mesh, W, X):
    return torch.mm(W, X)


def mm_allreduce_forward(device_mesh, W, X):
    partial_sum_tensor = torch.mm(W, X)
    reduced_tensor = funcol.all_reduce(partial_sum_tensor, "sum", device_mesh).wait()
    return reduced_tensor


def mul_forward(device_mesh, X, scalar):
    return torch.mul(X, scalar)


class TestLocalMap(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    # simple correctness check
    @with_comms
    def test_local_map_correctness(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # Y = W @ X
        W = torch.randn(12, 8, device=self.device_type, requires_grad=False)
        X = torch.randn(8, 16, device=self.device_type, requires_grad=False)
        Y = torch.mm(W, X)

        row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
        W_dt = distribute_tensor(
            W, device_mesh, col_wise
        )  # col-wisely sharded W tensor
        X_dt = distribute_tensor(
            X, device_mesh, row_wise
        )  # row-wisely sharded X tensor
        # get the function wrapped with DTensor/Tensor convertion
        # mm_allreduce_forward is a function that applies to Tensors with manual collective
        # local_mm_allreduce_forward is the function that does the same but applies to
        # DTensors' `_local_tensor`.
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=[Replicate()],
            in_placements=(col_wise, row_wise),
            device_mesh=device_mesh,
        )
        with comm_mode:
            Y_dt = local_mm_allreduce_forward(W_dt, X_dt)

        # output redistribution to Replicate
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # check output placements
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        # check output value
        self.assertEqual(Y_dt.to_local(), Y)

    # check for `out_placements`
    @with_comms
    def test_local_map_out_placements(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # X.equal(Y)
        X = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        Y = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        row_wise = [Shard(0)]
        X_dt = distribute_tensor(X, device_mesh, row_wise)
        Y_dt = distribute_tensor(Y, device_mesh, row_wise)
        local_equal_forward = local_map(equal_forward, out_placements=None)
        with comm_mode:
            equal_dt = local_equal_forward(X_dt, Y_dt)  # a bool

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertTrue(not equal_dt)
        self.assertTrue(not (X.equal(Y)))

    # check for `in_placements` handling
    @with_comms
    def test_local_map_in_placements(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # Y = W @ X
        W = torch.randn(12, 8, device=self.device_type, requires_grad=False)
        X = torch.randn(8, 16, device=self.device_type, requires_grad=False)
        Y = torch.mm(W, X)

        row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        replicate = [Replicate()]  # replicate placements on 1-d mesh
        W_dt = distribute_tensor(
            W, device_mesh, row_wise
        )  # row-wisely sharded W tensor
        X_dt = distribute_tensor(X, device_mesh, replicate)  # replicate X tensor

        # Test 1: explicitly pass `in_placements`
        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            in_placements=(row_wise, replicate),
            device_mesh=device_mesh,
        )
        with comm_mode:
            Y_dt = local_mm_forward(W_dt, X_dt)

        # no communication should occur in this case
        self.assertEqual(comm_mode.get_total_counts(), 0)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

        # Test 2: `in_placements=None`
        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            device_mesh=device_mesh,
        )
        with comm_mode:
            Y_dt = local_mm_forward(W_dt, X_dt)

        self.assertEqual(comm_mode.get_total_counts(), 0)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

        # Test 3: `None` placements for non-Tensor input argument
        local_mul_forward = local_map(
            mul_forward,
            in_placements=(row_wise, None),
            out_placements=row_wise,
            device_mesh=device_mesh,
        )
        Y = torch.mul(W, 2.0)
        with comm_mode:
            Y_dt = local_mul_forward(W_dt, 2.0)

        self.assertEqual(comm_mode.get_total_counts(), 0)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

    # check for `redistribute_inputs` handling
    @with_comms
    def test_local_map_redistribute(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # Y = W @ X
        W = torch.randn(12, 8, device=self.device_type, requires_grad=False)
        X = torch.randn(8, 16, device=self.device_type, requires_grad=False)
        Y = torch.mm(W, X)

        row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
        col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
        W_dt = distribute_tensor(
            W, device_mesh, row_wise
        )  # row-wisely sharded W tensor which will be redistributed
        X_dt = distribute_tensor(
            X, device_mesh, col_wise
        )  # col-wisely sharded X tensor which will be redistributed

        # Test 1: allow input redistribution
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=[Replicate()],
            in_placements=(col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=True,
        )
        with comm_mode:
            Y_dt = local_mm_allreduce_forward(W_dt, X_dt)

        # 2 for input redistribution and 1 for output
        self.assertEqual(comm_mode.get_total_counts(), 3)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        self.assertEqual(Y_dt.to_local(), Y)

        # Test 2: no input redistribution is allowed
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=[Replicate()],
            in_placements=(col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=False,
        )
        with self.assertRaisesRegex(ValueError, "set redistribute_inputs=True"):
            Y_dt = local_mm_allreduce_forward(W_dt, X_dt)


if __name__ == "__main__":
    run_tests()
