# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental import local_map
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


funcol_py = torch.ops.c10d_functional


row_wise = [Shard(0)]  # row-wise sharding placements on 1-d mesh
col_wise = [Shard(1)]  # col-wise sharding placements on 1-d mesh
replicate = [Replicate()]  # replicate placements on 1-d mesh


def equal_allgather_forward(device_mesh, X, Y):
    eq = torch.tensor([torch.equal(X, Y)], device=X.device)
    eq_gather = funcol.all_gather_tensor(eq, 0, device_mesh)
    return torch.all(eq_gather).item()


def mm_all_gather_forward(device_mesh, A, B):
    local_mm_result = torch.mm(A, B)
    return funcol.all_gather_tensor(local_mm_result, 0, device_mesh).wait()


def mm_forward(A, B):  # no device mesh needed since we don't do collective
    return torch.mm(A, B)


def mm_allreduce_forward(device_mesh, A, B):
    partial_sum_tensor = torch.mm(A, B)
    return funcol.all_reduce(partial_sum_tensor, "sum", device_mesh).wait()


@local_map(
    out_placements=replicate,
    in_placements=(None, col_wise, row_wise),
)
def mm_allreduce_forward_decorated(device_mesh, A, B):
    partial_sum_tensor = torch.mm(A, B)
    return funcol.all_reduce(partial_sum_tensor, "sum", device_mesh).wait()


def mul_forward(X, scalar):  # no device mesh needed since we don't do collective
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

        # Y = X @ W
        X = torch.randn(16, 8, device=self.device_type, requires_grad=False)
        W = torch.randn(8, 12, device=self.device_type, requires_grad=False)
        Y = torch.mm(X, W)

        X_dt = distribute_tensor(
            X, device_mesh, col_wise
        )  # col-wisely sharded X tensor
        W_dt = distribute_tensor(
            W, device_mesh, row_wise
        )  # row-wisely sharded W tensor

        # Test 1: use the function returned from calling local_map
        # get the function wrapped with DTensor/Tensor conversion
        # mm_allreduce_forward is a function that applies to Tensors with manual collective
        # local_mm_allreduce_forward is the function that does the same but applies to
        # DTensors' `_local_tensor`.
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
        )
        with comm_mode:
            Y_dt = local_mm_allreduce_forward(device_mesh, X_dt, W_dt)

        # output redistribution to Replicate
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # check output placements
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        # check output value
        self.assertEqual(Y_dt.to_local(), Y)

        # Test 2: use the local_map decorator
        with comm_mode:
            Y_dt = mm_allreduce_forward_decorated(device_mesh, X_dt, W_dt)

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
        # Test 1: wrap out into DTensor w/ `out_placements`
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # X.equal(Y)
        X = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        Y = torch.randn(8, 8, device=self.device_type, requires_grad=False)
        X_dt = distribute_tensor(X, device_mesh, row_wise)
        Y_dt = distribute_tensor(Y, device_mesh, row_wise)
        local_equal_allgather_forward = local_map(
            equal_allgather_forward,
            out_placements=None,
        )
        with comm_mode:
            equal_dt = local_equal_allgather_forward(device_mesh, X_dt, Y_dt)  # a bool

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertTrue(not equal_dt)
        self.assertTrue(not (X.equal(Y)))

        # Test 2: directly return out if no argument is DTensor
        # matmul in DDP
        X = torch.randn(
            4 // self.world_size, 4, device=self.device_type, requires_grad=False
        )
        W = torch.randn(4, 4, device=self.device_type, requires_grad=False)
        local_mm_all_gather_forward = local_map(
            mm_all_gather_forward,
            out_placements=row_wise,
            in_placements=(None, row_wise, replicate),
        )
        with comm_mode:
            Y = local_mm_all_gather_forward(device_mesh, X, W)

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(
            comm_mode.get_comm_counts()[funcol_py.all_gather_into_tensor], 1
        )
        X_replicate = funcol.all_gather_tensor(X, 0, device_mesh).wait()
        Y_replicate = torch.mm(X_replicate, W)
        self.assertEqual(Y, Y_replicate)  # Y is a torch.Tensor

    # check for `in_placements` handling
    @with_comms
    def test_local_map_in_placements(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # Y = X @ W
        X = torch.randn(16, 8, device=self.device_type, requires_grad=False)
        W = torch.randn(8, 12, device=self.device_type, requires_grad=False)
        Y = torch.mm(X, W)

        X_dt = distribute_tensor(
            X, device_mesh, row_wise
        )  # row-wisely sharded X tensor
        W_dt = distribute_tensor(W, device_mesh, replicate)  # replicate W tensor

        # Test 1: explicitly pass `in_placements`
        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            in_placements=(row_wise, replicate),
            device_mesh=device_mesh,
        )
        with comm_mode:
            Y_dt = local_mm_forward(X_dt, W_dt)

        # with uneven sharding support, one all-gather for shape computation
        self.assertEqual(comm_mode.get_total_counts(), 1)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

        # Test 2: `in_placements=None`
        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            device_mesh=device_mesh,
        )
        comm_mode_2 = CommDebugMode()
        with comm_mode_2:
            Y_dt = local_mm_forward(X_dt, W_dt)

        self.assertEqual(comm_mode_2.get_total_counts(), 1)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

        # Test 3: `None` placements for non-Tensor input argument
        # Y = X * 2.0
        local_mul_forward = local_map(
            mul_forward,
            in_placements=(row_wise, None),
            out_placements=row_wise,
            device_mesh=device_mesh,
        )
        Y = torch.mul(X, 2.0)
        comm_mode_3 = CommDebugMode()
        with comm_mode_3:
            Y_dt = local_mul_forward(X_dt, 2.0)

        self.assertEqual(comm_mode_3.get_total_counts(), 1)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))
        self.assertEqual(Y_dt.full_tensor(), Y)

        # Test 4: `None` placements for Tensor input argument
        local_mm_forward = local_map(
            mm_forward,
            out_placements=None,
            in_placements=(None, None),
            device_mesh=device_mesh,
        )
        comm_mode_4 = CommDebugMode()
        with comm_mode_4:
            Y_dt_local = local_mm_forward(X_dt.to_local(), W_dt.to_local())

        # Test 4 has out_placements=None, so no communication
        self.assertEqual(comm_mode_4.get_total_counts(), 0)
        self.assertEqual(
            DTensor.from_local(Y_dt_local, device_mesh, row_wise).full_tensor(),
            torch.mm(X, W),
        )

        # Test 5: Some placements for Tensor input argument
        local_mm_forward = local_map(
            mm_forward,
            out_placements=None,
            in_placements=(replicate, row_wise),
            device_mesh=device_mesh,
        )
        comm_mode_5 = CommDebugMode()
        with comm_mode_5:
            Y_dt_local = local_mm_forward(X_dt.to_local(), W_dt.to_local())

        # Test 5 has out_placements=None, so no communication
        self.assertEqual(comm_mode_5.get_total_counts(), 0)
        self.assertEqual(
            DTensor.from_local(Y_dt_local, device_mesh, row_wise).full_tensor(),
            torch.mm(X, W),
        )

        # Test 6: expect error - `None` placements for DTensor input argument
        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            in_placements=(row_wise, None),
            device_mesh=device_mesh,
        )
        with self.assertRaisesRegex(AssertionError, "expects placements"):
            Y_dt = local_mm_forward(X_dt, W_dt)

    # check for `redistribute_inputs` handling
    @with_comms
    def test_local_map_redistribute(self):
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        comm_mode = CommDebugMode()

        # Y = X @ W
        X = torch.randn(16, 8, device=self.device_type, requires_grad=False)
        W = torch.randn(8, 12, device=self.device_type, requires_grad=False)
        Y = torch.mm(X, W)

        X_dt = distribute_tensor(
            X, device_mesh, row_wise
        )  # row-wisely sharded X tensor which will be redistributed
        W_dt = distribute_tensor(
            W, device_mesh, col_wise
        )  # col-wisely sharded W tensor which will be redistributed

        # Test 1: allow input redistribution
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=True,
        )
        with comm_mode:
            Y_dt = local_mm_allreduce_forward(device_mesh, X_dt, W_dt)

        # 2 for input redistribution and 1 for output
        self.assertEqual(comm_mode.get_total_counts(), 3)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_replicate())
        self.assertEqual(Y_dt.to_local(), Y)

        # Test 2: no input redistribution is allowed
        local_mm_allreduce_forward = local_map(
            mm_allreduce_forward,
            out_placements=replicate,
            in_placements=(None, col_wise, row_wise),
            device_mesh=device_mesh,
            redistribute_inputs=False,
        )
        with self.assertRaisesRegex(ValueError, "set redistribute_inputs=True"):
            Y_dt = local_mm_allreduce_forward(device_mesh, X_dt, W_dt)

    # check for `in_grad_placements` handling
    @with_comms()
    def test_local_map_with_grad_placement(self):
        """
        Test the gradient result is correct when we specify the right
        `in_grad_placements`.
        """
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        torch.manual_seed(12)

        # ground truth output, consider X as a batch of 2 on dim 0.
        X = torch.randn(4, 2, device=self.device_type, requires_grad=True)
        X1, X2 = torch.chunk(X, 2, dim=0)
        X1 = X1.detach().requires_grad_()
        X2 = X2.detach().requires_grad_()
        W = torch.randn(2, 4, device=self.device_type, requires_grad=True)
        Y1 = torch.mm(X1, W)
        Y2 = torch.mm(X2, W)
        loss = Y1.sum() + Y2.sum()
        loss.backward()

        in_placement_mismatch_choice = (False, True)
        for is_in_placement_mismatch in in_placement_mismatch_choice:
            if is_in_placement_mismatch:
                # in_placements for local_map() will take effect
                X_dt = distribute_tensor(X, device_mesh, replicate)
            else:
                # in_placements for local_map() will not take effect
                X_dt = distribute_tensor(X, device_mesh, row_wise)
            W_dt = distribute_tensor(W, device_mesh, replicate)
            in_grad_placements = ([Shard(0)], [Partial()])

            local_mm_forward = local_map(
                mm_forward,
                out_placements=[Shard(0)],
                in_placements=(row_wise, replicate),
                in_grad_placements=in_grad_placements,
                device_mesh=device_mesh,
                redistribute_inputs=True,
            )
            Y_dt = local_mm_forward(X_dt, W_dt)
            self.assertEqual(Y_dt.full_tensor(), torch.cat([Y1, Y2], dim=0))

            # Note: this is a way to simulate how DPP works. We don't need to
            # all_gather the loss. Instead, we do all_reduce to each distributed
            # weight.
            loss = Y_dt.to_local().sum()
            loss.backward()

            if not is_in_placement_mismatch:
                self.assertEqual(X_dt.grad.placements, in_grad_placements[0])
                self.assertEqual(W_dt.grad.placements, in_grad_placements[1])
            # regardless of is_in_placement_mismatch, grad output should always
            # match
            self.assertEqual(
                X_dt.grad.full_tensor(), torch.cat([X1.grad, X2.grad], dim=0)
            )
            self.assertEqual(W_dt.grad.full_tensor(), W.grad)

    @with_comms
    def test_local_map_uneven_sharding(self):
        """
        Test that local_map correctly handles uneven sharding where
        local shards have different sizes across ranks.
        """
        device_mesh = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )

        uneven_size = self.world_size * 2 + 1  # e.g., 5 for world_size=2
        X = torch.randn(uneven_size, 4, device=self.device_type, requires_grad=False)
        W = torch.randn(4, 8, device=self.device_type, requires_grad=False)
        Y = torch.mm(X, W)

        X_dt = distribute_tensor(X, device_mesh, row_wise)
        W_dt = distribute_tensor(W, device_mesh, replicate)

        local_x_size = X_dt.to_local().shape[0]
        expected_sizes = [
            (uneven_size + self.world_size - 1 - i) // self.world_size
            for i in range(self.world_size)
        ]
        self.assertEqual(local_x_size, expected_sizes[self.rank])

        local_mm_forward = local_map(
            mm_forward,
            out_placements=row_wise,
            in_placements=(row_wise, replicate),
            device_mesh=device_mesh,
        )

        Y_dt = local_mm_forward(X_dt, W_dt)

        self.assertEqual(Y_dt.full_tensor().shape, Y.shape)
        self.assertEqual(Y_dt.full_tensor(), Y)
        for placement in Y_dt.placements:
            self.assertTrue(placement.is_shard(dim=0))


class TestLocalMap4GPU(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_multi_mesh_inputs(self):
        """
        Test the function can be applied to accept DTensors that lives
        on different device meshes.
        """
        mesh_full = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size,)
        )
        mesh_2d = init_device_mesh(
            device_type=self.device_type, mesh_shape=(self.world_size // 2, 2)
        )
        comm_mode = CommDebugMode()

        X = torch.randn(8, 32, device=self.device_type, requires_grad=False)
        x_placements = [Shard(1)]
        W = torch.randn(16, 8, device=self.device_type, requires_grad=False)
        w_placements = [Shard(0), Shard(1)]

        X_dt = distribute_tensor(X, mesh_full, x_placements)
        W_dt = distribute_tensor(W, mesh_2d, w_placements)

        # local output shape should be (8, 4)
        output_placements = [Replicate(), Shard(1)]

        local_mm_forward = local_map(
            mm_forward,
            out_placements=output_placements,
            in_placements=(x_placements, w_placements),
            device_mesh=mesh_2d,
        )

        with comm_mode:
            Y_dt = local_mm_forward(X_dt, W_dt)

        # with uneven sharding support, we do one all-gather to compute global shape
        # for outputs with Shard placements (output has Shard(1))
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # output local shape should be (8, 4)
        self.assertEqual(Y_dt.to_local().shape, (8, 4))
        # output lives in mesh_2d
        self.assertEqual(Y_dt.device_mesh, mesh_2d)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_local_map_uneven_sharding_2d_mesh(self):
        """
        Test uneven sharding with 2D mesh where multiple dimensions
        can be unevenly sharded.
        """
        device_mesh = init_device_mesh(device_type=self.device_type, mesh_shape=(2, 2))

        X = torch.randn(9, 5, device=self.device_type, requires_grad=False)
        scalar = 2.0
        Y = torch.mul(X, scalar)

        X_dt = distribute_tensor(X, device_mesh, [Shard(0), Shard(1)])

        local_mul_forward = local_map(
            mul_forward,
            out_placements=[Shard(0), Shard(1)],
            in_placements=([Shard(0), Shard(1)], None),
            device_mesh=device_mesh,
        )

        Y_dt = local_mul_forward(X_dt, scalar)

        self.assertEqual(Y_dt.full_tensor().shape, Y.shape)
        self.assertEqual(Y_dt.full_tensor(), Y)


if __name__ == "__main__":
    run_tests()
