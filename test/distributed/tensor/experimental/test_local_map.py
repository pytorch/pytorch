# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._utils import ExplicitRedistributionContext
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental import local_map
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


if torch.distributed._is_spmd_types_available():
    import spmd_types as spmd


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
            Y_dt = local_mm_forward(X_dt, W_dt)

        self.assertEqual(comm_mode.get_total_counts(), 0)
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
        with comm_mode:
            Y_dt = local_mul_forward(X_dt, 2.0)

        self.assertEqual(comm_mode.get_total_counts(), 0)
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
        with comm_mode:
            Y_dt_local = local_mm_forward(X_dt.to_local(), W_dt.to_local())

        self.assertEqual(comm_mode.get_total_counts(), 0)
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
        with comm_mode:
            Y_dt_local = local_mm_forward(X_dt.to_local(), W_dt.to_local())

        self.assertEqual(comm_mode.get_total_counts(), 0)
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

        self.assertEqual(comm_mode.get_total_counts(), 0)
        # output local shape should be (8, 4)
        self.assertEqual(Y_dt.to_local().shape, (8, 4))
        # output lives in mesh_2d
        self.assertEqual(Y_dt.device_mesh, mesh_2d)


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestLocalMapSpmdTypes(TestCase):
    """Single-process tests for local_map with spmd_types type checking."""

    WORLD_SIZE = 2

    @classmethod
    def setUpClass(cls):
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.mesh = init_device_mesh("cpu", (cls.WORLD_SIZE,), mesh_dim_names=("tp",))
        cls.tp_axis = spmd.MeshAxis.of(cls.mesh.get_group("tp"))

    @classmethod
    def tearDownClass(cls):
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        from spmd_types._mesh_axis import _reset

        _reset()
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()

    def tearDown(self):
        self.mode.__exit__(None, None, None)

    def test_local_spmd_types(self):
        """Verify local SPMD types inside the local_map region."""
        tp_axis = self.tp_axis

        # V@R -> V
        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Shard(0)], run_check=False
        )
        W_dt = DTensor.from_local(
            torch.randn(8, 4), self.mesh, [Replicate()], run_check=False
        )

        def mm_fn(X, W):
            self.assertIs(spmd.get_local_type(X)[tp_axis], spmd.V)
            self.assertIs(spmd.get_local_type(W)[tp_axis], spmd.R)
            out = torch.mm(X, W)
            self.assertIs(spmd.get_local_type(out)[tp_axis], spmd.V)
            return out

        wrapped = local_map(
            mm_fn,
            out_placements=[Shard(0)],
            in_placements=([Shard(0)], [Replicate()]),
            in_grad_placements=([Shard(0)], [Partial()]),
            device_mesh=self.mesh,
            spmd_types=True,
        )
        wrapped(X_dt, W_dt)

        # P -> P
        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Partial()], run_check=False
        )

        def check_p(X):
            X = X * 2
            self.assertIs(spmd.get_local_type(X)[tp_axis], spmd.P)
            return X

        wrapped = local_map(
            check_p,
            out_placements=[Partial()],
            in_placements=([Partial()],),
            in_grad_placements=([Replicate()],),
            device_mesh=self.mesh,
            spmd_types=True,
        )
        wrapped(X_dt)

    def test_replicate_grad_typing(self):
        """
        Replicate + grad Partial -> R (backward needs all-reduce).
        Replicate + grad Replicate -> I (backward is identity).
        """
        tp_axis = self.tp_axis
        W_dt = DTensor.from_local(
            torch.randn(8, 4), self.mesh, [Replicate()], run_check=False
        )

        def check_fn(W):
            self.assertIs(spmd.get_local_type(W)[tp_axis], spmd.R)
            return W

        # R w/ grad P: infers replicate local type
        wrapped = local_map(
            check_fn,
            out_placements=[Replicate()],
            in_placements=([Replicate()],),
            in_grad_placements=([Partial()],),
            device_mesh=self.mesh,
            spmd_types=True,
        )
        wrapped(W_dt)

        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Replicate()], run_check=False
        )

        # Replicate grad: infers invariant local type
        def check_fn(X):
            self.assertIs(spmd.get_local_type(X)[tp_axis], spmd.I)
            return X

        wrapped = local_map(
            check_fn,
            out_placements=[Replicate()],
            in_placements=([Replicate()],),
            in_grad_placements=([Replicate()],),
            device_mesh=self.mesh,
            spmd_types=True,
        )
        wrapped(X_dt)

    def test_incompatible_in_grad_placements(self):
        """Replicate fwd expects Partial or Replicate grad, not Shard."""
        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Replicate()], run_check=False
        )

        # R w/ grad S(0): raises incompatibility error
        wrapped = local_map(
            lambda X: X,
            out_placements=[Replicate()],
            in_placements=([Replicate()],),
            in_grad_placements=([Shard(0)],),
            device_mesh=self.mesh,
            spmd_types=True,
        )

        self.assertExpectedRaisesInline(
            ValueError,
            lambda: wrapped(X_dt),
            """in_grad_placements=S(0) is incompatible with in_placements=R. Valid grad placements for R: Partial or Replicate""",
        )

    def test_output_spmd_type_mismatch(self):
        """Output spmd type must be compatible with out_placements."""

        def check_mismatch(in_p, out_p, msg, in_grad_p=None):
            X_dt = DTensor.from_local(
                torch.randn(4, 8), self.mesh, in_p, run_check=False
            )
            kw = dict(
                out_placements=out_p,
                in_placements=(in_p,),
                device_mesh=self.mesh,
                spmd_types=True,
            )
            if in_grad_p is not None:
                kw["in_grad_placements"] = (in_grad_p,)
            wrapped = local_map(lambda X: X, **kw)
            self.assertExpectedRaisesInline(ValueError, lambda: wrapped(X_dt), msg)

        # V output vs Replicate
        check_mismatch(
            [Shard(0)],
            [Replicate()],
            """Output tensor placement mismatch on default_pg: out_placements=R but spmd_types inferred spmd.V""",
        )
        # R output vs Shard
        check_mismatch(
            [Replicate()],
            [Shard(0)],
            """Output tensor placement mismatch on default_pg: out_placements=S(0) but spmd_types inferred spmd.R""",
            in_grad_p=[Partial()],
        )
        # I output vs Shard
        check_mismatch(
            [Replicate()],
            [Shard(0)],
            """Output tensor placement mismatch on default_pg: out_placements=S(0) but spmd_types inferred spmd.I""",
            in_grad_p=[Replicate()],
        )
        # P output vs Replicate
        check_mismatch(
            [Partial()],
            [Replicate()],
            """Output tensor placement mismatch on default_pg: out_placements=R but spmd_types inferred spmd.P""",
            in_grad_p=[Replicate()],
        )
        # R output vs Partial
        check_mismatch(
            [Replicate()],
            [Partial()],
            """Output tensor placement mismatch on default_pg: out_placements=P(sum) but spmd_types inferred spmd.R""",
            in_grad_p=[Partial()],
        )
        # I output vs Partial
        check_mismatch(
            [Replicate()],
            [Partial()],
            """Output tensor placement mismatch on default_pg: out_placements=P(sum) but spmd_types inferred spmd.I""",
            in_grad_p=[Replicate()],
        )

    def test_output_missing_spmd_type(self):
        """Output tensor with no spmd_types annotation should raise."""
        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Shard(0)], run_check=False
        )
        unannotated = torch.randn(4, 4)

        def fn(X):
            return unannotated

        wrapped = local_map(
            fn,
            out_placements=[Shard(0)],
            in_placements=([Shard(0)],),
            device_mesh=self.mesh,
            spmd_types=True,
        )

        self.assertExpectedRaisesInline(
            ValueError,
            lambda: wrapped(X_dt),
            """Output tensor has no spmd_types annotation but out_placements expects one. Ensure the function's output is derived from annotated inputs or is explicitly annotated.""",
        )

    def test_output_spmd_type_is_stripped(self):
        from spmd_types._type_attr import _LOCAL_TYPE_ATTR

        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Shard(0)], run_check=False
        )
        outputs = []

        def fn(X):
            out = X * 2
            self.assertTrue(hasattr(out, _LOCAL_TYPE_ATTR))
            outputs.append(out)
            return out

        wrapped = local_map(
            fn,
            out_placements=[Shard(0)],
            in_placements=([Shard(0)],),
            device_mesh=self.mesh,
            spmd_types=True,
        )

        wrapped(X_dt)
        self.assertFalse(hasattr(outputs[0], _LOCAL_TYPE_ATTR))

    def test_out_spmd_types_to_grad_placements(self):
        from spmd_types._type_attr import _LOCAL_TYPE_ATTR

        from torch.distributed.tensor.experimental._func_map import (
            _out_spmd_types_to_grad_placements,
        )

        outs = (
            spmd.assert_type(torch.randn(4, 8), {self.tp_axis: spmd.R}),
            spmd.assert_type(torch.randn(4, 8), {self.tp_axis: spmd.I}),
            spmd.assert_type(torch.randn(4, 8), {self.tp_axis: spmd.P}),
            spmd.assert_type(torch.randn(4, 8), {self.tp_axis: spmd.V}),
            spmd.assert_type(torch.randn(4, 8), {self.tp_axis: spmd.V}),
        )
        out_placements = (
            [Replicate()],
            [Replicate()],
            [Partial()],
            [Shard(0)],
            [Partial()],
        )

        self.assertEqual(
            _out_spmd_types_to_grad_placements(
                outs,  # pyrefly: ignore [bad-argument-type]
                out_placements,
                self.mesh,
            ),
            (
                (Partial(),),
                (Replicate(),),
                (Replicate(),),
                (Shard(0),),
                (Replicate(),),
            ),
        )
        for out in outs:
            self.assertFalse(hasattr(out, _LOCAL_TYPE_ATTR))

    def test_unsupported_placement_type(self):
        """spmd_types=True should raise for unsupported placement types like _StridedShard."""
        from torch.distributed.tensor.placement_types import _StridedShard

        X_dt = DTensor.from_local(
            torch.randn(4, 8), self.mesh, [Shard(0)], run_check=False
        )

        wrapped = local_map(
            lambda X: X,
            out_placements=[Shard(0)],
            in_placements=([_StridedShard(0, split_factor=2)],),
            device_mesh=self.mesh,
            redistribute_inputs=True,
            spmd_types=True,
        )

        self.assertExpectedRaisesInline(
            ValueError,
            lambda: wrapped(X_dt),
            """local_map(spmd_types=True) does not support placement type _StridedShard: _S(0, 2)""",
        )


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestLocalMapSpmdTypesMultiGPU(DTensorTestBase):
    """Multi-GPU tests for local_map with spmd_types type checking."""

    @property
    def world_size(self):
        return 2

    @with_comms()
    def test_spmd_types_dp_matmul(self):
        """DP matmul with custom autograd functions.

        Data is Shard(0) (V), weights are Replicate (R). The correct backward
        all-reduces grad_W so every rank sees the full gradient. The buggy
        version skips the all-reduce.

        Step 1: BuggyDPMatmul without spmd_types -> wrong W gradient.
        Step 2: BuggyDPMatmul with spmd_types=True -> spmd.SpmdTypeError (V + I).
        Step 3: CorrectDPMatmul with spmd_types=True -> correct gradients.
        """
        from spmd_types import (
            assert_type,
            MeshAxis,
            register_autograd_function,
            register_local_autograd_function,
        )

        @register_local_autograd_function
        class BuggyDPMatmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, X, W):
                ctx.save_for_backward(X, W)
                return torch.mm(X, W)

            @staticmethod
            def backward(ctx, grad_output):
                X, W = ctx.saved_tensors
                grad_X = torch.mm(grad_output, W.t())
                grad_W = torch.mm(X.t(), grad_output)  # BUG: missing all-reduce
                return grad_X, grad_W

        @register_autograd_function
        class CorrectDPMatmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, X, W, mesh):
                ctx.save_for_backward(X, W)
                ctx.mesh = mesh
                return torch.mm(X, W)

            @staticmethod
            def backward(ctx, grad_output):
                X, W = ctx.saved_tensors
                grad_X = torch.mm(grad_output, W.t())
                grad_W = funcol.all_reduce(
                    torch.mm(X.t(), grad_output), "sum", ctx.mesh
                )
                return grad_X, grad_W, None

            @staticmethod
            def typecheck_forward(X, W, mesh):
                dp = MeshAxis.of(mesh.get_group("dp"))
                assert_type(X, {dp: spmd.V})
                assert_type(W, {dp: spmd.I})
                out = CorrectDPMatmul.apply(X, W, mesh)
                assert_type(out, {dp: spmd.V})
                return out

        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),
        )
        torch.manual_seed(42)

        X = torch.randn(4, 8, device=self.device_type, requires_grad=True)
        W = torch.randn(8, 4, device=self.device_type, requires_grad=True)

        # Single-node reference
        X_ref = X.detach().clone().requires_grad_()
        W_ref = W.detach().clone().requires_grad_()
        torch.mm(X_ref, W_ref).sum().backward()

        # Step 1: BuggyDPMatmul without spmd_types -> wrong W gradient.
        X.grad, W.grad = None, None
        X_dt = distribute_tensor(X, device_mesh, [Shard(0)])
        W_dt = distribute_tensor(W, device_mesh, [Replicate()])

        buggy = local_map(
            lambda X, W: BuggyDPMatmul.apply(X, W),
            out_placements=[Shard(0)],
            in_placements=([Shard(0)], [Replicate()]),
            device_mesh=device_mesh,
        )
        buggy(X_dt, W_dt).to_local().sum().backward()
        self.assertFalse(
            torch.allclose(W_dt.grad.full_tensor(), W_ref.grad),
            "Expected wrong W gradient from buggy (missing all-reduce)",
        )

        # Step 2: BuggyDPMatmul with spmd_types=True -> caught.
        X.grad, W.grad = None, None
        X_dt = distribute_tensor(X, device_mesh, [Shard(0)])
        W_dt = distribute_tensor(W, device_mesh, [Replicate()])

        buggy_typed = local_map(
            lambda X, W: BuggyDPMatmul.apply(X, W),
            out_placements=[Shard(0)],
            in_placements=([Shard(0)], [Replicate()]),
            in_grad_placements=([Shard(0)], [Replicate()]),
            device_mesh=device_mesh,
            spmd_types=True,
        )
        with self.assertRaises(spmd.SpmdTypeError):
            buggy_typed(X_dt, W_dt)

        # Step 3: CorrectDPMatmul with spmd_types=True -> correct gradients
        X.grad, W.grad = None, None
        X_dt = distribute_tensor(X, device_mesh, [Shard(0)])
        W_dt = distribute_tensor(W, device_mesh, [Replicate()])

        correct = local_map(
            lambda X, W: CorrectDPMatmul.apply(X, W, device_mesh),
            out_placements=[Shard(0)],
            in_placements=([Shard(0)], [Replicate()]),
            in_grad_placements=([Shard(0)], [Replicate()]),
            device_mesh=device_mesh,
            spmd_types=True,
        )
        correct(X_dt, W_dt).to_local().sum().backward()
        self.assertEqual(X_dt.grad.full_tensor(), X_ref.grad)
        self.assertEqual(W_dt.grad.full_tensor(), W_ref.grad)

    @with_comms()
    def test_spmd_types_backward_grad_placements(self):
        """
        Matching grad_out placements need no redistribution; mismatched grads get implicitly redistributed.
        """
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),
        )

        # S(0) @ R -> S(0) FWD: backward expects S(0) grad
        def s0_r_mm():
            X = torch.randn(4, 8, device=self.device_type, requires_grad=True)
            X_dt = distribute_tensor(X, device_mesh, [Shard(0)])
            W = torch.randn(8, 4, device=self.device_type, requires_grad=True)
            W_dt = distribute_tensor(W, device_mesh, [Replicate()])

            wrapped_shard = local_map(
                mm_forward,
                out_placements=[Shard(0)],
                in_placements=([Shard(0)], [Replicate()]),
                device_mesh=device_mesh,
                spmd_types=True,
            )
            return wrapped_shard(X_dt, W_dt), X_dt, W_dt

        Y_dt, X_dt, W_dt = s0_r_mm()
        grad_out = DTensor.from_local(
            torch.ones_like(Y_dt.to_local()), device_mesh, [Shard(0)]
        )
        with ExplicitRedistributionContext(strict=True):
            Y_dt.backward(grad_out)

        # Same, but S(1) grad -> all-to-all to redistribute to S(0)
        Y_dt, X_dt, W_dt = s0_r_mm()
        grad_out = distribute_tensor(
            torch.ones_like(Y_dt.full_tensor()), device_mesh, [Shard(1)]
        )
        with CommDebugMode() as comm_mode:
            Y_dt.backward(grad_out)
        # S(1) -> S(0) triggers shard_dim_alltoall
        self.assertEqual(
            comm_mode.get_comm_counts()[torch.ops._dtensor.shard_dim_alltoall], 1
        )

        # S(1) @ S(0) -> P: expects Replicate grad
        wrapped_partial = local_map(
            mm_forward,
            out_placements=[Partial()],
            in_placements=([Shard(1)], [Shard(0)]),
            device_mesh=device_mesh,
            spmd_types=True,
        )

        def s1_s0_mm():
            X = torch.randn(4, 8, device=self.device_type, requires_grad=True)
            X_dt = distribute_tensor(X, device_mesh, [Shard(1)])
            W = torch.randn(8, 4, device=self.device_type, requires_grad=True)
            W_dt = distribute_tensor(W, device_mesh, [Shard(0)])
            return wrapped_partial(X_dt, W_dt), X_dt, W_dt

        Y_dt, X_dt, W_dt = s1_s0_mm()
        grad_out = distribute_tensor(
            torch.ones_like(Y_dt.full_tensor()), device_mesh, [Replicate()]
        )
        with ExplicitRedistributionContext(strict=True):
            Y_dt.backward(grad_out)

        # Same, but Partial grad is mismatched -> redistributed (all-reduce)
        Y_dt, X_dt, W_dt = s1_s0_mm()
        grad_out = DTensor.from_local(
            torch.ones_like(Y_dt.to_local()), device_mesh, [Partial()]
        )
        with CommDebugMode() as comm_mode:
            Y_dt.backward(grad_out)
        self.assertEqual(comm_mode.get_comm_counts()[funcol_py.all_reduce], 1)

        # P output type with P output placement expects Replicate grad, so a
        # Partial grad_out redistributes via all-reduce.
        X = torch.randn(4, 8, device=self.device_type, requires_grad=True)
        X_dt = DTensor.from_local(X, device_mesh, [Partial()], run_check=False)
        wrapped_partial_out = local_map(
            lambda X: X * 2,
            out_placements=[Partial()],
            in_placements=([Partial()],),
            device_mesh=device_mesh,
            spmd_types=True,
        )
        Y_dt = wrapped_partial_out(X_dt)
        grad_out = DTensor.from_local(
            torch.ones_like(Y_dt.to_local()), device_mesh, [Partial()]
        )
        with CommDebugMode() as comm_mode:
            Y_dt.backward(grad_out)
        self.assertEqual(comm_mode.get_comm_counts()[funcol_py.all_reduce], 1)

        # V output type with S(1) output placement expects S(1) grad, so a
        # Partial grad_out redistributes via reduce-scatter along dim 1.
        X = torch.randn(4, 8, device=self.device_type, requires_grad=True)
        X_dt = distribute_tensor(X, device_mesh, [Shard(1)])
        wrapped_shard1_out = local_map(
            lambda X: X * 2,
            out_placements=[Shard(1)],
            in_placements=([Shard(1)],),
            device_mesh=device_mesh,
            spmd_types=True,
        )
        Y_dt = wrapped_shard1_out(X_dt)
        grad_out = DTensor.from_local(
            torch.ones_like(Y_dt.full_tensor()), device_mesh, [Partial()]
        )
        with CommDebugMode() as comm_mode:
            Y_dt.backward(grad_out)
        self.assertEqual(
            comm_mode.get_comm_counts()[funcol_py.reduce_scatter_tensor], 1
        )
        self.assertEqual(X_dt.grad.placements, (Shard(1),))
        self.assertEqual(X_dt.grad.to_local().shape, (4, 4))


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestLocalMapSpmdTypesMesh(TestCase):
    """Tests for local_map spmd_types with multi-dimensional meshes."""

    WORLD_SIZE = 4

    @classmethod
    def setUpClass(cls):
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        cls.mesh_ep = init_device_mesh("cpu", (4,), mesh_dim_names=("ep",))

    @classmethod
    def tearDownClass(cls):
        from spmd_types._mesh_axis import _reset

        if dist.is_initialized():
            dist.destroy_process_group()
        _reset()

    def setUp(self):
        from spmd_types._mesh_axis import _reset

        _reset()
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()

    def tearDown(self):
        self.mode.__exit__(None, None, None)

    def test_output_wrong_mesh(self):
        """Output reinterpreted to a different mesh should raise.

        TODO: support local_map with a different device_mesh + placements
        than the input DTensor's mesh -- reinterpret to that mesh, then check
        placements on it instead of raising.
        """
        X_dt = DTensor.from_local(
            torch.randn(4, 8),
            self.mesh_2d,
            [Shard(0), Shard(1)],
            run_check=False,
        )

        def fn_reinterpret(X):
            return spmd.reinterpret_mesh(X, self.mesh_ep)

        wrapped = local_map(
            fn_reinterpret,
            out_placements=[Shard(0), Shard(1)],
            in_placements=([Shard(0), Shard(1)],),
            device_mesh=self.mesh_2d,
            spmd_types=True,
        )

        self.assertExpectedRaisesInline(
            ValueError,
            lambda: wrapped(X_dt),
            """Output tensor has no spmd_types annotation on mesh_dp but out_placements expects S(0). Actual annotations are on: {default_pg}""",
        )


if __name__ == "__main__":
    run_tests()
