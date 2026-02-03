# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any, Optional
from unittest import skip

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch import Tensor
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops._math_ops import _NormPartial
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorOpTestBase,
    LocalDTensorOpTestBase,
    map_local_for_rank,
    skip_unless_torch_gpu,
    with_comms,
)


def no_op():
    return None


def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    Recursively convert (over Sequence and Dict types) Tensors into DTensors.

    :param device_mesh: the DeviceMesh to use.
    :param placements: the Placement list to use.
    :return: the transformed structure.
    """

    def f(x):
        if isinstance(x, Tensor) and not isinstance(x, DTensor):
            return distribute_tensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
        return x

    return pytree.tree_map(f, [val])[0]


def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    Recursive convert any DTensor to local Tensor.

    :param val: the structure to coerce.
    :return: the coerced structure.
    """

    def f(x):
        if isinstance(x, DTensor):
            return x.full_tensor()
        return x

    return pytree.tree_map(f, [val])[0]


class DistElementwiseOpsTest(DTensorOpTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        pre_op_fn: Optional[Callable] = None,
        args: Sequence[Any] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        if not kwargs:
            kwargs = {}

        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )

        pre_op_fn()

        # run the reference first, in case the call is broken;
        # it's better to debug an incorrect call at this point.
        reference_result = op(*args, **kwargs)

        pre_op_fn()

        dist_result = op(*dargs, **dkwargs)

        collected_result = deepcopy_convert_from_dtensor(dist_result)

        self.assertEqualOnRank(reference_result, collected_result)

    # TODO: We need to add CPU tests for ops in the future.
    def _run_sharded_elementwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        pre_op_fn: Optional[Callable] = None,
        input_size: Sequence[int],
        op: Callable,
        **kwargs,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        input_tensor = torch.randn(
            *input_size,
            device=self.device_type,
            requires_grad=True,
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            pre_op_fn=pre_op_fn,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    @with_comms
    def test_partial_add(self):
        device_mesh = self.build_device_mesh()
        d_1 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.rand(2, 2), device_mesh, [Partial()])
        d_3 = d_1 + d_2
        self.assertTrue(d_3._spec.placements[0].is_partial())

    @with_comms
    def test_partial_replicate_add(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        for reduce_op in ("sum", "avg"):
            d_1 = DTensor.from_local(
                torch.rand(2, 2),
                device_mesh,
                [Partial(reduce_op=reduce_op)],
            )
            d_2 = DTensor.from_local(
                torch.rand(2, 1),
                device_mesh,
                [Replicate()],
                run_check=True,
            )

            with comm_mode:
                d_3 = d_1 + d_2

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(d_3.placements, (Partial(reduce_op=reduce_op),))
            self.assertEqual(d_3.full_tensor(), d_1.full_tensor() + d_2.full_tensor())

    @with_comms
    def test_activations(self):
        device_mesh = self.build_device_mesh()
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 12),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    @with_comms
    @skip(
        "testing RNG based ops is broken: https://github.com/pytorch/PiPPy/issues/494"
    )
    def test_dropout(self):
        device_mesh = self.build_device_mesh()

        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @with_comms
    @skip_unless_torch_gpu
    def test_dropout_backward(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]

        input_size = (8, 5)

        grad_output = torch.rand(
            input_size,
            device=self.device_type,
            requires_grad=True,
        )
        mask = (
            torch.rand(
                input_size,
                device=self.device_type,
                requires_grad=False,
            )
            < 0.8
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            op=torch.ops.aten.native_dropout_backward,
            kwargs=dict(
                grad_output=grad_output,
                mask=mask,
                scale=0.3,
            ),
        )

    @with_comms
    @skip_unless_torch_gpu
    def test_dropout_errors(self):
        device_mesh = self.build_device_mesh()
        with self.assertRaisesRegex(RuntimeError, "supported"):
            self._run_sharded_elementwise_ops(
                device_mesh=device_mesh,
                placements=[Partial("sum")],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    @with_comms
    def test_mul_out(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(self.rank)
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())

    @with_comms
    def test_mul_partial(self):
        # we only test the partial behavior for mul op as other placement
        # behaviors should be well tested in test_dtensor_ops.py
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        # 1. simple test for partial * partial
        d_1 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        d_2 = DTensor.from_local(torch.ones(2, 2), device_mesh, [Partial()])
        with comm_mode:
            d_3 = d_1 * d_2
        comm_counts = comm_mode.get_total_counts()
        self.assertEqual(comm_counts, 1)
        self.assertTrue(isinstance(d_3, DTensor))
        self.assertEqual(d_3.placements, (Partial(),))
        self.assertEqual(d_3.to_local(), torch.ones(2, 2) * (self.world_size))

        # 2. test the partial input DTensor * scalar/replicate input
        input = torch.full((8, 8), 1.0, device=self.device_type)

        # test for different types of other inputs
        other_inps = (
            2.0,  # scalar
            torch.tensor(2.0, device=self.device_type),  # scalar tensor
            torch.full((8, 8), 2.0, device=self.device_type),  # tensor
        )

        for partial_op in ["sum", "avg"]:
            expected_p_out = (
                input * self.world_size * 2.0 if partial_op == "sum" else input * 2.0
            )

            d_input = DTensor.from_local(input, device_mesh, [Partial(partial_op)])

            for other_inp in other_inps:
                if isinstance(other_inp, Tensor) and other_inp.numel() > 1:
                    d_other = distribute_tensor(other_inp, device_mesh, [Replicate()])
                else:
                    d_other = other_inp

                with comm_mode:
                    z = d_input * d_other

                comm_counts = comm_mode.get_total_counts()
                self.assertEqual(comm_counts, 0)
                self.assertTrue(isinstance(z, DTensor))
                self.assertEqual(z.placements, (Partial(partial_op),))
                self.assertEqual(z.full_tensor(), expected_p_out)

        # test other partial to assert the partial not getting propagated
        d_input = DTensor.from_local(input, device_mesh, [Partial("max")])
        d_other = distribute_tensor(torch.ones(8, 8), device_mesh, [Replicate()])

        z = d_input * d_other
        self.assertEqual(z.placements, (Replicate(),))
        self.assertEqual(z.to_local(), input)

    @with_comms
    def test_div_partial(self):
        # we only test the partial behavior for div op as other placement
        # behaviors should be well tested in test_dtensor_ops.py
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        # 1. test the partial input DTensor / scalar/replicate input
        # This is mathematically sound: (a1 + a2) / b = a1/b + a2/b
        input = torch.full((8, 8), 2.0, device=self.device_type)

        # test for different types of other inputs
        other_inps = (
            2.0,  # scalar
            torch.tensor(2.0, device=self.device_type),  # scalar tensor
            torch.full((8, 8), 2.0, device=self.device_type),  # tensor
        )

        for partial_op in ["sum", "avg"]:
            expected_p_out = (
                input * self.world_size / 2.0 if partial_op == "sum" else input / 2.0
            )

            d_input = DTensor.from_local(input, device_mesh, [Partial(partial_op)])

            for other_inp in other_inps:
                if isinstance(other_inp, Tensor) and other_inp.numel() > 1:
                    d_other = distribute_tensor(other_inp, device_mesh, [Replicate()])
                else:
                    d_other = other_inp

                with comm_mode:
                    z = d_input / d_other

                comm_counts = comm_mode.get_total_counts()
                self.assertEqual(comm_counts, 0)
                self.assertTrue(isinstance(z, DTensor))
                self.assertEqual(z.placements, (Partial(partial_op),))
                self.assertEqual(z.full_tensor(), expected_p_out)

        # test other partial to assert the partial not getting propagated
        d_input = DTensor.from_local(input, device_mesh, [Partial("max")])
        d_other = distribute_tensor(
            torch.full((8, 8), 2.0, device=self.device_type), device_mesh, [Replicate()]
        )

        z = d_input / d_other
        self.assertEqual(z.placements, (Replicate(),))
        self.assertEqual(z.to_local(), input / 2.0)

    @with_comms
    def test_masked_fill_scalar(self):
        """Test masked_fill_ with scalar value."""
        device_mesh = self.build_device_mesh()

        # Test with deterministic values to avoid random seed issues in threaded tests
        # Test with Shard(0) placement
        input_tensor = torch.arange(
            40, dtype=torch.float32, device=self.device_type
        ).reshape(8, 5)
        mask = input_tensor > 20
        fill_value = -999.0

        # Create DTensor
        dt_input = distribute_tensor(input_tensor.clone(), device_mesh, [Shard(0)])
        dt_mask = distribute_tensor(mask, device_mesh, [Shard(0)])

        # Perform in-place masked_fill
        input_tensor.masked_fill_(mask, fill_value)
        dt_input.masked_fill_(dt_mask, fill_value)

        # Compare results
        self.assertEqual(input_tensor, dt_input.full_tensor())

        # Test with Replicate placement
        input_tensor2 = (
            torch.arange(40, dtype=torch.float32, device=self.device_type).reshape(8, 5)
            - 20
        )
        mask2 = input_tensor2 < 0
        fill_value2 = 42.0

        dt_input2 = distribute_tensor(input_tensor2.clone(), device_mesh, [Replicate()])
        dt_mask2 = distribute_tensor(mask2, device_mesh, [Replicate()])

        input_tensor2.masked_fill_(mask2, fill_value2)
        dt_input2.masked_fill_(dt_mask2, fill_value2)

        self.assertEqual(input_tensor2, dt_input2.full_tensor())

        # Test with Shard(1) placement
        input_tensor3 = torch.arange(
            48, dtype=torch.float32, device=self.device_type
        ).reshape(4, 12)
        mask3 = input_tensor3 % 2 == 0  # even numbers
        fill_value3 = 0.0

        dt_input3 = distribute_tensor(input_tensor3.clone(), device_mesh, [Shard(1)])
        dt_mask3 = distribute_tensor(mask3, device_mesh, [Shard(1)])

        input_tensor3.masked_fill_(mask3, fill_value3)
        dt_input3.masked_fill_(dt_mask3, fill_value3)

        self.assertEqual(input_tensor3, dt_input3.full_tensor())

    @with_comms
    def test_inplace_op_partial_to_replicate(self):
        # test that in-place operations that require redistribution raise an error
        # to preserve aliasing semantics (issue #163374)
        device_mesh = self.build_device_mesh()

        input_tensor = torch.tensor(64.0, device=self.device_type)
        partial_dt = DTensor.from_local(
            input_tensor, device_mesh, placements=(Partial(),)
        )

        self.assertTrue(partial_dt.placements[0].is_partial())

        # Inplace ops that require placement changes (Partial -> Replicate) should error
        with self.assertRaisesRegex(
            RuntimeError,
            "in-place operations that require placement changes are not supported",
        ):
            partial_dt.clamp_(max=10)

    @with_comms
    def test_mul_div_scalar_partial(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()

        # regular partial *,/ scalar
        local_tensor = map_local_for_rank(self.rank, lambda rank: torch.tensor([rank]))

        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )

        res = aten.mul.Scalar(dt, 2)
        self.assertEqual(
            res.to_local(),
            map_local_for_rank(self.rank, lambda rank: torch.tensor([rank * 2])),
        )

        self.assertTrue(res._spec.placements[0].is_partial())
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        expected = sum(i for i in range(self.world_size)) * 2
        self.assertEqual(res, expected)

        res = aten.div.Scalar(dt, 2)
        self.assertEqual(
            res.to_local(),
            map_local_for_rank(self.rank, lambda rank: torch.tensor([rank / 2])),
        )

        self.assertTrue(res._spec.placements[0].is_partial())
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        expected = sum(i for i in range(self.world_size)) / 2
        self.assertEqual(res, expected)

    @with_comms
    def test_mul_div_scalar_norm_partial(self):
        mesh = self.build_device_mesh()
        aten = torch.ops.aten
        local_tensor = torch.tensor([1.0, 1.0, 7.0, 7.0])
        dt = distribute_tensor(local_tensor, mesh, [Shard(0)])

        norm = dt.norm()
        self.assertTrue(isinstance(norm._spec.placements[0], _NormPartial))

        res = aten.mul.Scalar(norm, 2)
        self.assertTrue(isinstance(res._spec.placements[0], _NormPartial))
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        self.assertEqual(res, 20)

        res = aten.div.Scalar(norm, 2)
        self.assertTrue(isinstance(res._spec.placements[0], _NormPartial))
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        self.assertEqual(res, 5)

        res = aten.mul.Scalar(norm, -2)
        self.assertTrue(res._spec.placements[0].is_replicate())

        res = aten.div.Scalar(norm, -2)
        self.assertEqual(res, -5)
        self.assertTrue(res._spec.placements[0].is_replicate())

    @with_comms
    def test_add_sub_scalar_partial(self):
        mesh = self.build_device_mesh()

        rank = self.rank

        # regular partial + scalar -> replicate
        local_tensor = map_local_for_rank(rank, lambda rank: torch.tensor([rank]))

        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )

        res = dt + 1
        expected = sum(i for i in range(self.world_size)) + 1
        self.assertEqual(res, expected)
        self.assertTrue(res._spec.placements[0].is_replicate())

        # regular partial - scalar -> replicate
        local_tensor = map_local_for_rank(rank, lambda rank: torch.tensor([rank]))

        dt = DTensor.from_local(
            local_tensor, device_mesh=mesh, placements=[Partial("sum")]
        )

        res = dt - 1
        expected = sum(i for i in range(self.world_size)) - 1
        self.assertEqual(res, expected)
        self.assertTrue(res._spec.placements[0].is_replicate())

        res = 7 - dt
        expected = 7 - sum(i for i in range(self.world_size))
        self.assertEqual(res, expected)
        self.assertTrue(res._spec.placements[0].is_replicate())

        # regular partial + regular partial -> partial
        res = dt + dt
        self.assertEqual(res.to_local(), rank + rank)
        self.assertTrue(res._spec.placements[0].is_partial())
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        expected = sum(i for i in range(self.world_size)) * 2
        self.assertEqual(res, expected)

        # regular partial - regular partial -> partial
        res = dt - dt
        self.assertEqual(res.to_local(), rank - rank)
        self.assertTrue(res._spec.placements[0].is_partial())
        res = res.redistribute(dt.device_mesh, placements=[Replicate()])
        self.assertEqual(res, 0)

    @with_comms
    def test_add_sub_scalar_norm_partial(self):
        mesh = self.build_device_mesh()

        # norm partial + scalar
        local_tensor = torch.tensor([1.0, 1.0, 7.0, 7.0])
        dt = distribute_tensor(local_tensor, mesh, [Shard(0)])

        norm = dt.norm()
        self.assertTrue(isinstance(norm._spec.placements[0], _NormPartial))
        norm = norm + 1

        self.assertEqual(norm, 11)
        self.assertTrue(norm._spec.placements[0].is_replicate())

        dt = distribute_tensor(local_tensor, mesh, [Shard(0)])

        norm = dt.norm()
        self.assertTrue(isinstance(norm._spec.placements[0], _NormPartial))
        norm = norm - 1

        self.assertEqual(norm, 9)
        self.assertTrue(norm._spec.placements[0].is_replicate())

    @with_comms
    @parametrize("op,reduce_op", [(torch.maximum, "max"), (torch.minimum, "min")])
    def test_partial_propagation(self, op, reduce_op):
        # Test that torch.maximum/minimum preserves Partial("max"/"min") placements
        # since max(max(a), max(b)) == max(a, b) and min(min(a), min(b)) == min(a, b)
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        input1 = torch.rand(8, 8) * self.rank
        input2 = torch.rand(8, 8) * (self.world_size - self.rank)

        d_input1 = DTensor.from_local(input1, device_mesh, [Partial(reduce_op)])
        d_input2 = DTensor.from_local(input2, device_mesh, [Partial(reduce_op)])

        with comm_mode:
            result = op(d_input1, d_input2)

        # Should not require any communication
        self.assertEqual(comm_mode.get_total_counts(), 0)
        # Result should still be Partial with the same reduce_op
        self.assertEqual(result.placements, (Partial(reduce_op),))

    @with_comms
    def test_neg_partial(self):
        # test that neg preserves Partial placement without communication
        # math: -(A1 + A2) = -A1 + -A2
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        input = torch.full((8, 8), 2.0, device=self.device_type)

        for partial_op in ["sum", "avg"]:
            expected_full = (
                torch.full((8, 8), -2.0 * self.world_size, device=self.device_type)
                if partial_op == "sum"
                else torch.full((8, 8), -2.0, device=self.device_type)
            )

            d_input = DTensor.from_local(input, device_mesh, [Partial(partial_op)])

            with comm_mode:
                z = torch.neg(d_input)

            comm_counts = comm_mode.get_total_counts()
            self.assertEqual(comm_counts, 0)
            self.assertTrue(isinstance(z, DTensor))
            self.assertEqual(z.placements, (Partial(partial_op),))
            self.assertEqual(z.full_tensor(), expected_full)

        # test non-sum/avg partial to assert the partial not getting propagated
        # since -max(A1, A2) != max(-A1, -A2)
        d_input = DTensor.from_local(input, device_mesh, [Partial("max")])

        z = torch.neg(d_input)
        self.assertEqual(z.placements, (Replicate(),))
        self.assertEqual(
            z.to_local(), torch.full((8, 8), -2.0, device=self.device_type)
        )

    @with_comms
    def test_maximum_mixed_partials_redistribution(self):
        # Test that mixing Partial("max") with Partial("sum") correctly
        # redistributes the incompatible partial before computing maximum
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        input1 = torch.ones(4, 4) * (self.rank + 1)
        input2 = torch.ones(4, 4) * 0.1 * (self.rank + 1)

        d_input1 = DTensor.from_local(input1, device_mesh, [Partial("max")])
        d_input2 = DTensor.from_local(input2, device_mesh, [Partial("sum")])

        with comm_mode:
            result = torch.maximum(d_input1, d_input2)

        # Should require communication to reduce Partial("sum") to Replicate
        self.assertGreater(comm_mode.get_total_counts(), 0)
        # Result should be Partial("max") following the first operand
        self.assertEqual(result.placements, (Partial("max"),))

        # Verify correctness: d_input2's Partial("sum") should be reduced first
        # d_input2 full value = sum of all ranks' local values = 0.1 * (1+2+3+4) = 1.0
        # d_input1 stays as Partial("max"), so result.full_tensor() does max-reduce
        # max across ranks of max(rank_value, 1.0)
        # rank 0: max(1, 1) = 1, rank 1: max(2, 1) = 2, rank 2: max(3, 1) = 3, rank 3: max(4, 1) = 4
        # final max = 4
        expected_value = float(self.world_size)
        self.assertEqual(result.full_tensor()[0, 0].item(), expected_value)


class PointwisePartialsTest(TestCase):
    """
    Tests for Partial placement propagation in pointwise ops.

    These tests verify that Partial placements (P(max), P(min), P(avg), P(sum))
    are correctly preserved or transformed through various pointwise operations
    without requiring communication.
    """

    world_size = 2

    def setUp(self):
        super().setUp()
        if not dist.is_initialized():
            dist.init_process_group("fake", rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()

    @contextmanager
    def local_tensor_env(self):
        """Context manager that sets up LocalTensorMode, mesh, and CommDebugMode."""
        with LocalTensorMode(frozenset(range(self.world_size))):
            mesh = init_device_mesh("cpu", (self.world_size,))
            comm_mode = CommDebugMode()
            yield mesh, comm_mode

    def test_add_partial_max_with_replicate(self):
        """
        Test add: P(max) + R -> P(max)

        Mathematically: max(a0, a1) + r = max(a0 + r, a1 + r)
        Adding a replicated constant to P(max) preserves the max structure.
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            # Create P(max) input - each rank contributes to the max
            input1 = torch.ones(4, 4) * (0 + 1)  # rank 0
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            # Create replicated input
            input2 = torch.ones(4, 4) * 10.0
            d_input2 = distribute_tensor(input2, mesh, [Replicate()])

            with comm_mode:
                result = d_input1 + d_input2

            # Should preserve P(max) without communication
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # Verify: max(1,2) + 10 = 2 + 10 = 12
            expected = float(self.world_size) + 10.0
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_add_replicate_with_partial_max(self):
        """
        Test add: R + P(max) -> P(max)

        Mathematically: r + max(a0, a1) = max(r + a0, r + a1)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            # Create replicated input
            input1 = torch.ones(4, 4) * 10.0
            d_input1 = distribute_tensor(input1, mesh, [Replicate()])

            # Create P(max) input
            input2 = torch.ones(4, 4) * (0 + 1)
            d_input2 = DTensor.from_local(input2, mesh, [Partial("max")])

            with comm_mode:
                result = d_input1 + d_input2

            # Should preserve P(max) without communication
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            expected = 10.0 + float(self.world_size)
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_add_partial_min_with_replicate(self):
        """
        Test add: P(min) + R -> P(min)

        Mathematically: min(a0, a1) + r = min(a0 + r, a1 + r)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("min")])

            input2 = torch.ones(4, 4) * 10.0
            d_input2 = distribute_tensor(input2, mesh, [Replicate()])

            with comm_mode:
                result = d_input1 + d_input2

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # min(1,2) + 10 = 1 + 10 = 11
            expected = 1.0 + 10.0
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_add_partial_avg_with_replicate(self):
        """
        Test add: P(avg) + R -> P(avg)

        Mathematically: avg(a0, a1) + r = avg(a0 + r, a1 + r)
        Note: This only works when adding to partial, not when partial is added to R.
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("avg")])

            input2 = torch.ones(4, 4) * 10.0
            d_input2 = distribute_tensor(input2, mesh, [Replicate()])

            with comm_mode:
                result = d_input1 + d_input2

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("avg"),))

            # avg(1,2) + 10 = 1.5 + 10 = 11.5
            expected = (
                sum(i + 1 for i in range(self.world_size)) / self.world_size + 10.0
            )
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_sub_replicate_partial_max_gives_partial_min(self):
        """
        Test sub: R - P(max) -> P(min)

        Mathematically: r - max(a0, a1) = min(r - a0, r - a1)
        Subtracting a max gives a min (negation flips the ordering).
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * 10.0
            d_input1 = distribute_tensor(input1, mesh, [Replicate()])

            input2 = torch.ones(4, 4) * (0 + 1)
            d_input2 = DTensor.from_local(input2, mesh, [Partial("max")])

            with comm_mode:
                result = d_input1 - d_input2

            self.assertEqual(comm_mode.get_total_counts(), 0)
            # R - P(max) should give P(min)
            self.assertEqual(result.placements, (Partial("min"),))

            # 10 - max(1,2) = 10 - 2 = 8
            expected = 10.0 - float(self.world_size)
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_sub_replicate_partial_min_gives_partial_max(self):
        """
        Test sub: R - P(min) -> P(max)

        Mathematically: r - min(a0, a1) = max(r - a0, r - a1)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * 10.0
            d_input1 = distribute_tensor(input1, mesh, [Replicate()])

            input2 = torch.ones(4, 4) * (0 + 1)
            d_input2 = DTensor.from_local(input2, mesh, [Partial("min")])

            with comm_mode:
                result = d_input1 - d_input2

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # 10 - min(1,2) = 10 - 1 = 9
            expected = 10.0 - 1.0
            self.assertEqual(result.full_tensor()[0, 0].item(), expected)

    def test_maximum_partial_min_with_replicate(self):
        """
        Test maximum: P(min), R -> P(min)

        Mathematically: max(min(a0, a1), r) = min(max(a0, r), max(a1, r))
        When r >= all ai, this simplifies to r (replicate).
        But for the general case where each rank's value differs,
        the result is P(min) of the per-rank maximums.
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            # Each rank has a different value: rank 0 has 1, rank 1 has 2
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("min")])

            # Replicate input smaller than all rank values for interesting test
            input2 = torch.ones(4, 4) * 0.5
            d_input2 = distribute_tensor(input2, mesh, [Replicate()])

            with comm_mode:
                result = torch.maximum(d_input1, d_input2)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # min(1,2) vs 0.5 -> max(min(), 0.5) = max(1, 0.5) = 1
            # Actually: min(max(1, 0.5), max(2, 0.5)) = min(1, 2) = 1
            self.assertEqual(result.full_tensor()[0, 0].item(), 1.0)

    def test_minimum_partial_max_with_replicate(self):
        """
        Test minimum: P(max), R -> P(max)

        Mathematically: min(max(a0, a1), r) = max(min(a0, r), min(a1, r))
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            # Replicate input larger than all rank values
            input2 = torch.ones(4, 4) * 10.0
            d_input2 = distribute_tensor(input2, mesh, [Replicate()])

            with comm_mode:
                result = torch.minimum(d_input1, d_input2)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # max(min(1,10), min(2,10)) = max(1,2) = 2
            self.assertEqual(result.full_tensor()[0, 0].item(), float(self.world_size))

    def test_neg_partial_max_to_min(self):
        """
        Test neg: P(max) -> P(min)

        Mathematically: -max(a0, a1) = min(-a0, -a1)
        Negation flips max to min.
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            with comm_mode:
                result = torch.neg(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # -max(1,2) = -2 = min(-1,-2)
            self.assertEqual(result.full_tensor()[0, 0].item(), -float(self.world_size))

    def test_neg_partial_min_to_max(self):
        """
        Test neg: P(min) -> P(max)

        Mathematically: -min(a0, a1) = max(-a0, -a1)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("min")])

            with comm_mode:
                result = torch.neg(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # -min(1,2) = -1 = max(-1,-2)
            self.assertEqual(result.full_tensor()[0, 0].item(), -1.0)

    def test_floor_preserves_partial_max(self):
        """
        Test floor: P(max) -> P(max)

        floor is monotonic, so floor(max(a0, a1)) = max(floor(a0), floor(a1))
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 0.7)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            with comm_mode:
                result = torch.floor(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # max(floor(0.7), floor(1.7)) = max(0,1) = 1
            self.assertEqual(
                result.full_tensor()[0, 0].item(), float(self.world_size - 1)
            )

    def test_floor_preserves_partial_min(self):
        """
        Test floor: P(min) -> P(min)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 0.7)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("min")])

            with comm_mode:
                result = torch.floor(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # min(floor(0.7), floor(1.7)) = min(0,1) = 0
            self.assertEqual(result.full_tensor()[0, 0].item(), 0.0)

    def test_ceil_preserves_partial_max(self):
        """
        Test ceil: P(max) -> P(max)

        ceil is monotonic, so ceil(max(a0, a1)) = max(ceil(a0), ceil(a1))
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 + 0.3)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            with comm_mode:
                result = torch.ceil(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # max(ceil(0.3), ceil(1.3)) = max(1,2) = 2
            self.assertEqual(result.full_tensor()[0, 0].item(), float(self.world_size))

    def test_exp_preserves_partial_max(self):
        """
        Test exp: P(max) -> P(max)

        exp is monotonic (strictly increasing), so exp(max(a0, a1)) = max(exp(a0), exp(a1))
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            # Use small values to avoid overflow
            input1 = torch.ones(4, 4) * (0 * 0.1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("max")])

            with comm_mode:
                result = torch.exp(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("max"),))

            # exp(max(0, 0.1)) = exp(0.1)
            import math

            expected = math.exp((self.world_size - 1) * 0.1)
            self.assertAlmostEqual(
                result.full_tensor()[0, 0].item(), expected, places=5
            )

    def test_exp_preserves_partial_min(self):
        """
        Test exp: P(min) -> P(min)
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * (0 * 0.1)
            d_input1 = DTensor.from_local(input1, mesh, [Partial("min")])

            with comm_mode:
                result = torch.exp(d_input1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # exp(min(0, 0.1)) = exp(0) = 1
            self.assertAlmostEqual(result.full_tensor()[0, 0].item(), 1.0, places=5)

    def test_add_with_negative_alpha_flips_partial(self):
        """
        Test add with alpha=-1: R + alpha*P(max) -> P(min)

        When alpha is negative, adding P(max) becomes subtracting P(max),
        which flips it to P(min).
        """
        with self.local_tensor_env() as (mesh, comm_mode):
            input1 = torch.ones(4, 4) * 10.0
            d_input1 = distribute_tensor(input1, mesh, [Replicate()])

            input2 = torch.ones(4, 4) * (0 + 1)
            d_input2 = DTensor.from_local(input2, mesh, [Partial("max")])

            with comm_mode:
                result = torch.add(d_input1, d_input2, alpha=-1)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(result.placements, (Partial("min"),))

            # 10 + (-1)*max(1,2) = 10 - 2 = 8
            self.assertEqual(
                result.full_tensor()[0, 0].item(), 10.0 - float(self.world_size)
            )


instantiate_parametrized_tests(DistElementwiseOpsTest)
DistElementwiseOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistElementwiseOpsTest, base_class=LocalDTensorOpTestBase
)


if __name__ == "__main__":
    run_tests()
