# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed.ops as ops

from functools import partial
from torch.testing._internal.common_utils import TestCase, run_tests
from typing import Callable, Tuple
from unittest.mock import patch


class OpsTest(TestCase):
    def setUp(self) -> None:
        self._patches = []  # type: ignore[var-annotated]

        def patch_and_mock(fn: str) -> None:
            p = patch(f"torch.distributed.ops.dist.{fn}")

            mock = p.start()
            mock.side_effect = getattr(self, f"_mock_{fn}")

            self._patches.append(p)

        patch_and_mock("all_gather")
        patch_and_mock("all_reduce")
        patch_and_mock("broadcast")
        patch_and_mock("get_rank")
        patch_and_mock("get_world_size")
        patch_and_mock("reduce")

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()

    def _setup(self, rank: int, world_size: int, shape: Tuple[int, ...]) -> None:
        self._rank = rank  # Simulated rank.
        self._world_size = world_size  # Simulated world size.

        gen = torch.manual_seed(1)

        # Holds the inputs per rank.
        self._x = [
            torch.rand(shape, generator=gen, requires_grad=True) * 4.0 for _ in range(world_size)
        ]

        # Holds the intermediate activations per rank.
        self._activations = [x ** 3 for x in self._x]

        # Indicates whether we are in the forward or backward pass of the test.
        self._backprop = False

    def _mock_all_gather(self, outputs, input_, _) -> None:
        if self._backprop:
            # We know that the output gradients are identical on all ranks, so
            # we can simulate the all-gather operation.
            inputs = [input_] * self._world_size
        else:
            inputs = self._activations

        with torch.no_grad():
            for o, i in zip(outputs, inputs):
                o.copy_(i)

    def _mock_all_reduce(self, tensor, op, _) -> None:
        self._mock_reduce_core(tensor, op)

    def _mock_broadcast(self, tensor, src, _) -> None:
        if self._rank == src:
            return

        # We know that the output gradients are identical on all ranks, so we
        # can treat broadcasting as a noop.
        if not self._backprop:
            with torch.no_grad():
                tensor.copy_(self._activations[self._rank])

    def _mock_get_rank(self, _) -> int:
        return self._rank

    def _mock_get_world_size(self, _) -> int:
        return self._world_size

    def _mock_reduce(self, tensor, dst, op, _) -> None:
        if self._rank != dst:
            return

        self._mock_reduce_core(tensor, op)

    def _mock_reduce_core(self, tensor, op) -> None:
        if self._backprop:
            # We know that the output gradients are identical on all ranks, so
            # we can simulate the reduce operation.
            inputs = [tensor.detach().clone()] * self._world_size
        else:
            inputs = self._activations

        with torch.no_grad():
            if op == dist.ReduceOp.SUM:
                tensor.zero_()
                for i in inputs:
                    tensor.add_(i)
            elif op == dist.ReduceOp.PRODUCT:
                tensor.fill_(1.0)
                for i in inputs:
                    tensor.mul_(i)
            else:
                raise RuntimeError("Unsupported reduce operation.")

    def _run_test(self, set_state: Callable, op: Callable) -> None:
        for world_size in range(1, 5):
            for rank in range(world_size):
                for shape in [(1,), (2, 3), (4, 4), (6, 4)]:
                    with self.subTest(rank=rank, world_size=world_size, shape=shape):
                        self._setup(rank, world_size, shape)

                        set_state()

                        self._run_and_assert_op(op)

    def _set_state_for_copy(self, src_rank: int) -> None:
        self._expected_output = self._activations[self._rank]

        if self._rank == src_rank:
            # Simulate as if backprop (e.g. backward()) was run on all ranks.
            output = self._world_size * self._expected_output

            self._compute_expected_derivatives(output)
        else:
            zero_grad = torch.zeros_like(self._expected_output)

            # For all other ranks the gradient is always zero.
            self._expected_jacobian = self._expected_hessian = (zero_grad,)

    def _set_state_for_sum(self) -> None:
        self._expected_output = torch.zeros_like(self._activations[0])
        for a in self._activations:
            self._expected_output.add_(a)

        # Simulate as if backprop (e.g. backward()) was run on all ranks.
        output = self._world_size * self._expected_output

        self._compute_expected_derivatives(output)

    def _set_state_for_sum_on_rank(self, dst_rank: int) -> None:
        output = torch.zeros_like(self._activations[0])
        for a in self._activations:
            output.add_(a)

        self._compute_expected_derivatives(output)

        if self._rank == dst_rank:
            self._expected_output = output
        else:
            self._expected_output = torch.zeros_like(self._activations[0])

    def _set_state_for_prod(self) -> None:
        self._expected_output = torch.ones_like(self._activations[0])
        for a in self._activations:
            self._expected_output.mul_(a)

        # Simulate as if backprop (e.g. backward()) was run on all ranks.
        output = self._world_size * self._expected_output

        self._compute_expected_derivatives(output)

    def _set_state_for_minimum(self) -> None:
        self._expected_output = self._activations[0]
        for a in self._activations:
            self._expected_output = torch.minimum(self._expected_output, a)

        # Simulate as if backprop (e.g. backward()) was run on all ranks.
        output = self._world_size * self._expected_output

        self._compute_expected_derivatives(output)

    def _set_state_for_maximum(self) -> None:
        self._expected_output = self._activations[0]
        for a in self._activations:
            self._expected_output = torch.maximum(self._expected_output, a)

        # Simulate as if backprop (e.g. backward()) was run on all ranks.
        output = self._world_size * self._expected_output

        self._compute_expected_derivatives(output)

    def _compute_expected_derivatives(self, output: torch.Tensor) -> None:
        x = self._x[self._rank]

        o = (output,)

        grad_output = (torch.ones_like(output),)

        # Compute Jacobian.
        o = torch.autograd.grad(o, x, grad_output, create_graph=True)  # type: ignore[assignment]

        self._expected_jacobian = o

        # Compute Hessian. Note that we need to retain the autograd graph in
        # order to run backprop a second time for assertion.
        o = torch.autograd.grad(o, x, grad_output, retain_graph=True)  # type: ignore[assignment]

        self._expected_hessian = o

    def _run_and_assert_op(self, op: Callable) -> None:
        output = op(self._activations[self._rank])

        self._assert_op(output)

    def _assert_op(self, output: torch.Tensor) -> None:
        self.assertEqual(output, self._expected_output)

        try:
            self._backprop = True

            self._assert_derivatives(output)
        finally:
            self._backprop = False

    def _assert_derivatives(self, output: torch.Tensor) -> None:
        x = self._x[self._rank]

        o = (output,)

        grad_output = (torch.ones_like(output),)

        # Assert Jacobian
        o = torch.autograd.grad(o, x, grad_output, create_graph=True)  # type: ignore[assignment]

        self.assertEqual(o, self._expected_jacobian)

        # Assert Hessian.
        o = torch.autograd.grad(o, x, grad_output)  # type: ignore[assignment]

        self.assertEqual(o, self._expected_hessian)

    def test_copy(self) -> None:
        src_rank = 0

        set_state = partial(self._set_state_for_copy, src_rank)

        op = partial(ops.copy, src_rank)

        self._run_test(set_state, op)

    def test_sum(self) -> None:
        self._run_test(self._set_state_for_sum, ops.sum)

    def test_sum_on_rank(self) -> None:
        dst_rank = 0

        set_state = partial(self._set_state_for_sum_on_rank, dst_rank)

        op = partial(ops.sum_on_rank, dst_rank)

        self._run_test(set_state, op)

    def test_prod(self) -> None:
        self._run_test(self._set_state_for_prod, ops.prod)

    def test_minimum(self) -> None:
        self._run_test(self._set_state_for_minimum, ops.minimum)

    def test_maximum(self) -> None:
        self._run_test(self._set_state_for_maximum, ops.maximum)


if __name__ == "__main__":
    run_tests()
