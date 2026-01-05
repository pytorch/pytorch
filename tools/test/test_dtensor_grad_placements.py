# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""Tests for DTensor gradient placement transitions.

Tests the _DTensorGradPlacementHook which redistributes gradients to conjugate placements:
- Replicate in forward → Partial("sum") in backward
- Partial in forward → Replicate in backward
- Shard stays unchanged
"""

import torch
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    with_comms,
)


def get_conjugate_placement(placement):
    """Get the conjugate placement for gradient flow."""
    if isinstance(placement, Shard):
        return placement
    elif isinstance(placement, Partial):
        return Replicate()
    elif isinstance(placement, Replicate):
        return Partial("sum")
    else:
        return placement


class TestDTensorGradPlacements(DTensorOpTestBase):
    def _run_mm_and_backward(self, op1_placement, op2_placement):
        """Run matmul with given placements and return tensors after backward."""
        mesh = self.build_device_mesh()
        torch.manual_seed(0)

        M, N = self.world_size * 4, self.world_size * 4

        x_local = torch.randn(M, M, device=self.device_type, requires_grad=True)
        x = DTensor.from_local(x_local, mesh, [op1_placement], run_check=False)
        x.retain_grad()

        y_local = torch.randn(M, N, device=self.device_type, requires_grad=True)
        y = DTensor.from_local(y_local, mesh, [op2_placement], run_check=False)
        y.retain_grad()

        # Forward
        output = x @ y
        output.retain_grad()

        # Backward
        loss = output.sum()
        loss.backward(retain_graph=True)

        return x, y, output

    def _check_grad_placement(self, tensor, expected_placement):
        """Check that the gradient has the expected placement."""
        self.assertIsNotNone(tensor.grad)
        actual_placement = tensor.grad.placements[0]
        self.assertEqual(
            type(actual_placement),
            type(expected_placement),
            f"Expected {expected_placement}, got {actual_placement}",
        )
        if isinstance(expected_placement, Shard):
            self.assertEqual(actual_placement.dim, expected_placement.dim)

    # ==========================================================================
    # Tests with Shard(0) as first operand
    # ==========================================================================
    @with_comms
    def test_shard0_shard0(self):
        x, y, output = self._run_mm_and_backward(Shard(0), Shard(0))
        # Shard → Shard (unchanged)
        self._check_grad_placement(x, Shard(0))
        self._check_grad_placement(y, Shard(0))

    @with_comms
    def test_shard0_shard1(self):
        x, y, output = self._run_mm_and_backward(Shard(0), Shard(1))
        # Shard → Shard (unchanged)
        self._check_grad_placement(x, Shard(0))
        self._check_grad_placement(y, Shard(1))

    @with_comms
    def test_shard0_replicate(self):
        x, y, output = self._run_mm_and_backward(Shard(0), Replicate())
        # Shard → Shard, Replicate → Partial
        self._check_grad_placement(x, Shard(0))
        self._check_grad_placement(y, Partial("sum"))

    @with_comms
    def test_shard0_partial(self):
        x, y, output = self._run_mm_and_backward(Shard(0), Partial("sum"))
        # Shard → Shard, Partial → Replicate
        self._check_grad_placement(x, Shard(0))
        self._check_grad_placement(y, Replicate())

    # ==========================================================================
    # Tests with Shard(1) as first operand
    # ==========================================================================
    @with_comms
    def test_shard1_shard0(self):
        x, y, output = self._run_mm_and_backward(Shard(1), Shard(0))
        # Shard → Shard (unchanged)
        self._check_grad_placement(x, Shard(1))
        self._check_grad_placement(y, Shard(0))

    @with_comms
    def test_shard1_shard1(self):
        x, y, output = self._run_mm_and_backward(Shard(1), Shard(1))
        # Shard → Shard (unchanged)
        self._check_grad_placement(x, Shard(1))
        self._check_grad_placement(y, Shard(1))

    @with_comms
    def test_shard1_replicate(self):
        x, y, output = self._run_mm_and_backward(Shard(1), Replicate())
        # Shard → Shard, Replicate → Partial
        self._check_grad_placement(x, Shard(1))
        self._check_grad_placement(y, Partial("sum"))

    @with_comms
    def test_shard1_partial(self):
        x, y, output = self._run_mm_and_backward(Shard(1), Partial("sum"))
        # Shard → Shard, Partial → Replicate
        self._check_grad_placement(x, Shard(1))
        self._check_grad_placement(y, Replicate())

    # ==========================================================================
    # Tests with Replicate as first operand
    # ==========================================================================
    @with_comms
    def test_replicate_shard0(self):
        x, y, output = self._run_mm_and_backward(Replicate(), Shard(0))
        # Replicate → Partial, Shard → Shard
        self._check_grad_placement(x, Partial("sum"))
        self._check_grad_placement(y, Shard(0))

    @with_comms
    def test_replicate_shard1(self):
        x, y, output = self._run_mm_and_backward(Replicate(), Shard(1))
        # Replicate → Partial, Shard → Shard
        self._check_grad_placement(x, Partial("sum"))
        self._check_grad_placement(y, Shard(1))

    @with_comms
    def test_replicate_replicate(self):
        x, y, output = self._run_mm_and_backward(Replicate(), Replicate())
        # Replicate → Partial for both
        self._check_grad_placement(x, Partial("sum"))
        self._check_grad_placement(y, Partial("sum"))

    @with_comms
    def test_replicate_partial(self):
        x, y, output = self._run_mm_and_backward(Replicate(), Partial("sum"))
        # Replicate → Partial, Partial → Replicate
        self._check_grad_placement(x, Partial("sum"))
        self._check_grad_placement(y, Replicate())

    # ==========================================================================
    # Tests with Partial as first operand
    # ==========================================================================
    @with_comms
    def test_partial_replicate(self):
        x, y, output = self._run_mm_and_backward(Partial("sum"), Replicate())
        # Partial → Replicate, Replicate → Partial
        self._check_grad_placement(x, Replicate())
        self._check_grad_placement(y, Partial("sum"))


if __name__ == "__main__":
    run_tests()
