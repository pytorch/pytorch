# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.distributed._differentiable_collectives as dcols
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TEST_HPU,
    TEST_XPU,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# Determine available devices
DEVICE = "cuda"
devices = ["cpu"]
if TEST_HPU:
    devices.append("hpu")
    DEVICE = "hpu"
elif TEST_XPU:
    devices.append("xpu")
    DEVICE = "xpu"
elif TEST_CUDA:
    devices.append("cuda")


@instantiate_parametrized_tests
class TestDifferentiableCollectives(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    # ============================================================
    # Phase 1: Forward Correctness Tests
    # ============================================================

    @parametrize("device", devices)
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_forward_shape(self, device, gather_dim):
        """Test all_gather produces correct output shape."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, 3, device=device)
        output = dcols.all_gather(input_tensor, gather_dim=gather_dim, group=group_name)

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[gather_dim] *= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

    @parametrize("device", devices)
    def test_all_gather_forward_values(self, device):
        """Test all_gather produces correct values."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank has tensor with its rank value
        input_tensor = torch.full((2, 3), fill_value=float(rank), device=device)
        output = dcols.all_gather(input_tensor, gather_dim=0, group=group_name)

        # Verify output contains all ranks' data
        self.assertEqual(output.shape, (2 * self.world_size, 3))
        for r in range(self.world_size):
            chunk = output[r * 2 : (r + 1) * 2, :]
            expected_chunk = torch.full((2, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_forward_shape(self, device, scatter_dim):
        """Test reduce_scatter produces correct output shape."""
        group_name = dist.group.WORLD.group_name

        # Create input with appropriate size
        if scatter_dim == 0:
            input_tensor = torch.randn(4 * self.world_size, 3, device=device)
        else:  # scatter_dim == 1
            input_tensor = torch.randn(3, 4 * self.world_size, device=device)

        output = dcols.reduce_scatter(
            input_tensor, scatter_dim=scatter_dim, group=group_name, op="sum"
        )

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[scatter_dim] //= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

    @parametrize("device", devices)
    def test_reduce_scatter_forward_values(self, device):
        """Test reduce_scatter produces correct reduced values."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # All ranks send ones
        input_tensor = torch.full(
            (4 * self.world_size, 3), fill_value=float(rank), device=device
        )
        output = dcols.reduce_scatter(
            input_tensor, scatter_dim=0, group=group_name, op="sum"
        )

        # Each rank should receive sum of world_size tensors
        expected = torch.full(
            (4, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_reduce_sum_forward(self, device):
        """Test all_reduce_sum produces correct values."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = dcols.all_reduce_sum(input_tensor, group=group_name)

        # Verify reduced value (sum of 0 + 1 + 2 + 3 = 6)
        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_reduce_sum_invariant_forward(self, device):
        """Test all_reduce_sum_invariant does all_reduce in forward."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = dcols.all_reduce_sum_invariant(input_tensor, group=group_name)

        # With sum op on invariant, expect value * world_size
        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_mark_varying_forward(self, device):
        """Test mark_varying is identity in forward."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, device=device)
        output = dcols.mark_varying(input_tensor, group=group_name)

        # Forward should be identity
        self.assertEqual(output, input_tensor)

    @parametrize("device", devices)
    def test_all_to_all_forward(self, device):
        """Test all_to_all with uniform splits."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensor = torch.full(
            (2 * self.world_size, 3), fill_value=float(rank), device=device
        )

        # Uniform split
        output = dcols.all_to_all(
            input_tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=group_name,
        )

        # Output should have same shape as input for uniform splits
        self.assertEqual(output.shape, input_tensor.shape)

        for r in range(self.world_size):
            chunk = output[r * 2 : (r + 1) * 2, :]
            expected_chunk = torch.full((2, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    # ============================================================
    # Phase 2: Backward Correctness Tests
    # ============================================================

    @parametrize("device", devices)
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_backward(self, device, gather_dim):
        """Test all_gather backward produces correct gradients via reduce_scatter."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, 3, requires_grad=True, device=device)
        output = dcols.all_gather(input_tensor, gather_dim=gather_dim, group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be reduce_scatter of ones
        # Each rank contributes world_size gradient elements, reduced to 1 element per rank
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

        # Gradient should be all world_size (sum of ones from all ranks)
        expected_grad = torch.full(
            (3, 3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_backward(self, device, scatter_dim):
        """Test reduce_scatter backward produces correct gradients via all_gather."""
        group_name = dist.group.WORLD.group_name

        # Create input with appropriate size
        if scatter_dim == 0:
            input_tensor = torch.randn(
                4 * self.world_size, 3, requires_grad=True, device=device
            )
        else:
            input_tensor = torch.randn(
                3, 4 * self.world_size, requires_grad=True, device=device
            )

        output = dcols.reduce_scatter(
            input_tensor, scatter_dim=scatter_dim, group=group_name, op="sum"
        )

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

        # All gradients should be 1 (gathered from all ranks)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_reduce_sum_backward(self, device):
        """Test all_reduce_sum backward does all_reduce."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = dcols.all_reduce_sum(input_tensor, group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_reduce of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_reduce_sum_invariant_backward(self, device):
        """Test all_reduce_sum_invariant backward is identity (no-op)."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = dcols.all_reduce_sum_invariant(input_tensor, group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should NOT be aggregated (backward is identity)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones(3, 3, device=device)
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_mark_varying_backward(self, device):
        """Test mark_varying backward does all_reduce."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        input_tensor = torch.full(
            (3, 3), fill_value=float(rank), requires_grad=True, device=device
        )
        output = dcols.mark_varying(input_tensor, group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_reduce of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_to_all_backward(self, device):
        """Test all_to_all backward reverses split sizes."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output = dcols.all_to_all(
            input_tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=group_name,
        )

        # Backward
        output.sum().backward()

        # Gradient should have same shape as input
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)


if __name__ == "__main__":
    # from torch.testing._internal.common_utils import _print_test_names
    # _print_test_names()
    run_tests()
