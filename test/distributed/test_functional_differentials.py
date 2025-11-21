# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fcols
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
class TestFunctionalDifferentials(MultiThreadedTestCase):
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
    def test_all_reduce_sum_invariant_forward(self, device):
        """Test all_reduce_sum_invariant does all_reduce in forward.

        Tensor is VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value (tensor is varying)
        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = fcols.all_reduce_sum_invariant(input_tensor, group=group_name)

        # Forward does all_reduce: sum of 0+1+2+3 = 6
        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_reduce_forward(self, device):
        """Test all_reduce does all_reduce in forward.

        Tensor is VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value (tensor is varying)
        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = fcols.all_reduce(input_tensor, "sum", group=group_name)

        # Forward does all_reduce: sum of 0+1+2+3 = 6
        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_mark_varying_forward(self, device):
        """Test mark_varying is identity in forward.

        Tensor is INVARIANT (identical across ranks).
        Forward is no-op (identity).
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, device=device)
        output = fcols.mark_varying(input_tensor, group=group_name)

        # Forward should be identity
        self.assertEqual(output, input_tensor)

    # ============================================================
    # Phase 2: Backward Correctness Tests
    # ============================================================

    @parametrize("device", devices)
    def test_all_reduce_sum_invariant_backward(self, device):
        """Test all_reduce_sum_invariant backward is identity (no-op).

        Gradients are INVARIANT (identical across ranks).
        Backward is identity - no gradient aggregation needed.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = fcols.all_reduce_sum_invariant(input_tensor, group=group_name)

        # Backward with ones
        output.sum().backward()

        expected_grad = torch.ones(3, 3, device=device)
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is identity (no-op)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        self.assertEqual(grad_input, grad_outputs)

    @parametrize("device", devices)
    def test_all_reduce_backward(self, device):
        """Test all_reduce backward does all_reduce.

        Both tensor AND gradients are VARYING (different across ranks).
        Backward aggregates gradients via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = fcols.all_reduce(input_tensor, "sum", group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be aggregated (backward is all_reduce)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_reduce (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_reduce(grad_outputs, "sum", group=group_name)
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_mark_varying_backward(self, device):
        """Test mark_varying backward does all_reduce.

        Gradients are VARYING (different across ranks).
        Backward aggregates gradients via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank starts with same tensor (invariant)
        # but will produce different gradients (varying)
        input_tensor = torch.full(
            (3, 3), fill_value=float(rank), requires_grad=True, device=device
        )
        output = fcols.mark_varying(input_tensor, group=group_name)

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_reduce of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_reduce (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_reduce(grad_outputs, "sum", group=group_name)
        self.assertEqual(grad_input, expected_grad_input)

    # ============================================================
    # Phase 3: torch.library.opcheck Tests
    # ============================================================

    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        # "test_aot_dispatch_dynamic" - Open issue with check: TBD
    ]

    @parametrize("test_utils", test_utils)
    def test_all_reduce_sum_invariant_opcheck(self, test_utils):
        """Test custom op registration with torch.library.opcheck.

        Verifies all aspects of custom op registration including:
        - Fake tensor support
        - Autograd support
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        # opcheck verifies all aspects of custom op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_reduce_sum_invariant,
            (input_tensor, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_all_reduce_opcheck(self, test_utils):
        """Test all_reduce op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does all_reduce)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_reduce,
            (input_tensor, "sum", group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_mark_varying_opcheck(self, test_utils):
        """Test custom op registration with torch.library.opcheck.

        Verifies all aspects of custom op registration including:
        - Fake tensor support
        - Autograd support
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        # opcheck verifies all aspects of custom op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.mark_varying,
            (input_tensor, group_name),
            test_utils=test_utils,
        )

    # ============================================================
    # Phase 4: torch.compile Integration Tests
    # ============================================================

    def test_all_reduce_sum_invariant_compile(self):
        """Test that all_reduce_sum_invariant backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_reduce_sum_invariant(tensor, group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should NOT be aggregated
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones(3, 3)
        self.assertEqual(input_tensor.grad, expected_grad)

    def test_all_reduce_compile(self):
        """Test that all_reduce backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_reduce(tensor, "sum", group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be aggregated
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    def test_mark_varying_compile(self):
        """Test that mark_varying backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.mark_varying(tensor, group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be aggregated
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)


if __name__ == "__main__":
    run_tests()
