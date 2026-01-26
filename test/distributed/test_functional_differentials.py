# Owner(s): ["oncall: distributed"]

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as fcols
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    MultiThreadedTestCase,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# Determine available devices
DEVICE = "cuda"
devices = ["cpu"]
if acc := torch.accelerator.current_accelerator(True):
    devices += [acc.type]


def with_comms(func=None):
    if func is None:
        return partial(with_comms)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if (
            torch.cuda.is_available()
            and torch.accelerator.device_count() < self.world_size
        ):
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.pg = self.create_pg(device=DEVICE)
        self.device = DEVICE
        try:
            return func(self, *args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapper


@instantiate_parametrized_tests
class TestFunctionalDifferentials(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    # ============================================================
    # Forward Correctness Tests
    # ============================================================

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
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_tensor_forward(self, device, gather_dim):
        """Test all_gather_tensor produces correct output shape.

        Tensor is VARYING (different across ranks).
        Forward gathers tensors from all ranks along gather_dim.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank has tensor with its rank value
        input_tensor = torch.full((3, 3, 3), fill_value=float(rank), device=device)
        output = fcols.all_gather_tensor(
            input_tensor, gather_dim=gather_dim, group=group_name
        )

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[gather_dim] *= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

        # Verify output contains all ranks' data
        # Check each chunk along gather_dim contains the correct rank value
        for r in range(self.world_size):
            chunk = output.narrow(gather_dim, r * 3, 3)
            expected_chunk = torch.full((3, 3, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_tensor_forward(self, device, scatter_dim):
        """Test reduce_scatter_tensor produces correct output shape.

        Tensor is VARYING (different across ranks).
        Forward reduces and scatters chunks to ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Create input with appropriate size
        if scatter_dim == 0:
            input_tensor = torch.full(
                (4 * self.world_size, 3), fill_value=float(rank), device=device
            )
        else:  # scatter_dim == 1
            input_tensor = torch.full(
                (3, 4 * self.world_size), fill_value=float(rank), device=device
            )

        output = fcols.reduce_scatter_tensor(
            input_tensor, "sum", scatter_dim=scatter_dim, group=group_name
        )

        # Verify output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[scatter_dim] //= self.world_size
        self.assertEqual(list(output.shape), expected_shape)

        # Each rank should receive sum of all ranks' values: 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        expected = torch.full_like(output, fill_value=expected_value)
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_to_all_single_forward(self, device):
        """Test all_to_all_single with uniform splits.

        Tensor is VARYING (different across ranks).
        Forward exchanges tensor chunks between ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensor = torch.full(
            (2 * self.world_size, 3), fill_value=float(rank), device=device
        )

        # Uniform split
        output = fcols.all_to_all_single(
            input_tensor,
            output_split_sizes=None,
            input_split_sizes=None,
            group=group_name,
        )

        # Output should have same shape as input for uniform splits
        self.assertEqual(output.shape, input_tensor.shape)

        # Verify each rank receives data from all other ranks
        for r in range(self.world_size):
            chunk = output[r * 2 : (r + 1) * 2, :]
            expected_chunk = torch.full((2, 3), fill_value=float(r), device=device)
            self.assertEqual(chunk, expected_chunk)

    @parametrize("device", devices)
    def test_all_reduce_coalesced_forward(self, device):
        """Test all_reduce_coalesced does all_reduce on each tensor.

        Tensors are VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value
        input_tensors = [
            torch.full((3, 3), fill_value=float(rank), device=device),
            torch.full((2, 2), fill_value=float(rank), device=device),
        ]
        outputs = fcols.all_reduce_coalesced(input_tensors, "sum", group=group_name)

        # Forward does all_reduce: sum of 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        for output, input_tensor in zip(outputs, input_tensors):
            expected = torch.full_like(input_tensor, fill_value=expected_value)
            self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced_forward(self, device):
        """Test all_gather_into_tensor_coalesced gathers each tensor.

        Tensors are VARYING (different across ranks).
        Forward gathers tensors from all ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank has tensors with its rank value
        input_tensors = [
            torch.full((3, 3), fill_value=float(rank), device=device),
            torch.full((2, 2), fill_value=float(rank), device=device),
        ]
        outputs = fcols.all_gather_into_tensor_coalesced(
            input_tensors, group=group_name
        )

        # Verify output shapes
        for output, input_tensor in zip(outputs, input_tensors):
            expected_shape = list(input_tensor.shape)
            expected_shape[0] *= self.world_size
            self.assertEqual(list(output.shape), expected_shape)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_coalesced_forward(self, device):
        """Test reduce_scatter_tensor_coalesced reduces and scatters each tensor.

        Tensors are VARYING (different across ranks).
        Forward reduces and scatters chunks to ranks.
        """
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Create inputs with appropriate size (divisible by world_size)
        input_tensors = [
            torch.full((4 * self.world_size, 3), fill_value=float(rank), device=device),
            torch.full((2 * self.world_size, 2), fill_value=float(rank), device=device),
        ]
        scatter_dims = [0, 0]

        outputs = fcols.reduce_scatter_tensor_coalesced(
            input_tensors, "sum", scatter_dims, group=group_name
        )

        # Each rank should receive sum of all ranks' values: 0+1+2+3 = 6
        expected_value = self.world_size * (self.world_size - 1) / 2
        for output, input_tensor in zip(outputs, input_tensors):
            expected_shape = list(input_tensor.shape)
            expected_shape[0] //= self.world_size
            self.assertEqual(list(output.shape), expected_shape)
            expected = torch.full_like(output, fill_value=expected_value)
            self.assertEqual(output, expected)

    # ============================================================
    # Backward Correctness Tests
    # ============================================================

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
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_tensor_backward(self, device, gather_dim):
        """Test all_gather_tensor backward does reduce_scatter.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward gathers tensors, backward reduces and scatters gradients.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, 3, requires_grad=True, device=device)
        output = fcols.all_gather_tensor(
            input_tensor, gather_dim=gather_dim, group=group_name
        )

        # Backward with ones
        output.sum().backward()

        # Gradient should be reduce_scatter of ones
        self.assertIsNotNone(input_tensor.grad)
        # Gradient should be all world_size (sum from all ranks)
        expected_grad = torch.full(
            (3, 3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is reduce_scatter (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.reduce_scatter_tensor(
            grad_outputs, "sum", gather_dim, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    @parametrize("scatter_dim", [0, 1])
    def test_reduce_scatter_tensor_backward(self, device, scatter_dim):
        """Test reduce_scatter_tensor backward does all_gather.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward reduces and scatters, backward gathers gradients.
        """
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

        output = fcols.reduce_scatter_tensor(
            input_tensor, "sum", scatter_dim=scatter_dim, group=group_name
        )

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

        # All gradients should be 1 (gathered from all ranks)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

        # Backward is all_gather (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_gather_tensor(
            grad_outputs, scatter_dim, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_to_all_single_backward(self, device):
        """Test all_to_all_single backward reverses split sizes.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward does all_to_all, backward does all_to_all with reversed splits.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output = fcols.all_to_all_single(
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

        # Backward is all_gather (sum)
        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        expected_grad_input = fcols.all_to_all_single(
            grad_outputs, None, None, group=group_name
        )
        self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_reduce_coalesced_backward(self, device):
        """Test all_reduce_coalesced backward does all_reduce on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Backward aggregates each gradient via all_reduce(sum).
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(3, 3, requires_grad=True, device=device),
            torch.randn(2, 2, requires_grad=True, device=device),
        ]
        outputs = fcols.all_reduce_coalesced(input_tensors, "sum", group=group_name)

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be aggregated (backward is all_reduce)
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.full_like(
                input_tensor, fill_value=float(self.world_size)
            )
            self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_gather_into_tensor_coalesced_backward(self, device):
        """Test all_gather_into_tensor_coalesced backward does reduce_scatter on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Forward gathers each tensor, backward reduce_scatters each gradient.
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(3, 3, requires_grad=True, device=device),
            torch.randn(2, 2, requires_grad=True, device=device),
        ]
        outputs = fcols.all_gather_into_tensor_coalesced(
            input_tensors, group=group_name
        )

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be reduce_scatter of ones
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.full_like(
                input_tensor, fill_value=float(self.world_size)
            )
            self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_coalesced_backward(self, device):
        """Test reduce_scatter_tensor_coalesced backward does all_gather on each gradient.

        Tensors AND gradients are VARYING (different across ranks).
        Forward reduces and scatters each tensor, backward gathers each gradient.
        """
        group_name = dist.group.WORLD.group_name

        input_tensors = [
            torch.randn(4 * self.world_size, 3, requires_grad=True, device=device),
            torch.randn(2 * self.world_size, 2, requires_grad=True, device=device),
        ]
        scatter_dims = [0, 0]

        outputs = fcols.reduce_scatter_tensor_coalesced(
            input_tensors, "sum", scatter_dims, group=group_name
        )

        # Backward with ones
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        # Each gradient should be all_gather of ones
        for input_tensor in input_tensors:
            self.assertIsNotNone(input_tensor.grad)
            expected_grad = torch.ones_like(input_tensor)
            self.assertEqual(input_tensor.grad, expected_grad)

    # ============================================================
    # torch.library.opcheck Tests
    # ============================================================

    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
        # "test_aot_dispatch_dynamic" - Open issue with check: TBD
    ]

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
    def test_all_gather_into_tensor_opcheck(self, test_utils):
        """Test all_gather_into_tensor op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does reduce_scatter)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_gather_into_tensor,
            (input_tensor, self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_reduce_scatter_tensor_opcheck(self, test_utils):
        """Test reduce_scatter_tensor op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward does all_gather)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name

        # Input should be divisible by world_size
        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.reduce_scatter_tensor,
            (input_tensor, "sum", self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_all_to_all_single_opcheck(self, test_utils):
        """Test all_to_all_single op registration with torch.library.opcheck.

        Verifies all aspects of op registration including:
        - Fake tensor support
        - Autograd support (backward reverses split sizes)
        - Schema validation
        """
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        # Input should be divisible by world_size
        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        # opcheck verifies all aspects of op registration
        torch.library.opcheck(
            torch.ops._c10d_functional.all_to_all_single,
            (input_tensor, output_split_sizes, input_split_sizes, group_name),
            test_utils=test_utils,
        )


@instantiate_parametrized_tests
class TestFunctionalDifferentialsWithCompile(DistributedTestBase):
    # ============================================================
    # torch.compile Integration Tests
    # ============================================================

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_all_reduce_compile(self):
        """Test that all_reduce backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_reduce(tensor, "sum", group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be aggregated
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_gather_tensor_compile(self):
        """Test that all_gather_tensor backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_gather_tensor(tensor, gather_dim=0, group=group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_reduce_scatter_tensor_compile(self):
        """Test that reduce_scatter_tensor backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.reduce_scatter_tensor(
                tensor, "sum", scatter_dim=0, group=group_name
            )
            return output.sum()

        # Input should be divisible by world_size
        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_to_all_single_compile(self):
        """Test that all_to_all_single backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_to_all_single(
                tensor,
                output_split_sizes=None,
                input_split_sizes=None,
                group=group_name,
            )
            return output.sum()

        # Input should be divisible by world_size
        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be all_to_all with reversed splits (ones)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)


instantiate_device_type_tests(
    TestFunctionalDifferentialsWithCompile, globals(), only_for=DEVICE
)

if __name__ == "__main__":
    run_tests()
