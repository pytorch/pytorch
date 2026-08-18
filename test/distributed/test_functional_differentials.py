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
    @parametrize("reduce_op", ["sum", "avg", "min", "max"])
    def test_all_reduce_forward(self, device, reduce_op):
        """Test all_reduce does all_reduce in forward.

        Tensor is VARYING (different across ranks).
        Forward aggregates varying tensors via all_reduce.
        """
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        # Each rank contributes its rank value (tensor is varying)
        input_tensor = torch.full(shape, fill_value=float(rank), device=device)
        output = fcols.all_reduce(input_tensor, reduce_op, group=group_name)

        rank_values = [float(r) for r in range(self.world_size)]
        if reduce_op == "sum":
            expected_val = sum(rank_values)
        elif reduce_op == "avg":
            expected_val = sum(rank_values) / self.world_size
        elif reduce_op == "min":
            expected_val = min(rank_values)
        elif reduce_op == "max":
            expected_val = max(rank_values)
        expected = torch.full(shape, fill_value=expected_val, device=device)
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_all_reduce_premul_sum_forward(self, device):
        """Test all_reduce forward with PREMUL_SUM ReduceOp."""
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()
        factor = 0.5

        premul_sum_op = dist.ReduceOp.PREMUL_SUM(factor)
        input_tensor = torch.full(shape, fill_value=float(rank + 1), device=device)
        output = fcols.all_reduce(
            input_tensor, reduceOp=premul_sum_op, group=group_name
        )

        expected_val = factor * sum(float(r + 1) for r in range(self.world_size))
        expected = torch.full(shape, fill_value=expected_val, device=device)
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
        output = fcols.all_gather_single(
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

        output = fcols.reduce_scatter_single(
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
        outputs = fcols.all_gather_single_coalesced(input_tensors, group=group_name)

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

        outputs = fcols.reduce_scatter_single_coalesced(
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
    @parametrize("reduce_op", ["sum", "avg", "min", "max"])
    def test_all_reduce_backward(self, device, reduce_op):
        """Test all_reduce backward does all_reduce.

        Both tensor AND gradients are VARYING (different across ranks).
        Backward aggregates gradients via all_reduce().
        """
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name

        rank = dist.get_rank()

        if reduce_op in ("min", "max"):
            # Use distinct per-rank values so there's a unique extremum holder
            input_tensor = torch.full(
                shape, fill_value=float(rank), requires_grad=True, device=device
            )
        else:
            input_tensor = torch.randn(*shape, requires_grad=True, device=device)
        output = fcols.all_reduce(input_tensor, reduce_op, group=group_name)

        # Backward with ones
        output.sum().backward()

        if reduce_op == "sum":
            expected_grad = torch.full(
                shape, fill_value=float(self.world_size), device=device
            )
        elif reduce_op == "avg":
            expected_grad = torch.full(shape, fill_value=1.0, device=device)
        elif reduce_op == "min":
            # Grad flows only to the rank that held the min (rank 0)
            grad_val = float(self.world_size) if rank == 0 else 0.0
            expected_grad = torch.full(shape, fill_value=grad_val, device=device)
        elif reduce_op == "max":
            # Grad flows only to the rank that held the max (last rank)
            grad_val = float(self.world_size) if rank == self.world_size - 1 else 0.0
            expected_grad = torch.full(shape, fill_value=grad_val, device=device)
        self.assertEqual(input_tensor.grad, expected_grad)

        if reduce_op not in ("min", "max"):
            grad_outputs = torch.rand_like(output, device=device)
            (grad_input,) = torch.autograd.grad(
                output, input_tensor, grad_outputs=grad_outputs
            )
            expected_grad_input = fcols.all_reduce(
                grad_outputs, reduce_op, group=group_name
            )
            self.assertEqual(grad_input, expected_grad_input)

    @parametrize("device", devices)
    def test_all_reduce_premul_sum_backward(self, device):
        """Test all_reduce backward with PREMUL_SUM ReduceOp."""
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()
        factor = 0.5

        premul_sum_op = dist.ReduceOp.PREMUL_SUM(factor)

        input_tensor = torch.full(
            shape, fill_value=float(rank + 1), requires_grad=True, device=device
        )
        output = fcols.all_reduce(
            input_tensor, reduceOp=premul_sum_op, group=group_name
        )

        # Forward: premul_sum(0.5) premultiplies then sums
        expected_fwd = factor * sum(float(r + 1) for r in range(self.world_size))
        self.assertEqual(
            output, torch.full(shape, fill_value=expected_fwd, device=device)
        )

        output.sum().backward()
        # Backward: all_reduce(1, premul_sum(0.5)) = 0.5 * world_size
        expected_grad = torch.full(
            shape, fill_value=factor * float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    @parametrize("reduce_op", ["min", "max"])
    def test_all_reduce_nan_backward(self, device, reduce_op):
        """Test all_reduce backward passes through all-reduced grad for NaN inputs."""
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        input_tensor = torch.full(
            shape, fill_value=float(rank), requires_grad=True, device=device
        )
        # Inject a NaN on rank 0
        if rank == 0:
            with torch.no_grad():
                input_tensor[0, 0] = float("nan")

        output = fcols.all_reduce(input_tensor, reduce_op, group=group_name)
        output.sum().backward()

        grad = input_tensor.grad
        # NaN elements receive the all-reduced grad (world_size) rather than NaN
        all_reduced_grad = float(self.world_size)
        if rank == 0:
            self.assertEqual(grad[0, 0], all_reduced_grad)
            # Non-NaN elements on the extremum-holding rank get world_size
            if reduce_op == "min":
                self.assertEqual(grad[0, 1], all_reduced_grad)
            else:
                self.assertEqual(grad[0, 1], 0.0)
        else:
            self.assertFalse(grad.isnan().any())

    @parametrize("device", devices)
    @parametrize("gather_dim", [0, 1, 2])
    def test_all_gather_tensor_backward(self, device, gather_dim):
        """Test all_gather_tensor backward does reduce_scatter.

        Both tensor AND gradients are VARYING (different across ranks).
        Forward gathers tensors, backward reduces and scatters gradients.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, 3, requires_grad=True, device=device)
        output = fcols.all_gather_single(
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
        expected_grad_input = fcols.reduce_scatter_single(
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

        output = fcols.reduce_scatter_single(
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
        expected_grad_input = fcols.all_gather_single(
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
        outputs = fcols.all_gather_single_coalesced(input_tensors, group=group_name)

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

        outputs = fcols.reduce_scatter_single_coalesced(
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

    # ============================================================
    # _c10d_functional_autograd Backward Compatibility Tests
    # ============================================================

    @parametrize("device", devices)
    def test_all_gather_into_tensor_autograd_backward(self, device):
        """Test _c10d_functional_autograd.all_gather_into_tensor backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
            input_tensor, self.world_size, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward with ones
        output.sum().backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_reduce_scatter_tensor_autograd_backward(self, device):
        """Test _c10d_functional_autograd.reduce_scatter_tensor backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
            input_tensor, "sum", self.world_size, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward with ones
        output.sum().backward()

        # Gradient should be all_gather of ones
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    @parametrize("device", devices)
    def test_all_to_all_single_autograd_backward(self, device):
        """Test _c10d_functional_autograd.all_to_all_single backward.

        Verifies backward compatibility: the autograd ops should have the same
        backward behavior as _c10d_functional ops.
        """
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.randn(
            4 * self.world_size, 3, requires_grad=True, device=device
        )
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        output = torch.ops._c10d_functional_autograd.all_to_all_single(
            input_tensor, output_split_sizes, input_split_sizes, group_name
        )
        output = fcols.wait_tensor(output)

        # Backward
        output.sum().backward()

        # Gradient should have same shape as input
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        expected_grad = torch.ones_like(input_tensor)
        self.assertEqual(input_tensor.grad, expected_grad)

    # ============================================================
    # _c10d_functional_autograd opcheck Tests
    # ============================================================

    # Skip test_faketensor for autograd ops - they share impl with _c10d_functional
    autograd_test_utils = [
        "test_schema",
        "test_autograd_registration",
    ]

    @parametrize("test_utils", autograd_test_utils)
    def test_all_gather_into_tensor_autograd_opcheck(self, test_utils):
        """Test _c10d_functional_autograd.all_gather_into_tensor op registration."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.all_gather_into_tensor,
            (input_tensor, self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", autograd_test_utils)
    def test_reduce_scatter_tensor_autograd_opcheck(self, test_utils):
        """Test _c10d_functional_autograd.reduce_scatter_tensor op registration."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.reduce_scatter_tensor,
            (input_tensor, "sum", self.world_size, group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", autograd_test_utils)
    def test_all_to_all_single_autograd_opcheck(self, test_utils):
        """Test _c10d_functional_autograd.all_to_all_single op registration."""
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.ones(4 * self.world_size, 3, requires_grad=True)
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        torch.library.opcheck(
            torch.ops._c10d_functional_autograd.all_to_all_single,
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
        shape = (3, 3)
        group_name = dist.group.WORLD.group_name

        for reduce_op in ["sum", "avg", "min", "max"]:
            with self.subTest(reduce_op=reduce_op):

                @torch.compile(fullgraph=True)
                def compiled_fn(tensor):
                    output = fcols.all_reduce(tensor, reduce_op, group=group_name)
                    return output.sum()

                if reduce_op in ("min", "max"):
                    input_tensor = torch.full(
                        shape,
                        fill_value=float(self.rank),
                        device=self.device,
                        requires_grad=True,
                    )
                else:
                    input_tensor = torch.randn(
                        *shape, device=self.device, requires_grad=True
                    )

                loss = compiled_fn(input_tensor)
                loss.backward()

                self.assertIsNotNone(input_tensor.grad)
                if reduce_op == "sum":
                    expected_grad = torch.full(
                        shape, fill_value=float(self.world_size), device=self.device
                    )
                elif reduce_op == "avg":
                    expected_grad = torch.full(
                        shape, fill_value=1.0, device=self.device
                    )
                elif reduce_op == "min":
                    grad_val = float(self.world_size) if self.rank == 0 else 0.0
                    expected_grad = torch.full(
                        shape, fill_value=grad_val, device=self.device
                    )
                elif reduce_op == "max":
                    grad_val = (
                        float(self.world_size)
                        if self.rank == self.world_size - 1
                        else 0.0
                    )
                    expected_grad = torch.full(
                        shape, fill_value=grad_val, device=self.device
                    )
                self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_all_gather_tensor_compile(self):
        """Test that all_gather_tensor backward works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = fcols.all_gather_single(tensor, gather_dim=0, group=group_name)
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
            output = fcols.reduce_scatter_single(
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

    # ============================================================
    # _c10d_functional_autograd torch.compile Integration Tests
    # ============================================================

    @with_comms
    def test_all_gather_into_tensor_autograd_compile(self):
        """Test that _c10d_functional_autograd.all_gather_into_tensor works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
                tensor, self.world_size, group_name
            )
            output = fcols.wait_tensor(output)
            return output.sum()

        input_tensor = torch.randn(3, 3, device=self.device, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        # Gradient should be reduce_scatter of ones (all world_size)
        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)

    @with_comms
    def test_reduce_scatter_tensor_autograd_compile(self):
        """Test that _c10d_functional_autograd.reduce_scatter_tensor works with torch.compile."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
                tensor, "sum", self.world_size, group_name
            )
            output = fcols.wait_tensor(output)
            return output.sum()

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
    def test_all_to_all_single_autograd_compile(self):
        """Test that _c10d_functional_autograd.all_to_all_single works with torch.compile."""
        group_name = dist.group.WORLD.group_name
        group_size = dist.group.WORLD.size()

        input_tensor = torch.randn(
            4 * self.world_size, 3, device=self.device, requires_grad=True
        )
        # The raw op requires explicit split sizes (SymInt[], not optional); the
        # None default is only resolved by the all_to_all_single Python wrapper.
        output_split_sizes = [input_tensor.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = torch.ops._c10d_functional_autograd.all_to_all_single(
                tensor,
                output_split_sizes,
                input_split_sizes,
                group_name,
            )
            output = fcols.wait_tensor(output)
            return output.sum()

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
