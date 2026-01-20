# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import _varying_collectives as vcols
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
class TestVaryingFunctionalCollectives(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @parametrize("device", devices)
    def test_all_reduce_invariant_forward(self, device):
        """Forward does all_reduce(sum) on varying tensors."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        input_tensor = torch.full((3, 3), fill_value=float(rank), device=device)
        output = vcols.all_reduce_invariant(input_tensor, "sum", group_name)

        expected = torch.full(
            (3, 3),
            fill_value=self.world_size * (self.world_size - 1) / 2,
            device=device,
        )
        self.assertEqual(output, expected)

    @parametrize("device", devices)
    def test_mark_varying_forward(self, device):
        """Forward is identity (no-op)."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, device=device)
        output = vcols.mark_varying(input_tensor, group_name)

        self.assertEqual(output, input_tensor)

    @parametrize("device", devices)
    def test_all_reduce_invariant_backward(self, device):
        """Backward is identity (no gradient aggregation)."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.randn(3, 3, requires_grad=True, device=device)
        output = vcols.all_reduce_invariant(input_tensor, "sum", group_name)

        output.sum().backward()

        expected_grad = torch.ones(3, 3, device=device)
        self.assertEqual(input_tensor.grad, expected_grad)

        grad_outputs = torch.rand_like(output, device=device)
        (grad_input,) = torch.autograd.grad(
            output, input_tensor, grad_outputs=grad_outputs
        )
        self.assertEqual(grad_input, grad_outputs)

    @parametrize("device", devices)
    def test_mark_varying_backward(self, device):
        """Backward does all_reduce(sum) to aggregate varying gradients."""
        group_name = dist.group.WORLD.group_name
        rank = dist.get_rank()

        input_tensor = torch.full(
            (3, 3), fill_value=float(rank), requires_grad=True, device=device
        )
        output = vcols.mark_varying(input_tensor, group_name)

        output.sum().backward()

        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full(
            (3, 3), fill_value=float(self.world_size), device=device
        )
        self.assertEqual(input_tensor.grad, expected_grad)

        input_tensor2 = torch.full(
            (3, 3), fill_value=float(rank), requires_grad=True, device=device
        )
        output2 = vcols.mark_varying(input_tensor2, group_name)
        grad_outputs = torch.rand_like(output2, device=device)
        (grad_input,) = torch.autograd.grad(
            output2, input_tensor2, grad_outputs=grad_outputs
        )

        from torch.distributed import _functional_collectives as fcols

        expected_grad_input = fcols.all_reduce(grad_outputs, "sum", group_name)
        self.assertEqual(grad_input, expected_grad_input)

    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    @parametrize("test_utils", test_utils)
    def test_all_reduce_invariant_opcheck(self, test_utils):
        """Verify custom op registration (schema, autograd, fake tensor)."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional.all_reduce_invariant,
            (input_tensor, "sum", group_name),
            test_utils=test_utils,
        )

    @parametrize("test_utils", test_utils)
    def test_mark_varying_opcheck(self, test_utils):
        """Verify custom op registration (schema, autograd, fake tensor)."""
        group_name = dist.group.WORLD.group_name

        input_tensor = torch.ones(3, 3, requires_grad=True)

        torch.library.opcheck(
            torch.ops._c10d_functional.mark_varying,
            (input_tensor, group_name),
            test_utils=test_utils,
        )

    def test_all_reduce_invariant_compile(self):
        """Backward works with torch.compile (no gradient aggregation)."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = vcols.all_reduce_invariant(tensor, "sum", group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.ones(3, 3)
        self.assertEqual(input_tensor.grad, expected_grad)

    def test_mark_varying_compile(self):
        """Backward works with torch.compile (aggregates gradients)."""
        group_name = dist.group.WORLD.group_name

        @torch.compile(fullgraph=True)
        def compiled_fn(tensor):
            output = vcols.mark_varying(tensor, group_name)
            return output.sum()

        input_tensor = torch.randn(3, 3, requires_grad=True)

        loss = compiled_fn(input_tensor)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        expected_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(input_tensor.grad, expected_grad)


if __name__ == "__main__":
    run_tests()
