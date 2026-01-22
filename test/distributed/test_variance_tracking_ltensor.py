# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._ltensor import (
    _get_dim_name_from_group,
    LTensor,
    ltensor_custom_function_call,
    register_variance_tracking_strategy,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import run_tests


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


DEVICE = "cpu"


class TestLTensorVarianceTracking(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_invariant_union(self):
        """Invariant + Invariant = Invariant."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), set(), set(), mesh)
        y = LTensor(torch.randn(4, 8), set(), set(), mesh)

        result = x + y

        self.assertEqual(result.variant_dims, set())

    def test_invariant_with_variant_union(self):
        """Variant + Invariant = Variant (union rule)."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)
        y = LTensor(torch.randn(4, 8), set(), set(), mesh)

        result = x + y
        self.assertEqual(result.variant_dims, {"dp"})

    def test_variant_invariant_backward(self):
        """Backward inserts mark_varying for invariant input, aggregating its gradients."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )
        rank = dist.get_rank()

        variant_input = LTensor(
            torch.full((3, 3), fill_value=float(rank), requires_grad=True),
            {"dp"},
            set(),
            mesh,
        )

        invariant_input = LTensor(
            torch.randn(3, 3, requires_grad=True),
            set(),
            set(),
            mesh,
        )

        output = variant_input + invariant_input
        self.assertEqual(output.variant_dims, {"dp"})
        output.sum().backward()

        self.assertIsNotNone(invariant_input._local_tensor.grad)
        expected_invariant_grad = torch.full((3, 3), fill_value=float(self.world_size))
        self.assertEqual(invariant_input._local_tensor.grad, expected_invariant_grad)

    def test_variant_unary(self):
        """Unary ops preserve variance."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)
        result = x.sin()

        self.assertEqual(result.variant_dims, {"dp"})

    def test_from_dtensor(self):
        """Non-Replicate placements become variant_dims."""
        from torch.distributed.tensor import DTensor

        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        local_tensor = torch.randn(4, 8)

        dtensor = DTensor.from_local(
            local_tensor, mesh, placements=[Shard(0), Replicate()]
        )
        ltensor = LTensor(
            dtensor._local_tensor, **LTensor.compute_metadata_from_dtensor(dtensor)
        )

        self.assertEqual(ltensor.variant_dims, {"dp"})

    def test_mixed_bias_pattern(self):
        """LTensor @ weight + bias preserves variance."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        data = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)
        weight = torch.randn(8, 16)
        bias = torch.randn(16)

        hidden = data @ weight
        self.assertEqual(hidden.variant_dims, {"dp"})

        output = hidden + bias
        self.assertEqual(output.variant_dims, {"dp"})
        self.assertIsInstance(output, LTensor)

    def test_mixed_meshes_error(self):
        """Cannot mix LTensors from different meshes."""
        mesh1 = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        mesh2 = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("pp", "dp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh1)
        y = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh2)

        with self.assertRaisesRegex(
            AssertionError, "Cannot mix LTensors from different meshes"
        ):
            _ = x + y

    def test_all_reduce_removes_variant_dim(self):
        """all_reduce makes tensor invariant on the reduced dim."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp", "tp"}, set(), mesh)
        self.assertEqual(x.variant_dims, {"dp", "tp"})

        import torch.distributed._functional_collectives as fcols

        result = fcols.all_reduce(x, "sum", mesh.get_group("dp").group_name)

        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, {"tp"})

    def test_tuple_of_ltensors(self):
        """torch.cat with a tuple of LTensors preserves variance and returns LTensor."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)
        y = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)
        z = LTensor(torch.randn(4, 8), {"dp"}, set(), mesh)

        result = torch.cat((x, y, z), dim=0)

        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.shape, (12, 8))
        self.assertEqual(result.variant_dims, {"dp"})


# Custom autograd.Function for testing register_variance_for_function
class CustomAllReduceFunction(torch.autograd.Function):
    """Custom autograd.Function that wraps all_reduce."""

    @staticmethod
    def forward(ctx, input, reduce_op, group_name):
        ctx.reduce_op = reduce_op
        ctx.group_name = group_name
        output = all_reduce(input, reduce_op, group_name)
        return torch.ops._c10d_functional.wait_tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        # Identity backward - gradient is already reduced
        return grad_output, None, None


@register_variance_tracking_strategy(CustomAllReduceFunction)
def _custom_all_reduce_variance(
    input_variant_dims, input_reduced_dims, mesh, input, reduce_op, group_name
):
    """all_reduce makes output invariant on the reduced dim."""
    dim_name = _get_dim_name_from_group(mesh, group_name)
    if dim_name is None:
        raise ValueError(
            f"Could not find mesh dim for group {group_name}. "
            f"Available dims: {mesh.mesh_dim_names}"
        )
    return input_variant_dims - {dim_name}, input_reduced_dims


class TestRegisterVarianceForFunction(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_custom_function_variance_strategy(self):
        """Test custom all_reduce function makes output invariant."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        torch.manual_seed(42)

        # Create a variant LTensor (simulating sharded data)
        local_tensor = torch.randn(4, 8, device=DEVICE, requires_grad=True)
        ltensor = LTensor(
            local_tensor, variant_dims={"dp"}, reduced_dims=set(), mesh=mesh
        )

        # Apply custom all_reduce - should make output invariant
        group_name = mesh.get_group("dp").group_name
        result = CustomAllReduceFunction.apply(ltensor, "sum", group_name)

        # Result should be LTensor with no variant dims (invariant)
        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, set())

    def test_custom_function_multi_dim_variance_strategy(self):
        """Test custom function with multi-dim variance."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        torch.manual_seed(42)

        # Create LTensor variant on both dims
        local_tensor = torch.randn(4, 8, device=DEVICE)
        ltensor = LTensor(
            local_tensor, variant_dims={"dp", "tp"}, reduced_dims=set(), mesh=mesh
        )

        # Apply custom all_reduce on "dp" dim - should remove "dp" from variance
        group_name = mesh.get_group("dp").group_name
        result = CustomAllReduceFunction.apply(ltensor, "sum", group_name)

        # Result should only be variant on "tp" now
        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, {"tp"})

    def test_unregistered_function_default_variance(self):
        """Test that unregistered functions use default variance (union of inputs)."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        class UnregisteredFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        torch.manual_seed(42)

        # Create a variant LTensor
        local_tensor = torch.randn(4, 8, device=DEVICE, requires_grad=True)
        ltensor = LTensor(
            local_tensor, variant_dims={"dp"}, reduced_dims=set(), mesh=mesh
        )

        # Apply unregistered function - should preserve variance (default behavior)
        result = UnregisteredFunction.apply(ltensor)

        # Result should be LTensor with same variant dims (union = original)
        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, {"dp"})


class TestLTensorFunctionCall(MultiThreadedTestCase):
    """Tests for ltensor_custom_function_call with custom autograd.Function."""

    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_ltensor_custom_function_call_with_strategy(self):
        """Test ltensor_custom_function_call with a registered variance strategy."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        # Create variant LTensor
        local_tensor = torch.randn(4, 8, device=DEVICE)
        ltensor = LTensor(local_tensor, {"dp"}, set(), mesh)

        # Use the existing CustomAllReduceFunction which has a registered strategy
        group_name = mesh.get_group("dp").group_name
        result = ltensor_custom_function_call(
            CustomAllReduceFunction, ltensor, "sum", group_name
        )

        # Strategy removes 'dp' from variant dims
        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, set())
        # Context should be reset after call
        self.assertEqual(ltensor.custom_function_context, False)
        self.assertEqual(result.custom_function_context, False)

    def test_ltensor_custom_function_call_passthrough(self):
        """Test ltensor_custom_function_call with all_reduce that preserves LTensor."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        class AllReducePassthrough(torch.autograd.Function):
            """Custom all_reduce that operates on LTensor and returns LTensor.

            This pattern is used when the autograd.Function internally handles
            variance tracking by operating on LTensors directly. The variance
            tracking strategy for all_reduce automatically updates variant dims.
            """

            @staticmethod
            def forward(ctx, x, group_name):
                ctx.group_name = group_name
                # Perform all_reduce on the LTensor - this goes through
                # LTensor.__torch_function__ which applies variance tracking
                # strategy and wraps the result as LTensor with correct dims
                result = all_reduce(x, "sum", group_name)
                return torch.ops._c10d_functional.wait_tensor(result)

            @staticmethod
            def backward(ctx, grad_output):
                # Identity backward - gradient is already reduced
                return grad_output, None

        local_tensor = torch.randn(4, 8, device=DEVICE)
        ltensor = LTensor(local_tensor, {"dp"}, set(), mesh)

        group_name = mesh.get_group("dp").group_name
        result = ltensor_custom_function_call(AllReducePassthrough, ltensor, group_name)

        # Result should be LTensor with 'dp' removed from variant dims
        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, set())
        self.assertEqual(result.custom_function_context, False)

    def test_custom_function_context_skips_mark_varying(self):
        """Test that mark_varying is skipped when in custom_function_context."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        # Create LTensors with different variant dims
        tensor1 = torch.randn(4, 4, device=DEVICE)
        tensor2 = torch.randn(4, 4, device=DEVICE)

        ltensor1 = LTensor(tensor1, {"dp"}, set(), mesh, custom_function_context=True)
        ltensor2 = LTensor(tensor2, set(), set(), mesh, custom_function_context=True)

        # Normally adding these would insert mark_varying for ltensor2,
        # but in custom_function_context it should be skipped
        result = ltensor1 + ltensor2

        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_dims, {"dp"})
        # custom_function_context should be preserved
        self.assertEqual(result.custom_function_context, True)


if __name__ == "__main__":
    run_tests()
