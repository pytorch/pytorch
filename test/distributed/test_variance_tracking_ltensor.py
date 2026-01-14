# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._ltensor import LTensor
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

        x = LTensor(torch.randn(4, 8), set(), mesh)
        y = LTensor(torch.randn(4, 8), set(), mesh)

        result = x + y

        self.assertEqual(result.variant_dims, set())

    def test_invariant_with_variant_union(self):
        """Variant + Invariant = Variant (union rule)."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp"}, mesh)
        y = LTensor(torch.randn(4, 8), set(), mesh)

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
            mesh,
        )

        invariant_input = LTensor(
            torch.randn(3, 3, requires_grad=True),
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

        x = LTensor(torch.randn(4, 8), {"dp"}, mesh)
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

        data = LTensor(torch.randn(4, 8), {"dp"}, mesh)
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

        x = LTensor(torch.randn(4, 8), {"dp"}, mesh1)
        y = LTensor(torch.randn(4, 8), {"dp"}, mesh2)

        with self.assertRaisesRegex(
            RuntimeError, "Cannot mix LTensors from different meshes"
        ):
            _ = x + y

    def test_all_reduce_removes_variant_axis(self):
        """all_reduce makes tensor invariant on the reduced axis."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        x = LTensor(torch.randn(4, 8), {"dp", "tp"}, mesh)
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

        x = LTensor(torch.randn(4, 8), {"dp"}, mesh)
        y = LTensor(torch.randn(4, 8), {"dp"}, mesh)
        z = LTensor(torch.randn(4, 8), {"dp"}, mesh)

        result = torch.cat((x, y, z), dim=0)

        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.shape, (12, 8))
        self.assertEqual(result.variant_dims, {"dp"})


if __name__ == "__main__":
    run_tests()
