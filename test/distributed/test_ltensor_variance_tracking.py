# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor._ltensor import LTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
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

        self.assertEqual(result.variant_axes, set())

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
        self.assertEqual(result.variant_axes, {"dp"})

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
        self.assertEqual(output.variant_axes, {"dp"})
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

        self.assertEqual(result.variant_axes, {"dp"})

    def test_from_dtensor(self):
        """Non-Replicate placements become variant_axes."""
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

        self.assertEqual(ltensor.variant_axes, {"dp"})

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
        self.assertEqual(hidden.variant_axes, {"dp"})

        output = hidden + bias
        self.assertEqual(output.variant_axes, {"dp"})
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
        self.assertEqual(x.variant_axes, {"dp", "tp"})

        import torch.distributed._functional_collectives as fcols

        result = fcols.all_reduce(
            x, "sum", mesh.get_group("dp").group_name
        ).trigger_wait()

        self.assertIsInstance(result, LTensor)
        self.assertEqual(result.variant_axes, {"tp"})

    def test_data_parallel(self):
        """Data parallel with and without local_map produces same results."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("dp",),
        )

        # Test with DTensor
        torch.manual_seed(0)

        global_batch_size = 24
        d_in = 8
        x_global = torch.randn(
            global_batch_size, d_in, device=DEVICE, requires_grad=True
        )
        y_global = torch.randn(global_batch_size, 1, device=DEVICE, requires_grad=True)
        w_global = torch.randn(d_in, 1, device=DEVICE, requires_grad=True)

        x_dtensor = distribute_tensor(x_global, mesh, placements=[Shard(0)])
        y_dtensor = distribute_tensor(y_global, mesh, placements=[Shard(0)])
        w_dtensor = distribute_tensor(w_global, mesh, placements=[Replicate()])

        pred_dtensor = x_dtensor @ w_dtensor
        loss_dtensor = (pred_dtensor - y_dtensor).pow(2).mean()
        loss_dtensor.backward()
        w_dtensor_norm = w_dtensor.grad.full_tensor().norm()

        # Test WITH local_map
        x_local_map = distribute_tensor(
            x_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(0)]
        )
        y_local_map = distribute_tensor(
            y_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(0)]
        )
        w_local_map = distribute_tensor(
            w_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )

        @local_map(
            out_placements=[Shard(0)],
            in_placements=[(Shard(0),), (Replicate(),)],
            # Previously required this: in_grad_placements=[(Shard(0),), (Partial(),)],
            device_mesh=mesh,
            track_variant_axes=True,
        )
        def local_linear(x, weight):
            return x @ weight

        # Forward with local_map
        pred_local_map = local_linear(x_local_map, w_local_map)
        loss_local_map = (pred_local_map - y_local_map).pow(2).mean()
        loss_local_map.backward()
        w_local_map_norm = w_local_map.grad.full_tensor().norm()

        self.assertEqual(w_dtensor_norm, w_local_map_norm)

    def test_tensor_parallel(self):
        """Tensor parallel (ColumnParallel + RowParallel) with and without local_map produces same results."""
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("tp",),
        )

        # Test with DTensor
        torch.manual_seed(0)

        batch_size = 8
        d_in = 16
        d_hidden = 32
        d_out = 16
        x_global = torch.randn(batch_size, d_in, device=DEVICE, requires_grad=True)
        y_global = torch.randn(batch_size, d_out, device=DEVICE, requires_grad=True)
        # Column parallel: weight sharded along output dim (Shard(1))
        w1_global = torch.randn(d_in, d_hidden, device=DEVICE, requires_grad=True)
        # Row parallel: weight sharded along input dim (Shard(0))
        w2_global = torch.randn(d_hidden, d_out, device=DEVICE, requires_grad=True)

        # Replicate input, shard w1 along columns, shard w2 along rows
        x_dtensor = distribute_tensor(x_global, mesh, placements=[Replicate()])
        y_dtensor = distribute_tensor(y_global, mesh, placements=[Replicate()])
        w1_dtensor = distribute_tensor(w1_global, mesh, placements=[Shard(1)])
        w2_dtensor = distribute_tensor(w2_global, mesh, placements=[Shard(0)])

        # Forward: x @ w1 @ w2
        # After x @ w1: output is Shard(1) on last dim
        # After @ w2: output is Partial, then all-reduced to Replicate
        hidden_dtensor = x_dtensor @ w1_dtensor
        pred_dtensor = hidden_dtensor @ w2_dtensor
        loss_dtensor = (pred_dtensor - y_dtensor).pow(2).mean()
        loss_dtensor.backward()
        w1_dtensor_norm = w1_dtensor.grad.full_tensor().norm()
        w2_dtensor_norm = w2_dtensor.grad.full_tensor().norm()
        x_dtensor_norm = x_dtensor.grad.full_tensor().norm()

        # Test WITH local_map
        x_local_map = distribute_tensor(
            x_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )
        y_local_map = distribute_tensor(
            y_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )
        w1_local_map = distribute_tensor(
            w1_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(1)]
        )
        w2_local_map = distribute_tensor(
            w2_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(0)]
        )

        @local_map(
            out_placements=[Partial()],
            in_placements=[(Replicate(),), (Shard(1),), (Shard(0),)],
            device_mesh=mesh,
            track_variant_axes=True,
        )
        def local_matmul(x, w1, w2):
            h = x @ w1
            return h @ w2

        # Forward with local_map
        pred_local_map = local_matmul(x_local_map, w1_local_map, w2_local_map)
        loss_local_map = (pred_local_map - y_local_map).pow(2).mean()

        loss_local_map.backward()

        w1_local_map_norm = w1_local_map.grad.full_tensor().norm()
        w2_local_map_norm = w2_local_map.grad.full_tensor().norm()
        x_local_map_norm = x_local_map.grad.full_tensor().norm()

        self.assertEqual(w1_dtensor_norm, w1_local_map_norm)
        self.assertEqual(w2_dtensor_norm, w2_local_map_norm)
        self.assertEqual(x_dtensor_norm, x_local_map_norm)


if __name__ == "__main__":
    run_tests()
