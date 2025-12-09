# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import run_tests


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


DEVICE = "cpu"


class TestLTensorParallelisms(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

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

    def test_megatron_tp_parallelism(self):
        """
        Megatron-style tensor parallelism with paired ColumnParallel + RowParallel layers
        https://arxiv.org/pdf/1909.08053

        This test validates the Megatron TP pattern where:
        - Attention: Q,K,V projections use ColumnParallel (no comm), output projection uses RowParallel (all-reduce)
        - MLP: First linear uses ColumnParallel (no comm), second linear uses RowParallel (all-reduce)

        We use all_reduce for the forward to go from Partial to Replicate,
        which has identity backward (no gradient aggregation needed since inputs are already sharded)
        """
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("tp",),
        )

        torch.manual_seed(0)

        batch_size = 4
        seq_len = 8
        hidden_dim = 16
        num_heads = 4  # noqa: F841
        mlp_dim = 32

        # Input and target (replicated)
        x_global = torch.randn(
            batch_size, seq_len, hidden_dim, device=DEVICE, requires_grad=True
        )
        y_global = torch.randn(
            batch_size, seq_len, hidden_dim, device=DEVICE, requires_grad=True
        )

        # Attention weights
        # Q, K, V projections: Column Parallel (sharded on output dim)
        wq_global = torch.randn(hidden_dim, hidden_dim, device=DEVICE, requires_grad=True)
        wk_global = torch.randn(hidden_dim, hidden_dim, device=DEVICE, requires_grad=True)
        wv_global = torch.randn(hidden_dim, hidden_dim, device=DEVICE, requires_grad=True)

        # Output projection: Row Parallel (sharded on input dim)
        wo_global = torch.randn(hidden_dim, hidden_dim, device=DEVICE, requires_grad=True)

        # MLP weights
        # First linear (gate/up): Column Parallel (sharded on output dim)
        w1_global = torch.randn(hidden_dim, mlp_dim, device=DEVICE, requires_grad=True)
        # Second linear (down): Row Parallel (sharded on input dim)
        w2_global = torch.randn(mlp_dim, hidden_dim, device=DEVICE, requires_grad=True)

        # ========== Reference: DTensor-based computation ==========

        x_dtensor = distribute_tensor(x_global, mesh, placements=[Replicate()])
        y_dtensor = distribute_tensor(y_global, mesh, placements=[Replicate()])

        # Attention weights distributed
        wq_dtensor = distribute_tensor(wq_global, mesh, placements=[Shard(1)])
        wk_dtensor = distribute_tensor(wk_global, mesh, placements=[Shard(1)])
        wv_dtensor = distribute_tensor(wv_global, mesh, placements=[Shard(1)])
        wo_dtensor = distribute_tensor(wo_global, mesh, placements=[Shard(0)])

        # MLP weights distributed
        w1_dtensor = distribute_tensor(w1_global, mesh, placements=[Shard(1)])
        w2_dtensor = distribute_tensor(w2_global, mesh, placements=[Shard(0)])

        # Attention forward (simplified, no actual attention computation)
        _ = x_dtensor @ wq_dtensor
        _ = x_dtensor @ wk_dtensor
        v = x_dtensor @ wv_dtensor
        # Simplified attention: just use v for demonstration
        attn_out = v @ wo_dtensor

        # MLP forward
        mlp_hidden = attn_out @ w1_dtensor
        mlp_out = mlp_hidden @ w2_dtensor

        loss_dtensor = (mlp_out - y_dtensor).pow(2).mean()
        loss_dtensor.backward()

        # Collect reference gradients
        # wq, wk are placeholders and skipped
        wv_dtensor_grad_norm = wv_dtensor.grad.full_tensor().norm()
        wo_dtensor_grad_norm = wo_dtensor.grad.full_tensor().norm()
        w1_dtensor_grad_norm = w1_dtensor.grad.full_tensor().norm()
        w2_dtensor_grad_norm = w2_dtensor.grad.full_tensor().norm()
        x_dtensor_grad_norm = x_dtensor.grad.full_tensor().norm()

        # ========== Test: LTensor with all_reduce ==========
        x_local = distribute_tensor(
            x_global.detach().clone().requires_grad_(True), mesh, placements=[Replicate()]
        )
        y_local = distribute_tensor(
            y_global.detach().clone().requires_grad_(True), mesh, placements=[Replicate()]
        )

        # Distribute weights
        wq_local = distribute_tensor(
            wq_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(1)]
        )
        wk_local = distribute_tensor(
            wk_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(1)]
        )
        wv_local = distribute_tensor(
            wv_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(1)]
        )
        wo_local = distribute_tensor(
            wo_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(0)]
        )
        w1_local = distribute_tensor(
            w1_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(1)]
        )
        w2_local = distribute_tensor(
            w2_global.detach().clone().requires_grad_(True), mesh, placements=[Shard(0)]
        )

        tp_group_name = mesh.get_group("tp").group_name

        @local_map(
            out_placements=[Replicate()],
            in_placements=[
                (Replicate(),),  # x
                (Replicate(),),  # y
                (Shard(1),),  # wq
                (Shard(1),),  # wk
                (Shard(1),),  # wv
                (Shard(0),),  # wo
                (Shard(1),),  # w1
                (Shard(0),),  # w2
            ],
            device_mesh=mesh,
            track_variant_axes=True,
        )
        def megatron_tp_forward(x, y, wq, wk, wv, wo, w1, w2):
            # === Attention Block ===
            # Column Parallel: Q, K, V projections (no communication)
            _ = x @ wq  # [B, S, H] @ [H, H/TP] -> [B, S, H/TP]
            _ = x @ wk
            v = x @ wv

            # Row Parallel: Output projection (needs all-reduce)
            # After v @ wo: [B, S, H/TP] @ [H/TP, H] -> [B, S, H] (partial sum)
            attn_partial = v @ wo

            # import pdb
            # import torch.distributed as dist
            # if dist.get_rank() == 0:
            #     pdb.set_trace()

            # all_reduce: forward does all-reduce, backward is identity
            attn_out = all_reduce(attn_partial, "sum", tp_group_name)

            # === MLP Block ===
            # Column Parallel: First linear (no communication)
            mlp_hidden = attn_out @ w1  # [B, S, H] @ [H, MLP/TP] -> [B, S, MLP/TP]

            # Row Parallel: Second linear (needs all-reduce)
            # After mlp_hidden @ w2: [B, S, MLP/TP] @ [MLP/TP, H] -> [B, S, H] (partial sum)
            mlp_partial = mlp_hidden @ w2
            # all_reduce: forward does all-reduce, backward is identity
            mlp_out = all_reduce(mlp_partial, "sum", tp_group_name)

            return mlp_out

        # Forward with LTensor
        out_local = megatron_tp_forward(
            x_local, y_local, wq_local, wk_local, wv_local, wo_local, w1_local, w2_local
        )
        loss_local = (out_local - y_local).pow(2).mean()

        loss_local.backward()

        # Collect LTensor gradients
        # wq, wk are placeholders and skipped
        wv_local_grad_norm = wv_local.grad.full_tensor().norm()
        wo_local_grad_norm = wo_local.grad.full_tensor().norm()
        w1_local_grad_norm = w1_local.grad.full_tensor().norm()
        w2_local_grad_norm = w2_local.grad.full_tensor().norm()
        x_local_grad_norm = x_local.grad.full_tensor().norm()

        # Verify gradients match between DTensor and LTensor with all_reduce
        self.assertEqual(wv_dtensor_grad_norm, wv_local_grad_norm)
        self.assertEqual(wo_dtensor_grad_norm, wo_local_grad_norm)
        self.assertEqual(w1_dtensor_grad_norm, w1_local_grad_norm)
        self.assertEqual(w2_dtensor_grad_norm, w2_local_grad_norm)
        self.assertEqual(x_dtensor_grad_norm, x_local_grad_norm)


if __name__ == "__main__":
    run_tests()
