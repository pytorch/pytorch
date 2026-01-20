# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_gather_tensor_autograd,
    all_reduce,
    all_to_all_single,
)
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._func_map import local_map
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
            track_variant_dims=True,
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
            track_variant_dims=True,
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

    def test_megatron_tp_parallel(self):
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
        # wq, wk are placeholders and skipped
        wv_global = torch.randn(
            hidden_dim, hidden_dim, device=DEVICE, requires_grad=True
        )

        # Output projection: Row Parallel (sharded on input dim)
        wo_global = torch.randn(
            hidden_dim, hidden_dim, device=DEVICE, requires_grad=True
        )

        # MLP weights
        # First linear (gate/up): Column Parallel (sharded on output dim)
        w1_global = torch.randn(hidden_dim, mlp_dim, device=DEVICE, requires_grad=True)
        # Second linear (down): Row Parallel (sharded on input dim)
        w2_global = torch.randn(mlp_dim, hidden_dim, device=DEVICE, requires_grad=True)

        # ========== Reference: DTensor-based computation ==========

        x_dtensor = distribute_tensor(x_global, mesh, placements=[Replicate()])
        y_dtensor = distribute_tensor(y_global, mesh, placements=[Replicate()])

        # Attention weights distributed
        # wq, wk are placeholders and skipped
        wv_dtensor = distribute_tensor(wv_global, mesh, placements=[Shard(1)])
        wo_dtensor = distribute_tensor(wo_global, mesh, placements=[Shard(0)])

        # MLP weights distributed
        w1_dtensor = distribute_tensor(w1_global, mesh, placements=[Shard(1)])
        w2_dtensor = distribute_tensor(w2_global, mesh, placements=[Shard(0)])

        # Attention forward (simplified, no actual attention computation)
        # wq, wk are placeholders and skipped
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
            x_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )
        y_local = distribute_tensor(
            y_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )

        # Distribute weights
        # wq, wk are placeholders and skipped
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
                (Shard(1),),  # wv
                (Shard(0),),  # wo
                (Shard(1),),  # w1
                (Shard(0),),  # w2
            ],
            device_mesh=mesh,
            track_variant_dims=True,
        )
        def megatron_tp_forward(x, y, wv, wo, w1, w2):
            # === Attention Block ===
            # wq, wk are placeholders and skipped ((x @ wq) @ (x @ wk)^T) <-> [B, S, H]

            # Column Parallel: Q, K, V projections (no communication)
            v = x @ wv  # [B, S, H] @ [H, H/TP] -> [B, S, H/TP]

            # Row Parallel: Output projection (needs all-reduce)
            # After v @ wo: [B, S, H/TP] @ [H/TP, H] -> [B, S, H] (partial sum)
            attn_partial = v @ wo

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
            x_local, y_local, wv_local, wo_local, w1_local, w2_local
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

    def test_fsdp_tp_parallel(self):
        """
        FSDP + TP (Fully Sharded Data Parallel + Tensor Parallel) with local_map.

        - FSDP dim: Parameters are sharded and gathered via all_gather, gradients
          use reduce_scatter
        - TP dim: Activations are sharded across the TP dimension, using psum_scatter
          for forward and all_gather for backward
        """

        # 2D mesh: 2 FSDP groups x 2 TP groups = 4 total devices
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size).reshape(2, 2),
            mesh_dim_names=("tp", "fsdp"),
        )

        torch.manual_seed(0)

        # Dimensions
        batch_size = 8
        d_in = 16
        d_hidden = 32
        d_out = 8

        # Create global tensors
        # Input: sharded along batch (FSDP) and features (TP)
        x_global = torch.randn(batch_size, d_in, device=DEVICE, requires_grad=True)
        y_global = torch.randn(batch_size, d_out, device=DEVICE, requires_grad=True)

        # Weights: sharded along both FSDP and TP dims
        # W1: [d_in, d_hidden] - sharded along FSDP (dim 0) and TP (dim 1)
        w1_global = torch.randn(d_in, d_hidden, device=DEVICE, requires_grad=True)

        # W2: [d_hidden, d_out] - sharded along FSDP (dim 0) and TP (dim 1)
        w2_global = torch.randn(d_hidden, d_out, device=DEVICE, requires_grad=True)

        # ========== Test: FSDP+TP with regular tensors ==========

        x_ref = x_global.detach().clone().requires_grad_(True)
        y_ref = y_global.detach().clone().requires_grad_(True)
        w1_ref = w1_global.detach().clone().requires_grad_(True)
        w2_ref = w2_global.detach().clone().requires_grad_(True)

        # Forward pass (reference)
        h1_ref = x_ref @ w1_ref
        pred_ref = h1_ref @ w2_ref
        loss_ref = ((pred_ref - y_ref) ** 2).mean()
        loss_ref.backward()

        # ========== Test: FSDP+TP with DTensors ==========

        x_dtensor = distribute_tensor(
            x_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate(), Shard(0)],
        )
        y_dtensor = distribute_tensor(
            y_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate(), Shard(0)],
        )

        # Weights: Shard(1)/Shard(0) on TP and Shard(0) on FSDP
        # For FSDP all_gather, reduce_scatter occurs in the backward
        w1_dtensor = distribute_tensor(
            w1_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Shard(1), Shard(0)],  # Sharded on both dims
        )
        w2_dtensor = distribute_tensor(
            w2_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Shard(0), Shard(0)],  # Sharded on both dims (dim 0)
        )

        h1_dtensor = x_dtensor @ w1_dtensor
        pred_dtensor = h1_dtensor @ w2_dtensor
        loss_dtensor = (pred_dtensor - y_dtensor).pow(2).mean()
        loss_dtensor.backward()

        # ========== Test: FSDP+TP with local_map ==========

        # Input: Shard(1) on TP (heads) and Replicate() on FSDP (batch)
        x_dist = distribute_tensor(
            x_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate(), Shard(0)],
        )
        y_dist = distribute_tensor(
            y_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Replicate(), Shard(0)],
        )

        # Weights: Shard(1)/Shard(0) on TP and Shard(0) on FSDP
        # For FSDP all_gather, reduce_scatter occurs in the backward
        w1_dist = distribute_tensor(
            w1_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Shard(1), Shard(0)],  # Sharded on both dims
        )
        w2_dist = distribute_tensor(
            w2_global.detach().clone().requires_grad_(True),
            mesh,
            placements=[Shard(0), Shard(0)],  # Sharded on both dims (dim 0)
        )

        fsdp_group_name = mesh.get_group("fsdp").group_name
        tp_group_name = mesh.get_group("tp").group_name

        @local_map(
            out_placements=[Replicate(), Shard(0)],
            in_placements=[
                (Replicate(), Shard(0)),  # x: batch on FSDP, features on TP
                (Replicate(), Shard(0)),  # y: batch on FSDP
                (Shard(1), Shard(0)),  # w1: FSDP + TP sharded
                (Shard(0), Shard(0)),  # w2: FSDP + TP sharded
            ],
            device_mesh=mesh,
            track_variant_dims=True,
        )
        def fsdp_tp_forward(x_local, y_local, w1_frag, w2_frag):
            # FSDP: all_gather weights along FSDP dim
            w1 = all_gather_tensor_autograd(
                w1_frag, gather_dim=0, group=fsdp_group_name
            )
            w2 = all_gather_tensor_autograd(
                w2_frag, gather_dim=0, group=fsdp_group_name
            )

            # Layer 1: matmul
            # x_local: [batch/fsdp, d_in], w1_frag: [d_in/fsdp, d_hidden/tp]
            h1_shard = x_local @ w1

            # Layer 2: matmul
            # h_shard: [batch/fsdp, d_hidden/tp], w2_frag: [d_hidden/tp, dout]
            h2_partial = h1_shard @ w2
            output = all_reduce(h2_partial, "sum", group=tp_group_name)

            # Compute loss with two reductions
            return output

        # Forward with FSDP+TP
        pred_dist = fsdp_tp_forward(x_dist, y_dist, w1_dist, w2_dist)

        loss_dist = ((pred_dist - y_dist) ** 2).mean()
        loss_dist.backward()

        w1_dtensor_grad_norm = w1_dtensor.grad.full_tensor().norm()
        w1_dist_grad_norm = w1_dist.grad.full_tensor().norm()
        self.assertTrue(
            torch.allclose(w1_dtensor_grad_norm, w1_dist_grad_norm, atol=1e-4),
            f"W1 grad norm mismatch: ref={w1_dtensor_grad_norm.item()}, dist={w1_dist_grad_norm.item()}",
        )

        w2_dtensor_grad_norm = w2_dtensor.grad.full_tensor().norm()
        w2_dist_grad_norm = w2_dist.grad.full_tensor().norm()
        self.assertTrue(
            torch.allclose(w2_dtensor_grad_norm, w2_dist_grad_norm, atol=1e-4),
            f"W2 grad norm mismatch: ref={w2_dtensor_grad_norm.item()}, dist={w2_dist_grad_norm.item()}",
        )

    def test_context_parallel(self):
        """
        Context Parallelism (CP) - sequence dimension sharded across ranks.

        Setup (4 ranks):
            Rank 0       Rank 1       Rank 2       Rank 3
            ─────────    ─────────    ─────────    ─────────
            Seq[0:4]     Seq[4:8]     Seq[8:12]    Seq[12:16]  ← CP: seq sharded
            Q[0:4]       Q[4:8]       Q[8:12]      Q[12:16]    ← Q stays local
            K[0:4]       K[4:8]       K[8:12]      K[12:16]    ← K rotates
        """

        # 1D mesh for context parallelism
        cp_size = self.world_size  # 4
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(cp_size),
            mesh_dim_names=("cp",),
        )

        # Model dimensions
        B = 2  # batch_size
        L = 16  # seq_len (global, will be sharded so L/cp = 4 per rank)
        D = 8  # hidden_dim

        # Create global tensors with deterministic seed
        torch.manual_seed(42 + dist.get_rank())
        x_BLD = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        w_DD = torch.randn(D, D, device=DEVICE, requires_grad=True)

        # Broadcast from rank 0 to ensure all ranks have the same global tensors
        dist.broadcast(x_BLD, src=0)
        dist.broadcast(w_DD, src=0)

        # ========== Reference: Single-device computation ==========
        x_ref = x_BLD.clone().detach().requires_grad_(True)
        w_ref = w_DD.clone().detach().requires_grad_(True)

        # Simple attention-like op: each position attends to all positions
        # scores_BLL = softmax(x @ x.T), out = scores @ x @ w
        x_flat_TtD = x_ref.view(-1, D)  # [B*L, D]
        scores_TtTt = torch.softmax(x_flat_TtD @ x_flat_TtD.T, dim=-1)  # [B*L, B*L]
        attended_TtD = scores_TtTt @ x_flat_TtD  # [B*L, D]
        out_ref_TtD = attended_TtD @ w_ref  # [B*L, D]
        out_ref_BLD = out_ref_TtD.view(B, L, D)

        loss_ref = out_ref_BLD.sum()
        loss_ref.backward()

        # ========== Test: Context Parallel with local_map ==========
        x_dist_BLcD = distribute_tensor(
            x_BLD.clone().detach().requires_grad_(True),
            mesh,
            placements=[Shard(1)],  # shard on sequence dim
        )
        w_dist_DD = distribute_tensor(
            w_DD.clone().detach().requires_grad_(True),
            mesh,
            placements=[Replicate()],
        )

        cp_group_name = mesh.get_group("cp").group_name

        @local_map(
            out_placements=[Shard(1)],  # output seq-sharded
            in_placements=[
                (Shard(1),),  # x_BLcD: seq sharded
                (Replicate(),),  # w_DD: replicated
            ],
            device_mesh=mesh,
            track_variant_dims=True,
        )
        def context_parallel_forward(x_BLcD, w_DD):
            # x_BLcD: [B, L/cp, D] - local sequence chunk
            local_B = x_BLcD.shape[0]
            local_Lc = x_BLcD.shape[1]  # L/cp
            local_D = x_BLcD.shape[2]

            # all_gather to get full sequence: [B, L/cp, D] -> [B, L, D]
            x_BLD = all_gather_tensor_autograd(
                x_BLcD, gather_dim=1, group=cp_group_name
            )

            # Attention-like op on full sequence
            full_L = x_BLD.shape[1]
            x_flat_TtD = x_BLD.view(-1, local_D)  # [B*L, D]
            scores_TtTt = torch.softmax(x_flat_TtD @ x_flat_TtD.T, dim=-1)
            attended_TtD = scores_TtTt @ x_flat_TtD
            out_TtD = attended_TtD @ w_DD
            out_BLD = out_TtD.view(local_B, full_L, local_D)

            # Return only local chunk (seq sharded output)
            local_cp_rank = dist.get_rank()
            start_idx = local_cp_rank * local_Lc
            end_idx = start_idx + local_Lc
            return out_BLD[:, start_idx:end_idx, :]

        # Forward
        out_dist_BLcD = context_parallel_forward(x_dist_BLcD, w_dist_DD)

        # Loss and backward
        loss_dist = out_dist_BLcD.sum()
        loss_dist.backward()

        # Verify forward
        out_full_BLD = out_dist_BLcD.full_tensor()
        self.assertTrue(
            torch.allclose(out_ref_BLD, out_full_BLD, atol=1e-5),
            f"Output mismatch: max diff = {(out_ref_BLD - out_full_BLD).abs().max().item()}",
        )

        # Verify gradients
        w_ref_grad_norm = w_ref.grad.norm()
        w_dist_grad_norm = w_dist_DD.grad.full_tensor().norm()
        self.assertTrue(
            torch.allclose(w_ref_grad_norm, w_dist_grad_norm, atol=1e-4),
            f"W grad norm mismatch: ref={w_ref_grad_norm.item()}, dist={w_dist_grad_norm.item()}",
        )

        x_ref_grad_norm = x_ref.grad.norm()
        x_dist_grad_norm = x_dist_BLcD.grad.full_tensor().norm()
        self.assertTrue(
            torch.allclose(x_ref_grad_norm, x_dist_grad_norm, atol=1e-4),
            f"X grad norm mismatch: ref={x_ref_grad_norm.item()}, dist={x_dist_grad_norm.item()}",
        )

    def test_loss_parallel(self):
        """
        Loss Parallelism for NLL loss with class-sharded logits.

        Setup (4 ranks with 16 classes total):
            Rank 0       Rank 1       Rank 2       Rank 3
            ─────────    ─────────    ─────────    ─────────
            Classes      Classes      Classes      Classes
            [0:4]        [4:8]        [8:12]       [12:16]
        """
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(self.world_size),
            mesh_dim_names=("tp",),
        )

        torch.manual_seed(42)

        # Dimensions
        T = 8  # total tokens (B * L)
        C = 16  # num_classes (4 classes per rank with 4 ranks)
        Cp = C // self.world_size  # classes per rank = 4

        # Create logits and targets
        logits_TC = torch.randn(T, C, device=DEVICE, requires_grad=True)
        # Targets: [T] - replicated, contains class indices 0 to C-1
        targets_T = torch.randint(0, C, (T,), device=DEVICE)

        # Broadcast to ensure all ranks have same data
        dist.broadcast(logits_TC, src=0)
        dist.broadcast(targets_T, src=0)

        # ========== Reference: Single-device cross-entropy loss (numerically stable) ==========
        # Using the max trick: loss = log(sum(exp(z - m))) - (z_y - m)
        logits_ref_TC = logits_TC.clone().detach().requires_grad_(True)
        targets_ref_T = targets_T.clone()

        m_ref_T = logits_ref_TC.max(dim=1).values
        S_ref_T = torch.exp(logits_ref_TC - m_ref_T[:, None]).sum(dim=1)
        zy_minus_m_ref_T = (
            (logits_ref_TC - m_ref_T[:, None])
            .gather(1, targets_ref_T[:, None])
            .squeeze(1)
        )

        loss_ref = (torch.log(S_ref_T) - zy_minus_m_ref_T).mean()
        loss_ref.backward()

        # ========== Test: Loss parallel with class-sharded logits ==========
        logits_dist_TCp = distribute_tensor(
            logits_TC.clone().detach().requires_grad_(True),
            mesh,
            placements=[Shard(1)],  # Sharded on vocab/class dimension
        )

        tp_group_name = mesh.get_group("tp").group_name

        @local_map(
            out_placements=[Replicate()],  # Loss is replicated (scalar)
            in_placements=[
                (Shard(1),),  # logits_TCp: sharded on class dim
                None,  # targets_T: plain tensor (not DTensor)
                None,  # Cp: int (classes per rank)
                None,  # C: int (total classes)
            ],
            device_mesh=mesh,
            track_variant_dims=True,
        )
        def loss_parallel_cross_entropy(logits_TCp, targets_T, Cp, C):
            """
            Compute cross-entropy loss on class-sharded logits (numerically stable).
            """
            local_rank = dist.get_rank()

            # This rank handles classes [start_class, end_class)
            start_class = local_rank * Cp
            end_class = start_class + Cp

            # Step 1: Compute global max for numerical stability (detached - no gradient needed)
            m_local_T = logits_TCp.max(dim=1).values.detach()
            m_T = all_reduce(m_local_T, "max", tp_group_name).detach()

            # Step 2: Compute global sum of exp(z - m)
            S_local_T = torch.exp(logits_TCp - m_T[:, None]).sum(dim=1)
            S_T = all_reduce(S_local_T, "sum", tp_group_name)

            # Step 3: Gather target logit z_y (only one rank has it)
            # Check which targets are in this rank's range
            in_range_T = (targets_T >= start_class) & (targets_T < end_class)

            # Convert global indices to local indices (for targets in range)
            local_targets_T = torch.where(
                in_range_T,
                targets_T - start_class,
                torch.zeros_like(targets_T),
            )

            # Gather from local logits: [T, Cp] -> [T]
            gathered_T = torch.gather(
                logits_TCp, dim=1, index=local_targets_T.unsqueeze(1)
            ).squeeze(1)

            # Mask out-of-range contributions (set to 0)
            masked_gathered_T = torch.where(
                in_range_T,
                gathered_T,
                torch.zeros_like(gathered_T),
            )

            # All-reduce to combine: each target's logit comes from exactly one rank
            zy_T = all_reduce(masked_gathered_T, "sum", tp_group_name)

            # Step 4: Compute cross-entropy loss = log(S) - (z_y - m)
            loss = (torch.log(S_T) - (zy_T - m_T)).mean()
            return loss

        loss_dist = loss_parallel_cross_entropy(logits_dist_TCp, targets_T, Cp, C)
        loss_dist.backward()

        # Verify forward: loss values should match
        self.assertTrue(
            torch.allclose(loss_ref, loss_dist.full_tensor(), atol=1e-5),
            f"Loss mismatch: ref={loss_ref.item()}, dist={loss_dist.full_tensor().item()}",
        )

        # Verify backward: gradient norms should match
        logits_ref_grad_norm = logits_ref_TC.grad.norm()
        logits_dist_grad_norm = logits_dist_TCp.grad.full_tensor().norm()
        self.assertTrue(
            torch.allclose(logits_ref_grad_norm, logits_dist_grad_norm, atol=1e-4),
            f"Grad norm mismatch: ref={logits_ref_grad_norm.item()}, "
            f"dist={logits_dist_grad_norm.item()}",
        )

        # Also verify the actual gradient values match
        self.assertTrue(
            torch.allclose(
                logits_ref_TC.grad, logits_dist_TCp.grad.full_tensor(), atol=1e-5
            ),
            f"Grad mismatch: max diff = "
            f"{(logits_ref_TC.grad - logits_dist_TCp.grad.full_tensor()).abs().max().item()}",
        )

    def test_moe_expert_parallel(self):
        """
        Mixture of Experts (MoE) with flat DP + Expert Parallelism (EP).

        Tensor shape notation:
            B = batch_size (global)
            Bd = batch_size per rank (B / world_size)
            L = seq_len
            E = num_experts
            Ep = experts per rank (E / world_size)
            K = topk (selected experts per token)
            T = total tokens per rank (Bd * L)
            D = hidden_dim (d_model)
            F = expert_dim (FFN intermediate)

        Setup (4 ranks):
            Rank 0       Rank 1       Rank 2       Rank 3
            ─────────    ─────────    ─────────    ─────────
            Batch [0]    Batch [1]    Batch [2]    Batch [3]  ← Flat DP
            Exp [0,1]    Exp [2,3]    Exp [4,5]    Exp [6,7]  ← EP: 2 per rank

                         ←── all_to_all ──→
                         (exchange tokens)

        MoE Forward Pass:
        1. Router: compute gating scores and select top-k experts per token
        2. Dispatch: all_to_all to send tokens to their assigned expert ranks
        3. Expert FFN: each rank processes tokens assigned to its local experts
        4. Combine: all_to_all to gather results back to original token positions
        5. Weighted sum: combine expert outputs using gating weights
        """

        # 1D mesh with 4 ranks for flat DP + EP
        ep_size = self.world_size  # 4
        mesh = DeviceMesh(
            DEVICE,
            torch.arange(ep_size),
            mesh_dim_names=("ep",),
        )

        # Model dimensions
        B = 4  # batch_size (global, will be sharded so Bd=1 per rank)
        L = 8  # seq_len
        D = 16  # hidden_dim (d_model)
        E = 8  # num_experts (2 experts per rank with 4 ranks)
        F = 32  # expert_dim (FFN intermediate)
        K = 2  # topk (each token selects K experts)

        experts_per_rank = E // ep_size  # 2 experts per rank

        # Create global tensors with deterministic seed
        torch.manual_seed(42 + dist.get_rank())
        x_BLD = torch.randn(B, L, D, device=DEVICE)
        router_DE = torch.randn(D, E, device=DEVICE)
        w1_EDF = torch.randn(E, D, F, device=DEVICE)
        w2_EFD = torch.randn(E, F, D, device=DEVICE)

        # Broadcast from rank 0 to ensure all ranks have the same global tensors
        dist.broadcast(x_BLD, src=0)
        dist.broadcast(router_DE, src=0)
        dist.broadcast(w1_EDF, src=0)
        dist.broadcast(w2_EFD, src=0)

        # ========== Reference: Single-device computation ==========
        x_TD = x_BLD.view(-1, D)
        total_tokens = x_TD.shape[0]

        # Router scores
        logits_TE = x_TD @ router_DE
        probs_TE = torch.softmax(logits_TE, dim=-1)

        # Top-k selection
        weights_TK, indices_TK = torch.topk(probs_TE, K, dim=-1)
        weights_TK = weights_TK / weights_TK.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        out_TD = torch.zeros_like(x_TD)

        for t in range(total_tokens):
            token_1D = x_TD[t : t + 1]
            for k in range(K):
                expert_idx = indices_TK[t, k].item()
                weight = weights_TK[t, k]

                # Expert FFN: [1, D] @ [D, F] -> [1, F] -> [1, D]
                h_1F = token_1D @ w1_EDF[expert_idx]
                h_1F = torch.relu(h_1F)
                expert_out_1D = h_1F @ w2_EFD[expert_idx]

                out_TD[t] += weight * expert_out_1D.squeeze(0)

        out_ref_BLD = out_TD.view_as(x_BLD)

        # ========== Test: Flat DP + EP with local_map and all_to_all ==========
        # Placements (1D mesh):
        #   x_BLD: Shard(0) - batch sharded (each rank gets unique batch)
        #   router_DE: Replicate - same router on all ranks
        #   w1_EDF, w2_EFD: Shard(0) - experts sharded (2 per rank)

        x_dist_BdLD = distribute_tensor(
            x_BLD.clone(),
            mesh,
            placements=[Shard(0)],  # batch sharded
        )

        router_dist_DE = distribute_tensor(
            router_DE.clone(),
            mesh,
            placements=[Replicate()],
        )

        # Expert weights sharded: [E/ep, D, F] and [E/ep, F, D]
        w1_dist_EpDF = distribute_tensor(
            w1_EDF.clone(),
            mesh,
            placements=[Shard(0)],
        )
        w2_dist_EpFD = distribute_tensor(
            w2_EFD.clone(),
            mesh,
            placements=[Shard(0)],
        )

        ep_group_name = mesh.get_group("ep").group_name

        @local_map(
            out_placements=[Shard(0)],  # output batch-sharded
            in_placements=[
                (Shard(0),),  # x_BdLD: batch sharded
                (Replicate(),),  # router_DE: replicated
                (Shard(0),),  # w1_EpDF: experts sharded
                (Shard(0),),  # w2_EpFD: experts sharded
            ],
            device_mesh=mesh,
            track_variant_dims=True,
        )
        def moe_ep_forward(x_BdLD, router_DE, w1_EpDF, w2_EpFD):
            # x_BdLD: [B/ep, L, D] - local batch for this rank
            # w1_EpDF: [E/ep, D, F], w2_EpFD: [E/ep, F, D]

            local_ep_rank = dist.get_rank()

            local_B = x_BdLD.shape[0]  # B/ep = 2
            local_L = x_BdLD.shape[1]
            local_D = x_BdLD.shape[2]
            local_T = local_B * local_L

            x_TD = x_BdLD.view(-1, local_D)

            # Router: [T, D] @ [D, E] -> [T, E]
            logits_TE = x_TD @ router_DE
            probs_TE = torch.softmax(logits_TE, dim=-1)

            # Top-k selection
            weights_TK, indices_TK = torch.topk(probs_TE, K, dim=-1)
            weights_TK = weights_TK / weights_TK.sum(dim=-1, keepdim=True)

            # Expand tokens by topk: each token duplicated K times
            x_TKD = x_TD.unsqueeze(1).expand(-1, K, -1)
            x_TmKD = x_TKD.reshape(-1, local_D)  # [T*K, D]

            indices_TmK = indices_TK.view(-1)  # [T*K]
            weights_TmK = weights_TK.view(-1)  # [T*K]

            # Sort by expert index to group tokens destined for same expert
            sort_idx_TmK = torch.argsort(indices_TmK)
            sorted_x_TmKD = x_TmKD[sort_idx_TmK]
            sorted_ids_TmK = indices_TmK[sort_idx_TmK]

            # Count tokens per expert
            counts_E = torch.bincount(sorted_ids_TmK, minlength=E)

            # Compute split sizes for all_to_all (tokens per EP rank)
            input_split_sizes = []
            for ep_r in range(ep_size):
                start_e = ep_r * experts_per_rank
                end_e = (ep_r + 1) * experts_per_rank
                count = counts_E[start_e:end_e].sum().item()
                input_split_sizes.append(int(count))

            # Exchange counts via all_to_all to learn output_split_sizes
            send_counts_Ep = torch.tensor(
                input_split_sizes, dtype=torch.float32, device=x_BdLD.device
            )
            recv_counts_Ep = all_to_all_single(
                send_counts_Ep.unsqueeze(-1), None, None, ep_group_name
            ).squeeze(-1)
            output_split_sizes = [int(x) for x in recv_counts_Ep.tolist()]

            # Dispatch tokens via all_to_all: [T*K, D] -> [R, D] where R=recv_total
            recv_x_RD = all_to_all_single(
                sorted_x_TmKD, output_split_sizes, input_split_sizes, ep_group_name
            )

            # Dispatch expert IDs: [T*K] -> [R]
            recv_ids_R = (
                all_to_all_single(
                    sorted_ids_TmK.float().unsqueeze(-1),
                    output_split_sizes,
                    input_split_sizes,
                    ep_group_name,
                )
                .squeeze(-1)
                .long()
            )

            # Process with local experts (vectorized per expert)
            local_ids_R = recv_ids_R - local_ep_rank * experts_per_rank
            expert_outs_RD = torch.zeros_like(recv_x_RD)

            for local_e in range(experts_per_rank):
                mask_R = local_ids_R == local_e
                tokens_ND = recv_x_RD[mask_R]  # [N, D]
                h_NF = torch.relu(tokens_ND @ w1_EpDF[local_e])  # [N, F]
                expert_outs_RD[mask_R] = h_NF @ w2_EpFD[local_e]  # [N, D]

            # Combine back via all_to_all: [R, D] -> [T*K, D]
            combined_TmKD = all_to_all_single(
                expert_outs_RD, input_split_sizes, output_split_sizes, ep_group_name
            )

            # Unsort to restore original token order
            unsort_idx_TmK = torch.argsort(sort_idx_TmK)
            final_TmKD = combined_TmKD[unsort_idx_TmK]

            # Reshape and weight-sum across K experts per token
            final_TKD = final_TmKD.view(local_T, K, local_D)
            weights_TK1 = weights_TmK.view(local_T, K, 1)

            out_TD = (final_TKD * weights_TK1).sum(dim=1)

            return out_TD.view(local_B, local_L, local_D)

        # Forward
        out_dist_BdLD = moe_ep_forward(
            x_dist_BdLD, router_dist_DE, w1_dist_EpDF, w2_dist_EpFD
        )

        # Verify
        out_full_BLD = out_dist_BdLD.full_tensor()
        self.assertTrue(
            torch.allclose(out_ref_BLD, out_full_BLD, atol=1e-5),
            f"Output mismatch: max diff = {(out_ref_BLD - out_full_BLD).abs().max().item()}",
        )


if __name__ == "__main__":
    run_tests()
