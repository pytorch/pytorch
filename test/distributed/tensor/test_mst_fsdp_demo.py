# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Demo/test file showing MST (MemoryShardedDTensor) supports various FSDP-like sharding patterns.

This file demonstrates that MST can flexibly support different sharding strategies:
1. FSDP v2: Per-tensor dim 0 sharding
2. BSDP: Per-tensor 2D block sharding
3. FSDP v1: Flatten + concat, element-wise sharding
4. Muon: Tensor-level distribution (each rank owns complete tensors)
5. veScale: FSDP v1 with weighted/uneven sharding

Additionally tests FSDP+TP composition where tensors are already TP-sharded.

Run with 8 GPUs:
    python test/distributed/tensor/test_mst_fsdp_demo.py
"""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor, init_device_mesh, Replicate, Shard
from torch.distributed.memory_saving import (
    scatter_tensor_storage,
    MemoryShardedDTensor,
    ShardingBoundary,
    TensorGroupStorage,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


# ============================================================================
# Use Case 1: FSDP v2 - Per-tensor dim 0 sharding
# ============================================================================
def shard_fsdpv2(
    dtensors: list[DTensor], mesh_dim: str = "dp"
) -> list[MemoryShardedDTensor]:
    """
    Shard each tensor on dim 0 across dp mesh (FSDP v2 style).

    Each tensor is independently sharded along its first dimension.
    Different tensors may have different shard sizes.
    """
    return [scatter_tensor_storage(dt, dim=0, mesh_dim=mesh_dim) for dt in dtensors]


def unshard_fsdpv2(sharded: list[MemoryShardedDTensor]) -> list[DTensor]:
    """Unshard all tensors (all-gather per tensor)."""
    return [msdt.unshard() for msdt in sharded]


# ============================================================================
# Use Case 2: BSDP - Per-tensor 2D block sharding
# ============================================================================
def shard_bsdp(dtensors: list[DTensor]) -> list[MemoryShardedDTensor]:
    """
    Shard each tensor in 2D blocks across dp_row × dp_col mesh (BSDP style).

    Each rank holds a rectangular block of each tensor.
    Useful for distributed optimizers like Shampoo.
    """
    return [
        scatter_tensor_storage(dt, dim=[0, 1], mesh_dim=["dp_row", "dp_col"])
        for dt in dtensors
    ]


def unshard_bsdp(sharded: list[MemoryShardedDTensor]) -> list[DTensor]:
    """Unshard all tensors (all-gather on both dims)."""
    return [msdt.unshard() for msdt in sharded]


# ============================================================================
# Use Case 3: FSDP v1 - Flatten + concat, element-wise sharding
# ============================================================================
def shard_fsdpv1(
    dtensors: list[DTensor], mesh: DeviceMesh, mesh_dim: str = "dp"
) -> list[MemoryShardedDTensor]:
    """
    Flatten all tensors, concat into one buffer, shard across dp mesh (FSDP v1 style).

    All tensors share a single flat buffer. A single tensor may span
    multiple ranks. This enables a single all-gather for all parameters.
    """
    group = TensorGroupStorage(dtensors, mesh, mesh_dim=mesh_dim, boundary=ShardingBoundary.ELEMENT)
    return group.shard()


def unshard_fsdpv1(sharded: list[MemoryShardedDTensor]) -> list[DTensor]:
    """Unshard all tensors (single all-gather on flat buffer)."""
    return [msdt.unshard() for msdt in sharded]


# ============================================================================
# Use Case 4: Muon - Tensor-level distribution
# ============================================================================
def shard_muon(
    dtensors: list[DTensor], mesh: DeviceMesh, mesh_dim: str = "dp"
) -> list[MemoryShardedDTensor]:
    """
    Each dp rank holds complete tensors (Muon optimizer style).

    Tensors are distributed round-robin or by assignment to ranks.
    No tensor is split across ranks.
    """
    group = TensorGroupStorage(dtensors, mesh, mesh_dim=mesh_dim, boundary=ShardingBoundary.TENSOR)
    return group.shard()


def unshard_muon(sharded: list[MemoryShardedDTensor]) -> list[DTensor]:
    """Unshard all tensors (broadcast from owner)."""
    return [msdt.unshard() for msdt in sharded]


# ============================================================================
# Use Case 5: veScale - FSDP v1 with weighted sharding
# ============================================================================
def shard_vescale(
    dtensors: list[DTensor],
    mesh: DeviceMesh,
    mesh_dim: str = "dp",
    weights: list[int] | None = None,
) -> list[MemoryShardedDTensor]:
    """
    FSDP v1 with weighted sharding (veScale style).

    Different ranks can hold different proportions of the data.
    Useful for heterogeneous GPU memory scenarios.
    """
    group = TensorGroupStorage(
        dtensors, mesh, mesh_dim=mesh_dim, boundary=ShardingBoundary.ELEMENT, weights=weights
    )
    return group.shard()


def unshard_vescale(sharded: list[MemoryShardedDTensor]) -> list[DTensor]:
    """Unshard all tensors."""
    return [msdt.unshard() for msdt in sharded]


# ============================================================================
# Test Class
# ============================================================================
class TestMSTFlexibleFSDP(DTensorTestBase):
    """Test MST supports various FSDP-like sharding patterns."""

    @property
    def world_size(self) -> int:
        return 8

    def _create_test_dtensors(
        self, mesh: DeviceMesh
    ) -> tuple[list[DTensor], list[torch.Tensor]]:
        """Create 4 test DTensors with different shapes."""
        torch.manual_seed(42)
        original_tensors = [
            torch.randn(16, 8, device=self.device_type),
            torch.randn(32, 4, device=self.device_type),
            torch.randn(8, 16, device=self.device_type),
            torch.randn(24, 12, device=self.device_type),
        ]
        # Create replicated DTensors
        placements = [Replicate()] * mesh.ndim
        dtensors = [distribute_tensor(t, mesh, placements) for t in original_tensors]
        return dtensors, original_tensors

    def _assert_tensors_equal(
        self, unsharded: list[DTensor], original: list[torch.Tensor]
    ):
        """Assert unsharded tensors match originals."""
        for i, (dt, orig) in enumerate(zip(unsharded, original)):
            self.assertEqual(dt.to_local(), orig, f"Tensor {i} mismatch after unshard")

    @with_comms
    def test_fsdpv2(self):
        """Test FSDP v2 style: per-tensor dim 0 sharding."""
        mesh = init_device_mesh(
            self.device_type, (8,), mesh_dim_names=("dp",)
        )
        dtensors, originals = self._create_test_dtensors(mesh)

        sharded = shard_fsdpv2(dtensors)

        # Verify local shapes: dim 0 is divided by 8
        # (16, 8) -> (2, 8), (32, 4) -> (4, 4), (8, 16) -> (1, 16), (24, 12) -> (3, 12)
        expected_shapes = [(2, 8), (4, 4), (1, 16), (3, 12)]
        for i, (s, expected) in enumerate(zip(sharded, expected_shapes)):
            self.assertEqual(s.shape, torch.Size(expected), f"Tensor {i} sharded shape mismatch")

        unsharded = unshard_fsdpv2(sharded)
        self._assert_tensors_equal(unsharded, originals)

    @with_comms
    def test_bsdp(self):
        """Test BSDP style: per-tensor 2D block sharding."""
        mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("dp_row", "dp_col")
        )
        dtensors, originals = self._create_test_dtensors(mesh)

        sharded = shard_bsdp(dtensors)

        # Verify local shapes: dim 0 divided by 4, dim 1 divided by 2
        # (16, 8) -> (4, 4), (32, 4) -> (8, 2), (8, 16) -> (2, 8), (24, 12) -> (6, 6)
        expected_shapes = [(4, 4), (8, 2), (2, 8), (6, 6)]
        for i, (s, expected) in enumerate(zip(sharded, expected_shapes)):
            self.assertEqual(s.shape, torch.Size(expected), f"Tensor {i} sharded shape mismatch")

        unsharded = unshard_bsdp(sharded)
        self._assert_tensors_equal(unsharded, originals)

    @with_comms
    def test_fsdpv1(self):
        """Test FSDP v1 style: flatten + concat, element-wise sharding."""
        mesh = init_device_mesh(
            self.device_type, (8,), mesh_dim_names=("dp",)
        )
        dtensors, originals = self._create_test_dtensors(mesh)

        sharded = shard_fsdpv1(dtensors, mesh)

        # Verify total local elements: 672 total / 8 ranks = 84 elements per rank
        # Each MST's local numel depends on how the flat buffer intersects with that tensor
        total_local_numel = sum(s.local().numel() for s in sharded)
        self.assertEqual(total_local_numel, 84, "Total local elements mismatch")

        unsharded = unshard_fsdpv1(sharded)
        self._assert_tensors_equal(unsharded, originals)

    @with_comms
    def test_muon(self):
        """Test Muon style: tensor-level distribution."""
        mesh = init_device_mesh(
            self.device_type, (8,), mesh_dim_names=("dp",)
        )
        dtensors, originals = self._create_test_dtensors(mesh)

        sharded = shard_muon(dtensors, mesh)

        # Verify: 4 tensors distributed across 8 ranks
        # Ranks 0-3 each own 1 tensor (original shape), ranks 4-7 own nothing (empty)
        rank = mesh.get_local_rank()
        for i, s in enumerate(sharded):
            if s.storage_spec.owns_tensor:
                # Owner has original shape
                self.assertEqual(s.shape, originals[i].shape, f"Tensor {i} owner shape mismatch")
            else:
                # Non-owner has empty shape
                self.assertEqual(s.local().numel(), 0, f"Tensor {i} non-owner should be empty")

        unsharded = unshard_muon(sharded)
        self._assert_tensors_equal(unsharded, originals)

    @with_comms
    def test_vescale(self):
        """Test veScale style: FSDP v1 with weighted sharding."""
        mesh = init_device_mesh(
            self.device_type, (8,), mesh_dim_names=("dp",)
        )
        # Adjust tensor sizes to be divisible by weight sum (1+2+1+1+1+1+1+1=9)
        # Total elements must be divisible by 9
        torch.manual_seed(42)
        original_tensors = [
            torch.randn(18, 9, device=self.device_type),   # 162 elements
            torch.randn(27, 6, device=self.device_type),   # 162 elements
            torch.randn(9, 18, device=self.device_type),   # 162 elements
            torch.randn(18, 18, device=self.device_type),  # 324 elements
        ]  # Total: 810, divisible by 9
        placements = [Replicate()]
        dtensors = [
            distribute_tensor(t, mesh, placements) for t in original_tensors
        ]

        weights = [1, 2, 1, 1, 1, 1, 1, 1]
        sharded = shard_vescale(dtensors, mesh, weights=weights)

        # Verify total local elements based on weights
        # 810 total / 9 weight sum = 90 elements per weight unit
        # Rank 1 gets 180 elements (weight=2), others get 90 (weight=1)
        rank = mesh.get_local_rank()
        total_local_numel = sum(s.local().numel() for s in sharded)
        expected_numel = 90 * weights[rank]
        self.assertEqual(total_local_numel, expected_numel, f"Rank {rank} local elements mismatch")

        unsharded = unshard_vescale(sharded)
        self._assert_tensors_equal(unsharded, original_tensors)

    @with_comms
    def test_fsdp_plus_tp(self):
        """Test FSDP+TP: tensors already TP-sharded, then FSDP storage sharded.

        This demonstrates the common pattern where:
        1. Tensor parallelism shards weights across TP ranks (e.g., row or column-wise)
        2. FSDP then shards the storage of each TP shard across DP ranks

        Mixed TP sharding patterns:
        - Shard(1): Column-wise sharding (like ColumnParallelLinear weight)
        - Shard(0): Row-wise sharding (like RowParallelLinear weight)
        """
        # 2D mesh: 4 DP ranks x 2 TP ranks = 8 total
        mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("dp", "tp")
        )

        # Create tensors - dimensions must be divisible by TP size (2)
        torch.manual_seed(42)
        original_tensors = [
            torch.randn(16, 8, device=self.device_type),   # Will use Shard(1) - column
            torch.randn(32, 4, device=self.device_type),   # Will use Shard(0) - row
            torch.randn(8, 16, device=self.device_type),   # Will use Shard(1) - column
            torch.randn(24, 12, device=self.device_type),  # Will use Shard(0) - row
        ]

        # Create DTensors with mixed TP sharding patterns
        # [Replicate(), Shard(dim)] = replicated on dp, sharded on tp
        tp_placements = [
            [Replicate(), Shard(1)],  # Column-wise (like ColumnParallelLinear)
            [Replicate(), Shard(0)],  # Row-wise (like RowParallelLinear)
            [Replicate(), Shard(1)],  # Column-wise
            [Replicate(), Shard(0)],  # Row-wise
        ]
        tp_sharded_dtensors = [
            distribute_tensor(t, mesh, p)
            for t, p in zip(original_tensors, tp_placements)
        ]

        # Now apply FSDP storage sharding on the DP dimension
        # Each TP shard gets further split across DP ranks
        fsdp_sharded = shard_fsdpv2(tp_sharded_dtensors, mesh_dim="dp")

        # Verify local shapes:
        # Tensor 0: (16, 8) -> TP Shard(1) (16, 4) -> FSDP (4, 4)
        # Tensor 1: (32, 4) -> TP Shard(0) (16, 4) -> FSDP (4, 4)
        # Tensor 2: (8, 16) -> TP Shard(1) (8, 8)  -> FSDP (2, 8)
        # Tensor 3: (24, 12) -> TP Shard(0) (12, 12) -> FSDP (3, 12)
        expected_shapes = [(4, 4), (4, 4), (2, 8), (3, 12)]
        for i, (s, expected) in enumerate(zip(fsdp_sharded, expected_shapes)):
            self.assertEqual(s.shape, torch.Size(expected), f"Tensor {i} sharded shape mismatch")

        # Unshard FSDP (all-gather across DP) to get back TP-sharded DTensors
        unsharded = unshard_fsdpv2(fsdp_sharded)

        # Verify: unsharded should match the original TP-sharded DTensors
        for i, (unsh, tp_dt) in enumerate(zip(unsharded, tp_sharded_dtensors)):
            self.assertEqual(
                unsh.to_local().shape,
                tp_dt.to_local().shape,
                f"Tensor {i} shape mismatch",
            )
            self.assertTrue(
                torch.equal(unsh.to_local(), tp_dt.to_local()),
                f"Tensor {i} data mismatch after unshard",
            )

    @with_comms
    def test_local_shard_element_wise_update(self):
        """Test element-wise updates on local shards.

        MST does not support direct tensor operations. To perform local updates:
        1. Extract local tensor with .local()
        2. Perform element-wise operations on the local tensor
        3. Create new MST from the modified local tensor using from_local_shard()

        This pattern is useful for optimizer state updates, gradient modifications,
        or any computation that can be done independently on each shard.
        """
        mesh = init_device_mesh(
            self.device_type, (8,), mesh_dim_names=("dp",)
        )

        # Create and shard a tensor
        torch.manual_seed(42)
        original = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(original, mesh, [Replicate()])
        sharded = scatter_tensor_storage(dtensor, dim=0, mesh_dim="dp")

        # Verify initial sharded shape
        self.assertEqual(sharded.shape, torch.Size([2, 8]))  # 16 / 8 = 2

        # Step 1: Extract local tensor
        local_tensor = sharded.local()
        self.assertIsInstance(local_tensor, torch.Tensor)
        self.assertEqual(local_tensor.shape, torch.Size([2, 8]))

        # Step 2: Perform element-wise operations
        # Example: scale by 2 and add 1 (simulating optimizer update)
        updated_local = local_tensor * 2 + 1

        # Step 3: Create new MST from modified local tensor
        updated_sharded = MemoryShardedDTensor.from_local_shard(
            local_shard=updated_local,
            full_shape=sharded.full_shape,
            shard_dim=0,
            device_mesh=mesh,
            mesh_dim="dp",
        )

        # Verify the updated MST
        self.assertEqual(updated_sharded.shape, torch.Size([2, 8]))
        self.assertEqual(updated_sharded.full_shape, torch.Size([16, 8]))
        self.assertTrue(torch.equal(updated_sharded.local(), updated_local))

        # Unshard and verify the transformation was applied correctly
        unsharded = updated_sharded.unshard()
        expected_full = original * 2 + 1
        self.assertTrue(
            torch.allclose(unsharded.to_local(), expected_full),
            "Element-wise update not applied correctly",
        )

    @with_comms
    def test_shampoo_local_matmul(self):
        """Test Shampoo-style optimizer: BSDP sharding with local matmul.

        Shampoo optimizer pattern (from "Shampoo: Preconditioned Stochastic
        Tensor Optimization" paper):

        For weight W of shape (m, n) and gradient G of same shape:
        1. Accumulate preconditioners over iterations:
           L_t = β * L_{t-1} + (1-β) * G @ G.T  (shape m×m)
           R_t = β * R_{t-1} + (1-β) * G.T @ G  (shape n×n)
        2. Compute preconditioned gradient:
           precond_G = L^{-1/4} @ G @ R^{-1/4}
        3. Update weight:
           W = W - lr * precond_G

        With BSDP sharding, each rank holds a block of W and G, and can
        compute its local preconditioners and updates independently.
        """
        mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("dp_row", "dp_col")
        )

        # Create weight matrix and shard in 2D blocks
        torch.manual_seed(42)
        weight = torch.randn(16, 8, device=self.device_type)
        weight_dtensor = distribute_tensor(weight, mesh, [Replicate(), Replicate()])
        weight_sharded = shard_bsdp([weight_dtensor])[0]

        # Simulate gradient computation - gradient has same shape as weight
        # In practice, this comes from backward pass
        torch.manual_seed(123)
        grad = torch.randn(16, 8, device=self.device_type)
        grad_dtensor = distribute_tensor(grad, mesh, [Replicate(), Replicate()])
        grad_sharded = shard_bsdp([grad_dtensor])[0]

        # Verify block shapes: (16/4, 8/2) = (4, 4)
        self.assertEqual(weight_sharded.shape, torch.Size([4, 4]))
        self.assertEqual(grad_sharded.shape, torch.Size([4, 4]))

        # Step 1: Extract local blocks
        local_weight = weight_sharded.local()
        local_grad = grad_sharded.local()

        # Step 2: Compute Shampoo preconditioners from gradient
        # Approximation: real Shampoo accumulates over iterations with momentum
        # Here we show single-step computation for demonstration
        L = local_grad @ local_grad.T  # Left preconditioner (4x4)
        R = local_grad.T @ local_grad  # Right preconditioner (4x4)

        # Verify preconditioner shapes
        self.assertEqual(L.shape, torch.Size([4, 4]))
        self.assertEqual(R.shape, torch.Size([4, 4]))

        # Step 3: Apply preconditioned update to gradient
        # Approximation: real Shampoo uses L^{-1/4} @ grad @ R^{-1/4}
        # Here we use simple matmul to demonstrate the pattern
        precond_grad = L @ local_grad @ R

        # Step 4: Update weight with preconditioned gradient
        lr = 0.01
        updated_weight = local_weight - lr * precond_grad

        # Verify the update shape matches local weight
        self.assertEqual(updated_weight.shape, local_weight.shape)

        # Unshard original weight and verify integrity
        unsharded = weight_sharded.unshard()
        self.assertEqual(unsharded.to_local().shape, weight.shape)
        self.assertTrue(torch.allclose(unsharded.to_local(), weight))


if __name__ == "__main__":
    run_tests()
