# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Tests for MemoryShardedDTensor core class functionality.
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import (
    DTensor,
    distribute_tensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._memory_sharded import (
    BlockStorageShardingSpec,
    FlattenedStorageShardingSpec,
    MemoryShardedDTensor,
    ShardingBoundary,
    ShardParamInfo,
    TensorGroupShardingSpec,
    TensorGroupStorage,
)
from torch.distributed.memory_saving import (
    scatter_tensor_storage,
    scatter_tensor_group,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestBlockStorageShardingSpec(TestCase):
    """Unit tests for BlockStorageShardingSpec dataclass."""

    def test_block_storage_sharding_spec_creation(self):
        """Test that BlockStorageShardingSpec can be created with all fields."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 32]),
            orig_stride=(32, 1),
            shard_dims=(0,),
            mesh_dims=("dp",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )
        self.assertEqual(spec.orig_size, torch.Size([16, 32]))
        self.assertEqual(spec.orig_stride, (32, 1))
        self.assertEqual(spec.shard_dims, (0,))
        self.assertEqual(spec.mesh_dims, ("dp",))
        self.assertEqual(spec.padded_shard_sizes, (4,))
        self.assertEqual(spec.actual_shard_sizes, (4,))

    def test_block_storage_sharding_spec_uneven(self):
        """Test BlockStorageShardingSpec with uneven sharding (different actual vs padded)."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([13, 32]),
            orig_stride=(32, 1),
            shard_dims=(0,),
            mesh_dims=("dp",),
            padded_shard_sizes=(4,),  # ceiling(13/4) = 4
            actual_shard_sizes=(1,),  # last rank gets only 1 row
            mesh_dim_indices=(0,),
        )
        self.assertEqual(spec.padded_shard_sizes, (4,))
        self.assertEqual(spec.actual_shard_sizes, (1,))

    def test_block_storage_sharding_spec_multi_dim(self):
        """Test BlockStorageShardingSpec with multi-dimensional sharding."""
        spec = BlockStorageShardingSpec(
            orig_size=torch.Size([8, 4]),
            orig_stride=(4, 1),
            shard_dims=(0, 1),
            mesh_dims=("dp", "tp"),
            padded_shard_sizes=(2, 2),
            actual_shard_sizes=(2, 2),
            mesh_dim_indices=(0, 1),
        )
        self.assertEqual(spec.shard_dims, (0, 1))
        self.assertEqual(spec.mesh_dims, ("dp", "tp"))
        self.assertEqual(spec.padded_shard_sizes, (2, 2))
        self.assertEqual(spec.actual_shard_sizes, (2, 2))


class TestMemoryShardedDTensor(DTensorTestBase):
    """Distributed tests for MemoryShardedDTensor class."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_memory_sharded_dtensor_creation(self):
        """Verify MemoryShardedDTensor can be instantiated."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertIsInstance(msdt, DTensor)

    @with_comms
    def test_shape_returns_local(self):
        """Test that .shape returns local sharded shape."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.shape, torch.Size([4, 8]))

    @with_comms
    def test_full_shape_returns_original(self):
        """Test that .full_shape returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_size_with_dim(self):
        """Test that .size(dim) returns local size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.size(0), 4)
        self.assertEqual(msdt.size(1), 8)

    @with_comms
    def test_full_size_with_dim(self):
        """Test that .full_size(dim) returns original size."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.full_size(0), 16)
        self.assertEqual(msdt.full_size(1), 8)
        self.assertEqual(msdt.full_size(), torch.Size([16, 8]))

    @with_comms
    def test_local_returns_tensor(self):
        """Test that .local() returns torch.Tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        result = msdt.local()
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([4, 8]))

    @with_comms
    def test_ndim(self):
        """Test that .ndim is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.ndim, 2)

    @with_comms
    def test_dtype(self):
        """Test that .dtype is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, dtype=torch.float16)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.dtype, torch.float16)

    @with_comms
    def test_device(self):
        """Test that .device is correct."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.device.type, self.device_type)

    @with_comms
    def test_requires_grad(self):
        """Test that requires_grad is preserved."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type, requires_grad=True)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertTrue(msdt.requires_grad)

        # Test with requires_grad=False
        local_tensor_no_grad = torch.randn(4, 8, device=self.device_type)
        msdt_no_grad = MemoryShardedDTensor._create(
            local_tensor=local_tensor_no_grad,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )
        self.assertFalse(msdt_no_grad.requires_grad)

    @with_comms
    def test_storage_spec_property(self):
        """Test that storage_spec property returns the spec."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.storage_spec, storage_spec)
        self.assertEqual(msdt.storage_spec.shard_dims[0], 0)
        self.assertEqual(msdt.storage_spec.mesh_dims[0], "default")

    @with_comms
    def test_process_group_property(self):
        """Test that process_group property returns the PG."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        local_tensor = torch.randn(4, 8, device=self.device_type)

        storage_spec = BlockStorageShardingSpec(
            orig_size=torch.Size([16, 8]),
            orig_stride=(8, 1),
            shard_dims=(0,),
            mesh_dims=("default",),
            padded_shard_sizes=(4,),
            actual_shard_sizes=(4,),
            mesh_dim_indices=(0,),
        )

        pg = dist.distributed_c10d._get_default_group()
        msdt = MemoryShardedDTensor._create(
            local_tensor=local_tensor,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=pg,
            placements=(Replicate(),),
        )

        self.assertEqual(msdt.process_group, pg)

    @with_comms
    def test_tensor_ops_not_supported(self):
        """Test that tensor operations raise NotImplementedError.

        MemoryShardedDTensor does not support direct tensor operations.
        Users must use .local() or .unshard() instead.
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Arithmetic operations should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as cm:
            _ = msdt + msdt
        self.assertIn("does not support tensor operations", str(cm.exception))
        self.assertIn("local()", str(cm.exception))

        # Other arithmetic ops should also fail
        with self.assertRaises(NotImplementedError):
            _ = msdt * 2

        with self.assertRaises(NotImplementedError):
            _ = msdt - msdt

        with self.assertRaises(NotImplementedError):
            _ = -msdt

    @with_comms
    def test_correct_usage_pattern_local_and_back(self):
        """Test the correct usage pattern: local() -> operation -> from_local_shard().

        This demonstrates the proper way to perform operations with MemoryShardedDTensor:
        1. Get local shard with .local()
        2. Perform computation on local tensor
        3. Create new MemoryShardedDTensor from result using from_local_shard()
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Step 1: Get local shard
        local_shard = msdt.local()
        self.assertIsInstance(local_shard, torch.Tensor)
        self.assertNotIsInstance(local_shard, DTensor)

        # Step 2: Perform operation on local tensor
        result_local = local_shard * 2 + 1

        # Step 3: Create new MemoryShardedDTensor from result
        result_msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=result_local,
            full_shape=msdt.full_shape,
            device_mesh=device_mesh,
            shard_dim=0,
            mesh_dim=0,
        )

        # Verify result
        self.assertIsInstance(result_msdt, MemoryShardedDTensor)
        self.assertEqual(result_msdt.full_shape, msdt.full_shape)
        self.assertTrue(torch.equal(result_msdt.local(), result_local))

    @with_comms
    def test_autograd_requires_grad_preserved(self):
        """Test that requires_grad is preserved through sharding."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with requires_grad=True
        full_tensor = torch.randn(16, 8, device=self.device_type, requires_grad=True)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # requires_grad should be preserved
        self.assertTrue(msdt.requires_grad)
        self.assertTrue(msdt.local().requires_grad)

    @with_comms
    def test_autograd_backward_on_local(self):
        """Test that gradients flow correctly when computing on local shards.

        The pattern for autograd with MemoryShardedDTensor:
        1. Create MST from tensor with requires_grad
        2. Get local shard (maintains grad connection)
        3. Compute on local
        4. Backward on local result
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with requires_grad
        full_tensor = torch.randn(16, 8, device=self.device_type, requires_grad=True)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Get local shard - should maintain grad connection
        local_shard = msdt.local()
        self.assertTrue(local_shard.requires_grad)

        # Compute on local - use retain_grad to check gradients
        local_shard.retain_grad()
        output = (local_shard ** 2).sum()

        # Backward should work
        output.backward()

        # Local shard should have gradients
        self.assertIsNotNone(local_shard.grad)
        self.assertEqual(local_shard.grad.shape, local_shard.shape)

    @with_comms
    def test_autograd_unshard_and_compute(self):
        """Test gradients flow through unshard -> compute -> backward."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, requires_grad=True)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Unshard to get full DTensor
        unsharded = msdt.unshard()
        self.assertTrue(unsharded.requires_grad)
        self.assertIsInstance(unsharded, DTensor)

        # Retain grad on the local tensor since it's not a leaf
        unsharded_local = unsharded.to_local()
        unsharded_local.retain_grad()

        # Compute and backward
        output = (unsharded_local * 2).sum()
        output.backward()

        # The unsharded tensor should have gradients
        self.assertIsNotNone(unsharded_local.grad)
        # Grad should be 2 everywhere (from the *2 operation)
        expected_grad = torch.full_like(unsharded_local, 2.0)
        self.assertEqual(unsharded_local.grad, expected_grad)


class TestScatterStorage(DTensorTestBase):
    """Distributed tests for scatter_tensor_storage() factory function."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_scatter_tensor_storage_basic(self):
        """Test basic scatter_tensor_storage creates MemoryShardedDTensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a replicated DTensor
        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard storage along dimension 0
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))
        self.assertEqual(msdt.shape[0], 4)  # 16 / 4 ranks = 4 per rank
        self.assertEqual(msdt.shape[1], 8)

    @with_comms
    def test_scatter_tensor_storage_dim0(self):
        """Test sharding on dimension 0 (FSDP pattern)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(20, 10, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dims[0], 0)
        self.assertEqual(msdt.full_size(0), 20)
        # 20 / 4 = 5 per rank
        self.assertEqual(msdt.size(0), 5)

    @with_comms
    def test_scatter_tensor_storage_dim1(self):
        """Test sharding on dimension 1 (TP pattern)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(10, 20, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=1, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dims[0], 1)
        self.assertEqual(msdt.full_size(1), 20)
        # 20 / 4 = 5 per rank
        self.assertEqual(msdt.size(1), 5)

    @with_comms
    def test_scatter_tensor_storage_2d_mesh(self):
        """Test scatter_tensor_storage with 2D mesh using mesh_dim name."""
        # Create 2D mesh: 2x2
        device_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("dp", "tp")
        )

        full_tensor = torch.randn(8, 8, device=self.device_type)
        dtensor = distribute_tensor(
            full_tensor, device_mesh, [Replicate(), Replicate()]
        )

        # Shard on dp dimension (size 2)
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim="dp")

        self.assertEqual(msdt.storage_spec.mesh_dims[0], "dp")
        self.assertEqual(msdt.full_size(0), 8)
        # 8 / 2 (dp world size) = 4 per rank
        self.assertEqual(msdt.size(0), 4)

    @with_comms
    def test_scatter_tensor_storage_validation_dim(self):
        """Test error for invalid dim."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        with self.assertRaises(ValueError):
            scatter_tensor_storage(dtensor, dim=5, mesh_dim=0)  # Out of range

    @with_comms
    def test_scatter_tensor_storage_validation_mesh_dim(self):
        """Test error for invalid mesh_dim."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        with self.assertRaises(ValueError):
            scatter_tensor_storage(dtensor, dim=0, mesh_dim="invalid")  # Invalid name

    @with_comms
    def test_scatter_tensor_storage_negative_dim(self):
        """Test that negative dim is normalized correctly."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # dim=-1 should be equivalent to dim=1
        msdt = scatter_tensor_storage(dtensor, dim=-1, mesh_dim=0)

        self.assertEqual(msdt.storage_spec.shard_dims[0], 1)
        self.assertEqual(msdt.full_size(1), 8)

    @with_comms
    def test_uneven_sharding_dim0(self):
        """Test uneven sharding: 13 / 4 = uneven chunks."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows, 4 ranks: ceil(13/4) = 4 padded shard size
        # Ranks get: 4, 4, 4, 1 rows
        full_tensor = torch.randn(13, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.full_size(0), 13)
        self.assertEqual(msdt.storage_spec.padded_shard_sizes[0], 4)

        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 4)
        else:
            # Last rank gets only 1 row
            self.assertEqual(msdt.size(0), 1)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 1)

    @with_comms
    def test_uneven_sharding_data_integrity(self):
        """Test that data is correctly sharded with uneven sharding."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with known values
        full_tensor = (
            torch.arange(13 * 8, device=self.device_type).reshape(13, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        rank = dist.get_rank()
        local_data = msdt.local()

        # Verify each rank has the correct slice of data
        expected_start = rank * 4  # padded_shard_size = 4
        expected_end = min(expected_start + 4, 13)
        expected_data = full_tensor[expected_start:expected_end]

        self.assertEqual(local_data.shape, expected_data.shape)
        self.assertTrue(torch.equal(local_data, expected_data))

    @with_comms
    def test_scatter_tensor_storage_requires_replicated(self):
        """Test that scatter_tensor_storage raises error when placement is Shard."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a sharded DTensor (not replicated)
        full_tensor = torch.randn(16, 8, device=self.device_type)
        sharded_dtensor = distribute_tensor(full_tensor, device_mesh, [Shard(0)])

        # Should raise ValueError because placement is Shard, not Replicate
        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(sharded_dtensor, dim=0, mesh_dim=0)

        self.assertIn("Shard", str(ctx.exception))

    @with_comms
    def test_scatter_tensor_storage_rejects_partial(self):
        """Test that scatter_tensor_storage raises error when placement is Partial."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a Partial DTensor (pending reduction)
        full_tensor = torch.randn(16, 8, device=self.device_type)
        partial_dtensor = distribute_tensor(full_tensor, device_mesh, [Partial()])

        # Should raise ValueError because placement is Partial, not Replicate
        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(partial_dtensor, dim=0, mesh_dim=0)

        self.assertIn("Partial", str(ctx.exception))


class TestUnshard(DTensorTestBase):
    """Distributed tests for unshard() method."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_unshard_basic(self):
        """Test that unshard() returns DTensor with full data."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        self.assertIsInstance(unsharded, DTensor)
        self.assertEqual(unsharded.shape, torch.Size([16, 8]))

    @with_comms
    def test_unshard_shape(self):
        """Test that unsharded DTensor has original shape."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(20, 10, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Verify sharded shape
        self.assertEqual(msdt.size(0), 5)  # 20 / 4 = 5

        # Unshard and verify original shape
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([20, 10]))

    @with_comms
    def test_unshard_data_correctness(self):
        """Test that data matches original after unshard."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create tensor with known values
        full_tensor = (
            torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        # Verify data matches original
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_dim1(self):
        """Test unshard on dimension 1."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = (
            torch.arange(10 * 20, device=self.device_type).reshape(10, 20).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on dim 1
        msdt = scatter_tensor_storage(dtensor, dim=1, mesh_dim=0)

        self.assertEqual(msdt.size(1), 5)  # 20 / 4 = 5

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([10, 20]))

        # Verify data
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_preserves_requires_grad(self):
        """Test that requires_grad is preserved through unshard."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, requires_grad=True)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)
        self.assertTrue(msdt.requires_grad)

        unsharded = msdt.unshard()
        self.assertTrue(unsharded.requires_grad)

    @with_comms
    def test_unshard_uneven_sharding(self):
        """Test unshard with uneven sharding (13 / 4 = uneven)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows: ceil(13/4) = 4 per shard, ranks get 4, 4, 4, 1
        full_tensor = (
            torch.arange(13 * 8, device=self.device_type).reshape(13, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)
        unsharded = msdt.unshard()

        # Verify shape is original
        self.assertEqual(unsharded.shape, torch.Size([13, 8]))

        # Verify data matches original
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))

    @with_comms
    def test_unshard_3d_tensor(self):
        """Test unshard with 3D tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = (
            torch.arange(8 * 4 * 6, device=self.device_type).reshape(8, 4, 6).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on middle dimension
        msdt = scatter_tensor_storage(dtensor, dim=1, mesh_dim=0)

        # 4 / 4 = 1 per rank
        self.assertEqual(msdt.size(1), 1)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 4, 6]))

        # Verify data
        unsharded_local = unsharded.to_local()
        self.assertTrue(torch.equal(unsharded_local, full_tensor))


class TestEdgeCases(DTensorTestBase):
    """Edge case tests for MemoryShardedDTensor."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_1d_tensor(self):
        """Test scatter_tensor_storage works with 1D tensors."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.arange(16, device=self.device_type).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.ndim, 1)
        self.assertEqual(msdt.size(0), 4)  # 16 / 4 = 4
        self.assertEqual(msdt.full_size(0), 16)

        # Unshard and verify
        unsharded = msdt.unshard()
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_small_tensor_fewer_than_world_size(self):
        """Test tensor smaller than world_size (some ranks get no data)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Only 3 rows, 4 ranks - rank 3 gets nothing
        full_tensor = torch.arange(3 * 4, device=self.device_type).reshape(3, 4).float()
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 1)
        else:
            self.assertEqual(msdt.size(0), 0)

        # Unshard should still work
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([3, 4]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_various_dtypes_float16(self):
        """Test with float16 dtype."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.float16)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.dtype, torch.float16)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.dtype, torch.float16)
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_various_dtypes_bfloat16(self):
        """Test with bfloat16 dtype."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.bfloat16)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt.dtype, torch.bfloat16)

        unsharded = msdt.unshard()
        self.assertEqual(unsharded.dtype, torch.bfloat16)
        # Use allclose for bfloat16 due to precision
        self.assertTrue(torch.allclose(unsharded.to_local(), full_tensor))

    @with_comms
    def test_contiguous(self):
        """Test that sharded tensor local data is contiguous."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertTrue(msdt.local().is_contiguous())


class TestFSDPIntegration(DTensorTestBase):
    """Tests for FSDP integration methods."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_get_all_gather_input_basic(self):
        """Test get_all_gather_input returns 1D flattened tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Get all-gather input
        all_gather_input = msdt.get_all_gather_input()

        # Should be a plain tensor (not DTensor)
        self.assertIsInstance(all_gather_input, torch.Tensor)
        self.assertNotIsInstance(all_gather_input, DTensor)

        # Should be 1D
        self.assertEqual(all_gather_input.ndim, 1)

        # Should have correct size (4 rows * 8 cols = 32 elements per shard)
        self.assertEqual(all_gather_input.numel(), 4 * 8)

    @with_comms
    def test_get_all_gather_input_dtype_conversion(self):
        """Test get_all_gather_input with dtype conversion."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create float32 tensor
        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.float32)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Get all-gather input with float16 dtype
        all_gather_input = msdt.get_all_gather_input(dtype=torch.float16)

        self.assertEqual(all_gather_input.dtype, torch.float16)

    @with_comms
    def test_get_all_gather_input_preserves_dtype(self):
        """Test get_all_gather_input preserves dtype when None."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type, dtype=torch.bfloat16)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        all_gather_input = msdt.get_all_gather_input()

        self.assertEqual(all_gather_input.dtype, torch.bfloat16)

    @with_comms
    def test_get_all_gather_input_uneven_sharding(self):
        """Test get_all_gather_input with uneven sharding (pads correctly)."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 13 rows, 4 ranks: ceil(13/4) = 4 padded shard size
        # Ranks get: 4, 4, 4, 1 rows
        full_tensor = torch.randn(13, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        all_gather_input = msdt.get_all_gather_input()

        rank = dist.get_rank()
        # All ranks should return padded size (4 * 8 = 32 elements)
        # to ensure all-gather works correctly
        self.assertEqual(all_gather_input.numel(), 4 * 8)

        if rank == 3:
            # Last rank only has 1 actual row, rest is padding
            self.assertEqual(msdt.size(0), 1)

    @with_comms
    def test_from_local_shard_basic(self):
        """Test from_local_shard creates correct MemoryShardedDTensor."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        # Create a local shard as FSDP would
        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))
        self.assertEqual(msdt.storage_spec.shard_dims[0], 0)
        self.assertEqual(msdt.storage_spec.mesh_dims[0], "dp")

    @with_comms
    def test_from_local_shard_with_mesh_dim_index(self):
        """Test from_local_shard with mesh_dim as index."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim=0,  # Use index instead of name
        )

        self.assertIsInstance(msdt, MemoryShardedDTensor)
        self.assertEqual(msdt.full_shape, torch.Size([16, 8]))

    @with_comms
    def test_from_local_shard_uneven(self):
        """Test from_local_shard with uneven sharding (last rank smaller)."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        rank = dist.get_rank()
        # Simulate FSDP's uneven sharding: 13 / 4 = 4, 4, 4, 1
        if rank < 3:
            local_shard = torch.randn(4, 8, device=self.device_type)
        else:
            local_shard = torch.randn(1, 8, device=self.device_type)

        full_shape = torch.Size([13, 8])

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        self.assertEqual(msdt.full_shape, torch.Size([13, 8]))
        self.assertEqual(msdt.storage_spec.padded_shard_sizes[0], 4)

        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 4)
        else:
            self.assertEqual(msdt.size(0), 1)
            self.assertEqual(msdt.storage_spec.actual_shard_sizes[0], 1)

    @with_comms
    def test_from_local_shard_requires_grad(self):
        """Test from_local_shard respects requires_grad."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        # With requires_grad=True
        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
            requires_grad=True,
        )
        self.assertTrue(msdt.requires_grad)

        # With requires_grad=False (default) - create fresh tensor
        local_shard_no_grad = torch.randn(4, 8, device=self.device_type)
        msdt_no_grad = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard_no_grad,
            full_shape=full_shape,
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
            requires_grad=False,
        )
        self.assertFalse(msdt_no_grad.requires_grad)

    @with_comms
    def test_from_local_shard_roundtrip(self):
        """Test from_local_shard followed by unshard recovers correct data."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        # Create known full tensor and manually compute shards
        full_tensor = (
            torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        )

        rank = dist.get_rank()
        start_idx = rank * 4
        end_idx = start_idx + 4
        local_shard = full_tensor[start_idx:end_idx].contiguous()

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=torch.Size([16, 8]),
            shard_dim=0,
            device_mesh=device_mesh,
            mesh_dim="dp",
        )

        # Unshard and verify data matches original
        unsharded = msdt.unshard()
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_from_local_shard_invalid_mesh_dim_name(self):
        """Test from_local_shard raises error for invalid mesh_dim name."""
        device_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        with self.assertRaises(ValueError):
            MemoryShardedDTensor.from_local_shard(
                local_shard=local_shard,
                full_shape=full_shape,
                shard_dim=0,
                device_mesh=device_mesh,
                mesh_dim="invalid_name",
            )

    @with_comms
    def test_from_local_shard_invalid_mesh_dim_index(self):
        """Test from_local_shard raises error for invalid mesh_dim index."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        local_shard = torch.randn(4, 8, device=self.device_type)
        full_shape = torch.Size([16, 8])

        with self.assertRaises(ValueError):
            MemoryShardedDTensor.from_local_shard(
                local_shard=local_shard,
                full_shape=full_shape,
                shard_dim=0,
                device_mesh=device_mesh,
                mesh_dim=5,  # Out of range
            )


class TestIntegration(DTensorTestBase):
    """Integration tests for MemoryShardedDTensor."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_roundtrip_dim0(self):
        """Test scatter_tensor_storage -> unshard roundtrip on dim 0."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = (
            torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)
        self.assertEqual(msdt.shape, torch.Size([4, 8]))

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([16, 8]))

        # Verify data matches original
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_roundtrip_dim1(self):
        """Test scatter_tensor_storage -> unshard roundtrip on dim 1."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = (
            torch.arange(8 * 16, device=self.device_type).reshape(8, 16).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        # Shard on dim 1
        msdt = scatter_tensor_storage(dtensor, dim=1, mesh_dim=0)
        self.assertEqual(msdt.shape, torch.Size([8, 4]))  # 16 / 4 = 4

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 16]))

        # Verify data matches original
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_multiple_unshard(self):
        """Test that unshard can be called multiple times."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = (
            torch.arange(16 * 8, device=self.device_type).reshape(16, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Unshard multiple times
        unsharded1 = msdt.unshard()
        unsharded2 = msdt.unshard()

        # Both should have correct data
        self.assertTrue(torch.equal(unsharded1.to_local(), full_tensor))
        self.assertTrue(torch.equal(unsharded2.to_local(), full_tensor))

    @with_comms
    def test_different_mesh_dim_names(self):
        """Test with various mesh dimension names."""
        device_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("data_parallel", "model_parallel")
        )

        full_tensor = torch.arange(8 * 8, device=self.device_type).reshape(8, 8).float()
        dtensor = distribute_tensor(
            full_tensor, device_mesh, [Replicate(), Replicate()]
        )

        # Shard on "data_parallel" dimension
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim="data_parallel")

        self.assertEqual(msdt.storage_spec.mesh_dims[0], "data_parallel")
        self.assertEqual(msdt.size(0), 4)  # 8 / 2 = 4

        # Unshard
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([8, 8]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))

    @with_comms
    def test_roundtrip_uneven(self):
        """Test roundtrip with uneven sharding."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 15 rows, 4 ranks: ceil(15/4) = 4 per shard
        # Ranks get: 4, 4, 4, 3
        full_tensor = (
            torch.arange(15 * 8, device=self.device_type).reshape(15, 8).float()
        )
        dtensor = distribute_tensor(full_tensor, device_mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        # Verify uneven distribution
        rank = dist.get_rank()
        if rank < 3:
            self.assertEqual(msdt.size(0), 4)
        else:
            self.assertEqual(msdt.size(0), 3)

        # Unshard and verify
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.shape, torch.Size([15, 8]))
        self.assertTrue(torch.equal(unsharded.to_local(), full_tensor))


class TestShardParamInfo(TestCase):
    """Unit tests for ShardParamInfo dataclass."""

    def test_shard_param_info_in_shard(self):
        """Test ShardParamInfo when param is in shard."""
        spi = ShardParamInfo(
            in_shard=True,
            offset_in_shard=10,
            numel_in_shard=20,
            intra_param_start=5,
            intra_param_end=25,
        )
        self.assertTrue(spi.in_shard)
        self.assertEqual(spi.offset_in_shard, 10)
        self.assertEqual(spi.numel_in_shard, 20)
        self.assertEqual(spi.intra_param_start, 5)
        self.assertEqual(spi.intra_param_end, 25)

    def test_shard_param_info_not_in_shard(self):
        """Test ShardParamInfo when param is not in shard."""
        spi = ShardParamInfo(in_shard=False)
        self.assertFalse(spi.in_shard)
        self.assertIsNone(spi.offset_in_shard)
        self.assertIsNone(spi.numel_in_shard)


class TestFlattenedStorageShardingSpec(TestCase):
    """Unit tests for FlattenedStorageShardingSpec dataclass."""

    def test_flattened_spec_creation(self):
        """Test FlattenedStorageShardingSpec can be created."""
        spec = FlattenedStorageShardingSpec(
            param_shapes=(torch.Size([8, 4]), torch.Size([8])),
            param_strides=((4, 1), (1,)),
            param_numels=(32, 8),
            total_numel=40,
            padded_total_numel=40,
            mesh_dim="dp",
            local_offset=0,
            local_numel=10,
            shard_param_infos=(
                ShardParamInfo(in_shard=True, offset_in_shard=0, numel_in_shard=10, intra_param_start=0, intra_param_end=10),
                ShardParamInfo(in_shard=False),
            ),
            param_index=0,
        )
        self.assertEqual(len(spec.param_shapes), 2)
        self.assertEqual(spec.total_numel, 40)
        self.assertEqual(spec.param_index, 0)


class TestTensorGroupStorage(DTensorTestBase):
    """Tests for TensorGroupStorage with boundary=TENSOR (whole tensor sharding)."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tensor_mode_basic(self):
        """Test basic tensor mode sharding."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 4 tensors, 4 ranks: each rank owns exactly 1 tensor
        tensors = [
            torch.randn(8, 4, device=self.device_type),
            torch.randn(10, 6, device=self.device_type),
            torch.randn(4, device=self.device_type),
            torch.randn(16, 16, device=self.device_type),
        ]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        self.assertEqual(len(sharded), 4)

        rank = dist.get_rank()
        for i, s in enumerate(sharded):
            self.assertTrue(s.is_tensor_group_mode())
            self.assertEqual(s.storage_spec.param_index, i)
            if s.storage_spec.owns_tensor:
                self.assertEqual(s.storage_spec.param_to_rank[i], rank)

        # Unshard all
        unsharded = group.unshard_all()
        for orig, unsh in zip(tensors, unsharded):
            self.assertTrue(torch.equal(unsh.to_local(), orig))

    @with_comms
    def test_tensor_mode_more_tensors_than_ranks(self):
        """Test tensor mode with more tensors than ranks."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 7 tensors, 4 ranks: [2, 2, 2, 1] distribution
        tensors = [torch.randn(i + 1, device=self.device_type) for i in range(7)]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        self.assertEqual(len(sharded), 7)

        # Verify distribution is contiguous
        rank = dist.get_rank()
        owned_indices = []
        for i, s in enumerate(sharded):
            if s.storage_spec.owns_tensor:
                owned_indices.append(i)
                self.assertEqual(s.storage_spec.param_to_rank[i], rank)

        # Check contiguity
        if len(owned_indices) > 0:
            for j in range(len(owned_indices) - 1):
                self.assertEqual(owned_indices[j] + 1, owned_indices[j + 1])

        # Unshard all
        unsharded = group.unshard_all()
        for orig, unsh in zip(tensors, unsharded):
            self.assertTrue(torch.equal(unsh.to_local(), orig))

    @with_comms
    def test_tensor_mode_fewer_tensors_than_ranks(self):
        """Test tensor mode with fewer tensors than ranks."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 2 tensors, 4 ranks: ranks 0,1 own one tensor each, ranks 2,3 own nothing
        tensors = [
            torch.randn(8, 4, device=self.device_type),
            torch.randn(10, 6, device=self.device_type),
        ]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        self.assertEqual(len(sharded), 2)

        rank = dist.get_rank()
        if rank == 0:
            self.assertTrue(sharded[0].storage_spec.owns_tensor)
            self.assertFalse(sharded[1].storage_spec.owns_tensor)
        elif rank == 1:
            self.assertFalse(sharded[0].storage_spec.owns_tensor)
            self.assertTrue(sharded[1].storage_spec.owns_tensor)
        else:
            self.assertFalse(sharded[0].storage_spec.owns_tensor)
            self.assertFalse(sharded[1].storage_spec.owns_tensor)

        # Unshard all
        unsharded = group.unshard_all()
        for orig, unsh in zip(tensors, unsharded):
            self.assertTrue(torch.equal(unsh.to_local(), orig))

    @with_comms
    def test_tensor_mode_unshard_individual(self):
        """Test unsharding individual tensors in tensor mode."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensors = [
            torch.randn(8, 4, device=self.device_type),
            torch.randn(10, 6, device=self.device_type),
            torch.randn(4, device=self.device_type),
            torch.randn(16, 16, device=self.device_type),
        ]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        # Unshard each tensor individually
        for i, (s, orig) in enumerate(zip(sharded, tensors)):
            unsharded = s.unshard()
            self.assertTrue(
                torch.equal(unsharded.to_local(), orig),
                f"Mismatch at tensor {i}",
            )

    @with_comms
    def test_scatter_tensor_group_convenience(self):
        """Test the scatter_tensor_group convenience function."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensors = [torch.randn(i + 5, device=self.device_type) for i in range(4)]

        # Tensor mode
        sharded = scatter_tensor_group(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        self.assertEqual(len(sharded), 4)
        for s in sharded:
            self.assertTrue(s.is_tensor_group_mode())

        # Element mode (default)
        sharded_elem = scatter_tensor_group(tensors, mesh, 0)  # default boundary=ELEMENT
        self.assertEqual(len(sharded_elem), 4)
        for s in sharded_elem:
            self.assertTrue(s.is_flattened_mode())

    @with_comms
    def test_tensor_group_spec_properties(self):
        """Test TensorGroupShardingSpec properties."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensors = [
            torch.randn(8, 4, device=self.device_type),
            torch.randn(10, 6, device=self.device_type),
        ]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        spec = sharded[0].storage_spec
        self.assertEqual(spec.total_params, 2)
        self.assertEqual(spec.param_shapes[0], torch.Size([8, 4]))
        self.assertEqual(spec.param_shapes[1], torch.Size([10, 6]))
        self.assertEqual(spec.param_numels[0], 32)
        self.assertEqual(spec.param_numels[1], 60)

        # Check assignment
        self.assertEqual(spec.param_to_rank[0], 0)
        self.assertEqual(spec.param_to_rank[1], 1)

    @with_comms
    def test_tensor_mode_repr(self):
        """Test __repr__ for tensor mode."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensors = [torch.randn(8, 4, device=self.device_type)]

        group = TensorGroupStorage(tensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        repr_str = repr(sharded[0])
        self.assertIn("tensor_group_mode=True", repr_str)
        self.assertIn("owns_tensor=", repr_str)


class TestDTensorInput(DTensorTestBase):
    """Tests for DTensor input support in TensorGroupStorage and scatter_tensor_group."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_element_mode_with_dtensor_input(self):
        """Test element mode sharding with DTensor inputs."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create DTensors with Replicate placement
        t1 = torch.randn(8, 4, device=self.device_type)
        t2 = torch.randn(10, 6, device=self.device_type)
        dtensor1 = distribute_tensor(t1, mesh, [Replicate()])
        dtensor2 = distribute_tensor(t2, mesh, [Replicate()])

        # Shard using element mode
        group = TensorGroupStorage([dtensor1, dtensor2], mesh, 0, boundary=ShardingBoundary.ELEMENT)
        sharded = group.shard()

        self.assertEqual(len(sharded), 2)
        for s in sharded:
            self.assertTrue(s.is_flattened_mode())

        # Unshard and verify
        unsharded = group.unshard_all()
        self.assertTrue(torch.equal(unsharded[0].to_local(), t1))
        self.assertTrue(torch.equal(unsharded[1].to_local(), t2))

    @with_comms
    def test_tensor_mode_with_dtensor_input(self):
        """Test tensor mode sharding with DTensor inputs."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create DTensors with Replicate placement
        tensors = [
            torch.randn(8, 4, device=self.device_type),
            torch.randn(10, 6, device=self.device_type),
            torch.randn(4, device=self.device_type),
            torch.randn(16, 16, device=self.device_type),
        ]
        dtensors = [distribute_tensor(t, mesh, [Replicate()]) for t in tensors]

        # Shard using tensor mode
        group = TensorGroupStorage(dtensors, mesh, 0, boundary=ShardingBoundary.TENSOR)
        sharded = group.shard()

        self.assertEqual(len(sharded), 4)
        for s in sharded:
            self.assertTrue(s.is_tensor_group_mode())

        # Unshard and verify
        unsharded = group.unshard_all()
        for orig, unsh in zip(tensors, unsharded):
            self.assertTrue(torch.equal(unsh.to_local(), orig))

    @with_comms
    def test_mixed_tensor_and_dtensor_input(self):
        """Test mixing plain tensors and DTensors as input."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Mix of plain tensor and DTensor
        plain_tensor = torch.randn(8, 4, device=self.device_type)
        dtensor_base = torch.randn(10, 6, device=self.device_type)
        dtensor = distribute_tensor(dtensor_base, mesh, [Replicate()])

        # Shard using element mode
        sharded = scatter_tensor_group([plain_tensor, dtensor], mesh, 0, boundary=ShardingBoundary.ELEMENT)

        self.assertEqual(len(sharded), 2)

        # Unshard using individual unshard
        unsharded0 = sharded[0].unshard()
        unsharded1 = sharded[1].unshard()

        self.assertTrue(torch.equal(unsharded0.to_local(), plain_tensor))
        self.assertTrue(torch.equal(unsharded1.to_local(), dtensor_base))

    @with_comms
    def test_2d_mesh_dtensor_input(self):
        """Test TP+FSDP pattern: DTensor sharded on tp, storage sharded on dp."""
        # 2D mesh: 2 dp x 2 tp
        mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("dp", "tp")
        )

        # Create DTensor that is Shard on tp, Replicate on dp
        # This is the TP+FSDP pattern - TP shards the weight columns
        full_tensor = torch.randn(8, 4, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate(), Shard(1)])

        # Now shard storage on dp dimension (Replicate on dp, so valid)
        sharded = scatter_tensor_group([dtensor], mesh, "dp", boundary=ShardingBoundary.ELEMENT)

        self.assertEqual(len(sharded), 1)
        self.assertTrue(sharded[0].is_flattened_mode())

        # The local tensor should be the TP-sharded portion
        # Shape: [8, 2] (4 cols / 2 tp ranks)
        # After dp storage sharding, this is further split

    @with_comms
    def test_dtensor_wrong_mesh_raises(self):
        """Test that DTensor with different mesh raises ValueError."""
        # Create a 1D mesh for the DTensor
        mesh1 = init_device_mesh(self.device_type, (self.world_size,))

        # Create a 2D mesh for the storage sharding - this is different
        mesh2 = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("dp", "tp")
        )

        # Create DTensor with mesh1
        t = torch.randn(8, 4, device=self.device_type)
        dtensor = distribute_tensor(t, mesh1, [Replicate()])

        # Try to shard with mesh2 - should raise since meshes are different
        with self.assertRaises(ValueError) as ctx:
            TensorGroupStorage([dtensor], mesh2, "dp", boundary=ShardingBoundary.ELEMENT)

        self.assertIn("does not match", str(ctx.exception))

    @with_comms
    def test_dtensor_shard_on_target_raises(self):
        """Test that DTensor with Shard on target mesh_dim raises ValueError."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create DTensor with Shard placement on dim 0
        full_tensor = torch.randn(16, 4, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Shard(0)])

        # Try to shard storage on the same dim - should raise
        with self.assertRaises(ValueError) as ctx:
            TensorGroupStorage([dtensor], mesh, 0, boundary=ShardingBoundary.ELEMENT)

        self.assertIn("already has Shard placement", str(ctx.exception))

    @with_comms
    def test_dtensor_partial_on_target_raises(self):
        """Test that DTensor with Partial on target mesh_dim raises ValueError."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # Create a DTensor with Partial placement
        # We need to create this manually since distribute_tensor doesn't create Partial
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        local_tensor = torch.randn(8, 4, device=self.device_type)
        tensor_meta = TensorMeta(
            shape=local_tensor.shape,
            stride=local_tensor.stride(),
            dtype=local_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=mesh,
            placements=(Partial(),),
            tensor_meta=tensor_meta,
        )
        dtensor = DTensor(
            local_tensor,
            dtensor_spec,
            requires_grad=False,
        )

        # Try to shard storage - should raise
        with self.assertRaises(ValueError) as ctx:
            TensorGroupStorage([dtensor], mesh, 0, boundary=ShardingBoundary.ELEMENT)

        self.assertIn("Partial placement", str(ctx.exception))


class TestWeightedSharding(DTensorTestBase):
    """Tests for weighted (uneven) sharding functionality."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_weighted_sharding_basic(self):
        """Test basic weighted sharding with scatter_tensor_storage."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        # Create a tensor with size divisible by sum of weights (1+2+1+1=5)
        # 20 elements / 5 weights = 4 elements per weight unit
        full_tensor = torch.randn(20, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        # Apply weighted sharding: rank 1 gets 2x elements
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 2, 1, 1])

        # Verify local sizes: [4, 8, 4, 4]
        expected_sizes = [4, 8, 4, 4]
        self.assertEqual(msdt.shape[0], expected_sizes[rank])
        self.assertEqual(msdt.shape[1], 8)

        # Verify full_shape is correct
        self.assertEqual(msdt.full_shape, torch.Size([20, 8]))

        # Verify unshard recovers original tensor
        unsharded = msdt.unshard()
        self.assertEqual(unsharded.to_local().shape, full_tensor.shape)
        self.assertTrue(torch.allclose(unsharded.to_local(), full_tensor))

    @with_comms
    def test_weighted_sharding_not_divisible_error(self):
        """Test that dimension not divisible by weight sum raises error."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        # 17 is not divisible by 5 (1+2+1+1)
        full_tensor = torch.randn(17, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 2, 1, 1])

        self.assertIn("does not produce an integer shard size", str(ctx.exception))

    @with_comms
    def test_weighted_sharding_zero_weight_error(self):
        """Test that zero weight raises error."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 0, 1, 1])

        self.assertIn("positive", str(ctx.exception))

    @with_comms
    def test_weighted_sharding_negative_weight_error(self):
        """Test that negative weight raises error."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, -1, 1, 1])

        self.assertIn("positive", str(ctx.exception))

    @with_comms
    def test_weighted_sharding_wrong_length_error(self):
        """Test that wrong weights length raises error."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        with self.assertRaises(ValueError) as ctx:
            scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 2, 1])  # 3 != 4

        self.assertIn("must match", str(ctx.exception))

    @with_comms
    def test_weighted_sharding_float_weights(self):
        """Test that float weights work correctly."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        full_tensor = torch.randn(20, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        # Float weights [0.5, 1.0, 0.5, 0.5] is equivalent to [1, 2, 1, 1]
        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[0.5, 1.0, 0.5, 0.5])

        expected_sizes = [4, 8, 4, 4]
        self.assertEqual(msdt.shape[0], expected_sizes[rank])

    @with_comms
    def test_weighted_sharding_extreme_imbalance(self):
        """Test weighted sharding with extreme imbalance."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        # weights [1, 10, 1, 1] sum to 13, use 26 elements
        full_tensor = torch.randn(26, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        msdt = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 10, 1, 1])

        # 26 / 13 = 2 elements per weight unit
        expected_sizes = [2, 20, 2, 2]
        self.assertEqual(msdt.shape[0], expected_sizes[rank])

        # Verify unshard works
        unsharded = msdt.unshard()
        self.assertTrue(torch.allclose(unsharded.to_local(), full_tensor))

    @with_comms
    def test_weighted_sharding_uniform_weights(self):
        """Test that uniform weights behave same as even sharding."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        full_tensor = torch.randn(16, 8, device=self.device_type)
        dtensor = distribute_tensor(full_tensor, mesh, [Replicate()])

        # weights=[1,1,1,1] should be same as no weights
        msdt_weighted = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0, weights=[1, 1, 1, 1])
        msdt_even = scatter_tensor_storage(dtensor, dim=0, mesh_dim=0)

        self.assertEqual(msdt_weighted.shape, msdt_even.shape)

    @with_comms
    def test_from_local_shard_with_weights(self):
        """Test from_local_shard with weights."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        # Pre-computed local shards based on weights [1, 2, 1, 1]
        # Full shape (20, 8), sizes per rank: [4, 8, 4, 4]
        local_sizes = [4, 8, 4, 4]
        local_shard = torch.randn(local_sizes[rank], 8, device=self.device_type)

        msdt = MemoryShardedDTensor.from_local_shard(
            local_shard=local_shard,
            full_shape=torch.Size([20, 8]),
            shard_dim=0,
            device_mesh=mesh,
            mesh_dim=0,
            weights=[1, 2, 1, 1],
        )

        self.assertEqual(msdt.full_shape, torch.Size([20, 8]))
        self.assertEqual(msdt.shape[0], local_sizes[rank])

    @with_comms
    def test_tensor_group_element_mode_weighted(self):
        """Test TensorGroupStorage element mode with weights."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        # Create tensors with total numel divisible by weight sum
        # weights [1, 2, 1, 1] sum to 5
        # Total numel = 40 (divisible by 5) -> 8 elements per weight unit
        tensors = [
            torch.randn(10, device=self.device_type),
            torch.randn(20, device=self.device_type),
            torch.randn(10, device=self.device_type),
        ]

        group = TensorGroupStorage(
            tensors, mesh, mesh_dim=0, boundary=ShardingBoundary.ELEMENT, weights=[1, 2, 1, 1]
        )
        sharded = group.shard()

        self.assertEqual(len(sharded), 3)

        # Verify the flat buffer has correct size per rank
        expected_shard_sizes = [8, 16, 8, 8]  # 8 * [1, 2, 1, 1]
        flat_buffer = group.get_flat_buffer()
        self.assertEqual(flat_buffer.numel(), expected_shard_sizes[rank])

    @with_comms
    def test_tensor_group_tensor_mode_weighted(self):
        """Test TensorGroupStorage tensor mode with weights."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        rank = mesh.get_local_rank()

        # Create 10 tensors, distribute with weights [1, 2, 1, 1] sum=5
        # 10 / 5 = 2 tensors per weight unit
        tensors = [torch.randn(8, device=self.device_type) for _ in range(10)]

        group = TensorGroupStorage(
            tensors, mesh, mesh_dim=0, boundary=ShardingBoundary.TENSOR, weights=[1, 2, 1, 1]
        )
        sharded = group.shard()

        self.assertEqual(len(sharded), 10)

        # Count how many tensors this rank owns: [2, 4, 2, 2]
        owned_count = sum(1 for t in sharded if t._storage_spec.owns_tensor)
        expected_owned = [2, 4, 2, 2]
        self.assertEqual(owned_count, expected_owned[rank])


if __name__ == "__main__":
    run_tests()
