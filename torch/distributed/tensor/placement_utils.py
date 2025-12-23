# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.distributed.device_mesh import DeviceMesh


class PadType(Enum):
    """
    Type of padding operation based on collective direction and shard type.

    - OLD_*: Padding/unpadding for the source (existing) shard dimension
    - NEW_*: Padding/unpadding for the target (new) shard dimension
    - *_SHARD: Regular Shard placement
    - *_STRIDED: _StridedShard placement with split_factor
    """

    OLD_SHARD = "old_shard"
    NEW_SHARD = "new_shard"
    OLD_STRIDED = "old_strided"
    NEW_STRIDED = "new_strided"


@dataclass
class PaddingOp:
    """A paired padding/unpadding operation specification."""

    pad_type: PadType
    shard_dim: int
    dim_logical_size: int
    split_factor: int = 1  # Only used for strided shard types


@dataclass
class CollectivePaddingContext:
    """
    Context for managing padding/unpadding around collective operations.

    Padding operations are paired with their corresponding unpadding automatically.
    Unpadding is applied in reverse order (LIFO - last padded, first unpadded).

    This context simplifies the pad → collective → unpad pattern by:
    1. Automatically pairing each pad with its corresponding unpad
    2. Managing the order of operations (LIFO for unpadding)
    3. Providing a fluent API for easy composition

    Example usage::

        # Shard → Shard alltoall (pad old, unpad old+new)
        result = (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .pad_new_shard(new_shard_dim, current_logical_shape[new_shard_dim])
            .run(
                local_tensor,
                lambda t: shard_dim_alltoall(
                    t, self.dim, new_shard_dim, mesh, mesh_dim
                ),
            )
        )

        # Shard → Replicate all_gather (pad old, unpad old)
        result = (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .run(
                local_tensor,
                lambda t: funcol.all_gather_tensor(
                    t, gather_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        )
    """

    mesh: DeviceMesh
    mesh_dim: int
    _ops: list[PaddingOp] = field(default_factory=list)

    def __post_init__(self):
        self.num_chunks = self.mesh.size(mesh_dim=self.mesh_dim)
        coord = self.mesh.get_coordinate()
        self.current_rank = coord[self.mesh_dim] if coord else 0

    def pad_old_shard(
        self, shard_dim: int, dim_logical_size: int
    ) -> "CollectivePaddingContext":
        """
        Pad the source Shard dimension before a collective.

        Use this for the dimension that is currently sharded (source placement).
        Example: In Shard(0) → Replicate, dim 0 is the "old" shard dimension.
        """
        self._ops.append(PaddingOp(PadType.OLD_SHARD, shard_dim, dim_logical_size))
        return self

    def pad_new_shard(
        self, shard_dim: int, dim_logical_size: int
    ) -> "CollectivePaddingContext":
        """
        Pad the target Shard dimension before a collective.

        Use this for the dimension that will be sharded after the collective (target placement).
        Example: In Shard(0) → Shard(1), dim 1 is the "new" shard dimension.
        """
        self._ops.append(PaddingOp(PadType.NEW_SHARD, shard_dim, dim_logical_size))
        return self

    def pad_old_strided(
        self, shard_dim: int, dim_logical_size: int, split_factor: int
    ) -> "CollectivePaddingContext":
        """
        Pad the source _StridedShard dimension before a collective.

        Use this for the dimension that is currently sharded with _StridedShard.
        Example: In _StridedShard(0, sf=2) → Shard(1), dim 0 is the "old" strided shard dimension.
        """
        self._ops.append(
            PaddingOp(PadType.OLD_STRIDED, shard_dim, dim_logical_size, split_factor)
        )
        return self

    def pad_new_strided(
        self, shard_dim: int, dim_logical_size: int, split_factor: int
    ) -> "CollectivePaddingContext":
        """
        Pad the target _StridedShard dimension before a collective.

        Use this for the dimension that will be sharded with _StridedShard after the collective.
        Example: In Shard(0) → _StridedShard(1, sf=2), dim 1 is the "new" strided shard dimension.
        """
        self._ops.append(
            PaddingOp(PadType.NEW_STRIDED, shard_dim, dim_logical_size, split_factor)
        )
        return self

    def _apply_pad(self, tensor: torch.Tensor, op: PaddingOp) -> torch.Tensor:
        """Apply a single padding operation."""
        # Import here to avoid circular imports
        from torch.distributed.tensor.placement_types import _StridedShard, Shard

        if op.pad_type == PadType.OLD_SHARD:
            return Shard._pad_for_old_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.NEW_SHARD:
            return Shard._pad_for_new_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.OLD_STRIDED:
            return _StridedShard._pad_for_old_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        elif op.pad_type == PadType.NEW_STRIDED:
            return _StridedShard._pad_for_new_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        raise ValueError(f"Unknown pad type: {op.pad_type}")

    def _apply_unpad(self, tensor: torch.Tensor, op: PaddingOp) -> torch.Tensor:
        """Apply a single unpadding operation (paired with the corresponding pad)."""
        # Import here to avoid circular imports
        from torch.distributed.tensor.placement_types import _StridedShard, Shard

        if op.pad_type == PadType.OLD_SHARD:
            return Shard._unpad_for_old_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.NEW_SHARD:
            return Shard._unpad_for_new_shard_dim(
                tensor,
                op.shard_dim,
                op.dim_logical_size,
                self.num_chunks,
                self.current_rank,
            )
        elif op.pad_type == PadType.OLD_STRIDED:
            return _StridedShard._unpad_for_old_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        elif op.pad_type == PadType.NEW_STRIDED:
            return _StridedShard._unpad_for_new_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
                self.current_rank,
            )
        raise ValueError(f"Unknown pad type: {op.pad_type}")

    def run(
        self,
        local_tensor: torch.Tensor,
        collective_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute collective with automatic padding before and unpadding after.

        Args:
            local_tensor: The local tensor to process.
            collective_fn: The collective operation to run (e.g., alltoall, all_gather).

        Returns:
            The result tensor after the collective and unpadding.

        Note:
            Padding is applied in the order operations were added.
            Unpadding is applied in reverse order (LIFO - last padded, first unpadded).
        """
        # Apply all padding operations in order
        for op in self._ops:
            local_tensor = self._apply_pad(local_tensor, op)

        # Run collective
        result = collective_fn(local_tensor)

        # Apply all unpadding operations in reverse order
        for op in reversed(self._ops):
            result = self._apply_unpad(result, op)

        return result
