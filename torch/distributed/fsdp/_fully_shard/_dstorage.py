# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.hooks import RemovableHandle

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.autograd import Variable
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Placement
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = ["DStorage", "fully_shard_flat", "get_dstorage", "Owned"]


# Module attribute name for storing DStorage
_DSTORAGE_ATTR = "_dstorage"


def get_dstorage(module: nn.Module) -> DStorage | None:
    """Get the DStorage associated with a module, if any."""
    return getattr(module, _DSTORAGE_ATTR, None)


class Owned(Placement):
    """
    Placement indicating a parameter is fully owned by one rank.

    In parameter-boundary sharding, each parameter is assigned to exactly one
    rank (the owner). The owner has the full parameter data, while other ranks
    have an empty tensor.

    This enables sharding at parameter boundaries rather than within parameters,
    which can be useful for models where parameter sizes don't divide evenly.
    """

    def __init__(self, owner_rank: int):
        self.owner_rank = owner_rank
        super().__init__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Owned):
            return False
        return self.owner_rank == other.owner_rank

    def __hash__(self) -> int:
        return hash((type(self), self.owner_rank))

    def __repr__(self) -> str:
        return f"Owned({self.owner_rank})"


class ShardedState(Enum):
    """State of the parameters in DStorage."""

    SHARDED = auto()  # Parameters are sharded DTensors
    UNSHARDED = auto()  # Parameters are unsharded for forward/backward


@dataclass
class ParamInfo:
    """Metadata for a parameter in chunked storage."""

    fqn: str
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    placements: tuple[Placement, ...]
    local_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    local_numel: int = 0
    byte_offset: int = 0  # byte offset into the sharded storage
    global_numel: int = 0  # total elements in unsharded param
    unsharded_byte_offset: int = 0  # byte offset into the unsharded storage
    owner_rank: int | None = (
        None  # for param-boundary sharding: which rank owns this param
    )
    padded_local_numel: int = (
        0  # padded size for uniform buffer layout (max across ranks)
    )


def _get_dtype_alignment(dtype: torch.dtype) -> int:
    """Get alignment requirement in bytes for a dtype."""
    # Most dtypes need alignment equal to their element size
    return dtype.itemsize


def _align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next aligned boundary."""
    return (offset + alignment - 1) // alignment * alignment


class DStorage:
    """
    Manages a unified byte buffer that backs multiple DTensor parameters.

    All parameters, regardless of dtype, are stored in a single contiguous
    byte buffer. Each parameter's local shard is a typed view into this buffer
    at the appropriate byte offset with proper alignment.

    This enables a single all-gather operation for the entire parameter group.

    Lifecycle (automatic with hooks):
        1. SHARDED state: Parameters are sharded DTensors
        2. Forward pre-hook: unshard() - all-gather to get full params
        3. Forward: compute with unsharded params
        4. Forward post-hook: register backward hooks, optionally reshard
        5. Backward pre-hook: unshard() if resharded after forward
        6. Backward: compute gradients with unsharded params
        7. Post-backward: reshard() with reduce-scatter gradients
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        total_unsharded_bytes: int,
        module: nn.Module,
        reshard_after_forward: bool = True,
        register_hooks: bool = True,
        region_info: dict[str, int] | None = None,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._total_unsharded_bytes = total_unsharded_bytes
        self._module = module
        self._state = ShardedState.SHARDED
        self._reshard_after_forward = reshard_after_forward

        # Region info for batched collectives (Shard params first, then Owned)
        self._region_info = region_info or {
            "shard_region_start": 0,
            "shard_region_end": total_bytes,
            "owned_region_start": total_bytes,
            "owned_region_end": total_bytes,
        }

        # Unsharded buffer (allocated on demand)
        self._unsharded_byte_storage: torch.Tensor | None = None

        # Cache sharded DTensor parameters for reshard
        self._sharded_params: dict[str, nn.Parameter] = {}
        for fqn in param_infos:
            parts = fqn.split(".")
            mod = module
            for part in parts[:-1]:
                mod = getattr(mod, part)
            self._sharded_params[fqn] = getattr(mod, parts[-1])

        # Hook handles
        self._pre_forward_hook_handle: RemovableHandle | None = None
        self._post_forward_hook_handle: RemovableHandle | None = None

        # Track if post_backward has been called this iteration
        self._post_backward_called = False

        # Register forward hooks if requested
        if register_hooks:
            self._register_forward_hooks()

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor (sharded)."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the sharded storage."""
        return self._total_bytes

    @property
    def total_unsharded_bytes(self) -> int:
        """Total bytes needed for unsharded storage."""
        return self._total_unsharded_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    @property
    def state(self) -> ShardedState:
        """Current state (SHARDED or UNSHARDED)."""
        return self._state

    @property
    def world_size(self) -> int:
        """World size of the mesh."""
        return self._mesh.size()

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN (from sharded storage)."""
        info = self._param_infos[fqn]
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = self._byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.local_shape)

    def get_unsharded_view(self, fqn: str) -> torch.Tensor:
        """Get the unsharded tensor view for a parameter by FQN."""
        if self._unsharded_byte_storage is None:
            raise RuntimeError("Unsharded storage not allocated. Call unshard() first.")
        info = self._param_infos[fqn]
        num_bytes = info.global_numel * info.dtype.itemsize
        byte_view = self._unsharded_byte_storage[
            info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
        ]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.global_shape)

    def all_gather(self) -> torch.Tensor:
        """
        All-gather the sharded byte buffer to get the full unsharded buffer.

        Uses batched all-gather with padding for uniform buffer layout.
        For Owned params, uses batched all-gather with variable sizes.

        Returns:
            Unsharded byte buffer containing all parameter data.
        """
        # Allocate unsharded buffer if needed
        if self._unsharded_byte_storage is None:
            self._unsharded_byte_storage = torch.empty(
                self._total_unsharded_bytes,
                dtype=torch.uint8,
                device=self._byte_storage.device,
            )

        pg = self._mesh.get_group()
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        # Collect Shard and Owned params
        shard_infos = []
        owned_infos = []
        for fqn, info in self._param_infos.items():
            if isinstance(info.placements[0], Owned):
                owned_infos.append((fqn, info))
            else:
                shard_infos.append((fqn, info))

        # Batched all-gather for Shard params (padded uniform layout)
        if shard_infos:
            self._all_gather_shard_batched(shard_infos, pg, world_size, my_rank)

        # Batched all-gather for Owned params (variable sizes per rank)
        if owned_infos:
            self._all_gather_owned_batched(owned_infos, pg, world_size, my_rank)

        return self._unsharded_byte_storage

    def _all_gather_shard_batched(
        self,
        shard_infos: list[tuple[str, ParamInfo]],
        pg,
        world_size: int,
        my_rank: int,
    ) -> None:
        """Batched all-gather for all Shard params using padded uniform layout."""
        # Get the shard region from byte_storage (uniform across ranks due to padding)
        shard_start = self._region_info["shard_region_start"]
        shard_end = self._region_info["shard_region_end"]
        shard_region = self._byte_storage[shard_start:shard_end]

        if shard_region.numel() == 0:
            return

        # All-gather the entire shard region (uniform size across ranks)
        gathered = torch.empty(
            world_size * shard_region.numel(),
            dtype=torch.uint8,
            device=self._byte_storage.device,
        )
        dist.all_gather_into_tensor(gathered, shard_region.contiguous(), group=pg)

        # Extract and unpad data for each param
        for fqn, info in shard_infos:
            shard_dim = 0
            for placement in info.placements:
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    break

            # Compute local shapes for all ranks (actual, not padded)
            local_shapes = []
            local_numels = []
            for rank in range(world_size):
                shape = self._compute_local_shape_for_rank(
                    info.global_shape, info.placements, rank
                )
                local_shapes.append(shape)
                numel = 1
                for d in shape:
                    numel *= d
                local_numels.append(numel)

            # Extract chunks from gathered buffer (accounting for padding)
            chunks = []
            param_offset_in_region = info.byte_offset - shard_start

            for rank in range(world_size):
                # Each rank's region starts at rank * shard_region.numel()
                rank_region_start = rank * shard_region.numel()
                # Param is at the same offset within each rank's region
                chunk_start = rank_region_start + param_offset_in_region
                # Only read actual data (not padding)
                actual_bytes = local_numels[rank] * info.dtype.itemsize

                if local_numels[rank] > 0:
                    chunk_bytes = gathered[chunk_start : chunk_start + actual_bytes]
                    chunk = chunk_bytes.view(info.dtype).view(local_shapes[rank])
                    chunks.append(chunk)

            # Concatenate along shard dimension
            if chunks:
                unsharded = torch.cat(chunks, dim=shard_dim)
            else:
                unsharded = torch.empty(
                    info.global_shape,
                    dtype=info.dtype,
                    device=self._byte_storage.device,
                )

            # Copy to unsharded buffer
            num_bytes = info.global_numel * info.dtype.itemsize
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(unsharded.view(-1).view(torch.uint8))

    def _all_gather_owned_batched(
        self,
        owned_infos: list[tuple[str, ParamInfo]],
        pg,
        world_size: int,
        my_rank: int,
    ) -> None:
        """Batched all-gather for all Owned params using variable-size all-gather."""
        # Get the owned region from byte_storage (only owner has data)
        owned_start = self._region_info["owned_region_start"]
        owned_end = self._region_info["owned_region_end"]
        my_owned_region = self._byte_storage[owned_start:owned_end]

        # Gather sizes from all ranks
        my_size = torch.tensor(
            [my_owned_region.numel()],
            dtype=torch.long,
            device=self._byte_storage.device,
        )
        all_sizes = [
            torch.zeros(1, dtype=torch.long, device=self._byte_storage.device)
            for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, my_size, group=pg)
        all_sizes = [s.item() for s in all_sizes]

        # Skip if no owned params
        if sum(all_sizes) == 0:
            return

        # All-gather with variable sizes
        output_tensors = [
            torch.empty(size, dtype=torch.uint8, device=self._byte_storage.device)
            for size in all_sizes
        ]
        input_tensor = (
            my_owned_region.contiguous()
            if my_owned_region.numel() > 0
            else torch.empty(0, dtype=torch.uint8, device=self._byte_storage.device)
        )
        dist.all_gather(output_tensors, input_tensor, group=pg)

        # Build mapping from param to owner's contribution offset
        # For each owned param, find its data in the owner's output tensor
        for fqn, info in owned_infos:
            owner_rank = info.owner_rank
            assert owner_rank is not None

            # Find offset of this param in owner's owned region
            # Need to compute this based on param layout in owner's buffer
            owner_output = output_tensors[owner_rank]
            if owner_output.numel() == 0:
                continue

            # For the owner, info.byte_offset is within their byte_storage
            # The owned region starts at owned_start in owner's buffer
            # So param's offset within owner's owned region is: info.byte_offset - owned_start
            # But we need to handle that non-owners have byte_offset=0 for owned params
            # We need to compute the correct offset for the owner

            # Recompute offset for owner (this is a simplification - in practice we'd store this)
            if my_rank == owner_rank:
                param_offset_in_owned = info.byte_offset - owned_start
            else:
                # Need to figure out where this param is in owner's output
                # For now, iterate through owned_infos to find cumulative offset
                param_offset_in_owned = 0
                for other_fqn, other_info in owned_infos:
                    if other_info.owner_rank == owner_rank:
                        if other_fqn == fqn:
                            break
                        # Add this param's size
                        alignment = _get_dtype_alignment(other_info.dtype)
                        param_offset_in_owned = _align_offset(
                            param_offset_in_owned, alignment
                        )
                        param_offset_in_owned += (
                            other_info.global_numel * other_info.dtype.itemsize
                        )

            num_bytes = info.global_numel * info.dtype.itemsize
            param_data = owner_output[
                param_offset_in_owned : param_offset_in_owned + num_bytes
            ]

            # Copy to unsharded buffer
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(param_data)

    def _compute_local_shape_for_rank(
        self, global_shape: torch.Size, placements: tuple[Placement, ...], rank: int
    ) -> torch.Size:
        """Compute local shape for a specific rank using DTensor's sharding logic."""
        shard_dim = 0
        for placement in placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                break

        world_size = self.world_size
        dim_size = global_shape[shard_dim]

        # Match DTensor's Shard.local_shard_size_and_offset logic
        if dim_size % world_size == 0:
            # Even sharding
            local_dim_size = dim_size // world_size
        else:
            # Uneven sharding: ceil-based chunk size
            full_chunk_size = (dim_size + world_size - 1) // world_size
            shard_starting_idx = full_chunk_size * rank
            if dim_size < shard_starting_idx:
                local_dim_size = 0
            else:
                local_dim_size = (
                    min(dim_size, shard_starting_idx + full_chunk_size)
                    - shard_starting_idx
                )

        local_shape = list(global_shape)
        local_shape[shard_dim] = local_dim_size
        return torch.Size(local_shape)

    def unshard(self) -> None:
        """
        All-gather the byte buffer and register unsharded parameters on the module.

        After calling this, model.parameters() returns unsharded tensors for forward/backward.
        """
        if self._state == ShardedState.UNSHARDED:
            return  # Already unsharded

        # All-gather the byte buffer
        self.all_gather()

        # Register unsharded parameters on the module
        for fqn, info in self._param_infos.items():
            unsharded_view = self.get_unsharded_view(fqn)
            unsharded_param = nn.Parameter(
                unsharded_view, requires_grad=info.requires_grad
            )
            _set_param_on_module(self._module, fqn, unsharded_param)

        self._state = ShardedState.UNSHARDED

    def _sync_dtensor_to_storage(self) -> None:
        """
        Copy data from unsharded buffer back to sharded byte_storage.

        This is useful after calling reset_parameters() on unsharded params,
        to ensure the initialized values are persisted in the sharded storage.
        Must be called while in UNSHARDED state, before reshard().
        """
        if self._state != ShardedState.UNSHARDED:
            raise RuntimeError("Must be in UNSHARDED state to sync to storage")
        if self._unsharded_byte_storage is None:
            raise RuntimeError("Unsharded storage not allocated")

        my_rank = self._mesh.get_local_rank()
        world_size = self.world_size

        for fqn, info in self._param_infos.items():
            placement = info.placements[0]
            # Get the full param data from unsharded storage
            unsharded_view = self.get_unsharded_view(fqn)

            if isinstance(placement, Owned):
                # Only the owner rank has data in byte_storage
                owner_rank = info.owner_rank
                if my_rank == owner_rank:
                    local_view = self.get_local_view(fqn)
                    local_view.copy_(unsharded_view)
            else:
                # Shard: extract this rank's shard from unsharded data
                if info.local_numel > 0:
                    # Compute shard bounds
                    dim0_size = info.global_shape[0]
                    shard_size = (dim0_size + world_size - 1) // world_size
                    start = my_rank * shard_size
                    end = min(start + shard_size, dim0_size)
                    actual_shard_size = end - start

                    if actual_shard_size > 0:
                        # Extract shard from unsharded data
                        shard_data = unsharded_view[start:end]
                        local_view = self.get_local_view(fqn)
                        # Handle potential size mismatch due to padding
                        if shard_data.numel() == local_view.numel():
                            local_view.copy_(shard_data.view(local_view.shape))
                        else:
                            # Copy what we can
                            flat_shard = shard_data.view(-1)
                            flat_local = local_view.view(-1)
                            copy_size = min(flat_shard.numel(), flat_local.numel())
                            flat_local[:copy_size].copy_(flat_shard[:copy_size])

    def _sync_sharded_to_storage(self, device: torch.device | None = None) -> None:
        """
        Copy data from sharded DTensor local tensors to byte_storage.

        This is useful after calling to_empty() and reset_parameters() on a model
        that was sharded on meta device. The DTensor local tensors have been
        materialized and initialized, but byte_storage may still be on meta or
        have stale data.

        Args:
            device: Target device for byte_storage. If None, uses the device of
                    the first DTensor's local tensor.

        Must be called while in SHARDED state.
        """
        if self._state != ShardedState.SHARDED:
            raise RuntimeError("Must be in SHARDED state to sync to storage")

        # Get target device from first param if not specified
        if device is None:
            for fqn, sharded_param in self._sharded_params.items():
                if isinstance(sharded_param, DTensor):
                    device = sharded_param._local_tensor.device
                    break
                else:
                    device = sharded_param.device
                    break

        if device is None:
            raise RuntimeError("No parameters found to determine target device")

        # Materialize byte_storage if on meta device
        if self._byte_storage.device == torch.device("meta"):
            self._byte_storage = torch.empty(
                self._byte_storage.shape,
                dtype=self._byte_storage.dtype,
                device=device,
            )

        # Copy from each DTensor's local tensor to byte_storage
        for fqn, info in self._param_infos.items():
            sharded_param = self._sharded_params[fqn]

            # Get the local tensor data
            if isinstance(sharded_param, DTensor):
                local_data = sharded_param._local_tensor
            else:
                local_data = sharded_param.data

            if info.local_numel == 0:
                continue  # No data for this rank

            # Get the view into byte_storage for this param
            byte_offset = info.byte_offset
            num_bytes = info.local_numel * info.dtype.itemsize
            storage_slice = self._byte_storage[byte_offset : byte_offset + num_bytes]

            # View as the param's dtype and copy
            storage_view = storage_slice.view(info.dtype)
            storage_view.copy_(local_data.view(-1))

    def reshard(self) -> None:
        """
        Reduce-scatter gradients, free unsharded buffer, and restore sharded DTensor parameters.

        Gradients from the unsharded parameters are reduce-scattered across ranks,
        and the resulting sharded gradients are stored on the DTensor parameters.

        After calling this, model.parameters() returns sharded DTensors with gradients.
        """
        if self._state == ShardedState.SHARDED:
            return  # Already sharded

        # Reduce-scatter gradients before swapping parameters
        self._reduce_scatter_grads()

        # Restore sharded DTensor parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    def _reduce_scatter_grads(self) -> None:
        """
        Reduce-scatter gradients to sharded DTensor params.

        Handles mixed Shard/Owned placements per-parameter:
        - Shard: Uses reduce-scatter to distribute gradient shards
        - Owned: Uses reduce to send gradients to owner rank
        """
        pg = self._mesh.get_group()
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        # Separate params by placement type
        shard_grads: list[torch.Tensor] = []
        shard_infos: list[ParamInfo] = []
        shard_padded_sizes: list[torch.Size] = []

        for fqn, info in self._param_infos.items():
            unsharded_param = _get_param_from_module(self._module, fqn)
            if unsharded_param.grad is None:
                continue

            placement = info.placements[0]
            grad = unsharded_param.grad.contiguous()

            if isinstance(placement, Owned):
                # Handle Owned params with reduce to owner
                owner_rank = info.owner_rank
                assert owner_rank is not None
                dist.reduce(grad, dst=owner_rank, op=dist.ReduceOp.AVG, group=pg)
                if my_rank == owner_rank:
                    grad_dtensor = _create_dtensor_from_view(grad, info, self._mesh)
                    self._sharded_params[fqn].grad = grad_dtensor
            else:
                # Collect Shard params for batched reduce-scatter
                shard_grads.append(grad)
                shard_infos.append(info)
                padded_dim0 = (
                    (grad.size(0) + world_size - 1) // world_size
                ) * world_size
                padded_size = torch.Size([padded_dim0] + list(grad.shape[1:]))
                shard_padded_sizes.append(padded_size)

        # Batch reduce-scatter for Shard params
        if shard_grads:
            self._reduce_scatter_shard_grads(
                pg, world_size, shard_grads, shard_infos, shard_padded_sizes
            )

        torch.cuda.synchronize()

    def _reduce_scatter_shard_grads(
        self,
        pg,
        world_size: int,
        grads: list[torch.Tensor],
        grad_infos: list[ParamInfo],
        padded_sizes: list[torch.Size],
    ) -> None:
        """Reduce-scatter gradients for Shard params."""
        reduce_scatter_input_numel = sum(s.numel() for s in padded_sizes)
        reduce_scatter_output_numel = reduce_scatter_input_numel // world_size

        grad_dtype = grads[0].dtype
        device = grads[0].device

        # Allocate reduce-scatter input buffer
        reduce_scatter_input = torch.empty(
            reduce_scatter_input_numel, dtype=grad_dtype, device=device
        )

        # Copy-in with chunk_cat
        reduce_scatter_input_2d = reduce_scatter_input.view(world_size, -1)
        torch._chunk_cat(
            grads, dim=0, num_chunks=world_size, out=reduce_scatter_input_2d
        )

        # Allocate reduce-scatter output buffer
        reduce_scatter_output = torch.empty(
            reduce_scatter_output_numel, dtype=grad_dtype, device=device
        )

        # Perform reduce-scatter
        dist.reduce_scatter_tensor(
            output=reduce_scatter_output,
            input=reduce_scatter_input,
            op=dist.ReduceOp.AVG,
            group=pg,
        )

        # Extract sharded gradients from the output
        flat_grad_offset = 0
        for info, padded_size in zip(grad_infos, padded_sizes):
            sharded_size = info.local_shape
            padded_sharded_numel = padded_size.numel() // world_size

            if info.local_numel > 0:
                sharded_stride = make_contiguous_strides_for(sharded_size)
                sharded_grad = torch.as_strided(
                    reduce_scatter_output,
                    size=sharded_size,
                    stride=sharded_stride,
                    storage_offset=flat_grad_offset,
                ).contiguous()

                grad_dtensor = _create_dtensor_from_view(sharded_grad, info, self._mesh)
                self._sharded_params[info.fqn].grad = grad_dtensor

            flat_grad_offset += padded_sharded_numel

    @contextmanager
    def unsharded(self):
        """
        Context manager for automatic unshard/reshard around forward.

        Usage:
            with storage.unsharded():
                output = model(input)
        """
        self.unshard()
        try:
            yield
        finally:
            self.reshard()

    # ==================== Hook-based Scheduling ====================

    def _register_forward_hooks(self) -> None:
        """Register forward pre/post hooks on the module."""
        self._pre_forward_hook_handle = self._module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = self._module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _pre_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Forward pre-hook: unshard parameters."""
        self.unshard()
        return args, kwargs

    def _post_forward(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        """Forward post-hook: register backward hooks."""
        # Register backward hooks on output tensors
        output = self._register_pre_backward_hooks(output)

        # Reset post_backward flag for this iteration
        self._post_backward_called = False

        # NOTE: We do NOT reshard after forward even if reshard_after_forward=True
        # This is because the autograd graph references the unsharded params,
        # and we need those same param objects to receive gradients in backward.
        # Memory savings from reshard_after_forward would require more complex
        # tracking of unsharded params (like FSDP2's FSDPParam).

        return output

    def _register_pre_backward_hooks(self, output: Any) -> Any:
        """Register hooks on output tensors to trigger pre_backward."""
        if not torch.is_grad_enabled():
            return output

        flat_outputs, _ = tree_flatten(output)
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)

        return output

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pre-hook: register post-backward callback."""
        # Register post-backward callback (must be done during backward)
        self._register_post_backward_callback()
        # Params are already unsharded from forward, no need to unshard again
        return grad

    def _register_post_backward_callback(self) -> None:
        """Register callback to run after backward completes."""
        if self._post_backward_called:
            return
        Variable._execution_engine.queue_callback(self._post_backward)

    def _post_backward(self) -> None:
        """Post-backward callback: reshard and reduce-scatter gradients."""
        # Ensure we only run once per backward pass
        if self._post_backward_called:
            return
        self._post_backward_called = True

        # Only reshard if currently unsharded
        if self._state == ShardedState.UNSHARDED:
            self.reshard()

    def _reshard_params_only(self) -> None:
        """
        Reshard parameters without reduce-scatter (for use after forward).

        This restores sharded DTensor parameters but does NOT reduce-scatter
        gradients (since there are none yet after forward).
        """
        if self._state == ShardedState.SHARDED:
            return

        # Restore sharded DTensor parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    def remove_hooks(self) -> None:
        """Remove registered forward hooks."""
        if self._pre_forward_hook_handle is not None:
            self._pre_forward_hook_handle.remove()
            self._pre_forward_hook_handle = None
        if self._post_forward_hook_handle is not None:
            self._post_forward_hook_handle.remove()
            self._post_forward_hook_handle = None


def _compute_local_info(
    global_shape: torch.Size,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[torch.Size, int]:
    """Compute local shape and numel for a parameter on current rank."""
    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape, mesh, placements, skip_offset=True
    )
    local_numel = 1
    for dim in local_shape:
        local_numel *= dim
    return torch.Size(local_shape), local_numel


def _compute_max_local_numel(
    global_shape: torch.Size,
    placements: tuple[Placement, ...],
    world_size: int,
) -> int:
    """
    Compute the max local numel across all ranks for a Shard placement.

    This is used to pad local shards to uniform size for batched all-gather.
    """
    shard_dim = 0
    for placement in placements:
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            break

    dim_size = global_shape[shard_dim]

    # Compute max local dim size using DTensor's ceil-based sharding
    if dim_size % world_size == 0:
        max_local_dim = dim_size // world_size
    else:
        # Ceil-based: max is the full chunk size (first rank gets this)
        max_local_dim = (dim_size + world_size - 1) // world_size

    # Compute max numel
    max_numel = max_local_dim
    for i, d in enumerate(global_shape):
        if i != shard_dim:
            max_numel *= d

    return max_numel


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Parameters are laid out sequentially in the byte buffer with proper alignment.
    Uses padded sizes for uniform buffer layout across ranks (enables batched all-gather).

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
        region_info: dict with shard/owned region boundaries
    """
    world_size = mesh.size()
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
        dtype = param.dtype

        # Compute global numel
        global_numel = 1
        for dim in global_shape:
            global_numel *= dim

        # Compute padded local numel (max across all ranks for uniform layout)
        padded_local_numel = _compute_max_local_numel(
            global_shape, placements, world_size
        )

        # Align offset for this dtype (sharded buffer)
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)

        # Align offset for unsharded buffer
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=aligned_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            padded_local_numel=padded_local_numel,
        )
        param_infos[fqn] = info

        # Move offsets past this parameter's PADDED bytes (for uniform layout)
        param_bytes = padded_local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

    # For Shard-only: entire buffer is shard region
    region_info = {
        "shard_region_start": 0,
        "shard_region_end": current_byte_offset,
        "owned_region_start": current_byte_offset,
        "owned_region_end": current_byte_offset,
    }

    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


def _create_dtensor_from_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh: DeviceMesh,
) -> DTensor:
    """Create a DTensor whose _local_tensor is the given view."""
    spec = DTensorSpec(
        mesh=mesh,
        placements=info.placements,
        tensor_meta=TensorMeta(
            shape=info.global_shape,
            stride=info.global_stride,
            dtype=info.dtype,
        ),
    )
    return DTensor(
        local_view,
        spec,
        requires_grad=info.requires_grad,
    )


def _set_param_on_module(
    root_module: nn.Module,
    fqn: str,
    param: nn.Parameter,
) -> None:
    """Navigate to submodule by FQN and set parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], param)


def _get_param_from_module(
    root_module: nn.Module,
    fqn: str,
) -> nn.Parameter:
    """Navigate to submodule by FQN and get parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    return getattr(module, parts[-1])


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, int]:
    """
    Assign parameters to ranks using greedy bin-packing for balanced memory.

    Assigns larger parameters first to help balance the load.

    Returns:
        Dict mapping FQN to owner rank.
    """
    # Sort by size (descending) for better bin packing
    sorted_params = sorted(
        named_params,
        key=lambda x: x[1].numel() * x[1].element_size(),
        reverse=True,
    )

    rank_bytes: list[int] = [0] * world_size
    assignments: dict[str, int] = {}

    for fqn, param in sorted_params:
        # Assign to rank with least bytes
        target_rank = rank_bytes.index(min(rank_bytes))
        assignments[fqn] = target_rank
        rank_bytes[target_rank] += param.numel() * param.element_size()

    return assignments


def _create_param_infos_param_boundary(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    assignments: dict[str, int],
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for parameter-boundary sharding.

    Each parameter is assigned to one rank. The owner has full data,
    non-owners have empty storage for this param.

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer (this rank's owned params)
        total_unsharded_bytes: total bytes needed for the unsharded buffer (all params)
        region_info: dict with shard/owned region boundaries
    """
    my_rank = mesh.get_local_rank()
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        owner_rank = assignments[fqn]

        global_numel = param.numel()

        # For param-boundary: owner has full shape, others have empty
        if my_rank == owner_rank:
            local_shape = global_shape
            local_numel = global_numel
        else:
            # Empty tensor: shape with 0 in first dim
            local_shape = torch.Size([0] + list(global_shape[1:]))
            local_numel = 0

        placement = (Owned(owner_rank),)

        # Align offset for this dtype (sharded buffer - only for owned params)
        alignment = _get_dtype_alignment(dtype)

        # Only allocate space in sharded buffer if this rank owns the param
        if my_rank == owner_rank:
            aligned_offset = _align_offset(current_byte_offset, alignment)
            byte_offset = aligned_offset
            param_bytes = local_numel * dtype.itemsize
            current_byte_offset = aligned_offset + param_bytes
        else:
            byte_offset = 0  # Not stored locally

        # Unsharded buffer offset (all ranks need space for all params)
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placement,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=owner_rank,
        )
        param_infos[fqn] = info

    # For param_boundary: no shard region (all params are Owned)
    region_info = {
        "shard_region_start": 0,
        "shard_region_end": 0,
        "owned_region_start": 0,
        "owned_region_end": current_byte_offset,
    }

    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


def _create_param_infos_with_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    shard_placement_fn: Callable[[str, nn.Parameter], Shard | Owned | None],
    default_placement: Shard | Owned = Shard(0),
) -> tuple[dict[str, ParamInfo], int, int, dict[str, int]]:
    """
    Create ParamInfo for each parameter using a per-parameter placement function.

    Supports mixed Shard and Owned placements in the same storage.
    Uses padded sizes for Shard params to enable batched all-gather.

    Args:
        named_params: List of (fqn, param) tuples
        mesh: Device mesh for sharding
        shard_placement_fn: Function mapping (fqn, param) -> Shard | Owned | None
            Returns None for default placement
        default_placement: Default placement when function returns None

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
        region_info: dict with shard/owned region boundaries
    """
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    # First pass: categorize params by placement type
    shard_params: list[tuple[str, nn.Parameter, Shard]] = []
    owned_params: list[tuple[str, nn.Parameter, Owned]] = []

    for fqn, param in named_params:
        placement = shard_placement_fn(fqn, param)
        if placement is None:
            placement = default_placement

        # Normalize negative dims for Shard
        if isinstance(placement, Shard) and placement.dim < 0:
            placement = Shard(placement.dim + param.ndim)

        if isinstance(placement, Owned):
            owned_params.append((fqn, param, placement))
        else:
            assert isinstance(placement, Shard)
            shard_params.append((fqn, param, placement))

    # Second pass: layout buffer with Shard params first, then Owned params
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    # Layout Shard params first (contiguous region for batched all-gather)
    shard_region_start = 0
    for fqn, param, placement in shard_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        global_numel = param.numel()

        placements_tuple = (placement,)
        local_shape, local_numel = _compute_local_info(
            global_shape, mesh, placements_tuple
        )

        # Compute padded local numel for uniform buffer layout
        padded_local_numel = _compute_max_local_numel(
            global_shape, placements_tuple, world_size
        )

        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)
        byte_offset = aligned_offset
        # Use PADDED size for buffer allocation
        param_bytes = padded_local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        # Unsharded buffer offset
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements_tuple,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=None,
            padded_local_numel=padded_local_numel,
        )
        param_infos[fqn] = info

    shard_region_end = current_byte_offset

    # Layout Owned params (only owner stores data)
    owned_region_start = current_byte_offset
    for fqn, param, placement in owned_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        dtype = param.dtype
        global_numel = param.numel()
        owner_rank = placement.owner_rank

        if my_rank == owner_rank:
            local_shape = global_shape
            local_numel = global_numel
            alignment = _get_dtype_alignment(dtype)
            aligned_offset = _align_offset(current_byte_offset, alignment)
            byte_offset = aligned_offset
            param_bytes = local_numel * dtype.itemsize
            current_byte_offset = aligned_offset + param_bytes
        else:
            local_shape = torch.Size([0] + list(global_shape[1:]))
            local_numel = 0
            byte_offset = 0  # Not stored locally

        placements_tuple = (placement,)

        # Unsharded buffer offset
        alignment = _get_dtype_alignment(dtype)
        aligned_unsharded_offset = _align_offset(
            current_unsharded_byte_offset, alignment
        )
        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements_tuple,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
            owner_rank=owner_rank,
        )
        param_infos[fqn] = info

    owned_region_end = current_byte_offset

    # Return param_infos, total bytes, total unsharded bytes, and region info
    region_info = {
        "shard_region_start": shard_region_start,
        "shard_region_end": shard_region_end,
        "owned_region_start": owned_region_start,
        "owned_region_end": owned_region_end,
    }
    return param_infos, current_byte_offset, current_unsharded_byte_offset, region_info


def _copy_original_data_with_placement_fn(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh: DeviceMesh,
) -> None:
    """
    Copy original parameter data into byte storage, handling mixed Shard/Owned placements.
    """
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]

        if param.device.type == "meta":
            continue

        placement = info.placements[0]

        if isinstance(placement, Owned):
            # Only owner copies data
            if my_rank != info.owner_rank:
                continue
            # Copy full parameter into byte storage
            num_bytes = info.local_numel * info.dtype.itemsize
            byte_view = byte_storage[info.byte_offset : info.byte_offset + num_bytes]
            typed_dest = byte_view.view(info.dtype)
            typed_dest.copy_(param.data.view(-1))
        else:
            # Shard: chunk and copy local shard
            assert isinstance(placement, Shard)
            shard_dim = placement.dim
            chunks = list(torch.chunk(param.data, world_size, dim=shard_dim))
            while len(chunks) < world_size:
                chunks.append(chunks[0].new_empty(0))
            local_shard = chunks[my_rank]

            if local_shard.numel() > 0:
                num_bytes = info.local_numel * info.dtype.itemsize
                byte_view = byte_storage[
                    info.byte_offset : info.byte_offset + num_bytes
                ]
                typed_dest = byte_view.view(info.dtype)
                typed_dest.copy_(local_shard.view(-1))


def _copy_original_data_to_flat_storage_param_boundary(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh: DeviceMesh,
) -> None:
    """
    Copy original parameter data into byte storage for parameter-boundary sharding.

    Only the owner rank copies the full parameter data.
    """
    my_rank = mesh.get_local_rank()

    for fqn, param in named_params:
        info = param_infos[fqn]

        if param.device.type == "meta":
            continue

        # Only owner copies data
        if my_rank != info.owner_rank:
            continue

        # Copy full parameter into byte storage
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        typed_dest = byte_view.view(info.dtype)
        typed_dest.copy_(param.data.view(-1))


def _copy_original_data_to_flat_storage(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh: DeviceMesh,
) -> None:
    """
    Copy original parameter data into the unified byte storage, sharding as needed.

    For each parameter:
    1. If already on correct device, chunk and copy the local shard
    2. If on meta device, leave uninitialized (caller must initialize later)
    """
    if mesh.ndim != 1:
        raise NotImplementedError("Only 1D mesh (pure FSDP) is supported currently")

    shard_rank = mesh.get_local_rank()
    shard_world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]

        if param.device.type == "meta":
            continue

        # Determine shard dimension from placements
        shard_dim = 0
        for placement in info.placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                break

        # Chunk the parameter and get this rank's shard
        chunks = list(torch.chunk(param.data, shard_world_size, dim=shard_dim))
        while len(chunks) < shard_world_size:
            chunks.append(chunks[0].new_empty(0))
        local_shard = chunks[shard_rank]

        # Copy into byte storage
        if local_shard.numel() > 0:
            num_bytes = info.local_numel * info.dtype.itemsize
            byte_view = byte_storage[info.byte_offset : info.byte_offset + num_bytes]
            # View as the correct dtype, then copy
            typed_dest = byte_view.view(info.dtype)
            typed_dest.copy_(local_shard.view(-1))


def _get_managed_named_params(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """
    Collect parameters that should be managed by this module's DStorage.

    This excludes parameters from child modules that already have their own
    DStorage (i.e., already wrapped with fully_shard_flat).

    Similar to FSDP2's _get_managed_modules/_get_managed_states pattern.
    """
    managed_params: list[tuple[str, nn.Parameter]] = []

    # Find child modules that already have DStorage
    wrapped_prefixes: set[str] = set()
    for name, child in module.named_modules():
        if name and get_dstorage(child) is not None:
            # This child is already wrapped; skip its parameters
            wrapped_prefixes.add(name + ".")

    # Collect parameters not in wrapped submodules
    for fqn, param in module.named_parameters():
        is_wrapped = any(fqn.startswith(prefix) for prefix in wrapped_prefixes)
        if not is_wrapped:
            managed_params.append((fqn, param))

    return managed_params


def fully_shard_flat(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...] | None = None,
    reshard_after_forward: bool = True,
    register_hooks: bool = True,
    shard_strategy: Literal["per_param", "param_boundary"] = "per_param",
    shard_placement_fn: Callable[[str, nn.Parameter], Shard | Owned | None]
    | None = None,
) -> DStorage:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Creates a single unified byte buffer for all parameters (regardless of dtype)
    3. Replaces each parameter with a DTensor whose _local_tensor is a typed view
       into the byte buffer at the appropriate offset
    4. Optionally registers forward/backward hooks for automatic unshard/reshard
    5. Returns DStorage for managing the unified buffer

    The unified byte buffer enables a single all-gather operation for all parameters.

    Nested wrapping is supported: apply fully_shard_flat to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The device mesh for sharding. Currently only 1D mesh is supported.
        placements: The default sharding placements. Defaults to (Shard(0),).
            Used when shard_placement_fn is None or returns None for a param.
            Ignored when shard_strategy="param_boundary".
        reshard_after_forward: If True (default), reshard parameters after forward
            to save memory. Parameters will be re-unsharded in backward.
            If False, keep parameters unsharded between forward and backward.
        register_hooks: If True (default), register forward/backward hooks for
            automatic unshard/reshard. If False, caller must manually call
            unshard()/reshard().
        shard_strategy: The default sharding strategy (ignored if shard_placement_fn is provided):
            - "per_param" (default): Each parameter is sharded across all ranks
              along the specified dimension. Uses Shard placement.
            - "param_boundary": Each parameter is assigned to one rank (owner).
              The owner has full parameter data, others have empty tensors.
              Uses Owned placement and greedy bin-packing for balanced memory.
        shard_placement_fn: Optional callable for per-parameter placement control.
            Takes (fqn, param) and returns Shard | Owned | None.
            - Shard(dim): Shard this parameter along dimension dim
            - Owned(rank): Assign this parameter to the specified rank
            - None: Use default placement from `placements` parameter
            This enables mixed Shard/Owned placements in a single DStorage.

    Returns:
        DStorage instance containing the unified byte buffer and parameter metadata.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,))
        >>> model = Transformer(args)
        >>> # Nested wrapping: wrap layers first, then root
        >>> for layer in model.layers:
        ...     fully_shard_flat(layer, mesh)
        >>> storage = fully_shard_flat(model, mesh)  # Only wraps non-layer params
        >>> # Forward/backward now work automatically with hooks
        >>> output = model(input)
        >>> output.sum().backward()

    Example with per-parameter placement::

        >>> def placement_fn(fqn, param):
        ...     if "embed" in fqn:
        ...         return Owned(0)  # Embeddings owned by rank 0
        ...     return Shard(0)  # Other params sharded
        >>> storage = fully_shard_flat(model, mesh, shard_placement_fn=placement_fn)

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - The returned DStorage is also stored on module._dstorage
        - Forward/backward hooks are automatically registered for unshard/reshard
    """
    if placements is None:
        placements = (Shard(0),)

    # Check if module is already wrapped
    if get_dstorage(module) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has DStorage. "
            "Cannot apply fully_shard_flat twice to the same module."
        )

    # Collect parameters (excluding those from already-wrapped submodules)
    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )

    # Determine device - use param device if meta, otherwise use mesh device
    first_param = named_params[0][1]
    if first_param.device.type == "meta":
        device = torch.device("meta")
    else:
        device = mesh.device_type
        if device == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)

    # Create parameter infos with local shapes and byte offsets
    region_info = None
    if shard_placement_fn is not None:
        # Per-parameter placement function provided
        default_placement = placements[0] if placements else Shard(0)
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos_with_placement_fn(
                named_params, mesh, shard_placement_fn, default_placement
            )
        )
    elif shard_strategy == "param_boundary":
        # Assign params to ranks using bin-packing
        assignments = _assign_params_to_ranks(named_params, mesh.size())
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos_param_boundary(named_params, mesh, assignments)
        )
    else:
        param_infos, total_bytes, total_unsharded_bytes, region_info = (
            _create_param_infos(named_params, mesh, placements)
        )

    # Allocate unified byte storage
    byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # Copy original data into byte storage (handles sharding)
    if shard_placement_fn is not None:
        _copy_original_data_with_placement_fn(
            byte_storage, named_params, param_infos, mesh
        )
    elif shard_strategy == "param_boundary":
        _copy_original_data_to_flat_storage_param_boundary(
            byte_storage, named_params, param_infos, mesh
        )
    else:
        _copy_original_data_to_flat_storage(
            byte_storage, named_params, param_infos, mesh
        )

    # Replace each parameter with DTensor view (before creating DStorage)
    for fqn, info in param_infos.items():
        local_view = byte_storage[
            info.byte_offset : info.byte_offset + info.local_numel * info.dtype.itemsize
        ]
        typed_view = local_view.view(info.dtype).view(info.local_shape)
        dtensor = _create_dtensor_from_view(typed_view, info, mesh)
        new_param = nn.Parameter(dtensor, requires_grad=info.requires_grad)
        _set_param_on_module(module, fqn, new_param)

    # Create DStorage (after DTensor params are registered)
    # This also registers forward/backward hooks if requested
    storage = DStorage(
        byte_storage,
        param_infos,
        mesh,
        total_bytes,
        total_unsharded_bytes,
        module,
        reshard_after_forward=reshard_after_forward,
        register_hooks=register_hooks,
        region_info=region_info,
    )

    # Store DStorage on module for nested wrapping detection
    setattr(module, _DSTORAGE_ATTR, storage)

    return storage
