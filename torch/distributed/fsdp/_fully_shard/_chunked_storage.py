# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, TYPE_CHECKING

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


__all__ = ["DStorage", "fully_shard_flat", "get_dstorage"]


# Module attribute name for storing DStorage
_DSTORAGE_ATTR = "_dstorage"


def get_dstorage(module: nn.Module) -> "DStorage | None":
    """Get the DStorage associated with a module, if any."""
    return getattr(module, _DSTORAGE_ATTR, None)


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
        self._pre_forward_hook_handle = None
        self._post_forward_hook_handle = None

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

        # Get the process group from mesh
        pg = self._mesh.get_group()
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        # For each parameter, all-gather its chunks and reassemble
        for fqn, info in self._param_infos.items():
            # Get shard dimension
            shard_dim = 0
            for placement in info.placements:
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    break

            # Get local chunk for this rank
            local_view = self.get_local_view(fqn)

            # Compute max local numel across all ranks for padding
            max_local_numel = 0
            local_numels = []
            for rank in range(world_size):
                local_shape_for_rank = self._compute_local_shape_for_rank(
                    info.global_shape, info.placements, rank
                )
                numel = 1
                for d in local_shape_for_rank:
                    numel *= d
                local_numels.append(numel)
                max_local_numel = max(max_local_numel, numel)

            # Pad local tensor to max size
            if local_view.numel() < max_local_numel:
                padded = torch.zeros(
                    max_local_numel, dtype=info.dtype, device=self._byte_storage.device
                )
                if local_view.numel() > 0:
                    padded[: local_view.numel()] = local_view.view(-1)
            else:
                padded = local_view.view(-1)

            # All-gather the padded tensors
            all_gather_output = torch.empty(
                world_size * max_local_numel,
                dtype=info.dtype,
                device=self._byte_storage.device,
            )
            dist.all_gather_into_tensor(all_gather_output, padded.contiguous(), group=pg)

            # Extract and concatenate non-padded chunks
            chunks = []
            for rank in range(world_size):
                rank_numel = local_numels[rank]
                if rank_numel > 0:
                    start = rank * max_local_numel
                    chunk = all_gather_output[start : start + rank_numel]
                    local_shape_for_rank = self._compute_local_shape_for_rank(
                        info.global_shape, info.placements, rank
                    )
                    chunks.append(chunk.view(local_shape_for_rank))

            # Concatenate along shard dimension
            if chunks:
                unsharded_param = torch.cat(chunks, dim=shard_dim)
            else:
                unsharded_param = torch.empty(
                    info.global_shape,
                    dtype=info.dtype,
                    device=self._byte_storage.device,
                )

            # Copy into unsharded buffer
            num_bytes = info.global_numel * info.dtype.itemsize
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(unsharded_param.view(-1).view(torch.uint8))

        return self._unsharded_byte_storage

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
                local_dim_size = min(dim_size, shard_starting_idx + full_chunk_size) - shard_starting_idx

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
            unsharded_param = nn.Parameter(unsharded_view, requires_grad=info.requires_grad)
            _set_param_on_module(self._module, fqn, unsharded_param)

        self._state = ShardedState.UNSHARDED

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

        Uses the same approach as FSDP2's foreach_reduce:
        1. Pad each gradient's dim0 to be divisible by world_size
        2. Use chunk_cat to reorder: chunk each grad into world_size pieces,
           concatenate chunk[i] from all grads into row i
        3. Reduce-scatter the reordered buffer
        4. Extract sharded gradients from the output
        """
        pg = self._mesh.get_group()
        world_size = self.world_size

        # Collect all gradients and compute padded sizes
        grads: list[torch.Tensor] = []
        grad_infos: list[ParamInfo] = []
        padded_unsharded_sizes: list[torch.Size] = []

        for fqn, info in self._param_infos.items():
            unsharded_param = _get_param_from_module(self._module, fqn)
            if unsharded_param.grad is None:
                continue

            grad = unsharded_param.grad.contiguous()
            grads.append(grad)
            grad_infos.append(info)

            # Compute padded size (dim0 must be divisible by world_size)
            padded_dim0 = ((grad.size(0) + world_size - 1) // world_size) * world_size
            padded_size = torch.Size([padded_dim0] + list(grad.shape[1:]))
            padded_unsharded_sizes.append(padded_size)

        if not grads:
            return

        # Compute total numel for reduce-scatter buffers
        reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
        reduce_scatter_output_numel = reduce_scatter_input_numel // world_size

        # Determine dtype (all grads should have same dtype for reduce-scatter)
        grad_dtype = grads[0].dtype
        device = grads[0].device

        # Allocate reduce-scatter input buffer
        reduce_scatter_input = torch.empty(
            reduce_scatter_input_numel, dtype=grad_dtype, device=device
        )

        # Copy-in with chunk_cat: reorders data for reduce-scatter
        # chunk_cat chunks each tensor into world_size pieces along dim 0,
        # then concatenates chunk[i] from all tensors into row i of output
        reduce_scatter_input_2d = reduce_scatter_input.view(world_size, -1)
        torch._chunk_cat(grads, dim=0, num_chunks=world_size, out=reduce_scatter_input_2d)

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
        for info, padded_size in zip(grad_infos, padded_unsharded_sizes):
            # Compute the sharded size (actual local shape, not padded)
            sharded_size = info.local_shape

            # Extract gradient shard using as_strided for efficiency
            # The padded chunk size in the output
            padded_sharded_numel = padded_size.numel() // world_size

            if info.local_numel > 0:
                # Create sharded gradient by extracting from flat output
                # The actual shard may be smaller than padded chunk
                sharded_stride = make_contiguous_strides_for(sharded_size)
                sharded_grad = torch.as_strided(
                    reduce_scatter_output,
                    size=sharded_size,
                    stride=sharded_stride,
                    storage_offset=flat_grad_offset,
                ).contiguous()

                # Create DTensor gradient and assign to sharded param
                grad_dtensor = _create_dtensor_from_view(sharded_grad, info, self._mesh)
                self._sharded_params[info.fqn].grad = grad_dtensor

            flat_grad_offset += padded_sharded_numel

        torch.cuda.synchronize()

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


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[dict[str, ParamInfo], int, int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Parameters are laid out sequentially in the byte buffer with proper alignment.

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
    """
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

        # Align offset for this dtype (sharded buffer)
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)

        # Align offset for unsharded buffer
        aligned_unsharded_offset = _align_offset(current_unsharded_byte_offset, alignment)

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
        )
        param_infos[fqn] = info

        # Move offsets past this parameter's bytes
        param_bytes = local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

    return param_infos, current_byte_offset, current_unsharded_byte_offset


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
        placements: The sharding placements. Defaults to (Shard(0),).
        reshard_after_forward: If True (default), reshard parameters after forward
            to save memory. Parameters will be re-unsharded in backward.
            If False, keep parameters unsharded between forward and backward.
        register_hooks: If True (default), register forward/backward hooks for
            automatic unshard/reshard. If False, caller must manually call
            unshard()/reshard().

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

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - The returned DStorage is also stored on module._chunked_storage
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

    # Determine device
    device = mesh.device_type
    if device == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    # Create parameter infos with local shapes and byte offsets
    param_infos, total_bytes, total_unsharded_bytes = _create_param_infos(
        named_params, mesh, placements
    )

    # Allocate unified byte storage
    byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # Copy original data into byte storage (handles sharding)
    _copy_original_data_to_flat_storage(byte_storage, named_params, param_infos, mesh)

    # Replace each parameter with DTensor view (before creating DStorage)
    for fqn, info in param_infos.items():
        local_view = byte_storage[info.byte_offset : info.byte_offset + info.local_numel * info.dtype.itemsize]
        typed_view = local_view.view(info.dtype).view(info.local_shape)
        dtensor = _create_dtensor_from_view(typed_view, info, mesh)
        new_param = nn.Parameter(dtensor, requires_grad=info.requires_grad)
        _set_param_on_module(module, fqn, new_param)

    # Create DStorage (after DTensor params are registered)
    # This also registers forward/backward hooks if requested
    storage = DStorage(
        byte_storage, param_infos, mesh, total_bytes, total_unsharded_bytes, module,
        reshard_after_forward=reshard_after_forward,
        register_hooks=register_hooks,
    )

    # Store DStorage on module for nested wrapping detection
    setattr(module, _DSTORAGE_ATTR, storage)

    return storage
