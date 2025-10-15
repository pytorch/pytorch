# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import cast, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _chunk_with_empty,
    _get_dim_chunked_size,
    HSDPMeshInfo,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources


if TYPE_CHECKING:
    from torch.distributed.tensor.placement_types import Placement


class ReplicateParam(FSDPParam):
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.

    Inherits all attributes from FSDPParam. Only overrides _init_sharded_param
    to remove HSDP sharding logic and simplify for replicate-only use case.
    """

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        device: torch.device,
        shard_placement_fn: Optional[Callable],
    ):
        if param.device != device and param.device.type != "meta":
            raise AssertionError(
                f"Expects the parameter to already be moved to device {device} but got {param.device}"
            )
        if not param.is_contiguous():
            raise NotImplementedError(
                f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
            )
        fsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
        if fsdp_placement is None:
            fsdp_placement = Shard(0)
        elif fsdp_placement.dim < 0:
            fsdp_placement = Shard(fsdp_placement.dim + param.ndim)
        assert isinstance(fsdp_placement, Shard), f"{fsdp_placement}"
        self.fsdp_placement = fsdp_placement
        shard_dim = fsdp_placement.dim
        # TODO: Replace the sharded DTensor parameter construction logic with
        # `distribute_tensor` after https://github.com/pytorch/pytorch/issues/116101
        # TODO: Simplify the following sharded parameter padding logic after
        # https://github.com/pytorch/pytorch/issues/113045
        self.is_dtensor = isinstance(param, DTensor)
        if self.is_dtensor:
            self._tp_spec = cast(DTensor, param)._spec
            dp_mesh, tp_mesh = (self.mesh_info.mesh, self._tp_spec.mesh)
            dp_global_mesh = _mesh_resources.get_root_mesh(dp_mesh)
            tp_global_mesh = _mesh_resources.get_root_mesh(tp_mesh)
            if dp_global_mesh != tp_global_mesh or (
                dp_global_mesh is None or tp_global_mesh is None
            ):
                raise AssertionError(
                    "FSDP requires the DP and model parallel TP/EP mesh to have the same parent mesh but got: \n"
                    f"DP's global mesh: {dp_global_mesh}\nTP/EP's global mesh: {tp_global_mesh}"
                )
            name_dims_error = "FSDP requires named DeviceMesh dims for ND parallelism"
            assert dp_mesh.mesh_dim_names is not None, name_dims_error
            assert tp_mesh.mesh_dim_names is not None, name_dims_error
            submesh_names = dp_mesh.mesh_dim_names + tp_mesh.mesh_dim_names
            self._spmd_mesh = dp_global_mesh[submesh_names]
            if len(self._tp_spec.placements) > 2:
                raise NotImplementedError(
                    f"FSDP only supports 1D TP/EP or 2D EP+TP, not {self._tp_spec.placements}"
                )
            assert 2 <= self._spmd_mesh.ndim <= 4, (
                "_spmd_mesh.ndim can only be 2 (FSDP+TP/EP), 3 (FSDP+EP+TP, HSDP+TP/EP), "
                f"or 4 (HSDP+EP+TP) but got {self._spmd_mesh.ndim}."
            )
            self._spmd_placements: tuple[Placement, ...]
            dp_shard_tp_placement = (
                (Replicate()),
                *self._tp_spec.placements,
            )
            if dp_mesh.ndim == 1:  # FSDP
                self._spmd_placements = dp_shard_tp_placement
            else:  # HSDP
                assert self.mesh_info.replicate_mesh_dim == 0
                self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=self._tp_spec.tensor_meta,
            )
            param_data = cast(DTensor, param)._local_tensor
        else:
            self._spmd_mesh = self.mesh_info.mesh
            if isinstance(self.mesh_info, HSDPMeshInfo):
                self._spmd_placements = (Replicate(), fsdp_placement)
            else:
                self._spmd_placements = (Replicate(),)
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
            )
            param_data = param
        assert param_data.is_contiguous(), f"{param_data.shape=} {param_data.stride()=}"
        shard_dim = fsdp_placement.dim
        if shard_dim >= param_data.ndim:
            raise AssertionError(
                f"Shard dim {shard_dim} is invalid for {param_data.ndim}D tensor: {param.shape}"
            )
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
        shard_rank = 0
        shard_world_size = 1
        if shard_dim > 0 and param_data.size(shard_dim) % shard_world_size != 0:
            # If sharding on nonzero dim, require even sharding for now because
            # the uneven sharding (1) requires extra copies before/after FSDP
            # collectives and (2) introduces extra complexity to handle padding
            # and unpadding
            raise NotImplementedError(
                f"FSDP does not support uneven sharding on dim {shard_dim}: "
                f"{param_data.size()} (world size: {shard_world_size})"
            )
        chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
        sharded_param = chunks[shard_rank]
        self.sharded_size = _get_dim_chunked_size(
            sharded_param, param_data.size(), dim=shard_dim
        )
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        padded_sharded_size = chunks[0].size()  # 0th always padded
        self.padded_sharded_param_size = padded_sharded_size
        # Pre-pad the sharded parameter to avoid padding before all-gather
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)
        if sharded_param.numel() > 0:
            padded_sharded_param.narrow(
                dim=shard_dim, start=0, length=sharded_param.size(shard_dim)
            ).copy_(sharded_param)
        if self.offload_to_cpu and not padded_sharded_param.is_meta:
            padded_sharded_param = padded_sharded_param.cpu()
            if self.pin_memory:
                padded_sharded_param = padded_sharded_param.pin_memory()
        self._sharded_param_data = padded_sharded_param.view(-1)
        length = sharded_param.size(shard_dim) if sharded_param.numel() > 0 else 0
        sharded_param = padded_sharded_param.narrow(
            dim=shard_dim, start=0, length=length
        )
        assert sharded_param.is_contiguous(), f"{self.fsdp_placement=}"
        self.sharded_param = nn.Parameter(self.to_sharded_dtensor(sharded_param))
        self.sharded_param.requires_grad_(param.requires_grad)
        # Let `param_data` be freed normally when its ref count reaches 0 when
        # the `fully_shard` call returns to allow provided parameters to alias
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED


def alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    # Only call `requires_grad_` if needed to avoid the Python <> C++ context
    # switch overhead
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
