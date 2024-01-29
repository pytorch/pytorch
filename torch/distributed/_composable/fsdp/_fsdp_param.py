from dataclasses import dataclass, field
from enum import auto, Enum
from typing import cast, List, Tuple

import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, Placement, Replicate, Shard
from torch.distributed._tensor.device_mesh import _mesh_resources
from torch.distributed._tensor.placement_types import DTensorSpec

from ._fsdp_common import (
    _chunk_with_empty,
    _from_local_no_grad,
    _raise_assert_with_print,
    FSDPMeshInfo,
)

"""
[Note: FSDP tensors]
FSDP considers the following tensors:
- Original parameter: parameter passed to :class:`FSDPParam`, i.e. the one
  on the module when applying FSDP
- Sharded parameter: sharding the original parameter on dim-0 as a DTensor
  over the main mesh
- Unsharded parameter: parameter used for forward/backward computation
  constructed by all-gathering the sharded parameter
"""


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.
    """

    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    _sharded_param_data: torch.Tensor  # 1D
    sharded_param: nn.Parameter  # ND
    _global_placements: Tuple[Placement, ...]
    _global_size: torch.Size
    _global_stride: Tuple[int, ...]
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec
    _tp_global_size: torch.Size
    _tp_global_stride: Tuple[int, ...]

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        device: torch.device,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.device = device
        self._init_sharded_param(param, device)

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter, device: torch.device):
        if param.device != device:
            raise AssertionError(
                f"Expects the parameter to already be moved to device {device} but got {param.device}"
            )
        # TODO: Replace the sharded DTensor parameter construction logic with
        # `distribute_tensor` after https://github.com/pytorch/pytorch/issues/116101
        # TODO: Simplify the following sharded parameter padding logic after
        # https://github.com/pytorch/pytorch/issues/113045
        self.is_dtensor = isinstance(param, DTensor)
        if self.is_dtensor:
            self._tp_spec = cast(DTensor, param)._spec
            self._tp_global_size = param.size()
            self._tp_global_stride = param.stride()
            if (
                self.mesh_info.shard_mesh_dim != 0
                or self.mesh_info.replicate_mesh_dim is not None
            ):
                raise NotImplementedError("Using TP with HSDP is not supported")
            dp_mesh, tp_mesh = (self.mesh_info.mesh, self._tp_spec.mesh)
            dp_global_mesh = _mesh_resources.get_parent_mesh(dp_mesh)
            tp_global_mesh = _mesh_resources.get_parent_mesh(tp_mesh)
            if dp_global_mesh != tp_global_mesh or (
                dp_global_mesh is None or tp_global_mesh is None
            ):
                raise AssertionError(
                    "FSDP requires the DP and TP mesh to have the same parent mesh but got: \n"
                    f"DP's global mesh: {dp_global_mesh}\nTP's global mesh: {tp_global_mesh}"
                )
            self._global_mesh = dp_global_mesh
            if len(self._tp_spec.placements) != 1:
                raise NotImplementedError(
                    f"FSDP only supports 1D TP, not {self._tp_spec.placements}"
                )
            global_placements: List[Placement] = [Replicate(), Replicate()]
            global_dp_mesh_dim = _mesh_resources.get_parent_mesh_dim(dp_mesh)
            global_tp_mesh_dim = _mesh_resources.get_parent_mesh_dim(tp_mesh)
            assert global_dp_mesh_dim is not None  # mypy
            assert global_tp_mesh_dim is not None  # mypy
            # TODO: Hard code FSDP + TP; need to support HSDP + TP
            global_placements[global_dp_mesh_dim] = Shard(0)
            global_placements[global_tp_mesh_dim] = self._tp_spec.placements[0]
            self._global_placements = tuple(global_placements)
            self._global_size = self._tp_global_size
            self._global_stride = self._tp_global_stride
            param_data = cast(DTensor, param)._local_tensor
        else:
            if _mesh_resources.get_parent_mesh(self.mesh_info.mesh) is not None:
                raise NotImplementedError(
                    "Using a parent mesh with pure FSDP/HSDP is not supported"
                )
            self._global_mesh = self.mesh_info.mesh
            self._global_placements = (Shard(0),)
            self._global_size = param.size()
            self._global_stride = param.stride()
            param_data = param
        self._orig_size = param_data.size()
        shard_rank = self.mesh_info.shard_mesh_rank
        shard_world_size = self.mesh_info.shard_mesh_size
        chunks = _chunk_with_empty(param_data, shard_world_size, dim=0)
        sharded_param = chunks[shard_rank]
        self.sharded_size = sharded_param.size()
        padded_sharded_size = chunks[0].size()  # 0th always padded
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)
        if sharded_param.numel() > 0:
            padded_sharded_param[: sharded_param.size(0)].copy_(sharded_param)
        self._sharded_param_data = padded_sharded_param.view(-1)
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(padded_sharded_param[: sharded_param.size(0)])
        )
        self.sharded_param.requires_grad_(param.requires_grad)
        unsafe_free_storage(param_data)  # free immediately
        del param_data  # delete PyObject reference to avoid warning
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def _setattr_on_modules(self, tensor: torch.Tensor) -> None:
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, tensor
        )
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, tensor)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        if tensor.numel() == 0:
            # Normalize as (0) instead of possibly (0, *) for padding-only case
            tensor = tensor.view(0)
        if tensor.shape != self.sharded_size:
            _raise_assert_with_print(
                f"Expects a tensor with the sharded size {self.sharded_size} "
                f"but got {tensor.shape}"
            )
        return _from_local_no_grad(
            tensor,
            self._global_mesh,
            self._global_placements,
            self._global_size,
            self._global_stride,
        )


# NOTE: Unsafe here refers to not checking whether the storage is already
# allocated or freed, respectively. We should be safe to use them since we
# explicitly manage the state transition.
def unsafe_alloc_storage(tensor: torch.Tensor) -> None:
    # Skip the already-allocated check and assume that `tensor` is the base
    # tensor to save CPU overhead
    tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)


def unsafe_free_storage(tensor: torch.Tensor) -> None:
    # Skip the already-freed check to save CPU overhead
    tensor.untyped_storage().resize_(0)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: torch.Tensor
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = cast(nn.Parameter, param)
        super(nn.Module, module).__setattr__(param_name, param)
    else:  # slow path
        setattr(module, param_name, param)
