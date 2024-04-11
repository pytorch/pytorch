from dataclasses import dataclass, field
from enum import auto, Enum
from typing import cast, List, Optional, Tuple

import torch
import torch.nn as nn

from torch._prims_common import make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor import DTensor, Placement, Replicate, Shard
from torch.distributed._tensor.device_mesh import _mesh_resources
from torch.distributed._tensor.placement_types import DTensorSpec
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import (
    _chunk_with_empty,
    _from_local_no_grad,
    _get_dim0_chunked_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    FSDPMeshInfo,
    HSDPMeshInfo,
)

"""
[Note: FSDP tensors]
FSDP considers the following tensors:
- Original parameter: parameter passed to :class:`FSDPParam`, i.e. the one
  on the module when applying FSDP
- Sharded parameter: sharding the original parameter on dim-0 as a DTensor
  over the main mesh
- All-gather input: the ``torch.Tensor`` passed to all-gather, derived from the
  sharded parameter
- All-gather output: the ``torch.Tensor`` resulting from all-gathering the
  all-gather input
- Unsharded parameter: parameter used for forward/backward computation, derived
  from the all-gather output; autograd leaf

We define these tensors to describe the general framework that can accomodate
extensions, where:
- all-gather-input = pre-all-gather-transform(sharded-parameter)
- unsharded-parameter = post-all-gather-transform(all-gather-output)

For the default ``torch.Tensor`` case, the sharded parameter and all-gather
input share the same underlying tensor data, meaning that they can be thought
of as the same tensors. The same applies for the all-gather output and
unsharded parameter. For non-``torch.Tensor`` extensions, these equivalences
may no longer hold due to the pre/post-all-gather transforms.

[Note: FSDP and autograd]
FSDP dynamically frees and allocates the unsharded parameter. Since autograd
can pack a reference to it or a view to save for backward, we use storage
resizing to implement the freeing/allocation since that preserves the aliasing.
This implies that we construct the unsharded parameter object once and write to
it in-place thereafter. For the default ``torch.Tensor` original parameter
case, the all-gather output and unsharded parameter share the same
data, so we use storage resizing on the all-gather output.
"""


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
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

    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    contiguous_sharded_stride: Tuple[int, ...]
    padded_sharded_param_size: torch.Size  # ND
    sharded_post_forward_size: torch.Size  # ND
    contiguous_sharded_post_forward_stride: Tuple[int, ...]
    _sharded_param_data: torch.Tensor  # 1D
    sharded_param: nn.Parameter  # ND
    _sharded_post_forward_param_data: Optional[torch.Tensor]  # 1D
    _sharded_post_forward_param: Optional[nn.Parameter]  # ND
    _unsharded_param: nn.Parameter  # ND
    _global_placements: Tuple[Placement, ...]
    _global_size: torch.Size
    _global_stride: Tuple[int, ...]
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self._init_sharded_param(param, device)
        if self.post_forward_mesh_info:
            self._init_sharded_post_forward_param_metadata(param)
        self.all_gather_output = torch.empty(0)
        self.all_gather_output_storage_size = None
        self._param_fqn: Optional[str] = None  # prefixed from root module

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter, device: torch.device):
        if param.device != device and param.device.type != "meta":
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
            self._global_size = param.size()
            self._global_stride = param.stride()
            param_data = cast(DTensor, param)._local_tensor
        else:
            self._global_mesh = self.mesh_info.mesh
            if isinstance(self.mesh_info, HSDPMeshInfo):
                self._global_placements = (Replicate(), Shard(0))
            else:
                self._global_placements = (Shard(0),)
            self._global_size = param.size()
            self._global_stride = param.stride()
            param_data = param
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
        shard_rank = self.mesh_info.shard_mesh_rank
        shard_world_size = self.mesh_info.shard_mesh_size
        chunks = _chunk_with_empty(param_data, shard_world_size, dim=0)
        sharded_param = chunks[shard_rank]
        self.sharded_size = _get_dim0_chunked_size(sharded_param, param_data.size())
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        padded_sharded_size = chunks[0].size()  # 0th always padded
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)
        self.padded_sharded_param_size = padded_sharded_param.size()
        if sharded_param.numel() > 0:
            padded_sharded_param[: sharded_param.size(0)].copy_(sharded_param)
        self._sharded_param_data = padded_sharded_param.view(-1)
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(padded_sharded_param[: sharded_param.size(0)])
        )
        self.sharded_param.requires_grad_(param.requires_grad)
        # Let `param_data` be freed normally when its ref count reaches 0 when
        # the `fully_shard` call returns to allow provided parameters to alias
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None:
        mesh_info = self.post_forward_mesh_info
        assert mesh_info is not None  # mypy
        param_data = param._local_tensor if isinstance(param, DTensor) else param
        chunks = _chunk_with_empty(param_data, mesh_info.shard_mesh_size, dim=0)
        self.sharded_post_forward_size = _get_dim0_chunked_size(
            chunks[mesh_info.shard_mesh_rank], param_data.size()
        )
        self.contiguous_sharded_post_forward_stride = make_contiguous_strides_for(
            self.sharded_post_forward_size
        )

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    def init_all_gather_output(
        self,
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if self.all_gather_output.numel() > 0:
            return  # already initialized
        all_gather_output_size = torch.Size([all_gather_input_numel * world_size])
        self.all_gather_output = torch.empty(
            all_gather_output_size, dtype=dtype, device=device
        )
        self.all_gather_output_storage_size = self.all_gather_output.numel() * self.all_gather_output.itemsize

    def init_unsharded_param(self):
        if hasattr(self, "_unsharded_param"):
            return  # already initialized
        # For the default path (no post-all-gather), the all-gather output
        # gives the unsharded parameter data directly
        unsharded_param = torch.as_strided(
            self.all_gather_output,
            self._orig_size,
            make_contiguous_strides_for(self._orig_size),
            storage_offset=0,
        )
        if self.is_dtensor:
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                unsharded_param = _from_local_no_grad(
                    unsharded_param,
                    self._tp_spec.mesh,
                    self._tp_spec.placements,
                    self._global_size,
                    self._global_stride,
                )
            else:
                unsharded_param = DTensor.from_local(
                    unsharded_param,
                    self._tp_spec.mesh,
                    self._tp_spec.placements,
                    shape=self._global_size,
                    stride=self._global_stride,
                )
        self._unsharded_param = nn.Parameter(unsharded_param)
        self._unsharded_param.requires_grad_(self.sharded_param.requires_grad)

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_all_gather_output()
        self.sharded_state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.is_dtensor:
            raise NotImplementedError(
                "Resharding to smaller mesh with TP is not supported yet"
            )
        self._assert_in_states(ShardedState.UNSHARDED)
        assert self.post_forward_mesh_info is not None  # mypy
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        if (numel := self.all_gather_output.numel()) % shard_world_size != 0:
            _raise_assert_with_print(
                f"All-gather output size ({numel}) must be divisible by the shard "
                f"world size ({shard_world_size})"
            )
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        sharded_numel = numel // shard_world_size
        if torch.distributed._functional_collectives.is_torchdynamo_compiling():
            raise Exception("NYI(yf225): need to use ._unsharded_param instead of .all_gather_output for compile, to avoid having .all_gather_output as graph input")
        self._sharded_post_forward_param_data = (
            self.all_gather_output.narrow(0, sharded_numel * shard_rank, sharded_numel)
        ).clone()  # clone to be able to free all-gather output
        sharded_post_forward_tensor = torch.as_strided(
            self._sharded_post_forward_param_data,
            size=self.sharded_post_forward_size,
            stride=self.contiguous_sharded_post_forward_stride,
            storage_offset=0,
        )
        self._sharded_post_forward_param = nn.Parameter(
            self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)
        )
        self._setattr_on_modules(self._sharded_post_forward_param)
        self.free_all_gather_output()
        self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            # The data is allocated in the default stream via the post-forward
            # reshard and must be kept alive for the next all-gather copy-in.
            # Since we call this method after the copy-out, the data's lifetime
            # is ensured without further synchronization.
            self._sharded_post_forward_param = None
            self._sharded_post_forward_param_data = None  # free
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, param
        )
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        if tensor.shape != self.sharded_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_size} but got {tensor.shape}"
            )
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            ret = _from_local_no_grad(
                tensor,
                self._global_mesh,
                self._global_placements,
                self._global_size,
                self._global_stride,
            )
        else:
            ret = DTensor.from_local(
                tensor,
                self._global_mesh,
                self._global_placements,
                shape=self._global_size,
                stride=self._global_stride,
            )
        return ret

    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor:
        if tensor.shape != self.sharded_post_forward_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_post_forward_size} but got {tensor.shape}"
            )
        assert isinstance(self.post_forward_mesh_info, HSDPMeshInfo)
        # TODO: Prefer this DTensor to be read-only and generalize the
        # placement once we support TP.
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            ret = _from_local_no_grad(
                tensor,
                self.post_forward_mesh_info.mesh,
                (Replicate(), Shard(0)),
                self._global_size,
                self._global_stride,
            )
        else:
            ret = DTensor.from_local(
                tensor,
                self.post_forward_mesh_info.mesh,
                (Replicate(), Shard(0)),
                shape=self._global_size,
                stride=self._global_stride,
            )
        return ret

    def alloc_all_gather_output(self) -> None:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            unsafe_alloc_storage(self.all_gather_output, self.all_gather_output_storage_size)
        else:
            unsafe_alloc_storage(self._unsharded_param, self.all_gather_output_storage_size)

    def free_all_gather_output(self) -> None:
        # These two branches do the exact same thing underneath,
        # because .all_gather_output and ._unsharded_param share the same storage.
        # We use ._unsharded_param under compile just to avoid having .all_gather_output as graph input.
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            unsafe_free_storage(self.all_gather_output)
        else:
            unsafe_free_storage(self._unsharded_param)

    @property
    def all_gather_input(self) -> torch.Tensor:  # 1D
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
        if self.sharded_state == ShardedState.SHARDED:
            return _to_dtype_if_needed(self._sharded_param_data, self.param_dtype)
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            return _to_dtype_if_needed(
                cast(torch.Tensor, self._sharded_post_forward_param_data),
                self.param_dtype,
            )
        return torch.empty(0)  # mypy

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        self._assert_in_states(ShardedState.UNSHARDED)
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_param.grad
        assert grad is not None, "Expects unsharded_param.grad to not be None"
        return self._get_grad_inner_tensor(grad)

    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_dtensor:
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            assert isinstance(grad, DTensor), f"{type(grad)}"
            if any(pl.is_partial() for pl in grad.placements):
                placements = [
                    Replicate() if pl.is_partial() else pl for pl in grad.placements
                ]
                grad = grad.redistribute(placements=placements)
            grad = grad._local_tensor
        return grad

    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(
                f"Expects to be in one of {states}, not {self.sharded_state}"
            )


# NOTE: Unsafe here refers to not checking whether the storage is already
# allocated or freed, respectively. We should be safe to use them since we
# explicitly manage the state transition.
def unsafe_alloc_storage(tensor: torch.Tensor, new_size: int) -> None:
    # Skip the already-allocated check and assume that `tensor` is the base
    # tensor to save CPU overhead
    tensor.untyped_storage().resize_(new_size)


def unsafe_free_storage(tensor: torch.Tensor) -> None:
    # Skip the already-freed check to save CPU overhead
    tensor.untyped_storage().resize_(0)


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
