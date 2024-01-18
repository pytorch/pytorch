import itertools

from enum import auto, Enum
from typing import cast, List, Optional, Tuple

import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, Placement, Replicate, Shard
from torch.distributed._tensor.device_mesh import _mesh_resources
from torch.distributed._tensor.placement_types import DTensorSpec

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import (
    chunk_with_empty,
    from_local_no_grad,
    FSDP_SHARDED,
    FSDPMeshInfo,
    get_dim0_padded_size,
    ParamModuleInfo,
    print_and_raise_internal,
)


class ShardedState(Enum):
    """
    - ``SHARDED``: The parameter's data is sharded across the main mesh's data
      parallel sharding dimension. This data is always in memory for all
      sharded states.
    - ``SHARDED_POST_FORWARD``: The data is additionally sharded across the
      post-forward mesh's data parallel sharding dimension so that the
      pre-backward all-gather can use this post-forward mesh instead of the
    main mesh.
    - ``UNSHARDED``: The data is additionally unsharded across the main mesh.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


class FSDPParam:
    """
    This class manages a parameter with FSDP or FSDP variants applied,
    implementing dim-0 per-parameter sharding.

    Attributes:
        sharded_param (nn.Parameter): This the sharded parameter to be passed
            to the optimizer.
        unsharded_param (nn.Parameter): This is the unsharded parameter used
            for forward/backward computation.
    """

    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    # Sharded/unsharded parameter attributes
    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    _sharded_param_data: torch.Tensor  # 1D
    _sharded_post_forward_param_data: Optional[torch.Tensor]  # 1D
    sharded_param: nn.Parameter  # ND
    all_gather_output: torch.Tensor  # 1D
    _unsharded_param: nn.Parameter  # ND
    cpu_sharded_grad: torch.Tensor  # unpadded, ND, pinned memory
    # For splitting autograd-computed gradient
    unsharded_chunk_numels: List[int]
    # For splitting reduce-scatter input: the next two lists have N elements
    # for world size N, where each inner list has 1 element if pure or no
    # padding or 2 elements if partial padding
    padded_unsharded_chunk_numels: List[List[int]]
    is_padding_mask: List[List[bool]]
    unsharded_accumulated_grad: Optional[torch.Tensor]
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec
    _tp_global_size: torch.Size
    _tp_global_stride: Tuple[int, ...]

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device

        self._init_offload_attrs(offload_policy)
        self._init_dtype_attrs(param, mp_policy)

        # Debuggability
        self._param_fqn: Optional[str] = None  # prefixed from root module

        self._init_sharded_param(param, device)
        self.all_gather_output = torch.empty(0)

    def _init_offload_attrs(self, offload_policy: OffloadPolicy):
        self.offload_to_cpu: bool = offload_policy.offload_type == "cpu"
        self._grad_offload_event: Optional[torch.cuda.Event] = None

    def _init_dtype_attrs(self, param: nn.Parameter, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = param.dtype
        # Each saved dtype attribute should only be not `None` if it affects
        # behavior (e.g. requiring casting); otherwise, clamp to `None`
        if param_dtype is not None and param_dtype == self.orig_dtype:
            # E.g. orig=compute=bf16
            param_dtype = None
        # By default, gradients are computed in the compute dtype
        if reduce_dtype is not None and (
            # E.g. orig=compute=reduce=bf16
            (param_dtype is None and reduce_dtype == self.orig_dtype)
            # E.g. orig=fp32, compute=reduce=bf16
            or (param_dtype is not None and reduce_dtype == param_dtype)
        ):
            reduce_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter, device: torch.device):
        if param.device != device:
            raise AssertionError(
                "Expects parameter to already be moved to device "
                f"{device} but got {param.device}"
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
        chunks = chunk_with_empty(param_data, shard_world_size, dim=0)
        sharded_param = chunks[shard_rank]
        self.sharded_size = sharded_param.size()
        padded_sharded_size = chunks[0].size()  # 0th always padded
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)
        if sharded_param.numel() > 0:
            padded_sharded_param[: sharded_param.size(0)].copy_(sharded_param)
        if self.offload_to_cpu:
            padded_sharded_param = padded_sharded_param.cpu().pin_memory()
            self.cpu_sharded_grad = sharded_param.new_zeros(
                sharded_param.size(), device="cpu"
            ).pin_memory()
        self._sharded_param_data = padded_sharded_param.view(-1)
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(padded_sharded_param[: sharded_param.size(0)])
        )
        self.sharded_param.requires_grad_(param.requires_grad)
        unsafe_free_storage(param_data)  # free immediately
        del param_data  # delete PyObject reference to avoid warning
        setattr(self.sharded_param, FSDP_SHARDED, True)
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    @torch.no_grad()
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

    @torch.no_grad()
    def init_unsharded_param(self):
        if hasattr(self, "_unsharded_param"):
            return  # already initialized
        # For the default path (no post-all-gather), the all-gather output
        # gives the unsharded parameter data directly
        world_size = self.mesh_info.shard_mesh_size
        padded_unsharded_param_size = get_dim0_padded_size(self._orig_size, world_size)
        padded_unsharded_param = self.all_gather_output.view(
            padded_unsharded_param_size
        )
        unsharded_param = padded_unsharded_param[: self._orig_size[0]]
        if self.is_dtensor:
            unsharded_param = from_local_no_grad(
                unsharded_param,
                self._tp_spec.mesh,
                self._tp_spec.placements,
                self._tp_global_size,
                self._tp_global_stride,
            )
        self._unsharded_param = nn.Parameter(unsharded_param)
        self._unsharded_param.requires_grad_(self.sharded_param.requires_grad)
        # Unsharded accumulated gradient used for gradient accumulation without
        # reduce-scatter when `reduce_dtype` is specified
        self.unsharded_accumulated_grad = None
        # Compute some additional metadata to use `torch.split` in the
        # reduce-scatter copy-in
        self.padded_unsharded_chunk_numels: List[List[int]] = []
        self.is_padding_mask: List[List[bool]] = []
        self.unsharded_chunk_numels = []
        unsharded_param_tensor = (
            cast(DTensor, unsharded_param)._local_tensor
            if self.is_dtensor
            else unsharded_param
        )
        chunks = chunk_with_empty(unsharded_param_tensor, world_size, dim=0)
        padded_chunks = torch.chunk(padded_unsharded_param, world_size, dim=0)
        padded_chunk_numel = padded_chunks[0].numel()
        for chunk_idx, chunk in enumerate(chunks):
            chunk_numel = chunk.numel()
            self.unsharded_chunk_numels.append(chunk_numel)
            if chunk_numel != padded_chunk_numel:
                if chunk_numel == 0:  # pure padding
                    self.padded_unsharded_chunk_numels.append([padded_chunk_numel])
                    self.is_padding_mask.append([True])
                else:  # partial padding
                    padding_numel = (
                        padded_chunks[0].size(0) - chunk.size(0)
                    ) * padded_chunks[0][0].numel()
                    numels = [chunk_numel, padding_numel]
                    self.padded_unsharded_chunk_numels.append(numels)
                    self.is_padding_mask.append([False, True])
            else:  # no padding
                self.padded_unsharded_chunk_numels.append([chunk_numel])
                self.is_padding_mask.append([False])

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_all_gather_output()
        self.sharded_state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        self._assert_in_unsharded_state()
        assert self.post_forward_mesh_info is not None  # mypy
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        if self.all_gather_output.numel() % shard_world_size != 0:
            print_and_raise_internal(
                f"All-gather output size ({self.all_gather_output.numel()}) must "
                f"be divisible by the shard world size ({shard_world_size})"
            )
        chunks = torch.chunk(self.all_gather_output, shard_world_size, dim=0)
        # NOTE: This constructs a new Tensor object.
        self._sharded_post_forward_param_data = chunks[shard_rank].clone()
        self._setattr_on_modules(self._sharded_post_forward_param_data, as_param=False)
        # Do not strip padding here since this resharded parameter should never
        # be used in any ops and is only meant as a temporary storage
        self.free_all_gather_output()
        self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered (e.g. from
        # the foreach all-gather)
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            # The sharded post-forward parameter data is allocated in the
            # default stream via the post-forward reshard and needs to be kept
            # alive until after the copy-in for the next all-gather. Since this
            # method is only called in the copy-out after waiting for the
            # all-gather to finish (via an event wait), this data's lifetime is
            # ensured without needing further synchronization.
            self._sharded_post_forward_param_data = None  # free
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(
        self,
        tensor: torch.Tensor,
        name_override: Optional[str] = None,
        as_param: bool = True,
    ) -> None:
        setter = unsafe_setattr_param if as_param else unsafe_setattr_tensor
        setter(
            self._module_info.module,
            name_override or self._module_info.param_name,
            tensor,
        )
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            setter(shared_module, name_override or shared_param_name, tensor)

    def _delattr_on_modules(self, attr_name: str) -> None:
        for module in itertools.chain(
            [self._module_info.module], self._module_info.shared_modules
        ):
            if hasattr(module, attr_name):
                delattr(module, attr_name)

    def alloc_all_gather_output(self) -> None:
        unsafe_alloc_storage(self.all_gather_output)

    def free_all_gather_output(self) -> None:
        unsafe_free_storage(self.all_gather_output)

    def to_accumulated_grad_if_needed(self) -> None:
        # Access `_unsharded_param` to bypass the sharded state check since we
        # prefer to reshard before upcasting the gradient to save memory
        if (
            self.reduce_dtype is None
            or self._unsharded_param.grad is None
            or self._unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            self.unsharded_accumulated_grad += self.unsharded_param.grad
            self.unsharded_param.grad = None

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the *sharded* parameter or
        *sharded* gradient to DTensor.
        """
        if tensor.shape != self.sharded_size and not (
            # For size-0 padding, DTensor can flatten from (0, *) to (0)
            tensor.numel() == 0
            and self.sharded_size.numel() == 0
        ):
            print_and_raise_internal(
                f"Expects a tensor with the sharded size {self.sharded_size} "
                f"but got {tensor.shape}"
            )
        return from_local_no_grad(
            tensor,
            self._global_mesh,
            self._global_placements,
            self._global_size,
            self._global_stride,
        )

    @property
    def all_gather_input(self) -> torch.Tensor:  # 1D
        self._assert_in_sharded_state()
        if self.sharded_state == ShardedState.SHARDED:
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            return sharded_param_data
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            return cast(torch.Tensor, self._sharded_post_forward_param_data)
        return torch.empty(0)  # mypy

    @property
    def unsharded_param(self) -> nn.Parameter:
        # Include this assertion to avoid inadvertent accesses to the unsharded
        # parameter while the data is not present
        self._assert_in_unsharded_state()
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_param.grad
        assert grad is not None, "Expects unsharded_param.grad to not be None"
        if self.is_dtensor:
            grad = cast(DTensor, grad)._local_tensor
        return grad

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_accumulated_grad
        assert grad is not None, "Expects unsharded_accumulated_grad to not be None"
        if self.is_dtensor:
            grad = cast(DTensor, grad)._local_tensor
        return grad

    def _assert_in_sharded_state(self) -> None:
        if self.sharded_state not in (
            ShardedState.SHARDED,
            ShardedState.SHARDED_POST_FORWARD,
        ):
            print_and_raise_internal(
                f"Expects to be in a sharded state, not {self.sharded_state}"
            )

    def _assert_in_unsharded_state(self) -> None:
        if self.sharded_state != ShardedState.UNSHARDED:
            print_and_raise_internal(
                f"Expects to be in the UNSHARDED state, not {self.sharded_state}"
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


# NOTE: These are hacks to bypass `nn.Module.__setattr__` checks, which incur
# non-trivial CPU overhead. We do not need to do those checks repeatedly.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: torch.Tensor
) -> None:
    module._parameters[param_name] = cast(nn.Parameter, param)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, param)


def unsafe_setattr_tensor(
    module: nn.Module, param_name: str, tensor: torch.Tensor
) -> None:
    module._parameters.pop(param_name, None)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, tensor)


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    # Only call `requires_grad_` if needed to avoid the Python <> C++ context
    # switch overhead
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
