import itertools
import warnings

from enum import auto, Enum
from typing import Callable, cast, List, Optional, Tuple

import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, Placement, Shard
from torch.distributed._tensor.device_mesh import _mesh_resources
from torch.distributed._tensor.placement_types import DTensorSpec

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import (
    chunk_with_empty,
    from_local_no_grad,
    FSDP_SHARDED,
    FSDPMeshInfo,
    pad_tensor_on_dim0,
    ParamModuleInfo,
    print_and_raise_internal,
)


"""
[Note: Composing Per-Parameter FSDP with Tensor Subclasses]
FSDP manages the unsharded parameter data, all-gathering and freeing it as
needed. As such, FSDP prefers (1) to have a reference to the subclass's
underlying data and (2) a way to construct a new subclass object while
specifying the underlying data.

For DTensor, (1) is the local tensor, and (2) is ``DTensor.from_local``. For
Float8Tensor, (1) is the fp8 data, and (2) is ``Float8Tensor.__new__``.
"""


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
    This class manages a parameter with FSDP or FSDP variants applied. This
    implements dim-0 per-parameter sharding, which means that this class owns
    the sharded and unsharded tensor data.

    Attributes:
        sharded_param (nn.Parameter): This the sharded parameter to be passed
            to the optimizer.
        unsharded_param (nn.Parameter): This is the unsharded parameter used
            for forward/backward computation.
    """

    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    # Save the `<...>_view` tensors, which are always views into the
    # corresponding `<...>_data` tensor, to avoid slice calls per iteration
    _padded_unsharded_size: torch.Size  # ND
    _unsharded_size: torch.Size  # ND
    _padded_sharded_size: torch.Size  # ND
    _sharded_size: torch.Size  # ND
    _unsharded_param_data: torch.Tensor  # 1D
    _unsharded_param_view: torch.Tensor  # 1D
    _sharded_param_data: torch.Tensor  # 1D
    _sharded_param_view: torch.Tensor  # ND
    _sharded_post_forward_param_data: Optional[torch.Tensor]  # 1D
    _sharded_param: nn.Parameter
    _unsharded_param: nn.Parameter
    _cpu_sharded_grad: torch.Tensor  # pinned memory
    # For splitting autograd-computed gradient
    unsharded_chunk_numels: List[int]
    # For splitting reduce-scatter input: the next two lists have N elements
    # for world size N, where each inner list has 1 element if pure or no
    # padding or 2 elements if partial padding
    padded_unsharded_chunk_numels: List[List[int]]
    is_padding_mask: List[List[bool]]
    unsharded_accumulated_grad: Optional[torch.Tensor]
    padded_unsharded_numel: int
    padded_unsharded_bytes: int
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec
    _tp_global_size: torch.Size
    _tp_global_stride: Tuple[int, ...]
    # Float8Tensor attributes (only defined for Float8Tensor `param`)
    _float8_scale: torch.Tensor
    _float8_tensor_ctor: Callable
    _cast_w_to_float8_fn: Callable
    _is_amax_initialized_attr_name: str
    _float8_emulate_attr_name: str

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
        self._init_dtensor_attrs(param)
        self._init_float8tensor_attrs(param)
        if self._is_dtensor and self._is_float8tensor:
            # TODO: DTensor should wrap Float8Tensor, and we need to make sure
            # to unwrap twice to get the underlying data.
            raise NotImplementedError("DTensor + Float8Tensor is not supported yet")
        self._init_dtype_attrs(param, mp_policy)
        self._init_global_mesh_attrs(param)

        # Debuggability
        self._param_fqn: Optional[str] = None  # prefixed from root module

        self._init_sharded_param(param, device)
        self._init_unsharded_param()

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

    def _init_dtensor_attrs(self, param: nn.Parameter):
        self._is_dtensor = isinstance(param, DTensor)  # assume TP
        # These TP attributes are used for the unsharded parameter
        if self._is_dtensor:
            self._tp_spec = cast(DTensor, param)._spec
            self._tp_global_size = param.size()
            self._tp_global_stride = param.stride()

    def _init_float8tensor_attrs(self, param: nn.Parameter):
        self._is_float8tensor = getattr(param, "_is_fp8_weight", False)
        module = self._module_info.module
        # TODO: Leave this warning for now since we only integrate with
        # float8-experimental. We need to figure out meta-device story, where
        # tensor attributes are lost upon materializing on a real device.
        if (
            "Float8Linear" in str(type(module))
            and "weight" in self._module_info.param_name
        ) and not self._is_float8tensor:
            warnings.warn("Float8Linear.weight has _is_float8tensor == False")
        if self._is_float8tensor:
            float8_scale_name = "fp8_scale_w"
            float8_tensor_ctor_name = "_float8_tensor_ctor"
            cast_w_to_float8_fn_name = "cast_w_to_float8"
            is_amax_initialized_name = "is_amax_initialized"
            float8_emulate_name = "emulate"
            for attr_name in (
                float8_scale_name,
                float8_tensor_ctor_name,
                cast_w_to_float8_fn_name,
                is_amax_initialized_name,
                float8_emulate_name,
            ):
                if not hasattr(module, attr_name):
                    raise AssertionError(f"{type(module)} is missing {attr_name}")
            self._float8_scale = getattr(module, float8_scale_name)
            self._float8_tensor_ctor = getattr(module, float8_tensor_ctor_name)
            self._cast_w_to_float8_fn = getattr(module, cast_w_to_float8_fn_name)
            self._is_amax_initialized_attr_name = is_amax_initialized_name
            self._float8_emulate_attr_name = float8_emulate_name

    def _init_global_mesh_attrs(self, param: nn.Parameter):
        """
        Initializes attributes related to the global device mesh for the
        *sharded* parameter.
        """
        is_2d = self._is_dtensor
        assert (
            self.mesh_info.shard_mesh_dim == 0
            and self.mesh_info.replicate_mesh_dim is None
        ), "HSDP and DDP are not supported yet"
        if is_2d:
            dp_global_mesh = _mesh_resources.get_parent_mesh(self.mesh_info.mesh)
            tp_global_mesh = _mesh_resources.get_parent_mesh(self._tp_spec.mesh)
            if (
                dp_global_mesh != tp_global_mesh
                or dp_global_mesh is None
                or tp_global_mesh is None
            ):
                raise AssertionError(
                    "fully_shard requires the DP and TP mesh to have the same "
                    "parent mesh but got:\n"
                    f"DP's global mesh: {dp_global_mesh}\n"
                    f"TP's global mesh: {tp_global_mesh}"
                )
            self._global_mesh = dp_global_mesh
            if len(self._tp_spec.placements) != 1:
                raise NotImplementedError(
                    f"Only supports 1D TP, not ({self._tp_spec.placements})"
                )
            global_placements: List[Optional[Placement]] = [None, None]
            global_dp_mesh_dim = _mesh_resources.get_parent_mesh_dim(
                self.mesh_info.mesh
            )
            global_tp_mesh_dim = _mesh_resources.get_parent_mesh_dim(self._tp_spec.mesh)
            assert global_dp_mesh_dim is not None  # mypy
            assert global_tp_mesh_dim is not None  # mypy
            # TODO: Hard code inter-node FSDP + intra-node TP; need to support
            # HSDP and DDP in place of FSDP
            global_placements[global_dp_mesh_dim] = Shard(0)
            global_placements[global_tp_mesh_dim] = self._tp_spec.placements[0]
            self._global_placements = cast(
                Tuple[Placement, ...], tuple(global_placements)
            )
            self._global_size = self._tp_global_size
            self._global_stride = self._tp_global_stride
        else:
            if _mesh_resources.get_parent_mesh(self.mesh_info.mesh) is not None:
                raise NotImplementedError(
                    "For 1D DP, using a parent mesh is not supported"
                )
            self._global_mesh = self.mesh_info.mesh
            self._global_placements = (Shard(0),)
            self._global_size = param.size()
            self._global_stride = param.stride()

    @torch.no_grad()
    def _init_sharded_param(self, param: nn.Parameter, device: torch.device):
        if param.device != device:
            raise AssertionError(
                "Expects parameter to already be moved to device "
                f"{device} but got {param.device}"
            )
        param_data = param._local_tensor if isinstance(param, DTensor) else param
        self._unsharded_size = param_data.size()
        meta_param = torch.empty(
            self._unsharded_size,
            dtype=param.dtype,
            layout=param.layout,
            device="meta",
            requires_grad=param.requires_grad,
            # Assume contiguous memory layout
        )
        shard_rank = self.mesh_info.shard_mesh_rank
        shard_world_size = self.mesh_info.shard_mesh_size
        padded_param = pad_tensor_on_dim0(param_data, shard_world_size)
        padded_chunks = torch.chunk(padded_param, shard_world_size, dim=0)
        if self.offload_to_cpu:  # ignore `device`
            self._sharded_param_data = param_data.new_empty(
                padded_chunks[0].size(), device="cpu"
            ).pin_memory()
        else:
            self._sharded_param_data = param_data.new_empty(
                padded_chunks[0].size(), device=device
            )
        if device != torch.device("meta"):
            self._sharded_param_data.copy_(padded_chunks[shard_rank])
        chunks = chunk_with_empty(meta_param, shard_world_size, dim=0)
        self._sharded_size = chunks[shard_rank].size()
        self._padded_sharded_size = padded_chunks[0].size()
        self._padded_unsharded_size = padded_param.size()
        padded_chunk_numel = padded_chunks[0].numel()
        self.padded_unsharded_chunk_numels: List[List[int]] = []
        self.is_padding_mask: List[List[bool]] = []
        self.unsharded_chunk_numels = []
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
        self._sharded_param_view = self._sharded_param_data[: self._sharded_size[0]]
        self._sharded_param_data = self._sharded_param_data.view(-1)
        self._sharded_param = nn.Parameter(
            self.to_sharded_dtensor(self._sharded_param_view)
        )
        self._sharded_param.requires_grad_(param.requires_grad)
        self._unsharded_param_data = param_data  # HACK: for `to_sharded()`
        if not self.offload_to_cpu:
            assert self._sharded_param.device == device
        else:
            assert self._sharded_param.device == torch.device("cpu")
            self._cpu_sharded_grad = torch.empty_like(
                self._sharded_param_view
            ).pin_memory()  # no padding
        setattr(param, FSDP_SHARDED, True)
        setattr(self._sharded_param, FSDP_SHARDED, True)
        self.to_sharded()

    @torch.no_grad()
    def _init_unsharded_param(self):
        unsharded_param_dtype = (
            torch.float8_e4m3fn
            if self._is_float8tensor
            else (self.param_dtype or self.sharded_param.dtype)
        )
        self._unsharded_param_data = torch.empty(
            self._padded_unsharded_size,
            dtype=unsharded_param_dtype,
            layout=self.sharded_param.layout,
            device=self.device,
            requires_grad=False,
        ).view(-1)
        self._unsharded_param_view = self._unsharded_param_data[
            : self._unsharded_size.numel()
        ].view(self._unsharded_size)
        # TODO: Support DTensor + Float8Tensor, where we expect DTensor to wrap
        # Float8Tensor.
        if self._is_dtensor:
            self._unsharded_param = nn.Parameter(
                from_local_no_grad(
                    self._unsharded_param_view,
                    self._tp_spec.mesh,
                    self._tp_spec.placements,
                    self._tp_global_size,
                    self._tp_global_stride,
                )
            )
            self._unsharded_param.requires_grad_(self.sharded_param.requires_grad)
        elif self._is_float8tensor:
            # `orig_dtype` should be the dtype for autograd-computed gradients
            orig_dtype = self.param_dtype or self.orig_dtype
            emulate = getattr(self._module_info.module, self._float8_emulate_attr_name)
            float8_param = self._float8_tensor_ctor(
                data=self._unsharded_param_view,
                scale=self._float8_scale,
                orig_dtype=orig_dtype,
                emulate=emulate,
            )
            # Unlike the normal FSDP flow, the float8 weight is transient and
            # not an `nn.Parameter`.
            self._unsharded_param = float8_param.requires_grad_(
                self.sharded_param.requires_grad
            )
        else:
            self._unsharded_param = nn.Parameter(self._unsharded_param_view)
            self._unsharded_param.requires_grad_(self.sharded_param.requires_grad)
        # Unsharded accumulated gradient used for gradient accumulation without
        # reduce-scatter when `reduce_dtype` is specified
        self.unsharded_accumulated_grad = None
        # Precompute these (used for foreach all-gather) to reduce overhead
        self.padded_unsharded_numel = self._padded_unsharded_size.numel()
        self.padded_unsharded_bytes = (
            self.padded_unsharded_numel * self.unsharded_param_data_dtype.itemsize
        )
        unsafe_free_storage(self._unsharded_param_data)
        setattr(self._unsharded_param, FSDP_SHARDED, True)

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        if self._is_float8tensor:
            self._delattr_on_modules("_w_fp8")
        unsafe_free_storage(self._unsharded_param_data)
        self.state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.state != ShardedState.UNSHARDED:
            print_and_raise_internal(
                f"Expects state to be unsharded but got {self.state}"
            )
        assert self.post_forward_mesh_info is not None  # mypy
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        chunks = chunk_with_empty(
            self._unsharded_param_data.view(self._padded_unsharded_size),
            shard_world_size,
            dim=0,
        )
        # NOTE: This constructs a new Tensor object.
        self._sharded_post_forward_param_data = chunks[shard_rank].clone()
        self._setattr_on_modules(self._sharded_post_forward_param_data, as_param=False)
        if self._is_float8tensor:
            self._delattr_on_modules("_w_fp8")
        # Do not strip padding here since this resharded parameter should never
        # be used in any ops and is only meant as a temporary storage
        unsafe_free_storage(self._unsharded_param_data)
        self.state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered (e.g. from
        # the foreach all-gather)
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        if self._is_float8tensor:
            self._setattr_on_modules(
                self._unsharded_param, name_override="_w_fp8", as_param=False
            )
        else:
            self._setattr_on_modules(self._unsharded_param)
        if self.state == ShardedState.SHARDED_POST_FORWARD:
            # The sharded post-forward parameter data is allocated in the
            # default stream via the post-forward reshard and needs to be kept
            # alive until after the copy-in for the next all-gather. Since this
            # method is only called in the copy-out after waiting for the
            # all-gather to finish (via an event wait), this data's lifetime is
            # ensured without needing further synchronization.
            self._sharded_post_forward_param_data = None  # free
        self.state = ShardedState.UNSHARDED

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

    def alloc_unsharded_param(self) -> None:
        unsafe_alloc_storage(self._unsharded_param_data, self._padded_unsharded_size)

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
        if tensor.shape != self._sharded_size and not (
            # For size-0 padding, DTensor can flatten from (0, *) to (0)
            tensor.numel() == 0
            and self._sharded_size.numel() == 0
        ):
            print_and_raise_internal(
                f"Expects a tensor with the sharded size {self._sharded_size} "
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
        if self.state == ShardedState.SHARDED:
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            if self._is_float8tensor:
                # Unlike normal mixed precision, for fp8, we must explicitly
                # call a function to cast from fp32 -> fp8 rather than relying
                # on the cast from implicitly copying from e.g. fp32 -> bf16
                sharded_param_data_float8 = self._cast_w_to_float8_fn(
                    sharded_param_data,
                    getattr(
                        self._module_info.module, self._is_amax_initialized_attr_name
                    ),
                )
                sharded_param_data = sharded_param_data_float8._data
            return sharded_param_data
        elif self.state == ShardedState.SHARDED_POST_FORWARD:
            return cast(torch.Tensor, self._sharded_post_forward_param_data)
        return torch.empty(0)  # mypy

    @property
    def all_gather_input_numel(self) -> int:
        # NOTE: This property only exists to avoid recomputing the fp32 ->
        # fp8 cast in `all_gather_input` to get the numel for the fp8 path.
        self._assert_in_sharded_state()
        if self.state == ShardedState.SHARDED:
            return self._sharded_param_data.numel()
        elif self.state == ShardedState.SHARDED_POST_FORWARD:
            return cast(torch.Tensor, self._sharded_post_forward_param_data).numel()
        return 0  # mypy

    @property
    def sharded_param(self) -> nn.Parameter:
        return self._sharded_param

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
        if self._is_dtensor:
            grad = cast(DTensor, grad)._local_tensor
        return grad

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_accumulated_grad
        assert grad is not None, "Expects unsharded_accumulated_grad to not be None"
        if self._is_dtensor:
            grad = cast(DTensor, grad)._local_tensor
        return grad

    @property
    def unsharded_param_data_dtype(self) -> torch.dtype:
        # This represents the dtype of the *underlying data* and is used for
        # all-gathering. For fp8 weight matrices, this is e4m3fn. For non-fp8
        # parameters, this is `param_dtype` if it was specified and the
        # original dtype otherwise.
        if self._is_float8tensor:
            return self._unsharded_param._data.dtype  # type: ignore[attr-defined]
        return self._unsharded_param.dtype

    def _assert_in_sharded_state(self) -> None:
        if self.state not in (ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD):
            print_and_raise_internal(
                f"Expects to be in a sharded state, not {self.state}"
            )

    def _assert_in_unsharded_state(self) -> None:
        if self.state != ShardedState.UNSHARDED:
            print_and_raise_internal(
                f"Expects to be in the UNSHARDED state, not {self.state}"
            )


# NOTE: Unsafe here refers to not checking whether the storage is already
# allocated or freed, respectively. We should be safe to use them since we
# explicitly manage the state transition.
def unsafe_alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    # Skip the already-allocated check to save CPU overhead
    tensor.untyped_storage().resize_(size.numel() * tensor.element_size())


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
