# mypy: allow-untyped-defs
import inspect
import itertools
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from ._fsdp_common import (
    _chunk_with_empty,
    _from_local_no_grad,
    _get_dim_chunked_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
    FSDPMeshInfo,
    HSDPMeshInfo,
)


"""
[Note: FSDP tensors]
FSDP considers the following tensors:
- Original parameter: parameter passed to :class:`FSDPParam`, i.e. the one
  on the module when applying FSDP
- Sharded parameter: sharding the original parameter on dim-0 (or a
  user-specified dim) as a DTensor over the main mesh
- All-gather inputs: the ``torch.Tensor`` or ``Tensor`` s passed to all-gather,
  derived from the sharded parameter
- All-gather output: the ``torch.Tensor`` or ``Tensor`` s resulting from
  all-gathering the all-gather inputs
- Unsharded parameter: parameter used for forward/backward computation, derived
  from the all-gather output; autograd leaf

We define these tensors to describe the general framework that can accomodate
extensions, where:
- all-gather-inputs = pre-all-gather-transform(sharded-parameter)
- unsharded-parameter = post-all-gather-transform(all-gather-outputs)

For the default ``torch.Tensor`` case, there is only one all-gather input, and
it shares the same underlying tensor data as the sharded parameter, meaning
that they can be thought of as the same tensors. The same applies for the
all-gather output and unsharded parameter. For non-``torch.Tensor`` extensions,
these equivalences may no longer hold due to the pre/post-all-gather
transforms, and some may have multiple all-gather inputs/outputs (e.g.
quantized data and scales).

[Note: FSDP and autograd]
FSDP dynamically frees and allocates the unsharded parameter. Since autograd
can pack a reference to it or a view to save for backward, we use storage
resizing to implement the freeing/allocation since that preserves the aliasing.
This implies that we construct the unsharded parameter object once and write to
it in-place thereafter. For the default ``torch.Tensor` original parameter
case, the all-gather output and unsharded parameter share the same
data, so we use storage resizing on the all-gather output.
"""

lib = torch.library.Library("fsdp", "FRAGMENT")  # noqa: TOR901

lib.define("copy_(Tensor(a!) tensor, Tensor data) -> ()")


@torch.library.impl(lib, "copy_", "Meta")
@torch.library.impl(lib, "copy_", "CUDA")
@torch.library.impl(lib, "copy_", "CPU")
def copy_(tensor, data):
    tensor.copy_(data)


"""
[Note: Avoiding functionalization for fsdp.copy_ and inductor.resize_storage_bytes_]

Currently we don't functionalize `fsdp.copy_` op or `inductor.resize_storage_bytes_` op
(i.e. they show up as a mutation op in the middle of the AOT joint graph).

Reason:
Traceable FSDP2 compiled autograd BWD graph have the following traits:
(1) Two inputs of the graph were aliased to each other (one from hook closed-over tensors, one from FWD saved tensors).
(2) One of them is mutated (copy_ and resize_ to handle the all-gathered param).
(3) They are both subclasses.
The combination of these traits is not supported by AOTAutograd (it's difficult to reason about subclass aliasing).
So this doesn't work at all for Traceable FSDP2.

The compromise we use is to avoid functionalization for the FSDP2 copy_ and resize_ ops.
This avoids the problem above, because from AOTAutograd point-of-view there are no mutations
that functionalization needs to handle. (Although we need to be careful not to DCE those mutable ops.)

We can avoid this functionalization because:
(1) The nn.Parameter is never used before its .copy_() is called in eager code (i.e. no alias of it is created),
so it's safe to call .copy_() in the middle of the graph to update its content and start using the nn.Parameter downstream.
(2) We always re-allocate the buffer for nn.Parameter to store the AllGather output and to be used in downstream user ops.
So calling resize-to-0 in the middle of the graph to free nn.Parameter memory after use should always be okay
(since we always allocate anew next time we need it, we strictly don't need to keep the old tensor storage around anymore).

Q: Wouldn't the extra resize_ and copy_ ops hurt both memory usage and performance?
A: Yes it would. As an optimization, we have an Inductor post-grad FX pass to remove those resize_ and copy_ ops
for unsharded params that have this pattern: resize_(full) -> copy_ -> resize_(0).

TODO:
Now that we are maintaining the invariant of "no aliased + mutated graph inputs" in both the forward and backward,
it is now more feasible to functionalize all of the mutable FSDP ops. Some of the pros and cons are:

Cons (of functionalizing those ops):
(1) By not functionalizing them as we are today, we are making it more likely that they will run at the "correct" time
in the generated code. If we start to functionalize them, we will need to make sure that Inductor reinplaces them
in a way where it properly moves the mutations back to exactly where they should have run, or we risk suffering worse
peak memory than eager. (We probably already need to do something similar in Inductor's reinplacing for copy_:
https://github.com/pytorch/pytorch/issues/135305#issuecomment-2334888089)

Pros (of functionalizing):
(1) Better safety, we don't need to worry about the graph passes in inductor/partitioning handling input mutations
mid-graph quite as much (to be fair we've already done some amount of auditing, but we might have to do some more).
(2) Better perf: each mutation midway through the graph prevents Inductor from pattern matching across it.
But maybe there are few enough mutations induced by FSDP for this to matter.
"""


@torch.library.impl(lib, "copy_", "Functionalize")
def copy__functionalize(tensor, data):
    torch._sync(tensor)
    torch._sync(data)
    tensor_inner = torch._from_functional_tensor(tensor)
    data_inner = torch._from_functional_tensor(data)
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    ):
        torch.ops.fsdp.copy_.default(tensor_inner, data_inner)


torch.fx.node.has_side_effect(torch.ops.fsdp.copy_.default)


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


@dataclass
class ExtensionsData:
    # User-defined metadata passed from pre to post-all-gather
    all_gather_metadata: Optional[Any] = None
    # Save the all-gather input sizes to unflatten the all-gather outputs to ND
    all_gather_input_sizes: Sequence[torch.Size] = ()  # ND

    def clear(self):
        self.all_gather_metadata = None
        self.all_gather_input_sizes = ()


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
    unsharded_accumulated_grad: Optional[torch.Tensor]  # ND
    _sharding_spec: DTensorSpec
    # DTensor attributes (only defined for DTensor `param`):
    _tp_spec: DTensorSpec
    all_gather_outputs: List[torch.Tensor]  # 1D
    # All-gather extension attributes
    _extensions_data: ExtensionsData
    _unsharded_inner_tensors: List[torch.Tensor]

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]],
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: Optional[torch.Event] = None
        self._init_sharded_param(param, device, shard_placement_fn)
        if self.post_forward_mesh_info:
            self._init_sharded_post_forward_param_metadata(param)
        self._init_extensions()
        self.all_gather_outputs: List[torch.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None  # prefixed from root module
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

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
                    "FSDP requires the DP and TP mesh to have the same parent mesh but got: \n"
                    f"DP's global mesh: {dp_global_mesh}\nTP's global mesh: {tp_global_mesh}"
                )
            name_dims_error = "FSDP requires named DeviceMesh dims for ND parallelism"
            assert dp_mesh.mesh_dim_names is not None, name_dims_error
            assert tp_mesh.mesh_dim_names is not None, name_dims_error
            submesh_names = dp_mesh.mesh_dim_names + tp_mesh.mesh_dim_names
            self._spmd_mesh = dp_global_mesh[submesh_names]
            if len(self._tp_spec.placements) != 1:
                raise NotImplementedError(
                    f"FSDP only supports 1D TP, not {self._tp_spec.placements}"
                )
            split_factor = self._tp_spec.num_shards_map[shard_dim]
            assert (
                2 <= self._spmd_mesh.ndim <= 3
            ), f"_spmd_mesh.ndim can only be 2 or 3 but got {self._spmd_mesh.ndim}."
            self._spmd_placements: Tuple[Placement, ...]
            dp_shard_tp_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else fsdp_placement
                ),
                self._tp_spec.placements[0],
            )
            if self._spmd_mesh.ndim == 2:
                self._spmd_placements = dp_shard_tp_placement
            else:
                assert self.mesh_info.replicate_mesh_dim == 0
                self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=self._tp_spec.tensor_meta,
            )
            # TODO: Enable uneven sharding for FSDP+TP.
            if split_factor > 1:  # FSDP has strided sharding on tensor dim 0
                num_shards = self._sharding_spec.num_shards_map[0]
                tensor_size_dim_0 = self._sharding_spec.shape[0]
                if tensor_size_dim_0 % num_shards != 0:
                    raise NotImplementedError(
                        "FSDP+TP sharding does not support uneven sharding for now: "
                        f"tensor dim 0 has size {tensor_size_dim_0} which cannot be "
                        f"evenly sharded into {num_shards} shards."
                    )
            param_data = cast(DTensor, param)._local_tensor
        else:
            self._spmd_mesh = self.mesh_info.mesh
            if isinstance(self.mesh_info, HSDPMeshInfo):
                self._spmd_placements = (Replicate(), fsdp_placement)
            else:
                self._spmd_placements = (fsdp_placement,)
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
        shard_rank = self.mesh_info.shard_mesh_rank
        shard_world_size = self.mesh_info.shard_mesh_size
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
                padded_sharded_param = padded_sharded_param.pin_memory(
                    device=self.device
                )
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

    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None:
        mesh_info = self.post_forward_mesh_info
        assert mesh_info is not None  # mypy
        param_data = param._local_tensor if isinstance(param, DTensor) else param
        chunks = _chunk_with_empty(param_data, mesh_info.shard_mesh_size, dim=0)
        self.sharded_post_forward_size = _get_dim_chunked_size(
            chunks[mesh_info.shard_mesh_rank],
            param_data.size(),
            dim=self.fsdp_placement.dim,
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

    def _init_extensions(self) -> None:
        inner_tensor = self._sharded_local_tensor
        has_fsdp_pre_all_gather = hasattr(inner_tensor, "fsdp_pre_all_gather")
        has_fsdp_post_all_gather = hasattr(inner_tensor, "fsdp_post_all_gather")
        if has_fsdp_pre_all_gather != has_fsdp_post_all_gather:
            raise AssertionError(
                "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
                f"if using all-gather extensions: {inner_tensor}"
            )
        if has_fsdp_pre_all_gather:
            self._extensions_data = ExtensionsData()
        self._unsharded_inner_tensors: List[torch.Tensor] = []

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: List[int],
        all_gather_input_dtypes: List[torch.dtype],
        world_size: int,
        device: torch.device,
        force_recreate: bool = False,
    ):
        if not force_recreate and len(self.all_gather_outputs) > 0:
            return  # already initialized
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self):
        """
        [Note: Invariants for torch.compile Traceable FSDP2]
        1. Under compile, we always re-populate the content of `self._unsharded_param`
           per AllGather using the slow path.
        2. Under compile, we always recreate `self.all_gather_outputs` per AllGather.
           This is to ensure the buffer creation is internal to the graph and
           avoid `self.all_gather_outputs` being captured as a graph input.
        3. Under compile, at the end of `free_unsharded_param()`, we always clean up
           `self.all_gather_outputs` and `self._unsharded_inner_tensors`,
           to avoid them being captured as graph output.

        With these invariants, only these tensors will be inputs to the graph:
        - Sharded parameters
        - Placeholders for the `self._unsharded_param` nn.Parameter
        """
        if not compiled_autograd_enabled() and hasattr(
            self, "_unsharded_param"
        ):  # after the 1st all-gather
            inner_tensor = self._sharded_local_tensor
            if not hasattr(inner_tensor, "fsdp_post_all_gather"):
                return  # already initialized
            for tensor in self._unsharded_inner_tensors:
                alloc_storage(tensor)
            all_gather_outputs = self._unflatten_all_gather_outputs()
            inner_tensor.fsdp_post_all_gather(
                all_gather_outputs,
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
                out=self._unsharded_param,
            )
            self._extensions_data.clear()
            return
        inner_tensor = self._sharded_local_tensor
        if not compiled_autograd_enabled() and hasattr(
            inner_tensor, "fsdp_post_all_gather"
        ):
            all_gather_outputs = self._unflatten_all_gather_outputs()
            (
                unsharded_tensor,
                self._unsharded_inner_tensors,
            ) = inner_tensor.fsdp_post_all_gather(
                all_gather_outputs,
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
            )
            self._extensions_data.clear()
        else:
            # For the default path (no post-all-gather), the all-gather output
            # gives the unsharded parameter data directly
            assert len(self.all_gather_outputs) == 1, f"{len(self.all_gather_outputs)}"
            unsharded_tensor = self.all_gather_outputs[0]
        unsharded_param = torch.as_strided(
            unsharded_tensor,
            self._orig_size,
            self._contiguous_orig_stride,
            storage_offset=0,
        )
        if self.is_dtensor:
            unsharded_param = _from_local_no_grad(unsharded_param, self._tp_spec)
        if hasattr(self, "_unsharded_param"):
            assert compiled_autograd_enabled()
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
                self._unsharded_param
            ):
                # NOTE: Under compile, if an unsharded param goes through
                # resize_(full) -> copy_ -> resize_(0) pattern, we will remove those
                # resize_ and copy_ ops in a compiler graph pass
                # `remove_fsdp2_unsharded_param_graph_input_usage` to recover performance.
                self._unsharded_param.untyped_storage().resize_(
                    self._unsharded_param.numel() * self._unsharded_param.itemsize
                )
                torch.ops.fsdp.copy_(self._unsharded_param, unsharded_param)
        else:
            self._unsharded_param = nn.Parameter(
                unsharded_param, requires_grad=self.sharded_param.requires_grad
            )

    def _unflatten_all_gather_outputs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(
            t.view(-1, *s[1:])
            for t, s in zip(
                self.all_gather_outputs, self._extensions_data.all_gather_input_sizes
            )
        )

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.is_dtensor:
            raise NotImplementedError(
                "Resharding to smaller mesh with TP is not supported yet"
            )
        self._assert_in_states(ShardedState.UNSHARDED)
        assert self.post_forward_mesh_info is not None  # mypy
        assert len(self.all_gather_outputs) == 1
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        if (numel := self.all_gather_outputs[0].numel()) % shard_world_size != 0:
            _raise_assert_with_print(
                f"All-gather output size ({numel}) must be divisible by the shard "
                f"world size ({shard_world_size})"
            )
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        sharded_numel = numel // shard_world_size
        self._sharded_post_forward_param_data = (
            self.all_gather_outputs[0].narrow(
                0, sharded_numel * shard_rank, sharded_numel
            )
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
        self.free_unsharded_param()
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
        return _from_local_no_grad(
            tensor,
            self._sharding_spec,
        )

    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor:
        if tensor.shape != self.sharded_post_forward_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_post_forward_size} but got {tensor.shape}"
            )
        assert isinstance(self.post_forward_mesh_info, HSDPMeshInfo)
        # TODO: Prefer this DTensor to be read-only and generalize the
        # placement once we support TP.
        post_forward_sharding_spec = DTensorSpec(
            self.post_forward_mesh_info.mesh,
            (Replicate(), Shard(0)),
            tensor_meta=self._sharding_spec.tensor_meta,
        )
        return _from_local_no_grad(tensor, post_forward_sharding_spec)

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

    def alloc_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            alloc_storage(tensor)

    def free_unsharded_param(self) -> None:
        if compiled_autograd_enabled():
            """
            Assumptions under compile:
            - `self._unsharded_param` is NOT an alias of `self.all_gather_outputs`.
            Instead, we resize `self._unsharded_param` storage size to full and then
            explicitly *copy* the data from `self.all_gather_outputs` to `self._unsharded_param`
            in `init_unsharded_param()`. (For full-graph FSDP2 case, we will then remove
            the resize_ and copy_ ops in a compiler graph pass to recover performance.)
            - `self.all_gather_outputs` and `self._unsharded_inner_tensors` are NOT
            graph inputs. They are created within the graph and is guaranteed to be freed
            by the end of the graph. They don't leak outside of the graph.
            """
            self._unsharded_param.untyped_storage().resize_(0)
            self.all_gather_outputs = []
            self._unsharded_inner_tensors = []
        else:
            for tensor in itertools.chain(
                self.all_gather_outputs, self._unsharded_inner_tensors
            ):
                free_storage(tensor)

    @property
    def all_gather_inputs(self) -> List[torch.Tensor]:  # 1D
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
        if self.sharded_state == ShardedState.SHARDED:
            if not compiled_autograd_enabled() and hasattr(
                self._sharded_local_tensor, "fsdp_pre_all_gather"
            ):
                sharded_local_tensor = self._sharded_local_tensor
                if self.offload_to_cpu:
                    sharded_local_tensor = sharded_local_tensor.to(
                        self.device, non_blocking=True
                    )
                pre_all_gather_signature = inspect.signature(
                    sharded_local_tensor.fsdp_pre_all_gather
                )
                num_fn_params = len(pre_all_gather_signature.parameters)
                # Old signature only passes mesh; keep for BC for now
                assert num_fn_params in (
                    1,
                    5,
                ), (
                    f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                    "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                    "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
                )
                if num_fn_params == 1:
                    (
                        all_gather_inputs,
                        self._extensions_data.all_gather_metadata,
                    ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh)
                else:
                    (
                        all_gather_inputs,
                        self._extensions_data.all_gather_metadata,
                    ) = sharded_local_tensor.fsdp_pre_all_gather(
                        self.shard_mesh,
                        self._orig_size,
                        self._contiguous_orig_stride,
                        self._module_info.module,
                        self.mp_policy,
                    )
                    if (
                        sharded_local_tensor.size() != self.padded_sharded_param_size
                        and any(
                            all_gather_input.size() != self.padded_sharded_param_size
                            for all_gather_input in all_gather_inputs
                        )
                    ):
                        # NOTE: Since this error can only be raised on the
                        # ranks that have padding, this can manifest as a NCCL
                        # watchdog timeout, as the other ranks will not error.
                        raise AssertionError(
                            "When a parameter is unevenly sharded by FSDP "
                            f"(orig size={self._orig_size}, FSDP world size={self.mesh_info.mesh.size()}), "
                            "fsdp_pre_all_gather must return all-gather inputs with the padded sharded size "
                            f"{self.padded_sharded_param_size} but got {[t.size() for t in all_gather_inputs]}"
                        )
                self._extensions_data.all_gather_input_sizes = [
                    t.size() for t in all_gather_inputs
                ]
                return [t.view(-1) for t in all_gather_inputs]
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            if not compiled_autograd_enabled() and hasattr(
                self._sharded_local_tensor, "fsdp_pre_all_gather"
            ):
                raise NotImplementedError
            all_gather_input = _to_dtype_if_needed(
                cast(torch.Tensor, self._sharded_post_forward_param_data),
                self.param_dtype,
            )
            return [all_gather_input]
        return [torch.empty(0)]  # mypy

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_param.grad
        assert grad is not None, "Expects unsharded_param.grad to not be None"
        return self._get_grad_inner_tensor(grad)

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_accumulated_grad
        assert grad is not None, "Expects unsharded_accumulated_grad to not be None"
        return self._get_grad_inner_tensor(grad)

    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_dtensor:
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            assert isinstance(grad, DTensor), f"{type(grad)}"
            placements = self._tp_spec.placements
            if placements != grad.placements:
                assert len(self._tp_spec.placements) == len(
                    grad.placements
                ), f"{self._tp_spec=} {grad.placements=}"
                grad = grad.redistribute(placements=placements)
            grad = grad._local_tensor
        return grad

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    @property
    def shard_mesh(self):
        mesh = self.mesh_info.mesh
        if mesh.ndim == 1:
            return mesh
        elif mesh.ndim == 2:
            assert mesh.mesh_dim_names is not None
            return mesh[mesh.mesh_dim_names[-1]]
        raise ValueError(f"Invalid mesh: {mesh}")

    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(
                f"Expects to be in one of {states}, not {self.sharded_state}"
            )

    def reset_sharded_param(self):
        # For ops like `nn.Module._apply` or `load_state_dict(assign=True)`
        # that change the sharded parameter tensor, we may need to re-pad the
        # sharded local tensor and re-save the reference.
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} "
                    f"instead of {self.sharded_param}"
                )
            self.sharded_param = new_param
        local_tensor = new_param._local_tensor
        if local_tensor.is_meta:
            return
        updated_local_tensor = False
        padded_sharded_size = self.padded_sharded_param_size
        shard_dim = self.fsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0
        if local_tensor.size() != padded_sharded_size:
            assert (
                shard_dim == 0
            ), f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}"
            padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
            padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(
                local_tensor
            )
            local_tensor = padded_local_tensor
            updated_local_tensor = True
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory(device=self.device)
            updated_local_tensor = True
        self._sharded_param_data = local_tensor.view(-1)
        assert isinstance(self.sharded_param, DTensor)  # mypy
        if updated_local_tensor:
            # Only change the local tensor object if needed
            self.sharded_param._local_tensor = local_tensor.narrow(
                dim=shard_dim, start=0, length=length
            )
            assert self.sharded_param._local_tensor.is_contiguous()
        self._sharding_spec = self.sharded_param._spec

    def __repr__(self):
        return f"FSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


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
