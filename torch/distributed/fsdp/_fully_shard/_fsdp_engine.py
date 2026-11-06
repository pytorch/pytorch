# mypy: allow-untyped-defs
"""``fully_shard`` on the FlexShard engine's runtime.

Each ``fully_shard`` call's parameters (one group per mesh) become one bucket
of the FlexShard engine
(``torch.distributed.fsdp._flex_shard``): one byte buffer of padded shards, the
engine's ``Shard`` placement, and a ``BucketRuntime`` that schedules its
all-gathers, reshards, post-backward reduce-scatters and prefetches, shared by
all groups of the root module through one ``BucketCommContext``. ``flex_shard``
runs on the same runtime.

What stays here is the frontend: the DTensor representation of the
parameters (``FSDPParam``, which no longer owns storage or collectives) and the
glue that builds a bucket from a ``fully_shard`` call and joins it to the
root's runtime at lazy init.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .._flex_shard.bucket_runtime import (
    BucketCommContext,
    BucketParam,
    BucketRuntime,
    ParamOwnerRef,
    ParamRepresentation,
)
from .._flex_shard.bucket_storage import ParamInfo, ShardedBucketStorage
from .._flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from .._flex_shard.placements.shard import Shard as EngineShard
from ._all_gather_layout import (
    AllGatherInputMetadata,
    AllGatherLayout,
    DEFAULT_ALL_GATHER_LAYOUT,
    DefaultAllGatherLayout,
)
from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_common import (
    _from_local_no_grad,
    DDPMeshInfo,
    FSDPMeshInfo,
    TrainingState,
)
from ._fsdp_param import (
    _get_dtensor_spec_with_dtype,
    _get_param_module_infos,
    alloc_storage,
    FSDPParam,
    ShardedState,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
    from ._fsdp_common import DataParallelMeshInfo, ShardPlacementFnResult


__all__: list[str] = []


class _GroupMesh:
    """The 1-D view of a process group that engine placements and buckets use
    as their mesh."""

    ndim = 1

    def __init__(self, group: dist.ProcessGroup | None, device_type: str) -> None:
        # No group: this rank alone (replicate(), which does not shard)
        self._group = group
        self.device_type = device_type

    def get_group(self, mesh_dim: Any = None) -> dist.ProcessGroup | None:
        return self._group

    def size(self, mesh_dim: Any = None) -> int:
        return 1 if self._group is None else self._group.size()

    def get_local_rank(self, mesh_dim: Any = None) -> int:
        return 0 if self._group is None else self._group.rank()


class FSDPParamRepresentation(ParamRepresentation):
    """Exposes a bucket's parameters as FSDP2 does: sharded DTensor parameters
    whose local tensors view the bucket, and unsharded parameters that are
    plain tensors, or DTensors over the non-data-parallel mesh dims (TP/EP)."""

    def __init__(self, bucket: FullyShardBucket) -> None:
        self.bucket = bucket
        self.fsdp_params = bucket.fsdp_params

    def local_shard(
        self, index: int, sharded_param: nn.Parameter, *, use_autograd: bool
    ) -> torch.Tensor:
        if use_autograd:
            return sharded_param.to_local()
        return sharded_param._local_tensor.detach()

    def unsharded_param(self, index: int, full_param: torch.Tensor) -> torch.Tensor:
        fsdp_param = self.fsdp_params[index]
        spec = fsdp_param._unsharded_dtensor_spec
        if fsdp_param.is_spmd_types or spec is None or isinstance(full_param, DTensor):
            # FSDPParamShard's full params are FSDPParam's unsharded params
            return full_param
        return _from_local_no_grad(
            full_param, _get_dtensor_spec_with_dtype(spec, full_param.dtype)
        )

    def unsharded_local(self, index: int, unsharded_param: nn.Parameter) -> torch.Tensor:
        if isinstance(unsharded_param, DTensor):
            return unsharded_param._local_tensor
        return unsharded_param

    def unsharded_grad(self, index: int, grad: torch.Tensor) -> torch.Tensor:
        return self.fsdp_params[index]._get_grad_inner_tensor(grad)

    def check_post_forward_reshard(self) -> None:
        # FSDP2's limits for resharding to a smaller mesh after forward
        param_driven = self.bucket.param_driven_unshard()
        for fsdp_param in self.fsdp_params:
            if param_driven:  # FSDPParamShard's
                fsdp_param.check_post_forward_reshard()
                continue
            if fsdp_param.is_dtensor:
                raise NotImplementedError(
                    f"Resharding {fsdp_param._display_name} to smaller mesh is not "
                    "supported for DTensor parameters yet"
                )
            mesh_info = fsdp_param.post_forward_mesh_info
            shard_dim, orig_size = fsdp_param.fsdp_placement.dim, fsdp_param._orig_size
            world_size = mesh_info.shard_mesh_size if mesh_info is not None else 1
            if shard_dim > 0 and orig_size.numel() and orig_size[0] % world_size:
                raise NotImplementedError(
                    f"Resharding the Shard({shard_dim}) parameter "
                    f"{fsdp_param._display_name} to {world_size} ranks after forward "
                    f"requires dim 0 of its size {tuple(orig_size)} to be divisible "
                    f"by {world_size}"
                )

    def defers_grad_upcast(self, index: int) -> bool | None:
        return self.fsdp_params[index].defers_grad_upcast

    def adopt_unsharded_params(self, unsharded_params: list[nn.Parameter]) -> None:
        # Tools read FSDPParam.unsharded_param, as for FSDP2
        for fsdp_param, param in zip(self.fsdp_params, unsharded_params, strict=True):
            fsdp_param._unsharded_param = param

    def on_sharded_state(self, state: str) -> None:
        # FSDPParam.sharded_state, which tools read, as FSDP2 sets it
        sharded_state = {
            "unsharded": ShardedState.UNSHARDED,
            "sharded": ShardedState.SHARDED,
            "sharded_post_forward": ShardedState.SHARDED_POST_FORWARD,
        }[state]
        for fsdp_param in self.fsdp_params:
            fsdp_param.sharded_state = sharded_state

    def check_reduce_unused_params(self) -> None:
        # As FSDP2: check configuration rather than gradient presence so every
        # rank raises together
        for fsdp_param in self.fsdp_params:
            if (
                fsdp_param.sharded_param.requires_grad
                and fsdp_param.unsharded_grad_dtype is None
            ):
                raise ValueError(
                    "Reducing unused parameters with grad_dtype=None requires an "
                    "explicit MixedPrecisionPolicy.reduce_dtype so every rank can "
                    f"build zero gradients of the same dtype: {fsdp_param._param_fqn}. "
                    "Set reduce_dtype to the dtype its gradients already have."
                )

    def check_post_forward_unshard(self) -> None:
        for fsdp_param in self.fsdp_params:
            if hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather"):
                raise NotImplementedError(
                    f"All-gathering the post-forward shard of {fsdp_param._display_name} "
                    "is not supported for all-gather extensions yet"
                )

    def accumulate_sharded_grads(
        self,
        indices: list[int],
        sharded_params: list[nn.Parameter],
        sharded_grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        stored = []
        dtypes = [self.fsdp_params[index].sharded_grad_dtype for index in indices]
        casts = [
            dtype is not None and grad.dtype != dtype
            for grad, dtype in zip(sharded_grads, dtypes, strict=True)
        ]
        # As FSDP2: views of uncast grads keep the whole reduce output alive,
        # including the regions cast into their own buffers, so copy the uncast
        # grads when those regions are larger
        cast_numel = sum(g.numel() for g, cast in zip(sharded_grads, casts) if cast)
        total_numel = sum(g.numel() for g in sharded_grads)
        copy_uncast = cast_numel > total_numel - cast_numel
        new_grads = _cast_grad_runs(sharded_grads, dtypes, casts)
        for index, grad, new_grad, cast in zip(
            indices, sharded_grads, new_grads, casts, strict=True
        ):
            fsdp_param = self.fsdp_params[index]
            sharded_param = fsdp_param.sharded_param
            if cast:
                grad = new_grad
            elif copy_uncast:
                grad = grad.clone()
            stored.append(grad)
            to_accumulate = sharded_param.grad is not None
            hooks = getattr(sharded_param, "_post_accumulate_grad_hooks", None)
            if fsdp_param.offload_to_cpu:
                # As FSDP2: overlap the copy to pinned memory unless the CPU
                # accumulation or a post-accumulate-grad hook (e.g. optimizer
                # in backward) reads it; backward end waits for it
                non_blocking = fsdp_param.pin_memory and not to_accumulate and not hooks
                grad = grad.to(torch.device("cpu"), non_blocking=non_blocking)
                if non_blocking:
                    fsdp_param.grad_offload_event = (
                        torch.accelerator.current_stream().record_event()
                    )
            if to_accumulate:
                if not isinstance(sharded_param.grad, DTensor):
                    raise AssertionError(
                        "Expected the sharded gradient to be a DTensor, got "
                        f"{type(sharded_param.grad)}"
                    )
                sharded_param.grad._local_tensor += grad
            else:
                sharded_param.grad = fsdp_param.to_sharded_dtensor(grad)
            for hook in (hooks or {}).values():
                hook(sharded_param)
        return stored


def _cast_grad_runs(
    grads: list[torch.Tensor], dtypes: list[torch.dtype | None], casts: list[bool]
) -> list[torch.Tensor]:
    """Cast each run of adjacent grads to the same dtype in one buffer, as
    FSDP2 casts the reduce output per dtype run: one allocation per dtype."""
    out = list(grads)
    i = 0
    while i < len(grads):
        if not casts[i]:
            i += 1
            continue
        j = i + 1
        storage = grads[i].untyped_storage()
        while (
            j < len(grads)
            and casts[j]
            and dtypes[j] == dtypes[i]
            and grads[j].untyped_storage().data_ptr() == storage.data_ptr()
        ):
            j += 1
        run = grads[i:j]
        if all(g.is_contiguous() for g in run):
            start = min(g.storage_offset() for g in run)
            end = max(g.storage_offset() + g.numel() for g in run)
            flat = run[0].new_empty(0).set_(storage, start, (end - start,))
            cast_flat = flat.to(dtypes[i])
            for k, grad in enumerate(run, start=i):
                offset = grad.storage_offset() - start
                out[k] = cast_flat.narrow(0, offset, grad.numel()).view(grad.shape)
        else:
            for k in range(i, j):
                out[k] = grads[k].to(dtypes[k])
        i = j
    return out


@dataclass
class _ParamDrivenUnshardState:
    # On one rank, FSDP2 copies each parameter's all-gather inputs instead
    single_rank_inputs: list[list[torch.Tensor]] | None
    split_sizes: list[int]
    outer_sizes: list[int]
    param_numels: list[list[int]]
    param_dtypes: list[list[torch.dtype]]
    layout: AllGatherLayout
    output_metadata: object | None
    comm: Any
    group: dist.ProcessGroup | None


class FSDPParamShard(EngineShard):
    """The engine's ``Shard`` placement for buckets whose all-gather inputs and
    outputs ``FSDPParam`` defines: all-gather extensions
    (``fsdp_pre_all_gather``/``fsdp_post_all_gather``), tensor subclass
    parameters, and custom ``AllGatherLayout``s. The all-gather packs each
    parameter's ``all_gather_inputs``, and finish fills its all-gather outputs
    and unsharded parameter, as FSDP2 does; the gradient reduction is
    ``Shard``'s."""

    refills_persistent_buffers_in_finish = True

    def __init__(self, dim: int, bucket: FullyShardBucket) -> None:
        super().__init__(dim)
        self._bucket = bucket

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FSDPParamShard)
            and other._bucket is self._bucket
            and other.dim == self.dim
        )

    def __hash__(self) -> int:
        return hash((type(self), id(self._bucket), self.dim))

    def bucket_compatibility_key(self) -> object:
        return (type(self), id(self._bucket))

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: Any,
        debug_fqn: str | None,
        *,
        comm: Any = None,
        device: torch.device | None = None,
    ) -> PlacementPreparedUnshard:
        from ._fsdp_collectives import _get_all_gather_input_metadatas

        bucket = self._bucket
        world_size, rank = mesh.size(), mesh.get_local_rank()
        param_inputs = [fsdp_param.all_gather_inputs for fsdp_param in bucket.fsdp_params]
        if world_size == 1:
            return PlacementPreparedUnshard(
                placement=self,
                buffers=[],
                placement_state=_ParamDrivenUnshardState(
                    single_rank_inputs=param_inputs,
                    split_sizes=[],
                    outer_sizes=[],
                    param_numels=[],
                    param_dtypes=[],
                    layout=bucket.all_gather_layout,
                    output_metadata=None,
                    comm=comm,
                    group=mesh.get_group(),
                ),
            )
        # Extension layouts are set by the all_gather_inputs calls above
        outer_sizes = [
            layout.outer_size
            for fsdp_param in bucket.fsdp_params
            for layout in fsdp_param.all_gather_output_layouts
        ]
        param_dtypes, param_numels, dtype = _get_all_gather_input_metadatas(param_inputs)
        inputs = [
            t.view(torch.uint8) if dtype == torch.uint8 else t
            for ts in param_inputs
            for t in ts
        ]
        split_sizes = [t.numel() for t in inputs]
        numel = sum(split_sizes)
        layout = bucket.all_gather_layout
        if isinstance(layout, DefaultAllGatherLayout):
            copy_in, call_layout, output_metadata = layout.copy_in, layout, None
        else:
            copy_in, call_layout, output_metadata = layout.prepare(
                AllGatherInputMetadata(
                    input_split_sizes=list(split_sizes),
                    input_outer_sizes=list(outer_sizes),
                    input_numel=numel,
                    world_size=world_size,
                    dtype=dtype,
                    device=bucket.device,
                )
            )
        size = (numel * world_size,)
        output = (
            torch.empty(size, dtype=dtype, device=bucket.device)
            if comm is None
            else comm.allocate(size, dtype=dtype, device=bucket.device)
        )
        all_gather_input, output = copy_in(inputs, output, split_sizes, numel, rank)
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[all_gather_input, output],
            placement_state=_ParamDrivenUnshardState(
                single_rank_inputs=None,
                split_sizes=split_sizes,
                outer_sizes=outer_sizes,
                param_numels=param_numels,
                param_dtypes=param_dtypes,
                layout=call_layout,
                output_metadata=output_metadata,
                comm=comm,
                group=mesh.get_group(),
            ),
        )

    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        state = cast(_ParamDrivenUnshardState, prepared.placement_state)
        if state.single_rank_inputs is not None:
            return None
        all_gather_input, output = prepared.buffers
        if state.comm is None:
            return dist.all_gather_single(
                output, all_gather_input, group=state.group, async_op=async_op
            )
        kwargs = {}
        if state.layout.comm_type is not None:
            kwargs["output_metadata"] = state.output_metadata
        return state.comm(
            output_tensor=output,
            input_tensor=all_gather_input,
            group=state.group,
            async_op=async_op,
            **kwargs,
        )

    def finish_prepared_unshard(
        self, prepared: PlacementPreparedUnshard
    ) -> PlacementUnshardResult:
        from ._fsdp_collectives import AllGatherResult, foreach_all_gather_copy_out

        state = cast(_ParamDrivenUnshardState, prepared.placement_state)
        bucket = self._bucket
        fsdp_params = bucket.fsdp_params
        if state.single_rank_inputs is not None:
            self._copy_single_rank_inputs(state.single_rank_inputs)
            return self._finish_unsharded_params(prepared)
        result = AllGatherResult(
            prepared.buffers[1],
            None,
            None,
            state.param_dtypes,
            state.param_numels,
            state.split_sizes,
            state.outer_sizes,
            state.layout,
            state.output_metadata,
            prepared.buffers[0],
        )
        try:
            bucket.layout_buffers = foreach_all_gather_copy_out(
                result, fsdp_params, state.group, bucket.layout_buffers
            )
        except ValueError:
            # As FSDP2: free what a rejected refill re-allocated
            if bucket.layout_buffers is not None and not bucket.bucket.keep_persistent_storage:
                bucket.layout_buffers.free()
            for fsdp_param in fsdp_params:
                if not fsdp_param.keep_unsharded_storage:
                    fsdp_param.free_all_gather_outputs()
            raise
        return self._finish_unsharded_params(prepared)

    def _copy_single_rank_inputs(self, param_inputs: list[list[torch.Tensor]]) -> None:
        """As FSDP2's ``wait_for_unshard`` on one rank: copy the all-gather
        inputs into the all-gather outputs."""
        bucket = self._bucket
        for fsdp_param, inputs in zip(bucket.fsdp_params, param_inputs, strict=True):
            if not fsdp_param.all_gather_outputs:
                fsdp_param.init_all_gather_outputs(
                    [t.numel() for t in inputs], [t.dtype for t in inputs], 1, bucket.device
                )
            for tensor, inp in zip(fsdp_param.all_gather_outputs, inputs, strict=True):
                alloc_storage(tensor)
                with (
                    torch.autograd._unsafe_preserve_version_counter(tensor)
                    if not tensor.is_inference()
                    else contextlib.nullcontext()
                ):
                    if tensor.dtype == inp.dtype and tensor.numel() != inp.numel():
                        tensor = tensor[: inp.numel()]
                    elif inp.dtype == torch.uint8:
                        tensor = tensor.view(torch.uint8)
                    tensor.copy_(inp)

    def _finish_unsharded_params(
        self, prepared: PlacementPreparedUnshard
    ) -> PlacementUnshardResult:
        bucket = self._bucket
        fsdp_params = bucket.fsdp_params
        for fsdp_param in fsdp_params:
            fsdp_param.init_unsharded_param()
        if prepared.persistent_buffers is not None:
            return PlacementUnshardResult(
                full_params=[], persistent_buffers=prepared.persistent_buffers
            )
        persistent: dict[int, torch.Tensor] = {}
        for fsdp_param in fsdp_params:
            for tensor in (*fsdp_param.all_gather_outputs, *fsdp_param._unsharded_inner_tensors):
                persistent.setdefault(id(tensor), tensor)
        if bucket.layout_buffers is not None:
            for tensor in bucket.layout_buffers.tensors:
                persistent.setdefault(id(tensor), tensor)
        return PlacementUnshardResult(
            full_params=[fsdp_param._unsharded_param for fsdp_param in fsdp_params],
            persistent_buffers=list(persistent.values()) if prepared.persistent else [],
        )


def _aligned_offsets(tensors: list[torch.Tensor]) -> list[int]:
    """Byte offsets of ``tensors`` packed in one buffer, each aligned to its
    element size so that it can be viewed in its dtype."""
    offsets = []
    offset = 0
    for tensor in tensors:
        itemsize = tensor.element_size()
        offset = -(-offset // itemsize) * itemsize
        offsets.append(offset)
        offset += tensor.numel() * itemsize
    return offsets


class FullyShardBucket:
    """One ``fully_shard`` group (a call's parameters on one mesh) as an engine
    bucket: its ``FSDPParam`` representations, its storage and its runtime.
    It holds no scheduling of its own: the ``BucketRuntime`` does that."""

    def __init__(
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ) -> None:
        self.modules = modules
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.fsdp_params = [
            FSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy._resolve_for_param(param),
                offload_policy,
            )
            for param, module_info in zip(
                params, _get_param_module_infos(params, modules)
            )
        ]
        self.mesh = _GroupMesh(
            mesh_info.shard_process_group
            if isinstance(mesh_info, FSDPMeshInfo)
            else None,
            device.type,
        )
        self.storage = ShardedBucketStorage(
            torch.empty(0, dtype=torch.uint8, device=device),
            {},
            self.mesh,  # type: ignore[arg-type]
            0,
            modules[0],
            fsdp2_compatible=True,
        )
        self.storage._fsdp2_semantics = True
        self.set_post_forward_mesh_info(post_forward_mesh_info)
        if isinstance(mesh_info, DDPMeshInfo):  # HSDP or replicate()
            self.storage.set_replicate_group(mesh_info.replicate_process_group)
        self.offload = isinstance(offload_policy, CPUOffloadPolicy)
        if self.offload:
            self.storage._compute_device = device
        self.views: list[torch.Tensor] = []
        self.module_fqn: str | None = None
        self._lazy_initialized = False
        # FSDPModule.set_all_gather_layout, and the persistent buffers a custom
        # layout chose (see FSDPParamShard)
        self.all_gather_layout: AllGatherLayout = DEFAULT_ALL_GATHER_LAYOUT
        self.layout_buffers: Any = None
        from ._fsdp_collectives import DefaultAllGather, DefaultReduceScatter

        self._default_all_gather_comm = DefaultAllGather()
        self._default_reduce_scatter_comm = DefaultReduceScatter()
        # Parameters whose all-gather FSDPParam defines stay in their own
        # storage (see FSDPParamShard)
        self.own_storage = any(
            hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
            or type(fsdp_param._sharded_param_data) is not torch.Tensor
            for fsdp_param in self.fsdp_params
        )
        # Re-read at lazy init, as FSDP2 reads extensions then (see refresh)
        self.has_extensions = self.own_storage
        # FSDP2's set_reduce_scatter_max_input_buffers; the root's engine context
        # keeps the largest of its groups'
        self.max_input_buffers = 1
        self.has_spmd_types = any(p.is_spmd_types for p in self.fsdp_params)
        self._home()
        self.bucket = BucketRuntime(
            bucket_storage=self.storage,
            bucket_params=self._bucket_params(),
            context=BucketCommContext.for_device(device),
            debug_fqn=None,
            representation=FSDPParamRepresentation(self),
            logger=logging.getLogger("torch.distributed.fsdp.fully_shard"),
            log_name="FSDP",
        )
        self.bucket.context.buckets.append(self.bucket)
        if len(modules) > 1:
            self.bucket.group_modules = modules
        # As FSDP2: state dicts read and load the sharded parameters
        for module in modules:
            module.register_state_dict_pre_hook(self._reshard_hook)
            module._register_load_state_dict_pre_hook(self._reshard_hook)

    def _reshard_hook(self, *args: Any, **kwargs: Any) -> None:
        if self.bucket.is_unsharded:
            self.bucket.reshard()

    # Group attributes that tools and tests inspect
    @property
    def _module_fqn(self) -> str | None:
        return self.module_fqn

    @property
    def is_unsharded(self) -> bool:
        return self.bucket.is_unsharded

    @property
    def is_sharded(self) -> bool:
        return not self.bucket.is_unsharded and self.bucket.post_forward_storage is None

    @property
    def _reshard_after_forward(self) -> bool:
        return self.storage._reshard_after_forward

    @property
    def reshard_after_backward(self) -> bool:
        return self.storage._reshard_after_backward

    @reshard_after_backward.setter
    def reshard_after_backward(self, value: bool) -> None:
        self.storage.set_reshard_after_backward(value)

    @property
    def reduce_grads(self) -> bool:
        return self.storage._requires_gradient_sync

    @reduce_grads.setter
    def reduce_grads(self, value: bool) -> None:
        self.storage.set_requires_gradient_sync(value)

    @property
    def all_reduce_grads(self) -> bool:
        return self.storage._requires_all_reduce

    @all_reduce_grads.setter
    def all_reduce_grads(self, value: bool) -> None:
        self.storage._requires_all_reduce = value

    @property
    def unshard_in_backward(self) -> bool:
        return self.storage._unshard_in_backward

    @property
    def unshard_async_op(self) -> bool:
        return self.storage._unshard_async_op

    @property
    def gradient_divide_factor(self) -> float | None:
        return self.storage.gradient_divide_factor

    @property
    def force_sum_reduction_for_comms(self) -> bool:
        return self.storage.gradient_reduction.force_sum

    @property
    def reduce_scatter_unused_params(self) -> bool:
        return self.storage._reduce_scatter_unused_params

    @property
    def reduce_scatter_max_input_buffers(self) -> int:
        return self.max_input_buffers

    @property
    def _all_gather_result(self) -> Any:
        """The bucket's pending (prefetched) unshard, if any."""
        runtime = self.bucket
        for pending in runtime.context.pending_unshards:
            if pending.bucket is runtime:
                return pending.result
        return None

    @property
    def is_sharded_post_forward(self) -> bool:
        return self.bucket.post_forward_storage is not None

    @property
    def _all_gather_layout(self) -> AllGatherLayout:
        return self.all_gather_layout

    @property
    def _all_gather_comm(self) -> Any:
        return self.storage._all_gather_comm or self._default_all_gather_comm

    @property
    def _reduce_scatter_comm(self) -> Any:
        return self.storage._reduce_scatter_comm or self._default_reduce_scatter_comm

    @property
    def _all_gather_buffers(self) -> Any:
        return self.layout_buffers

    @property
    def _all_gather_output_fn(self) -> Any:
        raise AttributeError(
            "_all_gather_output_fn was removed; use "
            "FSDPModule.set_all_gather_layout(DefaultAllGatherLayout(fn))"
        )

    @_all_gather_output_fn.setter
    def _all_gather_output_fn(self, fn: Any) -> None:
        raise AttributeError(
            "_all_gather_output_fn was removed; use "
            "FSDPModule.set_all_gather_layout(DefaultAllGatherLayout(fn))"
        )

    @property
    def _prepare_reduce_scatter_inputs(self) -> Callable:
        from ._fsdp_collectives import _default_reduce_scatter_input_fn

        return self.storage._reduce_scatter_input_fn or _default_reduce_scatter_input_fn

    @_prepare_reduce_scatter_inputs.setter
    def _prepare_reduce_scatter_inputs(self, fn: Callable) -> None:
        from ._fsdp_collectives import _default_reduce_scatter_input_fn

        self.storage._reduce_scatter_input_fn = (
            None if fn is _default_reduce_scatter_input_fn else fn
        )

    @property
    def _partial_reduce_output(self) -> torch.Tensor | None:
        return self.bucket.partial_reduce_output

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup | None:
        return self.storage.reduce_mesh.get_group()

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup | None:
        if self.bucket.post_forward_storage is not None:
            return self.bucket.post_forward_group
        return self.mesh.get_group()

    def param_driven_unshard(self) -> bool:
        """Whether the all-gather needs ``FSDPParamShard``: parameters with
        their own all-gather, or a layout other than the native rank-major
        one."""
        from ..experimental import all_gather_output_fn_with_native_copy

        layout = self.all_gather_layout
        return (
            self.own_storage
            or self.has_extensions
            or not (
                type(layout) is DefaultAllGatherLayout
                and layout.output_fn in (None, all_gather_output_fn_with_native_copy)
            )
        )

    def set_post_forward_mesh_info(
        self, post_forward_mesh_info: FSDPMeshInfo | None
    ) -> None:
        """Reshard after forward (``post_forward_mesh_info`` not None), to the
        smaller post-forward mesh for an int ``reshard_after_forward``."""
        self.post_forward_mesh_info = post_forward_mesh_info
        self.storage._reshard_after_forward = post_forward_mesh_info is not None
        self.storage._post_forward_group = (
            post_forward_mesh_info.shard_process_group
            if post_forward_mesh_info is not None
            and post_forward_mesh_info is not self.mesh_info
            else None
        )

    def wait_for_grad_offload(self) -> None:
        for fsdp_param in self.fsdp_params:
            if (event := fsdp_param.grad_offload_event) is not None:
                event.synchronize()
                fsdp_param.grad_offload_event = None

    # Storage #
    def homed(self) -> bool:
        """Whether each parameter's data still views the bucket storage.
        ``FSDPParam.reset_sharded_param`` rebinds it when the sharded parameter
        changed (``to_empty``, ``load_state_dict(assign=True)``, a dtype
        change)."""
        return self.own_storage or (
            len(self.views) == len(self.fsdp_params)
            and all(
                fsdp_param._sharded_param_data is view
                for fsdp_param, view in zip(self.fsdp_params, self.views)
            )
        )

    def _home(self) -> None:
        """Move each parameter's padded sharded data into one byte buffer and
        point its ``_sharded_param_data`` and sharded parameter at it."""
        datas = [fsdp_param._sharded_param_data for fsdp_param in self.fsdp_params]
        if self.own_storage or any(data.is_meta for data in datas):
            # Meta parameters stay on meta until to_empty(), as FSDP2's; lazy
            # init homes them
            self._build_infos()
            return
        nbytes = [data.numel() * data.element_size() for data in datas]
        offsets = _aligned_offsets(datas)
        total = offsets[-1] + nbytes[-1] if datas else 0
        if self.offload:
            pin_memory = cast(CPUOffloadPolicy, self.offload_policy).pin_memory
            storage = torch.empty(
                total, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
            )
        else:
            storage = torch.empty(total, dtype=torch.uint8, device=self.device)
        views = []
        for fsdp_param, data, size, offset in zip(
            self.fsdp_params, datas, nbytes, offsets, strict=True
        ):
            view = storage[offset : offset + size].view(data.dtype)
            if not data.is_meta and data.numel():
                view.copy_(data)
            fsdp_param._sharded_param_data = view
            views.append(view)
            shard_dim = fsdp_param.fsdp_placement.dim
            sharded_param = fsdp_param.sharded_param
            local = sharded_param._local_tensor
            length = local.size(shard_dim) if local.numel() > 0 else 0
            sharded_param._local_tensor = view.view(
                fsdp_param.padded_sharded_param_size
            ).narrow(shard_dim, 0, length)
        self.views = views
        self.storage._byte_storage = storage
        self.storage._total_bytes = total
        self._build_infos()

    def _build_infos(self) -> None:
        datas = [fsdp_param._sharded_param_data for fsdp_param in self.fsdp_params]
        infos: dict[str, ParamInfo] = {}
        for index, (fsdp_param, data, offset) in enumerate(
            zip(self.fsdp_params, datas, _aligned_offsets(datas), strict=True)
        ):
            has_override = fsdp_param._has_sharded_grad_dtype_override
            fqn = fsdp_param._param_fqn or f"{index}.{fsdp_param._module_info.param_name}"
            infos[fqn] = ParamInfo(
                fqn=fqn,
                global_shape=fsdp_param._orig_size,
                global_stride=fsdp_param._contiguous_orig_stride,
                dtype=data.dtype,
                requires_grad=fsdp_param.sharded_param.requires_grad,
                placements=(
                    FSDPParamShard(fsdp_param.fsdp_placement.dim, self)
                    if self.param_driven_unshard()
                    else EngineShard(fsdp_param.fsdp_placement.dim),
                ),
                param_dtype=getattr(fsdp_param, "param_dtype", None),
                reduce_dtype=getattr(fsdp_param, "reduce_dtype", None),
                local_shape=fsdp_param.sharded_size,
                local_numel=fsdp_param.sharded_size.numel(),
                byte_offset=offset,
                storage_nbytes=data.numel() * data.element_size(),
                global_numel=fsdp_param._orig_size.numel(),
                has_explicit_grad_dtype=has_override,
                grad_dtype_override=fsdp_param.sharded_grad_dtype if has_override else None,
            )
        trainable = [
            info.unsharded_grad_dtype or info.unsharded_dtype
            for info in infos.values()
            if info.requires_grad
        ]
        if trainable:
            reduce_dtype = trainable[0]
            for dtype in trainable[1:]:
                reduce_dtype = torch.promote_types(reduce_dtype, dtype)
            for info in infos.values():
                info.bucket_reduce_dtype = reduce_dtype
        self.storage._param_infos = infos

    def _bucket_params(self) -> list[BucketParam]:
        return [
            BucketParam(
                param_owner=ParamOwnerRef(
                    fsdp_param._module_info.module, fsdp_param._module_info.param_name
                ),
                param_info=info,
                sharded_param=fsdp_param.sharded_param,
                shared_owners=tuple(
                    ParamOwnerRef(module, name)
                    for module, name in zip(
                        fsdp_param._module_info.shared_modules,
                        fsdp_param._module_info.shared_param_names,
                        strict=True,
                    )
                ),
            )
            for fsdp_param, info in zip(
                self.fsdp_params, self.storage._param_infos.values(), strict=True
            )
        ]

    # Lazy init #
    def refresh(self) -> None:
        """Re-home parameters whose sharded data no longer views the bucket
        (see ``homed``) and re-read their engine metadata. Runs before an
        unshard that may follow such a change."""
        if self.bucket.is_unsharded:
            return
        self.has_extensions = any(
            hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
            for fsdp_param in self.fsdp_params
        )
        if not self.homed():
            self._home()
        else:
            self._build_infos()
        self.bucket.bucket_params = self._bucket_params()

    def lazy_init(self) -> None:
        """Re-read parameters replaced since construction and resolve mixed
        precision dtypes, as FSDP2's group lazy init does, once."""
        if self._lazy_initialized:
            return
        self._lazy_initialized = True
        for fsdp_param in self.fsdp_params:
            fsdp_param.reset_sharded_param()
            fsdp_param._init_extensions()
        meta = [
            fsdp_param._param_fqn
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type == "meta"
        ]
        if meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before "
                f"training, but the following were still on meta device: {meta}\n"
                "For example, call module.to_empty(device) to materialize to "
                "device and call module.reset_parameters() on each module to "
                "initialize values."
            )
        self._init_mp_dtypes()
        self.refresh()

    def _init_mp_dtypes(self) -> None:
        """Resolve the params' mixed precision dtypes and check that the group's
        original and reduce dtypes are uniform."""
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(fsdp_param.mp_policy)
        trainable = [p for p in self.fsdp_params if p.sharded_param.requires_grad]
        params_for_dtype = trainable or [
            p for p in self.fsdp_params if p.orig_dtype.is_floating_point
        ]
        orig_dtypes = {p.orig_dtype for p in params_for_dtype}
        reduce_dtypes = {p.reduce_dtype for p in params_for_dtype}
        if trainable and len(orig_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        if trainable and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )

    def join(self, context: BucketCommContext, *, reshard_after_forward: bool) -> None:
        """Join the root module's engine context."""
        self.lazy_init()
        self.storage._reshard_after_forward = reshard_after_forward
        context.max_pending_reduce_grads = max(
            context.max_pending_reduce_grads, self.max_input_buffers
        )
        old_context = self.bucket.context
        if old_context is not context:
            old_context.take_pending_unshard(None)
            self.bucket.context = context
            context.buckets.append(self.bucket)
        self.bucket.debug_fqn = self.module_fqn


# FSDPModule methods on the engine #
def _buckets(state: Any) -> list[FullyShardBucket]:
    return list(state._engine_buckets)


def reshard(state: Any) -> None:
    for bucket in _buckets(state):
        runtime = bucket.bucket
        runtime.context.release_pending_unshard(runtime)
        if runtime.is_unsharded:
            runtime.reshard()


def unshard(state: Any, async_op: bool) -> list[BucketRuntime]:
    """Start (``async_op``) or finish the module's unshards on the current
    stream, as ``FSDPModule.unshard`` does; return the buckets to wait on."""
    runtimes = []
    for bucket in _buckets(state):
        bucket.lazy_init()
        bucket.refresh()
        runtime = bucket.bucket
        if runtime.is_unsharded:
            continue
        if async_op:
            runtime.context.prefetch(runtime, explicit=True, on_current_stream=True)
        else:
            runtime.unshard(on_current_stream=True)
        runtimes.append(runtime)
    return runtimes


def wait_for_unshard(runtimes: list[BucketRuntime]) -> None:
    for runtime in runtimes:
        if any(p.bucket is runtime for p in runtime.context.pending_unshards):
            runtime.unshard()


def set_requires_gradient_sync(state: Any, requires_gradient_sync: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage.set_requires_gradient_sync(requires_gradient_sync)
        bucket.storage._requires_all_reduce = requires_gradient_sync


def set_reshard_after_forward(state: Any, reshard_after_forward: bool | int) -> None:
    from ._fsdp_init import _get_post_forward_mesh_info

    for bucket in _buckets(state):
        if not isinstance(bucket.mesh_info, FSDPMeshInfo):
            raise AssertionError(f"Expected FSDPMeshInfo, got {bucket.mesh_info}")
        bucket.set_post_forward_mesh_info(
            _get_post_forward_mesh_info(reshard_after_forward, bucket.mesh_info)
        )


def set_requires_all_reduce(state: Any, requires_all_reduce: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage._requires_all_reduce = requires_all_reduce


def set_all_reduce_hook(
    state: Any,
    hook: Callable[[torch.Tensor], None],
    stream: torch.Stream | None,
) -> None:
    buckets = _buckets(state)
    if len(buckets) > 1:
        raise NotImplementedError(
            "set_all_reduce_hook is not supported with multiple parameter groups "
            "(per-param mesh)"
        )
    for bucket in buckets:
        bucket.storage._all_reduce_hook = hook
        if stream is not None:
            if bucket.storage._replicate_group is not None:
                raise ValueError("stream cannot be set when using native HSDP")
            bucket.storage._all_reduce_hook_stream = stream


def set_force_sum_reduction_for_comms(state: Any, enable: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage.set_force_sum_reduction_for_comms(enable)


def set_reshard_after_backward(state: Any, reshard_after_backward: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage.set_reshard_after_backward(reshard_after_backward)


def set_keep_unsharded_storage(state: Any, keep: bool) -> None:
    for bucket in _buckets(state):
        bucket.bucket.keep_persistent_storage = keep
        for fsdp_param in bucket.fsdp_params:
            fsdp_param.keep_unsharded_storage = keep


def set_modules_to_prefetch(state: Any, target_states: list[Any], *, forward: bool) -> None:
    targets = [
        target.storage
        for target_state in target_states
        for target in (
            _buckets(target_state) if forward else reversed(_buckets(target_state))
        )
    ]
    for bucket in _buckets(state):
        if forward:
            bucket.storage.set_buckets_to_forward_prefetch(targets)
        else:
            bucket.storage.set_buckets_to_backward_prefetch(targets)


def set_gradient_divide_factor(state: Any, factor: float) -> None:
    for bucket in _buckets(state):
        bucket.storage.set_gradient_divide_factor(factor)


def _single_bucket(state: Any, method: str) -> list[FullyShardBucket]:
    buckets = _buckets(state)
    if len(buckets) > 1:
        raise ValueError(
            f"{method} is not supported with multiple param groups (from "
            "per-param mesh via shard_placement_fn). The custom comm would be "
            "ambiguous across groups with different meshes."
        )
    return buckets


def _check_layout_comm(bucket: FullyShardBucket, layout: Any, comm_type: type) -> None:
    if layout.comm_type is not None and not issubclass(comm_type, layout.comm_type):
        raise ValueError(
            f"all-gather layout {type(layout).__name__} requires a "
            f"{layout.comm_type.__name__} comm, not {comm_type.__name__}; "
            "install one with set_custom_all_gather() before the layout"
        )


def set_custom_all_gather(state: Any, comm: Any) -> None:
    buckets = _single_bucket(state, "set_custom_all_gather")
    for bucket in buckets:
        _check_layout_comm(bucket, bucket.all_gather_layout, type(comm))
    for bucket in buckets:
        bucket.storage._all_gather_comm = (
            None if comm is bucket._default_all_gather_comm else comm
        )


def set_custom_reduce_scatter(state: Any, comm: Any) -> None:
    for bucket in _single_bucket(state, "set_custom_reduce_scatter"):
        bucket.storage._reduce_scatter_comm = (
            None if comm is bucket._default_reduce_scatter_comm else comm
        )


def set_allocate_memory_from_process_group(state: Any, enable: bool) -> None:
    from ._fsdp_collectives import (
        DefaultAllGather,
        DefaultReduceScatter,
        ProcessGroupAllocAllGather,
        ProcessGroupAllocReduceScatter,
    )

    for bucket in _buckets(state):
        _check_layout_comm(
            bucket,
            bucket.all_gather_layout,
            ProcessGroupAllocAllGather if enable else type(bucket._default_all_gather_comm),
        )
    for bucket in _buckets(state):
        storage = bucket.storage
        for comm, allowed in (
            (storage._all_gather_comm, (DefaultAllGather, ProcessGroupAllocAllGather)),
            (
                storage._reduce_scatter_comm,
                (DefaultReduceScatter, ProcessGroupAllocReduceScatter),
            ),
        ):
            if comm is not None and not isinstance(comm, allowed):
                raise AssertionError(
                    "cannot call set_allocate_memory_from_process_group() when "
                    f"the comm is custom: {comm.__class__.__name__}"
                )
        group = bucket.mesh.get_group()
        reduce_group = storage.reduce_mesh.get_group()
        storage._all_gather_comm = ProcessGroupAllocAllGather(group) if enable else None
        storage._reduce_scatter_comm = (
            ProcessGroupAllocReduceScatter(reduce_group) if enable else None
        )


def set_symm_mem(state: Any, backend: str) -> None:
    from ._fsdp_collectives import (
        DefaultAllGather,
        DefaultReduceScatter,
        SymmMemAllGather,
        SymmMemReduceScatter,
    )

    for bucket in _buckets(state):
        _check_layout_comm(bucket, bucket.all_gather_layout, SymmMemAllGather)
    for bucket in _buckets(state):
        storage = bucket.storage
        if storage._all_gather_comm is not None and not isinstance(
            storage._all_gather_comm, (DefaultAllGather, SymmMemAllGather)
        ):
            raise AssertionError(
                "cannot call set_symm_mem() when all gather comm is custom: "
                f"{storage._all_gather_comm.__class__.__name__}"
            )
        storage._all_gather_comm = SymmMemAllGather(bucket.mesh.get_group(), backend)
        if storage._reduce_scatter_comm is not None and not isinstance(
            storage._reduce_scatter_comm, (DefaultReduceScatter, SymmMemReduceScatter)
        ):
            raise AssertionError(
                "cannot call set_symm_mem() when reduce scatter comm is custom: "
                f"{storage._reduce_scatter_comm.__class__.__name__}"
            )
        if storage.gradient_reduction.force_sum:
            # As of NCCL 2.29.3, NCCL symmetric reduce-scatter only supports SUM
            storage._reduce_scatter_comm = SymmMemReduceScatter(
                storage.reduce_mesh.get_group(), backend
            )


def set_reduce_scatter_input_fn(state: Any, fn: Callable | None) -> None:
    from ._fsdp_collectives import _default_reduce_scatter_input_fn

    for bucket in _buckets(state):
        bucket.storage._reduce_scatter_input_fn = (
            None if fn is None or fn is _default_reduce_scatter_input_fn else fn
        )


def set_reduce_scatter_unused_params(state: Any, enable: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage._reduce_scatter_unused_params = enable


def set_reduce_scatter_max_input_buffers(state: Any, max_input_buffers: int) -> None:
    for bucket in _buckets(state):
        bucket.max_input_buffers = max_input_buffers
    if state._engine_context is not None:
        root_states = state._state_ctx.all_states
        state._engine_context.max_pending_reduce_grads = max(
            (b.max_input_buffers for s in root_states for b in _buckets(s)), default=1
        )


def set_separate_reduce_scatter_group(
    state: Any, enable: bool, new_groups: dict[tuple[int, ...], dist.ProcessGroup]
) -> None:
    for bucket in _buckets(state):
        storage = bucket.storage
        mesh_info = bucket.mesh_info
        if not enable or bucket.mesh.size() == 1:
            storage._reduce_mesh = None
            if isinstance(mesh_info, FSDPMeshInfo):
                mesh_info.reduce_scatter_process_group = None
            continue
        group = bucket.mesh.get_group()
        ranks = tuple(dist.get_process_group_ranks(group))
        if ranks not in new_groups:
            if storage._reduce_mesh is not None:
                new_groups[ranks] = storage._reduce_mesh.get_group()
            else:
                new_groups[ranks] = dist.new_group(
                    list(ranks), use_local_synchronization=True
                )
        storage._reduce_mesh = _GroupMesh(new_groups[ranks], bucket.device.type)  # type: ignore[assignment]
        if isinstance(mesh_info, FSDPMeshInfo):
            # Where FSDP2 keeps it
            mesh_info.reduce_scatter_process_group = new_groups[ranks]


def set_unshard_async_op(state: Any, async_op: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage._unshard_async_op = async_op


def set_unshard_in_backward(state: Any, unshard_in_backward: bool) -> None:
    for bucket in _buckets(state):
        bucket.storage._unshard_in_backward = unshard_in_backward


def validate_finalize_backward(state: Any) -> None:
    """FSDP2's ``FSDPState._validate_finalize_backward`` on the engine."""
    state_ctx, context = state._state_ctx, state._engine_context
    name = state._state_name
    if state_ctx.gradient_reduction_pending:
        raise RuntimeError(
            "The previous gradient reduction must be waited on before "
            "finalizing backward again"
        )
    if context is not None and context.pending_finalization is not None:
        raise RuntimeError(
            "Another gradient reduction sharing this communication context "
            "must be waited on before finalizing backward"
        )
    if not state._is_root:
        raise RuntimeError(f"finalize_backward must be called on the root {name} module")
    if not state_ctx.manual_backward_finalization:
        raise RuntimeError(
            f"{name} finalize_backward requires manual backward finalization. "
            "Call set_manual_backward_finalization(True) before forward"
        )
    if context is None:
        return
    if torch._C._current_graph_task_id() != -1:
        raise RuntimeError(f"{name} finalize_backward must be called after backward completes")
    buckets = [b for s in state_ctx.all_states for b in _buckets(s)]
    if partial := next((b for b in buckets if b.bucket.partial_reduce_output is not None), None):
        raise RuntimeError(
            f"{name} manual backward finalization does not support partial "
            f"gradient reductions for parameter group {partial.module_fqn or '<root>'!r}. "
            "Use automatic backward finalization for this configuration"
        )
    if no_all_reduce := next(
        (
            b
            for b in buckets
            if b.bucket.needs_finish()
            and b.reduce_grads
            and isinstance(b.mesh_info, DDPMeshInfo)
            and not b.all_reduce_grads
        ),
        None,
    ):
        raise RuntimeError(
            f"{name} manual backward finalization requires all-reduce for "
            f"parameter group {no_all_reduce.module_fqn or '<root>'!r}. Call "
            "set_requires_all_reduce(True) before finalization"
        )


def finalize_backward(state: Any) -> None:
    """Issue the reductions backward left and reshard, without waiting."""
    if (context := state._engine_context) is not None:
        context.start_finalize_backward()


def end_backward_iteration(state: Any) -> None:
    """Wait for the gradient reductions and reset per-iteration state, as
    FSDP2's ``FSDPState._end_backward_iteration``."""
    if (context := state._engine_context) is not None:
        context.wait_and_clear_reduce_grad_states(debug_fqn=None)
        context.pending_finalization = None
    state._engine_backward_end()
    state._state_ctx.manual_backward_finalization_active = None
    state._state_ctx.gradient_reduction_pending = False


def reset_iter_state(state: Any) -> None:
    """FSDP2's ``FSDPState._reset_iter_state`` on the engine: order in-flight
    collectives, reshard, and drop in-flight and accumulated gradients."""
    if state._is_root is False:
        raise RuntimeError("reset_iter_state must be called on the root FSDP module")
    state_ctx, context = state._state_ctx, state._engine_context
    if (
        context is not None
        and (pending := context.pending_finalization) is not None
        and getattr(pending, "_state", None) is not None
        and pending._state._state_ctx is not state_ctx
    ):
        raise RuntimeError(
            "reset_iter_state cannot reset a gradient reduction owned by "
            "another FSDP root sharing the communication context"
        )
    if context is not None:
        context.take_pending_unshard(None)
        context.wait_and_clear_reduce_grad_states(debug_fqn=None)
        context.pending_finalization = None
        context.post_backward_callback_queued = False
        context.finish_at_backward_end = False
        for runtime in context.buckets:
            if runtime.is_unsharded:
                runtime.reshard()
            runtime.post_forward_storage = None
            runtime.partial_reduce_output = None
            runtime.partial_reduce_indices = frozenset()
            runtime.needs_sync = False
            for param in runtime.unsharded_params or []:
                param.grad = None
            runtime.reset_backward_state()
            runtime._set_unsharded_grad_dtypes(defer_upcast=False)
    for s in state_ctx.all_states:
        s._training_state = TrainingState.IDLE
        for bucket in _buckets(s):
            for fsdp_param in bucket.fsdp_params:
                fsdp_param.sharded_param.grad = None
    state_ctx.iter_forward_root = None
    state_ctx.manual_backward_finalization_active = None
    state_ctx.gradient_reduction_pending = False


def share_context(states: list[Any]) -> None:
    """``share_comm_ctx`` on the engine: the roots share one engine context
    (streams, prefetches, pending reductions), as FSDP2's share one comm
    context."""
    first = states[0]
    context = first._engine_context or first._shared_engine_context
    if context is None:
        context = BucketCommContext.for_device(first._device)
        context.fsdp2_prefetch_release = True
    for state in states:
        old = state._engine_context
        if old is not None and old is not context:
            for runtime in old.buckets:
                runtime.context = context
                context.buckets.append(runtime)
            for callbacks, old_callbacks in (
                (context.backward_start_callbacks, old.backward_start_callbacks),
                (context.backward_end_callbacks, old.backward_end_callbacks),
            ):
                callbacks.extend(c for c in old_callbacks if c not in callbacks)
            for other in state._state_ctx.all_states:
                other._engine_context = context
        state._shared_engine_context = context


def set_all_gather_layout(states: list[Any], layout: AllGatherLayout) -> None:
    buckets = [bucket for state in states for bucket in _buckets(state)]
    for bucket in buckets:
        _check_layout_comm(bucket, layout, type(bucket._all_gather_comm))
        if layout is bucket.all_gather_layout:
            continue
        runtime = bucket.bucket
        if runtime.is_unsharded or any(
            pending.bucket is runtime for pending in runtime.context.pending_unshards
        ):
            raise RuntimeError(
                "cannot replace the all-gather layout while an all-gather is "
                "pending or parameters are unsharded; reshard, or call "
                "reset_iter_state() after a failed unshard"
            )
        if any(p._keep_all_gather_output_storage for p in bucket.fsdp_params):
            raise RuntimeError(
                "cannot replace the all-gather layout after parameters adopted "
                "outputs from the installed custom layout"
            )
    for bucket in buckets:
        was_param_driven = bucket.param_driven_unshard()
        bucket.all_gather_layout = layout
        if bucket.param_driven_unshard() != was_param_driven:
            # The new placement creates its own persistent buffers
            bucket.bucket.unsharded_params = None
            bucket.bucket.persistent_buffers = []
            bucket.bucket.persistent_buffer_nbytes = []
            bucket.layout_buffers = None
            for fsdp_param in bucket.fsdp_params:
                fsdp_param.all_gather_outputs = []
                if hasattr(fsdp_param, "_unsharded_param"):
                    del fsdp_param._unsharded_param
        bucket.refresh()
