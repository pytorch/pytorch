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

import functools
import logging
import weakref
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .._flex_shard import module_ops
from .._flex_shard.bucket_runtime import (
    BucketCommContext,
    BucketRuntime,
    ParamRepresentation,
)
from .._flex_shard.bucket_storage import ParamMeta, ShardedBucketStorage
from .._flex_shard.module_ops import GroupMesh
from .._flex_shard.module_state import BucketGroup
from .._flex_shard.param_homing import build_homed_bucket, param_owners, ParamSource
from .._flex_shard.placements.shard import Shard as EngineShard
from .._flex_shard.placements.shard_all_gather import ShardAllGather
from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_common import _from_local_no_grad, FSDPMeshInfo
from ._fsdp_param import (
    _get_dtensor_spec_with_dtype,
    FSDPParam,
    ParamModuleInfo,
    ShardedState,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from .._flex_shard.all_gather_layout import AllGatherLayout
    from .._flex_shard.bucket_storage import ParamInfo
    from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
    from ._fsdp_common import DataParallelMeshInfo, ShardPlacementFnResult


__all__: list[str] = []


class FSDPParamRepresentation(ParamRepresentation):
    """Exposes a bucket's parameters as FSDP2 does: sharded DTensor parameters
    whose local tensors view the bucket, and unsharded parameters that are
    plain tensors, or DTensors over the non-data-parallel mesh dims (TP/EP)."""

    def __init__(self, fsdp_params: list[FSDPParam]) -> None:
        self.fsdp_params = fsdp_params

    def local_shard(
        self, index: int, sharded_param: nn.Parameter, *, use_autograd: bool
    ) -> torch.Tensor:
        dtensor = cast(DTensor, sharded_param)
        if use_autograd:
            return dtensor.to_local()
        return dtensor._local_tensor.detach()

    def local_tensor(self, sharded_param: nn.Parameter) -> torch.Tensor:
        return cast(DTensor, sharded_param)._local_tensor

    def set_local_tensor(
        self, index: int, sharded_param: nn.Parameter, local: torch.Tensor
    ) -> None:
        cast(DTensor, sharded_param)._local_tensor = local

    def new_sharded_param(
        self, index: int, local: torch.Tensor, info: ParamInfo
    ) -> nn.Parameter:
        param = nn.Parameter(
            self.fsdp_params[index].to_sharded_dtensor(local),
            requires_grad=info.requires_grad,
        )
        if info.has_explicit_grad_dtype:
            param.grad_dtype = info.grad_dtype_override
        return param

    def refresh_param(self, index: int, sharded_param: nn.Parameter) -> None:
        self.fsdp_params[index].refresh(sharded_param)

    def unsharded_param(self, index: int, full_param: torch.Tensor) -> torch.Tensor:
        fsdp_param = self.fsdp_params[index]
        spec = fsdp_param._unsharded_dtensor_spec
        if fsdp_param.is_spmd_types or spec is None or isinstance(full_param, DTensor):
            return full_param
        return _from_local_no_grad(
            full_param, _get_dtensor_spec_with_dtype(spec, full_param.dtype)
        )

    def unsharded_local(
        self, index: int, unsharded_param: nn.Parameter
    ) -> torch.Tensor:
        if isinstance(unsharded_param, DTensor):
            return unsharded_param._local_tensor
        return unsharded_param

    def unsharded_grad(self, index: int, grad: torch.Tensor) -> torch.Tensor:
        return self.fsdp_params[index]._get_grad_inner_tensor(grad)

    def check_post_forward_reshard(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.check_post_forward_reshard()
        super().check_post_forward_reshard()

    def defers_grad_upcast(self, index: int) -> bool | None:
        # Gradients reduced outside DP keep autograd's upcast so that
        # reduction runs in the wider dtype too
        if self.fsdp_params[index].may_reduce_grad_outside_dp:
            return False
        return super().defers_grad_upcast(index)

    def on_sharded_state(self, state: str) -> None:
        # FSDPParam.sharded_state, which tools read, as FSDP2 sets it
        sharded_state = {
            "unsharded": ShardedState.UNSHARDED,
            "sharded": ShardedState.SHARDED,
            "sharded_post_forward": ShardedState.SHARDED_POST_FORWARD,
        }[state]
        for fsdp_param in self.fsdp_params:
            fsdp_param.sharded_state = sharded_state

    def sharded_grads(
        self,
        indices: list[int],
        sharded_params: list[nn.Parameter],
        grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Each sharded grad's dtype: an explicit grad_dtype, or the parameter's
        dtypes = [
            param.grad_dtype if param._has_grad_dtype_override else param.dtype
            for param in sharded_params
        ]
        casts = [
            dtype is not None and grad.dtype != dtype
            for grad, dtype in zip(grads, dtypes, strict=True)
        ]
        # As FSDP2: views of uncast grads keep the whole reduce output alive,
        # including the regions cast into their own buffers, so copy the uncast
        # grads when those regions are larger
        cast_numel = sum(g.numel() for g, cast in zip(grads, casts) if cast)
        copy_uncast = cast_numel > sum(g.numel() for g in grads) - cast_numel
        cast_grads = _cast_grad_runs(grads, dtypes, casts)
        return [
            cast_grad if needs_cast else grad.clone() if copy_uncast else grad
            for grad, cast_grad, needs_cast in zip(
                grads, cast_grads, casts, strict=True
            )
        ]

    def store_sharded_grads(
        self,
        indices: list[int],
        sharded_params: list[nn.Parameter],
        grads: list[torch.Tensor],
    ) -> None:
        for index, sharded_param, grad in zip(
            indices, sharded_params, grads, strict=True
        ):
            hooks = getattr(sharded_param, "_post_accumulate_grad_hooks", None)
            if sharded_param.grad is not None:
                if not isinstance(sharded_param.grad, DTensor):
                    raise AssertionError(
                        "Expected the sharded gradient to be a DTensor, got "
                        f"{type(sharded_param.grad)}"
                    )
                sharded_param.grad._local_tensor += grad
            else:
                sharded_param.grad = self.fsdp_params[index].to_sharded_dtensor(grad)
            for hook in (hooks or {}).values():
                hook(sharded_param)


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
            # pyrefly: ignore [no-matching-overload]
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


class FullyShardBucket(BucketGroup):
    """One ``fully_shard`` group (a call's parameters on one mesh): an engine
    bucket that stores, gathers and reduces the parameters, and their DTensor
    views (``FSDPParam``)."""

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
        self.mesh_info = mesh_info
        self.device = device
        owners = param_owners(params, modules)
        self.fsdp_params = [
            FSDPParam(
                param,
                ParamModuleInfo(
                    owner.module,
                    owner.param_name,
                    [shared.module for shared in shared_owners],
                    [shared.param_name for shared in shared_owners],
                ),
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy._resolve_for_param(param),
            )
            for param, (owner, shared_owners) in zip(params, owners, strict=True)
        ]
        shard_info = mesh_info.shard_info
        self.mesh = GroupMesh(
            shard_info.shard_process_group if shard_info is not None else None,
            device.type,
        )
        self._lazy_initialized = False
        representation = FSDPParamRepresentation(self.fsdp_params)
        for index, fsdp_param in enumerate(self.fsdp_params):
            fsdp_param._group = self
            fsdp_param._index = index
        replicate_info = mesh_info.replicate_info
        runtime, self.homing = build_homed_bucket(
            [
                ParamSource(
                    param,
                    owner,
                    shared_owners,
                    f"{index}.{owner.param_name}",
                    fsdp_param.shard_source,
                    EngineShard(fsdp_param.fsdp_placement.dim),
                    ParamMeta(
                        requires_grad=fsdp_param._requires_grad,
                        has_explicit_grad_dtype=fsdp_param._init_grad_dtype_override,
                        grad_dtype_override=fsdp_param._init_grad_dtype
                        if fsdp_param._init_grad_dtype_override
                        else None,
                    ),
                    fsdp_param.mp_policy,
                    # The DTensor's local shape, which keeps the nonzero dims
                    # of an empty shard (_get_dim_chunked_size)
                    fsdp_param.sharded_size,
                )
                for index, (param, (owner, shared_owners), fsdp_param) in enumerate(
                    zip(params, owners, self.fsdp_params, strict=True)
                )
            ],
            mesh=self.mesh,
            device=device,
            module=modules[0],
            representation=representation,
            extension_mesh=self.fsdp_params[0].shard_mesh_from_root,
            context=new_engine_context(device),
            offload_policy=offload_policy
            if isinstance(offload_policy, CPUOffloadPolicy)
            else None,
            replicate_group=replicate_info.replicate_process_group
            if replicate_info is not None
            else None,
            reshard_after_forward=post_forward_mesh_info is not None,
            post_forward_group=self._post_forward_group(post_forward_mesh_info),
            logger=logging.getLogger("torch.distributed.fsdp.fully_shard"),
            log_name="FSDP",
        )
        for fsdp_param in self.fsdp_params:
            del fsdp_param.shard_source
        super().__init__(runtime, modules)
        # FSDP2's set_reduce_scatter_max_input_buffers
        self.max_input_buffers = 1
        self.has_spmd_types = any(p.is_spmd_types for p in self.fsdp_params)
        self.bucket.context.buckets.append(self.bucket)
        if len(modules) > 1:
            self.bucket.group_modules = modules
        # As FSDP2: state dicts read and load the sharded parameters, which a
        # load with assign=True replaces
        for module in modules:
            module.register_state_dict_pre_hook(self._reshard_hook)
            module._register_load_state_dict_pre_hook(self._reshard_hook)
        # A weak reference, so that a submodule kept alive does not keep the
        # whole module alive
        reload_hook = functools.partial(_reload_params, weakref.ref(self))
        for module in {owner.module for owner, _ in owners}:
            module.register_load_state_dict_post_hook(reload_hook)

    def _reshard_hook(self, *args: Any, **kwargs: Any) -> None:
        if self.bucket.is_unsharded:
            self.bucket.reshard()

    @property
    def all_gather(self) -> ShardAllGather:
        """The group's all-gather, whose outputs and layout persist across
        unshards (the engine ``Shard``'s bucket state)."""
        return cast(ShardAllGather, self.bucket.placement_state)

    @property
    def all_gather_layout(self) -> AllGatherLayout:
        """``FSDPModule.set_all_gather_layout``'s layout."""
        return self.all_gather.layout

    # FSDP2's parameter groups had this hook (e.g. set by APF), which an
    # assignment would otherwise silently replace with an unused attribute
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

    def set_post_forward_mesh_info(
        self, post_forward_mesh_info: FSDPMeshInfo | None
    ) -> None:
        """Reshard after forward (``post_forward_mesh_info`` not None), to the
        smaller post-forward mesh for an int ``reshard_after_forward``."""
        self.storage._reshard_after_forward = post_forward_mesh_info is not None
        self.storage._post_forward_group = self._post_forward_group(
            post_forward_mesh_info
        )

    def _post_forward_group(
        self, post_forward_mesh_info: FSDPMeshInfo | None
    ) -> dist.ProcessGroup | None:
        if post_forward_mesh_info is None or post_forward_mesh_info is self.mesh_info:
            return None
        return post_forward_mesh_info.shard_process_group

    def homed(self) -> bool:
        return self.homing.homed()

    def refresh(self) -> None:
        """Re-home parameters replaced since the last unshard (see
        ``ParamHoming``)."""
        self.homing.refresh()

    def lazy_init(self) -> None:
        """Re-read parameters replaced since construction and their mixed
        precision dtypes, as FSDP2's group lazy init does, once."""
        if self._lazy_initialized:
            return
        self._lazy_initialized = True
        self.homing.reset_sharded_params()
        self.homing.init_extensions()
        meta = [
            fsdp_param._display_name
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
        self.homing.refresh()
        self._check_uniform_dtypes()

    def _check_uniform_dtypes(self) -> None:
        """FSDP2's check that the group's original and reduce dtypes are
        uniform over its trainable parameters."""
        infos = self.bucket.infos
        trainable = [info for info in infos if info.requires_grad]
        infos_for_dtype = trainable or [
            info for info in infos if info.dtype.is_floating_point
        ]
        orig_dtypes = {info.dtype for info in infos_for_dtype}
        reduce_dtypes = {info.reduce_dtype for info in infos_for_dtype}
        if trainable and len(orig_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        if trainable and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )

    def join(self, context: BucketCommContext, *, reshard_after_forward: bool) -> None:
        """Join the root module's engine context, keying the parameters' infos
        by the FQNs that the root gave them."""
        self.lazy_init()
        self.homing.rename(
            [
                fsdp_param._param_fqn or info.fqn
                for fsdp_param, info in zip(
                    self.fsdp_params, self.bucket.infos, strict=True
                )
            ]
        )
        super().join(context, reshard_after_forward=reshard_after_forward)


def _reload_params(group_ref: weakref.ref[FullyShardBucket], *args: Any) -> None:
    """Re-read a group's parameters after ``load_state_dict``, which may
    replace them (``assign=True``); lazy init or the next explicit unshard
    copies replaced data into the storage (``refresh``), as in FSDP2."""
    if (group := group_ref()) is not None:
        group.homing.reset_sharded_params()


# FSDPModule methods on the engine #
def _buckets(state: Any) -> list[FullyShardBucket]:
    return list(state._fsdp_param_groups)


def runtimes(state: Any) -> list[BucketRuntime]:
    """The engine runtimes of the state's groups."""
    return [bucket.bucket for bucket in state._fsdp_param_groups]


def storages(state: Any) -> list[ShardedBucketStorage]:
    """The engine storages of the state's groups."""
    return [bucket.storage for bucket in state._fsdp_param_groups]


def first_storage(state: Any) -> ShardedBucketStorage | None:
    """The storage of the state's first group, which holds the settings that
    the ``FSDPModule`` setters give all its groups."""
    buckets = _buckets(state)
    return buckets[0].storage if buckets else None


def unshard(state: Any, async_op: bool) -> list[BucketRuntime]:
    """Start (``async_op``) or finish the module's unshards on the current
    stream, as ``FSDPModule.unshard`` does; return the buckets to wait on."""
    for bucket in _buckets(state):
        bucket.lazy_init()
        bucket.refresh()
    return module_ops.unshard(runtimes(state), async_op)


def set_reshard_after_forward(state: Any, reshard_after_forward: bool | int) -> None:
    from ._fsdp_init import _get_post_forward_mesh_info

    for bucket in _buckets(state):
        if (shard_info := bucket.mesh_info.shard_info) is None:
            raise AssertionError(
                f"Expected a mesh that shards parameters, got {bucket.mesh_info}"
            )
        bucket.set_post_forward_mesh_info(
            _get_post_forward_mesh_info(reshard_after_forward, shard_info)
        )


def set_modules_to_prefetch(
    state: Any, target_states: list[Any], *, forward: bool
) -> None:
    targets = [
        runtime
        for target_state in target_states
        for runtime in (
            runtimes(target_state) if forward else reversed(runtimes(target_state))
        )
    ]
    module_ops.set_prefetch(runtimes(state), targets, forward=forward)


def set_reduce_scatter_max_input_buffers(state: Any, max_input_buffers: int) -> None:
    for bucket in _buckets(state):
        bucket.max_input_buffers = max_input_buffers
    if state._engine_context is not None:
        root_states = state._state_ctx.all_states
        state._engine_context.max_pending_reduce_grads = max(
            (b.max_input_buffers or 1 for s in root_states for b in _buckets(s)),
            default=1,
        )


def new_engine_context(device: torch.device) -> BucketCommContext:
    """The comm context of a root's buckets."""
    return BucketCommContext.for_device(device)
