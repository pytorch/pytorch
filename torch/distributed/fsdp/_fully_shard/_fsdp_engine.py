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

import logging
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .._flex_shard import module_ops
from .._flex_shard.bucket_runtime import BucketCommContext, ParamRepresentation
from .._flex_shard.bucket_storage import ParamMeta
from .._flex_shard.module_ops import GroupMesh
from .._flex_shard.module_state import BucketGroup
from .._flex_shard.param_homing import build_bucket, param_owners, ParamSource
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

    def refresh_param(
        self, index: int, sharded_param: nn.Parameter, info: ParamInfo
    ) -> None:
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
        runtime = build_bucket(
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
                )
                for index, (param, (owner, shared_owners), fsdp_param) in enumerate(
                    zip(params, owners, self.fsdp_params, strict=True)
                )
            ],
            mesh=self.mesh,  # type: ignore[arg-type]
            module=modules[0],
            device=device,
            context=new_engine_context(device),
            representation=representation,
            mp_policy=mp_policy,
            extension_mesh=self.fsdp_params[0].shard_mesh_from_root,
            storage_options={
                "reshard_after_forward": post_forward_mesh_info is not None,
                "fsdp2_compatible": True,
            },
            offload_policy=offload_policy
            if isinstance(offload_policy, CPUOffloadPolicy)
            else None,
            replicate_group=replicate_info.replicate_process_group
            if replicate_info is not None
            else None,
            post_forward_group=self._post_forward_group(post_forward_mesh_info),
            logger=logging.getLogger("torch.distributed.fsdp.fully_shard"),
            log_name="FSDP",
        )
        self.homing = runtime.homing
        for fsdp_param in self.fsdp_params:
            del fsdp_param.shard_source
        super().__init__(runtime, modules)
        self.has_spmd_types = any(p.is_spmd_types for p in self.fsdp_params)

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

    def lazy_init(self) -> None:
        if self._lazy_initialized:
            return
        super().lazy_init()
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
        self.homing.rename(
            [
                fsdp_param._param_fqn or info.fqn
                for fsdp_param, info in zip(
                    self.fsdp_params, self.bucket.infos, strict=True
                )
            ]
        )
        super().join(context, reshard_after_forward=reshard_after_forward)


# FSDPModule methods on the engine #
def set_reshard_after_forward(state: Any, reshard_after_forward: bool | int) -> None:
    from ._fsdp_init import _get_post_forward_mesh_info

    for bucket in state._fsdp_param_groups:
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
            target_state.runtimes if forward else reversed(target_state.runtimes)
        )
    ]
    module_ops.set_prefetch(state.runtimes, targets, forward=forward)


def new_engine_context(device: torch.device) -> BucketCommContext:
    """The comm context of a root's buckets."""
    return BucketCommContext.for_device(device)
