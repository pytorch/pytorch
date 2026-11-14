# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed._composable_state import _get_module_state

from . import module_ops
from .api import CPUOffloadPolicy, GradientReductionHandle, MixedPrecisionPolicy
from .bucket_comm import begin_explicit_bucket_unshard
from .bucket_runtime import BucketCommContext, ParamOwnerRef, ParamRepresentation
from .bucket_storage import (
    _assign_params_to_buckets,
    _fsdp2_param_order,
    _hook_module_fqns,
    BucketParamFQNsByIndex,
    BucketSpec,
    GradientReduceOp,
    ParamInfo,
    PlacementFn,
    ShardedBucketStorage,
)
from .module_state import BucketGroup, ShardedModuleState
from .param_homing import build_bucket, ParamSource
from .placements.shard import _engine_placements
from .sharded_param import is_flex_shard_param, set_sharding_info
from .utils import (
    _bucket_debug_fqn,
    _get_device_from_mesh,
    _get_managed_named_params,
    _get_shared_param_names,
    _validate_bucket_uniform_dtype_and_placement,
    _validate_eager_params,
    _validate_flex_shard_mesh,
    _validate_placements,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_runtime import BucketRuntime
    from .placement_contract import Placement


__all__ = [
    "flex_shard",
    "relayout_bucket",
]


class FlexShardRepresentation(ParamRepresentation):
    """flex_shard's parameters: their local tensors, with the sharding
    metadata that flex_shard's checkpointing reads (``set_sharding_info``)."""

    def __init__(self, mesh: DeviceMesh) -> None:
        self.mesh = mesh

    def new_sharded_param(
        self, index: int, local: torch.Tensor, info: ParamInfo
    ) -> nn.Parameter:
        param = super().new_sharded_param(index, local, info)
        self._set_sharding_info(param, info)
        return param

    def refresh_param(
        self, index: int, sharded_param: nn.Parameter, info: ParamInfo
    ) -> None:
        # e.g. load_state_dict(assign=True) registered a plain parameter
        if not is_flex_shard_param(sharded_param):
            self._set_sharding_info(sharded_param, info)

    def _set_sharding_info(self, param: nn.Parameter, info: ParamInfo) -> None:
        set_sharding_info(
            param,
            placements=tuple(p._api_placement() for p in info.placements),
            global_shape=info.global_shape,
            global_stride=info.global_stride,
            mesh=self.mesh,
            outer_layout=info.outer_layout,
        )


class FlexShardState(ShardedModuleState):
    """The state that ``flex_shard()`` puts on its module: its buckets, each
    a group that unshards on the forward of its modules."""

    def __init__(self) -> None:
        super().__init__()
        self.bucket_storages: list[ShardedBucketStorage] = []

    def add_buckets(self, bucket_storages: list[ShardedBucketStorage]) -> None:
        """Run each bucket on the forward of its modules: those its patterns
        name (``BucketSpec.patterns``), or the module that holds all its
        parameters, casting their inputs and outputs as its mixed precision
        policy says."""
        self.bucket_storages = bucket_storages
        module_positions = {
            module: position
            for position, module in enumerate(bucket_storages[0]._module.modules())
        }
        for storage in bucket_storages:
            runtime = cast("BucketRuntime", storage.runtime)
            modules = (
                [
                    _unwrap_checkpoint(storage._module.get_submodule(fqn))
                    for fqn in storage._hook_module_fqns
                ]
                if storage._hook_module_fqns
                else [runtime.forward_hook_module()]
            )
            runtime.module_position = min(module_positions[m] for m in modules)
            casts = (None, None)
            if (policy := storage._mp_policy) is not None:
                input_dtype = policy.param_dtype if policy.cast_forward_inputs else None
                casts = (input_dtype, policy.output_dtype)
            group = BucketGroup(runtime, tuple(modules), named_by_modules=False)
            self.add_group(group, casts)


class FlexShardModule:
    """Mixin added to modules after flex_shard()."""

    @property
    def sharded_bucket_storages(self) -> list[ShardedBucketStorage]:
        """All bucket storage objects, one per non-empty BucketSpec, in order.

        Each takes per-bucket settings, e.g. ``set_requires_gradient_sync``.
        """
        return self._state().bucket_storages

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Any:
        # The buckets re-read the parameters that nn.Module._apply replaced
        # (to_empty(), module.to(dtype)) and keep their data in their storage
        apply = super()._apply  # type: ignore[misc]
        return module_ops.apply_to_params(
            self._runtimes(), lambda: apply(fn, recurse=recurse)
        )

    def _state(self) -> FlexShardState:
        return cast(FlexShardState, _get_module_state(cast(nn.Module, self)))

    def _runtimes(self) -> list[BucketRuntime]:
        return self._state().runtimes

    def _contexts(self) -> list[BucketCommContext]:
        return list({id(r.context): r.context for r in self._runtimes()}.values())

    def reshard(self) -> None:
        """Reshard every unsharded bucket, like FSDP2's ``FSDPModule.reshard()``.

        Buckets with ``reshard_after_forward=False`` stay unsharded after a
        forward until its backward; call this when that backward will not run.
        Until then, ``module.parameters()`` returns their unsharded params, so
        zero grads through the optimizer rather than ``module.zero_grad()``.
        Grads accumulated without sync are kept, in their unsharded dtype.
        """
        module_ops.reshard(self._runtimes())

    def set_manual_backward_finalization(self, enabled: bool) -> None:
        """Set whether the caller finalizes backward, like FSDP2's
        ``set_manual_backward_finalization``.

        When enabled, a backward does not finish FlexShard's work at its end;
        call :meth:`finalize_backward` after the step's last backward. Until
        then, buckets a backward left unsharded stay unsharded and their grads
        stay on the unsharded params. It cannot change during a backward.
        """
        self._state()._set_manual_backward_finalization(enabled)

    def finalize_backward(
        self, *, async_op: bool = False
    ) -> GradientReductionHandle | None:
        """Finish backward on the calling thread, like FSDP2's
        ``finalize_backward``.

        Reduce-scatters, if gradient sync is on, the grads that backwards
        without sync accumulated, finishes buckets a backward left unsharded,
        and resets per-backward state, following the current sync and reshard
        settings. For example, a pipeline stage can reduce its last
        microbatch's grads after its backward, while other stages compute.
        It requires :meth:`set_manual_backward_finalization`, as FSDP2's does.
        Call it after a backward, not between a forward and its backward. With
        ``async_op=True``, it returns a handle without waiting; call its
        ``wait()`` before using the grads or running the next forward. Eager
        only.
        """
        return self._state()._finalize_backward(async_op)

    def unshard(
        self, *, async_op: bool = False
    ) -> module_ops.ModuleUnshardHandle | None:
        """Unshard every bucket and keep it unsharded, like FSDP2's
        ``FSDPModule.unshard()``.

        For callers that run submodules' computation directly, bypassing the
        forward hooks that gather buckets, e.g. Megatron-LM's EP overlap
        schedule, or that gather ahead of a forward, e.g. a multi-stage
        pipeline schedule. Buckets whose hooks never run then stay unsharded,
        with their grads, until the end of a backward that re-gathers a bucket
        in its pre-backward hook, or ``finalize_backward``, finishes them,
        following the sync and reshard settings, or until ``reshard()``. Call
        it outside backward and after waiting on any
        ``finalize_backward`` handle. Eager only.

        As FSDP2's, the all-gathers run on the current stream, with their
        buffers from its memory pool. With ``async_op=True``, the collectives
        are asynchronous and it returns a handle without waiting, so the current
        stream runs other work in between; call the handle's ``wait()`` before
        using the unsharded params directly. A forward finishes its buckets'
        all-gathers without it.
        """
        started = self._state()._unshard(async_op)
        return module_ops.ModuleUnshardHandle(started) if async_op else None

    def finish_deferred_backward(self, param: nn.Parameter) -> None:
        """Finish the backward of the bucket that holds ``param`` and defers its
        post-backward (``BucketSpec.defer_post_backward``).

        With gradient sync on, this reduce-scatters the bucket's grads into the
        local shards and reshards, as its post-backward would have once its
        module's backward was done; without, the grads stay for the next
        syncing backward. Call it once the late grads exist, e.g. after
        TransformerEngine's ``backward_dw()``, inside that backward or before
        ``finalize_backward``. ``param`` is the local shard or, while the
        bucket is unsharded, the unsharded param. Calls after the bucket is
        finished do nothing, so a hook per param may call it. Eager only.
        """
        bucket = self._bucket_holding(param)
        if not bucket.bucket_storage._defer_post_backward:
            raise ValueError(
                f"FlexShard: bucket {bucket.debug_fqn} does not defer its "
                "post-backward (BucketSpec.defer_post_backward)."
            )
        if bucket.needs_finish():
            bucket.post_backward()

    def bucket_storage_of(self, param: nn.Parameter) -> ShardedBucketStorage:
        """The bucket storage that holds ``param``, the local shard or, while
        its bucket is unsharded, the unsharded param. It takes per-bucket
        calls, e.g. ``unshard()``, like the ``fully_shard`` group of an FSDP2
        module."""
        return self._bucket_holding(param).bucket_storage

    def _bucket_holding(self, param: nn.Parameter) -> BucketRuntime:
        for bucket in self._runtimes():
            if any(
                p is param
                for p in (*bucket.sharded_params, *(bucket.unsharded_params or ()))
            ):
                return bucket
        raise ValueError("FlexShard: param is not in this module's buckets.")

    def _bucket_storages(self, recurse: bool) -> list[ShardedBucketStorage]:
        modules = cast(nn.Module, self).modules() if recurse else [self]
        return [
            bucket_storage
            for module in modules
            if isinstance(module, FlexShardModule)
            for bucket_storage in module.sharded_bucket_storages
        ]

    def set_gradient_reduce_op(
        self,
        op: GradientReduceOp,
        *,
        recurse: bool = True,
    ) -> None:
        """Set gradient reduction semantics for this module's FlexShard buckets."""
        for bucket_storage in self._bucket_storages(recurse):
            bucket_storage.set_gradient_reduce_op(op)

    def set_gradient_divide_factor(
        self,
        factor: float | None,
        *,
        recurse: bool = True,
    ) -> None:
        """Set what ``AVG`` divides this module's FlexShard buckets' summed
        gradients by, like FSDP2's ``set_gradient_divide_factor``. None restores
        the default, each bucket's mesh size."""
        module_ops.set_gradient_divide_factor(self._bucket_storages(recurse), factor)

    def set_requires_gradient_sync(
        self,
        requires_gradient_sync: bool,
        *,
        recurse: bool = True,
    ) -> None:
        """Set whether backward reduce-scatters gradients, like FSDP2's.

        With ``False``, backward keeps each bucket's full gradients on its
        unsharded params and later backwards accumulate into them, in
        ``MixedPrecisionPolicy.reduce_dtype`` if set and otherwise as each
        param's ``grad_dtype`` specifies (its dtype if unset). The next
        backward with ``True`` reduce-scatters the accumulated gradients, also
        for buckets it does not use. A backward that raises drops the
        gradients accumulated so far on that rank, since they mix with its
        partial ones.
        It applies to the backwards after the call, e.g.
        ``model.set_requires_gradient_sync(is_last_microbatch)`` before each
        microbatch; like FSDP2, there is no ``no_sync()`` context manager.
        Eager only: torch.compile raises ``NotImplementedError``.

        To set one bucket, e.g. to keep reduce-scattering expert buckets whose
        full gradients are large, call ``set_requires_gradient_sync`` on its
        entry in ``sharded_bucket_storages``.
        """
        module_ops.set_requires_gradient_sync(
            self._bucket_storages(recurse), requires_gradient_sync
        )

    def set_separate_reduce_scatter_group(
        self,
        enable: bool = True,
        *,
        recurse: bool = True,
    ) -> None:
        """Run gradient reduce-scatters on a process group of their own, like
        FSDP2's ``set_separate_reduce_scatter_group`` (experimental).

        A bucket's all-gathers and reduce-scatters run on separate streams but
        by default through its mesh's process group: one NCCL communicator,
        which runs one collective at a time, so a backward all-gather waits
        behind the reduce-scatter issued before it. With ``True``, the
        reduce-scatters run on a dedicated process group over the same ranks
        (``dist.new_group``, one per distinct set of ranks), so the two can
        progress together when the network allows. Like other process-group
        setup, call it on every rank of the buckets' meshes, before the first
        backward. ``False`` goes back to the meshes' groups. 1-D meshes only.
        """
        module_ops.set_separate_reduce_scatter_group(
            self._bucket_storages(recurse), enable
        )

    def set_reshard_after_backward(
        self,
        reshard_after_backward: bool,
        *,
        recurse: bool = True,
    ) -> None:
        """Set whether backward reshards, like FSDP2's.

        With ``False``, buckets stay unsharded after backward, so the next
        forward skips the all-gather: with sync off for all but the last
        microbatch and without reshard-after-forward, each bucket all-gathers
        and reduce-scatters once per optimizer step. Until they reshard,
        ``module.parameters()`` returns their unsharded params. As in FSDP2,
        this also holds in backwards with gradient sync: set ``True`` before the
        last microbatch's backward, or call ``reshard()``, before the optimizer
        step changes the local shards, or the next forward uses the stale
        unsharded params. Entries of ``sharded_bucket_storages`` take the same
        setting per bucket.
        """
        module_ops.set_reshard_after_backward(
            self._bucket_storages(recurse), reshard_after_backward
        )

    def set_reduce_scatter_unused_params(
        self,
        reduce_scatter_unused_params: bool,
        *,
        recurse: bool = True,
    ) -> None:
        """Reduce-scatter zeros for params without grads, like FSDP2's
        ``set_reduce_scatter_unused_params``.

        By default, as in FSDP2, a bucket whose forward did not run since its
        last reduction contributes no grads. A bucket whose forward ran
        reduces every trainable param, with zeros for missing grads, even if
        this rank's backward did not use its outputs, so that every rank issues
        the same collective; a ``fsdp2_compatible`` bucket reduces only the
        params that have grads, as FSDP2. If ranks can skip a module's forward
        entirely, set ``True`` so that every rank reduce-scatters every bucket,
        with zeros for missing grads. Entries of ``sharded_bucket_storages``
        take the same setting per bucket.
        """
        module_ops.set_reduce_scatter_unused_params(
            self._bucket_storages(recurse), reduce_scatter_unused_params
        )

    def set_reduce_scatter_max_input_buffers(self, max_input_buffers: int) -> None:
        """Set how many packed gradient reduce-scatter inputs may be alive at
        once, like FSDP2's ``set_reduce_scatter_max_input_buffers``: by default
        1, so the next bucket's copy-in waits on the previous reduce-scatter;
        more lets the copy-in write a fresh buffer instead, at the cost of
        memory, which helps when the reduce-scatters are exposed."""
        self._state()._set_reduce_scatter_max_input_buffers(max_input_buffers)


@contract(state_cls=FlexShardState)  # type: ignore[misc]
def flex_shard(
    module: nn.Module,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Installs per-bucket forward hooks that unshard into persistent unsharded
       parameters (FSDP2-style) swapped into the owning modules
    6. Stores ShardedBucketStorage objects on the module

    Each bucket gets its own byte buffer and ShardedBucketStorage, enabling
    independent unshard operations per bucket.

    Args:
        module: The module to shard. CPU modules are moved to the bucket mesh's
            CUDA device before sharding. Meta parameters keep uninitialized
            storage.
        buckets: Required list of bucket specifications. Each bucket owns its
            1D CUDA mesh and its placement function. A bucket is one collective
            over one process group, so the mesh lives on the bucket; different
            buckets may use different meshes (all sharing one device type). A
            single whole-module bucket can be expressed as
            ``[BucketSpec(["*"], placement_fn=per_param_placements, mesh=mesh)]``.

    Parameters must be plain tensors. Parameters that are local shards of an
    outer (e.g. TP/EP) sharding declare their position in the full parameter
    with ``set_global_layout``; ``flex_shard.layout_adapters.dtensor_to_global_layout``
    does this for DTensor parameters. A parameter whose grad each rank of a
    group computes only in part declares the group with
    ``set_partial_grad_group``, and its grad is summed over it before the
    reduce-scatter.

    Returns:
        The module (mutated in-place). Use
        module.sharded_bucket_storages to inspect bucket storage internals.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> # Single bucket without reshard-after-forward:
        >>> flex_shard(
        ...     model,
        ...     buckets=[
        ...         BucketSpec(
        ...             ["*"],
        ...             placement_fn=per_param_placements,
        ...             mesh=mesh,
        ...             reshard_after_forward=False,
        ...         )
        ...     ],
        ... )
        >>> # Explicit buckets, each carrying its own mesh:
        >>> flex_shard(
        ...     model,
        ...     buckets=[
        ...         BucketSpec(["attn.*"], placement_fn=per_param_placements, mesh=mesh),
        ...         BucketSpec(["ffn.*"], placement_fn=per_param_placements, mesh=mesh),
        ...     ],
        ... )
    Note:
        - Each parameter must have exactly one placement.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - A module whose parameters are all on meta gets meta storage until
          ``to_empty()``; parameters on meta while others are not must be
          replaced (e.g. ``load_state_dict(assign=True)``) before the first
          forward
    """
    inputs = _prepare_flex_shard_inputs(
        module,
        buckets,
    )

    bucket_storages = _materialize_bucket_storages(
        module,
        inputs,
        buckets,
    )

    return _install_state(module, flex_shard.state(module), bucket_storages)


def _install_state(
    module: nn.Module,
    state: FlexShardState,
    bucket_storages: list[ShardedBucketStorage],
) -> FlexShardModule:
    """Put ``state``, running ``bucket_storages``, on ``module``, which
    becomes a ``FlexShardModule``."""
    state.add_buckets(bucket_storages)
    state.init(
        (module,),
        _get_device_from_mesh(bucket_storages[0]._mesh),
        (None, None),
        auto_reshard_after_forward=False,
        root_module=module,
        post_hook_always_call=True,
    )
    cls = type(module)
    module.__class__ = type(cls.__name__, (FlexShardModule, cls), {})
    return cast(FlexShardModule, module)


def relayout_bucket(
    module: FlexShardModule,
    bucket_storage: ShardedBucketStorage,
    placement_fn: PlacementFn,
) -> ShardedBucketStorage:
    """Move one bucket's parameters to the placements ``placement_fn`` gives,
    moving their values once.

    For a layout that initialization or checkpoint loading can't use, e.g.
    whole-parameter owners (``BucketedOwned``) where initialization goes
    through FSDP2-style DTensor views of ``Shard`` placements: initialize or
    load in the first layout, then move. Call it after the module is
    materialized and before the bucket's first unshard, and before an
    optimizer holds its parameters: the bucket's parameters are replaced, and
    the old storage is freed.

    The parameters are all-gathered in their storage dtype and copied into the
    new storage, so their values move bitwise. The memory peak is this
    bucket's old shards, its full parameters and its new storage. The bucket
    keeps its mesh, mixed precision, reshard and gradient settings, prefetch
    and forward hooks, and is no longer ``fsdp2_compatible``. Returns the new
    storage, which replaces the old one in ``module.sharded_bucket_storages``.
    """
    storages = module.sharded_bucket_storages
    index = next(
        (i for i, storage in enumerate(storages) if storage is bucket_storage), None
    )
    if index is None:
        raise ValueError("bucket_storage is not one of the module's bucket storages.")
    device = bucket_storage.byte_storage.device
    if device.type == "meta":
        raise ValueError(
            "relayout_bucket needs materialized storage; call to_empty() first."
        )
    if bucket_storage._cpu_offload is not None:
        raise NotImplementedError("relayout_bucket does not support CPU offload yet.")
    runtime = cast("BucketRuntime", bucket_storage.runtime)
    if runtime.is_unsharded or runtime.unsharded_params:
        raise RuntimeError(
            "relayout_bucket must run before the bucket's first unshard, e.g. "
            "right after initialization or checkpoint loading."
        )
    infos = list(bucket_storage.param_infos.values())
    sharded_params = [
        _module_param(cast(nn.Module, module), info.fqn) for info in infos
    ]
    if any(param.grad is not None for param in sharded_params):
        raise RuntimeError("relayout_bucket needs a bucket without gradients.")

    mesh = bucket_storage._mesh
    with torch.no_grad():
        # Gather in the storage dtype, not the forward dtype, so master weights
        # move without rounding.
        result = begin_explicit_bucket_unshard(
            [param.detach() for param in sharded_params],
            [replace(info, param_dtype=None) for info in infos],
            mesh,
            async_op=False,
        ).finish()
    named_full_params: list[tuple[str, nn.Parameter]] = []
    for info, full in zip(infos, result.full_params, strict=True):
        param = nn.Parameter(full.contiguous(), requires_grad=info.requires_grad)
        # The original parameter's attributes, with any outer layout and
        # partial-grad group it declared, as flex_shard() recorded them.
        for name, value in info.param_attrs.items():
            setattr(param, name, value)
        if info.has_explicit_grad_dtype:
            param.grad_dtype = info.grad_dtype_override
        named_full_params.append((info.fqn, param))
    del result

    param_placements = _engine_placements(placement_fn(named_full_params, mesh))
    _validate_placements(param_placements, named_full_params)
    full_params = [param for _, param in named_full_params]
    build_bucket(
        _param_sources(
            bucket_storage._module,
            named_full_params,
            param_placements,
            {info.fqn: list(info.shared_fqns) for info in infos},
        ),
        mesh=mesh,
        module=bucket_storage._module,
        device=_get_device_from_mesh(mesh),
        context=runtime.context,
        representation=runtime.representation,
        mp_policy=_mixed_precision_policy_of(infos, full_params),
        extension_mesh=mesh,
        storage_options={
            "reshard_after_forward": bucket_storage.reshard_after_forward,
            "gradient_reduce_op": bucket_storage.gradient_reduce_op,
            "pre_backward_hook": bucket_storage._pre_backward_hook,
            "post_reduce_hook": bucket_storage._post_reduce_hook,
            "gradient_divide_factor": bucket_storage.gradient_divide_factor,
            "defer_post_backward": bucket_storage._defer_post_backward,
            "hook_module_fqns": bucket_storage._hook_module_fqns,
            "mp_policy": bucket_storage._mp_policy,
        },
        runtime=runtime,
    )
    new_storage = runtime.bucket_storage
    new_storage._requires_gradient_sync = bucket_storage._requires_gradient_sync
    new_storage._reshard_after_backward = bucket_storage._reshard_after_backward
    for storage in storages:
        for attr in ("_forward_prefetch", "_backward_prefetch"):
            prefetch = getattr(storage, attr)
            if prefetch is not None:
                setattr(
                    storage,
                    attr,
                    [new_storage if s is bucket_storage else s for s in prefetch],
                )
    new_storage._forward_prefetch = bucket_storage._forward_prefetch
    new_storage._backward_prefetch = bucket_storage._backward_prefetch
    storages[index] = new_storage
    return new_storage


def _unwrap_checkpoint(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def _module_param(module: nn.Module, fqn: str) -> nn.Parameter:
    return ParamOwnerRef.resolve(module, fqn).param


def _mixed_precision_policy_of(
    infos: list[ParamInfo], params: list[nn.Parameter]
) -> MixedPrecisionPolicy:
    """The policy that gives each of ``params`` the dtypes that flex_shard()
    resolved for its ``infos`` entry."""
    param_dtypes = {info.param_dtype for info in infos}
    reduce_dtypes = {info.reduce_dtype for info in infos}
    if len(reduce_dtypes) != 1:
        raise AssertionError(f"Expected one reduce dtype, got {reduce_dtypes}")
    reduce_dtype = reduce_dtypes.pop()
    if len(param_dtypes) == 1:
        return MixedPrecisionPolicy(
            param_dtype=param_dtypes.pop(), reduce_dtype=reduce_dtype
        )
    # Overrides may name the parameter's own dtype, the only value other than
    # param_dtype they allow.
    param_dtype = next(
        info.param_dtype for info in infos if info.param_dtype != info.dtype
    )
    overrides = {
        id(param): info.unsharded_dtype
        for info, param in zip(infos, params, strict=True)
        if info.param_dtype != param_dtype
    }
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        param_dtype_override_fn=lambda param: overrides.get(id(param)),
    )


def _check_not_already_flex_sharded(module: nn.Module) -> None:
    """Raise if applying FlexShard would create nested ownership."""
    for child_fqn, child in module.named_modules():
        if child_fqn and isinstance(child, FlexShardModule):
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Child module {child_fqn!r} is already FlexSharded. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
            )
    for param_fqn, param in module.named_parameters(remove_duplicate=False):
        if is_flex_shard_param(param):
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Parameter {param_fqn!r} is already managed by FlexShard. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
            )


@dataclass(frozen=True)
class PreparedFlexShardInputs:
    """Validated inputs and derived setup state for flex_shard()."""

    named_params: list[tuple[str, nn.Parameter]]
    device: torch.device
    param_placements: dict[str, tuple[Placement, ...]]
    bucket_assignments: BucketParamFQNsByIndex
    # Shared parameters: first name -> other names, all in the same bucket.
    shared_param_names: dict[str, list[str]] = field(default_factory=dict)


def _materialize_bucket_storages(
    module: nn.Module,
    inputs: PreparedFlexShardInputs,
    buckets: list[BucketSpec],
) -> list[ShardedBucketStorage]:
    """Create ShardedBucketStorage objects, with their runtimes in one
    context, and install sharded parameters."""
    named_params_dict = dict(inputs.named_params)
    context = BucketCommContext.for_device(_get_device_from_mesh(buckets[0].mesh))
    bucket_storages: list[ShardedBucketStorage] = []

    for bucket_idx, bucket_fqns in enumerate(inputs.bucket_assignments):
        if not bucket_fqns:
            continue

        bucket_spec = buckets[bucket_idx]
        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        if bucket_spec.fsdp2_compatible:
            bucket_named_params = _fsdp2_param_order(
                module, bucket_spec.patterns, bucket_named_params
            )
        bucket_storages.append(
            build_flex_bucket(
                module,
                bucket_named_params,
                inputs.param_placements,
                bucket_spec,
                inputs.shared_param_names,
                context,
            )
        )

    return bucket_storages


def build_flex_bucket(
    module: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    param_placements: dict[str, tuple[Placement, ...]],
    bucket_spec: BucketSpec,
    shared_names: dict[str, list[str]] | None = None,
    context: BucketCommContext | None = None,
) -> ShardedBucketStorage:
    """One ``flex_shard`` bucket of ``named_params`` and its runtime in
    ``context`` (a new one by default), keeping the parameters' shards in its
    storage and installing sharded parameters that view them."""
    mesh = bucket_spec.mesh
    device = _get_device_from_mesh(mesh)
    offload = bucket_spec.offload_policy
    runtime = build_bucket(
        _param_sources(module, named_params, param_placements, shared_names or {}),
        mesh=mesh,
        module=module,
        device=device,
        context=context or BucketCommContext.for_device(device),
        representation=FlexShardRepresentation(mesh),
        mp_policy=bucket_spec.mp_policy or MixedPrecisionPolicy(),
        extension_mesh=mesh,
        storage_options={
            "reshard_after_forward": bucket_spec.reshard_after_forward,
            "gradient_reduce_op": bucket_spec.gradient_reduce_op,
            "pre_backward_hook": bucket_spec.pre_backward_hook,
            "post_reduce_hook": bucket_spec.post_reduce_hook,
            "gradient_divide_factor": bucket_spec.gradient_divide_factor,
            "defer_post_backward": bucket_spec.defer_post_backward,
            "hook_module_fqns": _hook_module_fqns(module, bucket_spec.patterns),
            "fsdp2_compatible": bucket_spec.fsdp2_compatible,
            "mp_policy": bucket_spec.mp_policy,
        },
        offload_policy=offload if isinstance(offload, CPUOffloadPolicy) else None,
        debug_fqn=_bucket_debug_fqn(module, [fqn for fqn, _ in named_params]),
    )
    return runtime.bucket_storage


def _param_sources(
    module: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    param_placements: dict[str, tuple[Placement, ...]],
    shared_names: dict[str, list[str]],
) -> list[ParamSource]:
    sources = []
    for fqn, param in named_params:
        shared = tuple(shared_names.get(fqn, ()))
        sources.append(
            ParamSource(
                param,
                ParamOwnerRef.resolve(module, fqn),
                tuple(ParamOwnerRef.resolve(module, name) for name in shared),
                fqn,
                param,
                param_placements[fqn][0],
                shared_fqns=shared,
            )
        )
    return sources


def _resolve_bucket_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    bucket_assignments: BucketParamFQNsByIndex,
    buckets: list[BucketSpec],
) -> dict[str, tuple[Placement, ...]]:
    """Call each bucket's placement function for the params assigned to it."""
    named_params_dict = dict(named_params)
    param_placements: dict[str, tuple[Placement, ...]] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue
        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_param_placements = _engine_placements(
            buckets[bucket_idx].placement_fn(
                bucket_named_params, buckets[bucket_idx].mesh
            )
        )
        _validate_placements(bucket_param_placements, bucket_named_params)
        param_placements.update(bucket_param_placements)

    _validate_placements(param_placements, named_params)
    return param_placements


def _validate_fsdp2_compatible_buckets(
    bucket_assignments: BucketParamFQNsByIndex,
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[BucketSpec],
) -> None:
    for bucket, bucket_fqns in zip(buckets, bucket_assignments, strict=True):
        if not bucket.fsdp2_compatible:
            continue
        for fqn in bucket_fqns:
            (placement,) = param_placements[fqn]
            if not placement.matches_fsdp2_collectives:
                raise ValueError(
                    f"BucketSpec {bucket.patterns} sets fsdp2_compatible, which "
                    "requires placements whose collectives match FSDP2's, such as "
                    f"Shard, but {fqn!r} uses {placement!r}."
                )


def _validate_offloaded_buckets(
    bucket_assignments: BucketParamFQNsByIndex,
    param_placements: dict[str, tuple[Placement, ...]],
    buckets: list[BucketSpec],
) -> None:
    for bucket, bucket_fqns in zip(buckets, bucket_assignments, strict=True):
        if not isinstance(bucket.offload_policy, CPUOffloadPolicy):
            continue
        for fqn in bucket_fqns:
            (placement,) = param_placements[fqn]
            if not placement.supports_cpu_offload:
                raise NotImplementedError(
                    f"BucketSpec {bucket.patterns} sets CPUOffloadPolicy, which "
                    "requires placements that support CPU offload, such as Shard, "
                    f"but {fqn!r} uses {placement!r}."
                )


def _prepare_flex_shard_inputs(
    module: nn.Module,
    buckets: list[BucketSpec],
) -> PreparedFlexShardInputs:
    """Validate inputs and derive setup state for flex_shard()."""
    _check_not_already_flex_sharded(module)

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")
    for bucket in buckets:
        if not callable(bucket.placement_fn):
            raise TypeError("BucketSpec.placement_fn must be callable.")
        _validate_flex_shard_mesh(bucket.mesh)
    # A bucket is one collective over one process group, so the mesh is a
    # per-bucket property. Buckets may use different meshes, but this rank has a
    # single local device, so require all bucket meshes to share a device type.
    device_types = {bucket.mesh.device_type for bucket in buckets}
    if len(device_types) > 1:
        raise ValueError(
            "All BucketSpec meshes in one flex_shard() call must share a device "
            f"type, but got {sorted(device_types)}."
        )

    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )

    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    device = (
        torch.device("meta")
        if all_params_meta
        else _get_device_from_mesh(buckets[0].mesh)
    )
    if not all_params_meta and all(
        param.device.type == "cpu" for _, param in named_params
    ):
        module.to(device)
        named_params = _get_managed_named_params(module)

    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    param_fqns = [fqn for fqn, _ in named_params]
    shared_param_names = _get_shared_param_names(module)
    bucket_assignments = _assign_params_to_buckets(
        param_fqns, buckets, shared_param_names
    )
    param_placements = _resolve_bucket_param_placements(
        named_params,
        bucket_assignments,
        buckets,
    )
    _validate_bucket_uniform_dtype_and_placement(
        bucket_assignments,
        param_placements,
        buckets,
        named_params,
    )
    _validate_fsdp2_compatible_buckets(bucket_assignments, param_placements, buckets)
    _validate_offloaded_buckets(bucket_assignments, param_placements, buckets)

    return PreparedFlexShardInputs(
        named_params=named_params,
        device=device,
        param_placements=param_placements,
        bucket_assignments=bucket_assignments,
        shared_param_names=shared_param_names,
    )
