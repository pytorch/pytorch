# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._pytree import tree_leaves, tree_map_only
from torch.utils.hooks import RemovableHandle

from .bucket_comm import (
    begin_bucket_unshard,
    begin_explicit_bucket_unshard,
    begin_reduce_grad,
    ReduceGradHandle,
    UnshardHandle,
)
from .bucket_storage import ParamInfo, ShardedBucketStorage
from .utils import (
    _get_bucket_storage_debug_fqn,
    _module_path_common_prefix,
    _outside_selective_checkpoint,
    _record_function_if_eager,
    alloc_grads_in_param_layout,
)


_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_MAX_PENDING_REDUCE_GRADS_ATTR = "_flex_shard_max_pending_reduce_grads"


@dataclass
class ParamOwnerRef:
    """Owning module and local parameter name for a managed parameter."""

    module: nn.Module
    param_name: str

    @classmethod
    def resolve(cls, root_module: nn.Module, fqn: str) -> ParamOwnerRef:
        """Resolve a parameter FQN from root, unwrapping checkpoint wrappers."""
        parts = fqn.split(".")
        leaf_module = root_module
        for part in parts[:-1]:
            child = getattr(leaf_module, part, None)
            if child is None:
                wrapped = getattr(leaf_module, "_checkpoint_wrapped_module", None)
                if wrapped is not None:
                    leaf_module = getattr(wrapped, part)
                else:
                    leaf_module = getattr(leaf_module, part)
            else:
                leaf_module = child
        if hasattr(leaf_module, "_checkpoint_wrapped_module"):
            leaf_module = leaf_module._checkpoint_wrapped_module
        return cls(leaf_module, parts[-1])


@dataclass(frozen=True)
class BucketParam:
    """Per-parameter bucket runtime state.

    param_owner locates the module slot that holds the local shard, or the
    unsharded param while the bucket is unsharded. sharded_param is the local
    shard used for unshard input and grad writes, captured at runtime install
    and re-read after to_empty(). param_info carries immutable bucket storage and
    placement metadata for collectives. Keeping them together preserves bucket
    order and avoids repeated FQN resolution in hooks. shared_owners locates the
    other slots of a shared parameter, which swap together with param_owner.
    """

    param_owner: ParamOwnerRef
    param_info: ParamInfo
    sharded_param: nn.Parameter
    shared_owners: tuple[ParamOwnerRef, ...] = ()


def _in_backward() -> bool:
    """Return whether the autograd engine is executing a backward pass."""
    return torch._C._current_graph_task_id() != -1


def _alloc_storage(tensor: torch.Tensor, nbytes: int) -> None:
    storage = tensor.untyped_storage()
    if storage.size() != nbytes:
        storage.resize_(nbytes)


def _free_storage(tensor: torch.Tensor) -> None:
    storage = tensor.untyped_storage()
    if storage.size() != 0:
        storage.resize_(0)


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Tensors in a pytree, including dataclass fields (as FSDP2 collects them)."""
    tensors: list[torch.Tensor] = []
    for leaf in tree_leaves(value):
        if isinstance(leaf, torch.Tensor):
            tensors.append(leaf)
        elif dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            tensors.extend(
                _tensor_leaves(
                    [getattr(leaf, f.name) for f in dataclasses.fields(leaf)]
                )
            )
    return tensors


def _storage_ptr(tensor: torch.Tensor) -> int | None:
    if is_traceable_wrapper_subclass(tensor):  # no storage of its own
        return None
    return tensor.untyped_storage().data_ptr()


def _inner_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """The plain tensors holding a (possibly wrapper-subclass) tensor's data."""
    if is_traceable_wrapper_subclass(tensor):
        attrs, _ = tensor.__tensor_flatten__()
        return [
            inner for attr in attrs for inner in _inner_tensors(getattr(tensor, attr))
        ]
    return [tensor]


@dataclass
class PendingUnshard:
    """A prefetched unshard in flight until its bucket's hook takes it."""

    bucket: BucketRuntime
    result: UnshardHandle
    # From an explicit prefetch list (ShardedBucketStorage.
    # set_buckets_to_forward_prefetch), not the learned order.
    explicit: bool = False


@dataclass
class PendingReduceGrad:
    """One in-flight reduce-grad result."""

    result: ReduceGradHandle


@dataclass
class GroupForwardPass:
    """One forward call of a bucket that hooks several modules: from the call
    of the first of them until the last finishes (or the root module's
    post-forward completes it), as FSDP2's group state stays FORWARD."""

    # Whether the call's inputs carry a post-backward trigger.
    has_trigger: bool = False
    # Whether the group's post-forward ran for it. A trigger of a call that
    # never completed (one of the modules run on its own) runs post-backward
    # as soon as it fires.
    completed: bool = False


class GradientReductionHandle:
    """Returned by ``FlexShardModule.finalize_backward(async_op=True)``.

    ``wait()`` makes the current stream wait for the reduce-scatters, so the
    local-shard grads are ready for the optimizer, and releases their buffers.
    """

    def __init__(self, contexts: list[BucketCommContext]) -> None:
        self._contexts = contexts
        for context in contexts:
            context.pending_finalization = self

    def wait(self) -> None:
        for context in self._contexts:
            if context.pending_finalization is self:
                context.wait_and_clear_reduce_grad_states(debug_fqn=None)
                context.pending_finalization = None
        self._contexts = []


class ModuleUnshardHandle:
    """Returned by ``FlexShardModule.unshard(async_op=True)``, like FSDP2's
    ``UnshardHandle``.

    ``wait()`` finishes the all-gathers that ``unshard`` started and that are
    still pending into the unsharded params. A forward that runs first
    finishes its buckets' all-gathers itself.
    """

    def __init__(self, buckets: list[tuple[BucketCommContext, BucketRuntime]]) -> None:
        self._buckets = buckets

    def wait(self) -> None:
        for context, bucket in self._buckets:
            if any(pending.bucket is bucket for pending in context.pending_unshards):
                bucket.unshard()
        self._buckets = []


@dataclass
class BucketCommContext:
    """Streams and scheduling state shared by buckets on one root module/device."""

    device_handle: ModuleType
    unshard_stream: torch.Stream
    reduce_grad_stream: torch.Stream
    reduce_grad_release_stream: torch.Stream
    max_pending_reduce_grads: int
    # Buckets in registration (BucketSpec) order.
    buckets: list[BucketRuntime] = field(default_factory=list)
    # Buckets in the order their pre- and post-forward hooks first ran, learned
    # from the first forward (FSDP2 records a post-forward order). Forward
    # prefetch follows the first; backward prefetch walks the second in reverse.
    forward_order: list[BucketRuntime] = field(default_factory=list)
    post_forward_order: list[BucketRuntime] = field(default_factory=list)
    pending_unshards: list[PendingUnshard] = field(default_factory=list)
    reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    retired_reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    post_backward_callback_queued: bool = False
    # Whether the queued callback finishes the buckets, not only waits (see
    # queue_post_backward_callback).
    finish_at_backward_end: bool = False
    # See FlexShardModule.set_manual_backward_finalization and finalize_backward.
    manual_backward_finalization: bool = False
    pending_finalization: GradientReductionHandle | None = None

    @classmethod
    def get(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext | None:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            return None
        return contexts.get(device)

    @classmethod
    def create(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            contexts = {}
            setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)
        if device in contexts:
            raise AssertionError(
                f"Communication context for device {device} already exists."
            )

        device_handle = _get_device_handle(device.type)
        context = cls(
            device_handle=device_handle,
            unshard_stream=device_handle.Stream(priority=-1),
            reduce_grad_stream=device_handle.Stream(priority=-1),
            reduce_grad_release_stream=device_handle.Stream(priority=-1),
            max_pending_reduce_grads=getattr(
                root_module, _MAX_PENDING_REDUCE_GRADS_ATTR
            ),
        )
        contexts[device] = context
        return context

    def next_forward_bucket(self, bucket: BucketRuntime) -> BucketRuntime | None:
        """Return the bucket whose forward followed ``bucket``'s."""
        idx = bucket.forward_index
        if idx is None or idx + 1 >= len(self.forward_order):
            return None
        return self.forward_order[idx + 1]

    def next_backward_bucket(self, bucket: BucketRuntime) -> BucketRuntime | None:
        """Return the bucket whose backward re-gather follows ``bucket``'s.

        Backward visits buckets in reverse post-forward order. Buckets without
        reshard-after-forward, or already unsharded, need no re-gather.
        """
        idx = bucket.post_forward_index
        if idx is None:
            return None
        for candidate in reversed(self.post_forward_order[:idx]):
            if (
                candidate.bucket_storage._reshard_after_forward
                and not candidate.is_unsharded
            ):
                return candidate
        return None

    def prefetch(
        self,
        bucket: BucketRuntime | None,
        *,
        explicit: bool = False,
        on_current_stream: bool = False,
    ) -> None:
        """Start ``bucket``'s unshard ahead of its hook. Learned-order prefetches
        run one at a time, explicit ones as many as the lists name. With
        ``on_current_stream``, for ``FlexShardModule.unshard(async_op=True)``,
        the all-gather runs asynchronously on the current stream."""
        if (
            bucket is None
            or bucket.is_unsharded
            or any(pending.bucket is bucket for pending in self.pending_unshards)
            or (
                not explicit
                and any(not pending.explicit for pending in self.pending_unshards)
            )
        ):
            return
        self.pending_unshards.append(
            PendingUnshard(
                bucket,
                bucket.begin_unshard(
                    on_current_stream=on_current_stream, async_op=on_current_stream
                ),
                explicit,
            )
        )

    def bucket_runtime(self, bucket_storage: ShardedBucketStorage) -> BucketRuntime:
        """The runtime of ``bucket_storage``, e.g. an explicit prefetch target."""
        for bucket in self.buckets:
            if bucket.bucket_storage is bucket_storage:
                return bucket
        raise ValueError(
            "FlexShard: a bucket storage is not a bucket of this flex_shard "
            "module on this device."
        )

    def check_outside_backward(self, method: str) -> None:
        """Raise unless ``method`` (e.g. ``"unshard"``) may run now: outside
        backward and after waiting on any ``finalize_backward`` handle."""
        if _in_backward():
            raise RuntimeError(f"FlexShard: {method}() cannot run in backward.")
        self.check_no_raised_backward()
        if self.pending_finalization is not None:
            raise RuntimeError(
                f"FlexShard: wait on the finalize_backward() handle before {method}()."
            )

    def release_pending_unshard(self, bucket: BucketRuntime) -> None:
        """Release ``bucket``'s prefetched unshard, if any."""
        kept = []
        for pending in self.pending_unshards:
            if pending.bucket is bucket:
                self._release_prefetch(pending)
            else:
                kept.append(pending)
        self.pending_unshards = kept

    def finish_other_prefetches(self, bucket: BucketRuntime) -> None:
        """Finish other buckets' learned-order prefetches into their storage,
        where their own unshards find them, instead of releasing them."""
        kept, finished = [], []
        for pending in self.pending_unshards:
            if pending.bucket is not bucket and not pending.explicit:
                finished.append(pending)
            else:
                kept.append(pending)
        self.pending_unshards = kept
        for pending in finished:
            pending.bucket.finish_unshard(pending.result)

    def take_pending_unshard(
        self,
        bucket: BucketRuntime | None,
    ) -> UnshardHandle | None:
        """Return ``bucket``'s prefetched unshard and release other buckets'
        learned-order prefetches; ``None`` releases every prefetch.

        A learned-order prefetch for another bucket means execution diverged
        from the learned order; releasing it bounds memory and frees the
        prefetch slot. Explicit prefetches stay until their buckets take them.
        """
        result = None
        kept = []
        for pending in self.pending_unshards:
            if bucket is not None and pending.bucket is bucket:
                result = pending.result
            elif bucket is not None and pending.explicit:
                kept.append(pending)
            else:
                self._release_prefetch(pending)
        self.pending_unshards = kept
        return result

    def _release_prefetch(self, pending: PendingUnshard) -> None:
        with _record_function_if_eager(
            "FlexShard::release_unused_prefetch",
            pending.bucket.debug_fqn,
        ):
            pending.result.wait()
            pending.result.release_buffers()
            # A refill's begin re-allocated the storage (a first unshard has none).
            pending.bucket.free_persistent_storage()

    def queue_post_backward_callback(self, *, finish: bool) -> None:
        """Queue the end-of-backward callback (once per backward).

        It waits on all reduce-grad work. With ``finish``, requested by a
        pre-backward hook, it also finishes the buckets the backward left
        (``finish_buckets``) and releases unused prefetches; FSDP2 likewise
        queues its final callback only from a pre-backward hook. A backward
        that only ran post-backward triggers, e.g. of a module called on its
        own on chunks of an output, so leaves the other buckets' forward to a
        later backward. With manual backward finalization, or outside a
        backward (``finalize_backward``), nothing is queued.
        """
        if self.manual_backward_finalization or not _in_backward():
            return
        self.finish_at_backward_end |= finish
        if self.post_backward_callback_queued:
            return
        self.post_backward_callback_queued = True

        def _post_backward_callback() -> None:
            with dist._spmd_no_typecheck():
                finish = self.finish_at_backward_end
                if finish:
                    self.finish_buckets()
                self.wait_and_clear_reduce_grad_states(debug_fqn=None)
                if finish:
                    self.take_pending_unshard(None)
                self.finish_at_backward_end = False
                self.post_backward_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(
            _post_backward_callback
        )

    def finish_buckets(self, *, reverse_module_order: bool = False) -> None:
        """Finish the buckets backwards left (see ``BucketRuntime.needs_finish``),
        then reset the per-backward trigger counts. ``reverse_module_order``
        finishes them in reverse order of their modules in the root module, and
        a module's buckets last first, as FSDP2's ``finalize_backward`` goes
        through its states and their param groups.

        A bucket that defers its post-backward must be finished before a
        syncing backward that ran its module's backward ends, since its late
        grads may not exist yet; outside backward (``finalize_backward``), they
        do."""
        buckets = (
            reversed(sorted(self.buckets, key=lambda bucket: bucket.module_position))
            if reverse_module_order
            else self.buckets
        )
        for bucket in buckets:
            if bucket.needs_finish():
                if (
                    bucket.bucket_storage._defer_post_backward
                    and bucket.bucket_storage._requires_gradient_sync
                    and bucket.input_triggers_run
                    and _in_backward()
                ):
                    raise RuntimeError(
                        f"FlexShard: bucket {bucket.debug_fqn} defers its "
                        "post-backward, but a syncing backward ended before "
                        "finish_deferred_backward() finished it."
                    )
                bucket.post_backward()
            bucket.reset_backward_state()

    def start_finalize_backward(self) -> None:
        """``FlexShardModule.finalize_backward`` without the wait: issue the
        remaining reduce-scatters and reshard on the calling thread."""
        if self.pending_finalization is not None:
            raise RuntimeError(
                "FlexShard: wait on the previous finalize_backward() handle before "
                "finalizing backward again."
            )
        if _in_backward():
            raise RuntimeError("FlexShard: finalize_backward() cannot run in backward.")
        self.check_no_raised_backward()
        # In reverse module order, as FSDP2's finalize_backward, like backward:
        # a block's routed experts reduce before its dense bucket, whose smaller
        # buffers then reuse the memory they freed.
        self.finish_buckets(reverse_module_order=True)
        self.take_pending_unshard(None)

    def check_no_raised_backward(self) -> None:
        """Raise if a backward raised before its final callback ran.

        Autograd drops queued callbacks when a backward raises, which leaves
        partial gradients and unfinished state. FlexShard does not recover
        from errors: training has to stop.
        """
        if self.post_backward_callback_queued:
            raise RuntimeError(
                "FlexShard: a previous backward raised before finishing, which "
                "leaves FlexShard's state undefined; restart training."
            )

    def reset(self) -> None:
        """Reshard every bucket and clear per-backward state, for
        ``FlexShardModule.reshard()``. Unsharded params keep their grads, which
        may be accumulated without sync.

        A backward left unfinished, e.g. under manual backward finalization,
        may also have left grad upcasts deferred. Restore them, as FSDP2's
        post-backward does after every call, so the next backward that starts
        a grad without sync keeps the upcast.
        """
        self.check_no_raised_backward()
        self.take_pending_unshard(None)
        self.wait_and_clear_reduce_grad_states(debug_fqn=None)
        self.pending_finalization = None
        for bucket in self.buckets:
            if bucket.is_unsharded:
                bucket.reshard()
            bucket.reset_backward_state()
            bucket._set_unsharded_grad_dtypes(defer_upcast=False)

    def wait_and_clear_reduce_grad_states(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Order reduced gradients before optimizer use and release buffers."""
        if not self.reduce_grad_states and not self.retired_reduce_grad_states:
            return
        with _record_function_if_eager(
            "FlexShard::post_backward_reduce_grad_wait",
            debug_fqn,
        ):
            for pending in self.retired_reduce_grad_states:
                pending.result.wait()
            self.retired_reduce_grad_states.clear()
            for pending in self.reduce_grad_states:
                self.wait_and_release_reduce_grad_state(
                    pending,
                    wait_current_stream=True,
                )
            self.reduce_grad_states.clear()

    def wait_and_release_reduce_grad_state(
        self,
        pending: PendingReduceGrad,
        *,
        wait_current_stream: bool,
    ) -> None:
        if wait_current_stream:
            pending.result.wait()
        with self.device_handle.stream(self.reduce_grad_release_stream):
            pending.result.wait()
            pending.result.release_buffers(
                release_sharded_grads=True,
            )

    def drain_reduce_grad_states_if_needed(
        self,
        debug_fqn: str | None,
    ) -> None:
        if self.max_pending_reduce_grads <= 0:
            return
        while len(self.reduce_grad_states) >= self.max_pending_reduce_grads:
            pending = self.reduce_grad_states.pop(0)
            with _record_function_if_eager(
                "FlexShard::post_backward_reduce_grad_retire",
                debug_fqn,
            ):
                self.wait_and_release_reduce_grad_state(
                    pending,
                    wait_current_stream=False,
                )
                self.retired_reduce_grad_states.append(pending)


@dataclass
class BucketRuntime:
    """Runtime state and hooks for one FlexShard bucket.

    Eager (FSDP2-style): each managed parameter has a persistent unsharded
    ``nn.Parameter``, created from the first unshard. It is swapped into the
    owning module's ``_parameters`` while unsharded, so attribute reads and
    ``module.parameters()`` see it in forward and backward. Reshard swaps the
    local shard back and frees the placement's persistent buffers backing the
    unsharded params with ``untyped_storage().resize_(0)``; later unshards
    re-allocate and refill them in place. Grads accumulate into the unsharded
    params through autograd and are reduce-scattered into the local shards once
    the hooked module's backward is done (its input grads are ready, as in
    FSDP2), or at the end of backward.

    Compile: ``_BucketUnshard`` outputs are swapped into ``_parameters`` for the
    forward, so Dynamo traces one all-gather and one reduce-scatter per bucket.
    """

    bucket_storage: ShardedBucketStorage
    bucket_params: list[BucketParam]
    context: BucketCommContext
    debug_fqn: str | None
    # Position in the context's learned forward and post-forward orders.
    forward_index: int | None = None
    post_forward_index: int | None = None
    # Whether the unsharded params hold data and are swapped into their modules.
    is_unsharded: bool = False
    # Forward calls since the last backward whose outputs need backward, how
    # many carried a post-backward trigger on their inputs, and how many of
    # those triggers ran; and whether each in-progress call carries one.
    backward_calls: int = 0
    input_triggers: int = 0
    input_triggers_run: int = 0
    call_has_trigger: list[bool] = field(default_factory=list)
    # A bucket hooking several modules (BucketSpec.patterns): the modules, the
    # ones whose forward the current pass still awaits, and its open forward
    # call, if any (see GroupForwardPass).
    group_modules: tuple[nn.Module, ...] = field(default=(), repr=False)
    # Position of its first hooked module in the root module's modules(), as
    # FSDP2 orders its states (see EagerCommContext.finish_buckets).
    module_position: int = 0
    modules_to_run: set[nn.Module] = field(default_factory=set, repr=False)
    group_pass: GroupForwardPass | None = None
    # Whether a backward without gradient sync finished this bucket since its
    # last reduce-scatter, so the next syncing backward reduces it even if it
    # does not use the bucket. Unlike grad presence, the same on every rank.
    needs_sync: bool = False
    # Persistent unsharded params, created from the first unshard, and the
    # placement's persistent buffers backing them with their allocated storage
    # size in bytes. Left out of repr: their storage is freed while resharded,
    # so printing them would read freed memory.
    unsharded_params: list[nn.Parameter] | None = field(default=None, repr=False)
    persistent_buffers: list[torch.Tensor] = field(default_factory=list, repr=False)
    persistent_buffer_nbytes: list[int] = field(default_factory=list)
    # With BucketSpec.fsdp2_compatible: bucket_params indices in FSDP2's
    # reduce-scatter order, grouped by sharded grad dtype. Set at the first
    # syncing backward, as FSDP2 caches it at lazy init.
    fsdp2_reduce_scatter_order: list[int] | None = None
    # Whether the placement's gradient reduction reads grads in the parameter
    # layout as is (see _run_pre_backward_hook). Set at the first backward.
    grads_in_param_layout: bool | None = None

    @classmethod
    def from_bucket_storage(
        cls,
        bucket_storage: ShardedBucketStorage,
        context: BucketCommContext | None = None,
    ) -> BucketRuntime:
        """Create runtime state for one bucket storage."""
        bucket_params: list[BucketParam] = []
        for info in bucket_storage._param_infos.values():
            param_owner = ParamOwnerRef.resolve(bucket_storage._module, info.fqn)
            bucket_params.append(
                BucketParam(
                    param_owner=param_owner,
                    param_info=info,
                    sharded_param=param_owner.module._parameters[
                        param_owner.param_name
                    ],
                    shared_owners=tuple(
                        ParamOwnerRef.resolve(bucket_storage._module, shared_fqn)
                        for shared_fqn in info.shared_fqns
                    ),
                )
            )
        comm_device = bucket_storage.byte_storage.device
        if context is None:
            context = BucketCommContext.get(bucket_storage._module, comm_device)
            if context is None:
                context = BucketCommContext.create(bucket_storage._module, comm_device)
        return cls(
            bucket_storage=bucket_storage,
            bucket_params=bucket_params,
            context=context,
            debug_fqn=_get_bucket_storage_debug_fqn(bucket_storage),
        )

    @property
    def infos(self) -> list[ParamInfo]:
        return [bucket_param.param_info for bucket_param in self.bucket_params]

    @property
    def sharded_params(self) -> list[nn.Parameter]:
        return [bucket_param.sharded_param for bucket_param in self.bucket_params]

    def _local_shards(self, *, use_autograd: bool) -> list[torch.Tensor]:
        # detach() shares the parameter version counter, which versioned
        # placement caches need for optimizer-update invalidation.
        return [
            param if use_autograd else param.detach() for param in self.sharded_params
        ]

    def reset_sharded_params(self) -> None:
        """Re-read the local-shard params after their module slots changed.

        ``to_empty()`` re-installs them; FSDP2 re-reads them the same way in
        ``reset_sharded_param()``.
        """
        self.bucket_params = [
            replace(
                bucket_param,
                sharded_param=bucket_param.param_owner.module._parameters[
                    bucket_param.param_owner.param_name
                ],
            )
            for bucket_param in self.bucket_params
        ]

    def rebind_storage(self, bucket_storage: ShardedBucketStorage) -> None:
        """Take over the storage that replaced this bucket's before its first
        unshard (``relayout_bucket``): the same parameters, hooks and context,
        with the new storage's parameter infos and sharded parameters."""
        assert self.unsharded_params is None and not self.is_unsharded
        rebound = BucketRuntime.from_bucket_storage(bucket_storage, self.context)
        self.bucket_storage = rebound.bucket_storage
        self.bucket_params = rebound.bucket_params
        self.fsdp2_reduce_scatter_order = None
        self.grads_in_param_layout = None

    def forward_hook_module(self) -> nn.Module:
        """Return the module whose forward triggers this bucket.

        It is the deepest common ancestor of the bucket's params, so one
        pre-forward unshard covers their accesses: a bucket with
        "layers.0.attn.wq.weight" and "layers.0.mlp.w1.weight" hooks "layers.0".
        Containers without a forward, such as ``ModuleList``, give way to their
        nearest ancestor that runs one.
        """
        # Every name of a shared parameter counts, so the hooked module contains
        # each of its uses.
        path = _module_path_common_prefix(
            [
                ".".join(fqn.split(".")[:-1])
                for info in self.infos
                for fqn in (info.fqn, *info.shared_fqns)
            ]
        )
        modules = [self.bucket_storage._module]
        for part in path.split(".") if path else []:
            modules.append(getattr(modules[-1], part))
        target = next(
            (
                module
                for module in reversed(modules)
                if type(module).forward is not nn.Module.forward
            ),
            modules[0],
        )
        return getattr(target, "_checkpoint_wrapped_module", target)

    def begin_unshard(
        self,
        local_shards: list[torch.Tensor] | None = None,
        *,
        on_current_stream: bool = False,
        async_op: bool = False,
    ) -> UnshardHandle:
        """Begin this bucket's unshard on the shared stream, or, with
        ``on_current_stream``, for ``FlexShardModule.unshard()``, on the current
        stream, asynchronously with ``async_op``, as FSDP2's
        ``FSDPModule.unshard()`` does (``begin_explicit_bucket_unshard``).

        A refill re-allocates the persistent storage on the current stream (the
        one that frees it on reshard), and the placement writes the new values
        into it, during the unshard or when it is finished. If it writes them
        only when finished (``Placement.refills_persistent_buffers_in_finish``),
        ``finish_unshard`` allocates the storage right before, as FSDP2
        allocates its unsharded parameters at copy-out; otherwise this does,
        before the unshard stream waits on it.
        """
        if local_shards is None:
            local_shards = self._local_shards(use_autograd=False)
        persistent_buffers = None
        if self.unsharded_params is not None and not torch.compiler.is_compiling():
            if not self.infos[0].placement.refills_persistent_buffers_in_finish:
                self._alloc_persistent_storage()
            persistent_buffers = self.persistent_buffers
        if on_current_stream:
            return begin_explicit_bucket_unshard(
                local_shards,
                self.infos,
                self.bucket_storage._mesh,
                async_op=async_op,
                debug_fqn=self.debug_fqn,
                persistent_buffers=persistent_buffers,
            )
        return begin_bucket_unshard(
            local_shards,
            self.infos,
            self.bucket_storage._mesh,
            self.context.unshard_stream,
            debug_fqn=self.debug_fqn,
            persistent_buffers=persistent_buffers,
        )

    def reduce_grads(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        sharded_params: list[nn.Parameter],
    ) -> None:
        """Reduce full-parameter grads and accumulate local sharded grads."""
        if not grads:
            return
        with torch.no_grad():
            self.context.drain_reduce_grad_states_if_needed(self.debug_fqn)
            # A 1-rank mesh has nothing to communicate, so it reduces on the
            # current stream, as FSDP2 does for a 1-rank group. On the reduce-grad
            # stream, its cast sharded grads would come from that stream's
            # allocator pool, which memory freed on the current stream cannot
            # refill, and with one rank they are full size.
            reduce_grad_stream = (
                self.context.device_handle.current_stream(grads[0].device)
                if self.bucket_storage._mesh.size() == 1
                else self.context.reduce_grad_stream
            )
            result = begin_reduce_grad(
                grads,
                infos,
                self.bucket_storage.reduce_mesh,
                self.bucket_storage.gradient_reduction,
                reduce_grad_stream,
                debug_fqn=self.debug_fqn,
            )
            with self.context.device_handle.stream(reduce_grad_stream):
                with _record_function_if_eager(
                    "FlexShard::reduce_grad_accumulate",
                    self.debug_fqn,
                ):
                    sharded_grads = result.finish()
                    result.record_sharded_grads(
                        _accumulate_sharded_grads(
                            sharded_params,
                            sharded_grads,
                        ),
                        reduce_grad_stream,
                    )
            self.context.reduce_grad_states.append(PendingReduceGrad(result))
            self.context.queue_post_backward_callback(finish=False)

    # ------------------------------------------------------------------
    # Eager: persistent unsharded parameters
    # ------------------------------------------------------------------

    def unshard(
        self, *, finish_other_prefetch: bool = False, on_current_stream: bool = False
    ) -> None:
        """Make the unsharded params hold data and swap them into the modules.

        With ``finish_other_prefetch``, other buckets' learned-order prefetches
        are finished into those buckets' storage, where their own unshards find
        them, instead of being released. Without a pending unshard to finish,
        it all-gathers on the shared stream, or, with ``on_current_stream``,
        for ``FlexShardModule.unshard()``, synchronously on the current stream.
        """
        if not self.is_unsharded:
            if finish_other_prefetch:
                self.context.finish_other_prefetches(self)
            result = self.context.take_pending_unshard(self)
            if result is None:
                result = self.begin_unshard(on_current_stream=on_current_stream)
            self.finish_unshard(result)
        self._swap_in_params(self.unsharded_params)

    def finish_unshard(self, result: UnshardHandle) -> None:
        """Finish ``result`` into the persistent buffers.

        The first unshard creates the unsharded params from the placement's
        full params and keeps the persistent buffers backing them. Later ones
        (see ``begin_unshard``) write those buffers without bumping their
        version counters, which the unsharded params share (autograd may have
        saved them in forward before a reshard), as FSDP2 does for its
        all-gather outputs. The storage is created outside inference mode, so a
        model first run under ``torch.inference_mode()`` can still train.
        """
        with torch.inference_mode(False), torch.no_grad():
            if self.unsharded_params is None:
                unshard = result.finish()
                self._create_unsharded_params(
                    unshard.full_params, unshard.persistent_buffers
                )
            else:
                if self.infos[0].placement.refills_persistent_buffers_in_finish:
                    self._alloc_persistent_storage()
                with torch.autograd._unsafe_preserve_version_counter(
                    tuple(self.persistent_buffers)
                ):
                    refill = result.finish()
                if len(refill.persistent_buffers) != len(
                    self.persistent_buffers
                ) or any(
                    a is not b
                    for a, b in zip(refill.persistent_buffers, self.persistent_buffers)
                ):
                    raise AssertionError(
                        f"Placement {self.infos[0].placement!r} did not refill the "
                        "persistent buffers it was given."
                    )
            self._sync_requires_grad()
        self.is_unsharded = True

    def _create_unsharded_params(
        self,
        full_params: list[torch.Tensor],
        persistent_buffers: list[torch.Tensor],
    ) -> None:
        if not persistent_buffers:
            raise AssertionError(
                f"Placement {self.infos[0].placement!r} returned no persistent "
                "buffers for a persistent unshard."
            )
        buffer_storages = {_storage_ptr(buffer) for buffer in persistent_buffers}
        unsharded_params: list[nn.Parameter] = []
        for bucket_param, full_param in zip(
            self.bucket_params, full_params, strict=True
        ):
            # Refills write into the persistent buffers, so the params (or a
            # tensor subclass's inner tensors) must view them.
            if any(
                inner.numel() and _storage_ptr(inner) not in buffer_storages
                for inner in _inner_tensors(full_param)
            ):
                raise AssertionError(
                    f"Placement {bucket_param.param_info.placement!r} returned full "
                    f"param {bucket_param.param_info.fqn!r}, which does not view its "
                    "persistent buffers, so refills would not reach it."
                )
            unsharded_param = nn.Parameter(
                full_param, requires_grad=bucket_param.sharded_param.requires_grad
            )
            for name, value in bucket_param.param_info.param_attrs.items():
                setattr(unsharded_param, name, value)
            # Like FSDP2's unsharded_param.grad_dtype: autograd casts each
            # incoming grad before accumulating multiple uses.
            unsharded_param.grad_dtype = bucket_param.param_info.unsharded_grad_dtype
            unsharded_params.append(unsharded_param)
        self.unsharded_params = unsharded_params
        self.persistent_buffers = list(persistent_buffers)
        self.persistent_buffer_nbytes = [
            persistent_buffer.untyped_storage().size()
            for persistent_buffer in persistent_buffers
        ]

    def _sync_requires_grad(self) -> None:
        """Follow the local shards' ``requires_grad``, as FSDP2 does per unshard."""
        for bucket_param, unsharded_param in zip(
            self.bucket_params, self.unsharded_params, strict=True
        ):
            requires_grad = bucket_param.sharded_param.requires_grad
            if unsharded_param.requires_grad != requires_grad:
                unsharded_param.requires_grad_(requires_grad)

    def _swap_in_params(self, params: list[torch.Tensor]) -> None:
        """Expose ``params`` through their modules' ``_parameters``."""
        for bucket_param, param in zip(self.bucket_params, params, strict=True):
            for owner in (bucket_param.param_owner, *bucket_param.shared_owners):
                owner.module._parameters[owner.param_name] = param

    def reshard(self) -> None:
        """Swap the local shards back in and free the persistent storage."""
        self._swap_in_params(self.sharded_params)
        self.free_persistent_storage()
        self.is_unsharded = False

    def _alloc_persistent_storage(self) -> None:
        """Re-allocate the storage behind the unsharded params for a refill,
        on the current stream, outside inference mode."""
        with torch.inference_mode(False):
            for persistent_buffer, nbytes in zip(
                self.persistent_buffers, self.persistent_buffer_nbytes, strict=True
            ):
                _alloc_storage(persistent_buffer, nbytes)

    def free_persistent_storage(self) -> None:
        """Free the storage behind the unsharded params.

        Consumers of the storage were queued on this stream, which also
        allocated it (a refill's unshard stream writes it only after waiting on
        this stream), so the caching allocator orders the reuse.
        """
        for persistent_buffer in self.persistent_buffers:
            _free_storage(persistent_buffer)

    def reset_backward_state(self) -> None:
        """Clear the post-backward trigger counts and the group's forward
        state at the end of a backward, as FSDP2's final callback does."""
        self.backward_calls = self.input_triggers = self.input_triggers_run = 0
        self.modules_to_run.clear()
        self.group_pass = None

    def needs_finish(self) -> bool:
        """Whether backwards left this bucket to finish: still unsharded (no
        post-backward trigger fired) or, with gradient sync on, holding grads
        that backwards without sync accumulated since its last reduce-scatter."""
        return self.is_unsharded or (
            self.needs_sync and self.bucket_storage._requires_gradient_sync
        )

    @_outside_selective_checkpoint
    def pre_backward_hook(self, grad_outputs: Any) -> None:
        """Re-unshard before this bucket's module runs backward.

        A pre-hook on the autograd node that produced an output. Autograd runs
        a tensor's hooks before its node's pre-hooks, so a consumer bucket's
        post-backward trigger (a hook on that same output) frees its params
        before this re-gather, as FSDP2's ordering does.

        In a syncing backward of a bucket called more than once in the forward,
        the first of these hooks reduce-scatters the grads kept from backwards
        without sync on their own. FSDP2 runs a post-backward per call, and the
        first one fires before autograd sums the calls' grads into the kept
        ones, so it reduce-scatters the kept grads alone and this backward's
        in a second reduce-scatter.
        """
        with dist._spmd_no_typecheck():
            self.context.queue_post_backward_callback(finish=True)
            self.unshard()
            if (
                self.backward_calls > 1
                and self.needs_sync
                and self.bucket_storage._requires_gradient_sync
                and not self.bucket_storage._defer_post_backward
            ):
                self._post_backward(reshard=False)
            self._set_unsharded_grad_dtypes(defer_upcast=True)
            self._run_pre_backward_hook()
            self._prefetch(forward=False)

    def _run_pre_backward_hook(self) -> None:
        """Run the bucket's pre_backward_hook. A bucket without one whose
        placement reduces grads in the parameter layout as is (whole-parameter
        BucketedOwned's owner rows) gets its grads allocated there, so the
        reduce copies none in. Missing grads are zero-filled before the reduce
        in such buckets anyway, so the values don't change."""
        hook = self.bucket_storage._pre_backward_hook
        if hook is None:
            if self.grads_in_param_layout is None:
                infos = [bucket_param.param_info for bucket_param in self.bucket_params]
                self.grads_in_param_layout = infos[
                    0
                ].placement.reduces_grads_in_param_layout(
                    infos, self.bucket_storage._mesh.size()
                )
            if self.grads_in_param_layout:
                hook = alloc_grads_in_param_layout
        if hook is not None:
            hook(self._named_unsharded_params())

    def _prefetch(self, *, forward: bool) -> None:
        """Prefetch this bucket's explicit list, if it has one, else the next
        bucket in the learned order."""
        storage = self.bucket_storage
        targets = storage._forward_prefetch if forward else storage._backward_prefetch
        if targets is None:
            self.context.prefetch(
                self.context.next_forward_bucket(self)
                if forward
                else self.context.next_backward_bucket(self)
            )
            return
        for target in targets:
            self.context.prefetch(self.context.bucket_runtime(target), explicit=True)

    def _set_unsharded_grad_dtypes(self, *, defer_upcast: bool) -> None:
        """Defer or restore the upcast of grads accumulating in a wider dtype.

        As in FSDP2 (pytorch/pytorch#198668 and #199242), a param whose
        unsharded grad dtype is wider than its compute dtype keeps the grads
        autograd produces while deferred (``grad_dtype=None``), in a backward
        that syncs or that adds to a kept grad: the reduce-scatter copy-in
        widens them as it copies, and AccumulateGrad adds them in place to a
        wider kept grad. That saves a cast kernel per param and the wider
        unsharded grads until the copy-in. A backward without sync that starts
        a grad keeps the upcast, so a param used more than once sums its uses
        in the wider dtype, as FSDP2 does.
        Restoring upcasts a new grad that is not reduced, once, after the
        bucket reshards, so the accumulation across backwards stays wider. As
        FSDP2 does for grads it reduces outside data parallelism, a param with
        a partial-grad group keeps autograd's upcast, so its all-reduce runs in
        the wider dtype too.
        """
        for bucket_param, param in zip(
            self.bucket_params, self.unsharded_params or [], strict=False
        ):
            info = bucket_param.param_info
            dtype = info.unsharded_grad_dtype
            compute_dtype = info.unsharded_dtype
            if (
                dtype is None
                or not param.requires_grad
                or not dtype.is_floating_point
                or not compute_dtype.is_floating_point
                or dtype.itemsize <= compute_dtype.itemsize
                or info.partial_grad_group is not None
            ):
                continue
            if defer_upcast:
                # As FSDP2: a backward without sync that starts the grad keeps
                # the upcast, so that a param used more than once accumulates
                # its uses in the wider dtype.
                if (
                    param.grad is not None
                    or self.bucket_storage._requires_gradient_sync
                ):
                    param.grad_dtype = None
                continue
            grad = param.grad
            if grad is not None and grad.dtype != dtype:
                param.grad = grad.to(dtype)
            param.grad_dtype = dtype

    def _named_unsharded_params(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (bucket_param.param_info.fqn, param)
            for bucket_param, param in zip(
                self.bucket_params, self.unsharded_params or [], strict=False
            )
        ]

    def _run_post_reduce_hook(self) -> None:
        """Run ``BucketSpec.post_reduce_hook`` once the unsharded grads are gone."""
        hook = self.bucket_storage._post_reduce_hook
        if hook is not None and self.unsharded_params is not None:
            hook(self._named_unsharded_params())

    def on_input_grads(self) -> None:
        """Post-backward trigger from the hooked module's input grads.

        Like FSDP2's ``RegisterPostBackwardFunction``, it reduces and reshards
        once the module's backward is done, after every op that reads its
        params (frozen ones included), even if some params got no grad. It fires
        only when every forward call since the last backward carried a trigger
        and all of them ran, so no other call's backward still needs the params.
        A bucket that defers its post-backward waits for
        ``FlexShardModule.finish_deferred_backward`` instead. Once it fires,
        the counts restart, as a backward that runs no pre-backward hook does
        not reset them (see ``BucketCommContext.queue_post_backward_callback``).
        """
        self.input_triggers_run += 1
        if self.bucket_storage._defer_post_backward:
            return
        if self.backward_calls == self.input_triggers == self.input_triggers_run:
            self.post_backward()
            self.backward_calls = self.input_triggers = self.input_triggers_run = 0

    def on_partial_input_grads(self, forward_pass: GroupForwardPass) -> None:
        """Post-backward trigger of a forward call that did not complete the
        group: a module of the bucket called on its own, e.g. an output
        projection applied to chunks of the hidden states.

        Such a call has no pre-backward hook and no other call to wait for, so
        as in FSDP2's partial group backward, it reduces right away, or keeps
        the grads without gradient sync, and the next call starts a new pass.
        """
        if self.group_pass is forward_pass:
            self.group_pass = None
        if self.bucket_storage._defer_post_backward:
            return
        self.post_backward()

    def post_backward(self) -> None:
        """Reduce-scatter this backward's grads into the shards and reshard.

        Every trainable param joins the reduce-scatter, with zeros if it got no
        grad (e.g. an expert that saw no tokens on this rank), so every rank
        issues the same collective, once per bucket per backward. With
        ``BucketSpec.fsdp2_compatible``, only the params that got a grad join,
        in FSDP2's order, as FSDP2's reduce-scatter does.

        Without gradient sync, the grads stay on the unsharded params for later
        backwards to accumulate into, and the params stay unsharded unless
        ``reshard_after_backward`` is set. A syncing backward always reshards,
        since the optimizer step then changes the local shards.
        """
        with dist._spmd_no_typecheck():
            self._post_backward()

    def _post_backward(self, *, reshard: bool = True) -> None:
        # reshard=False keeps the unsharded params for a backward still to run.
        if not self.bucket_storage._requires_gradient_sync:
            self.needs_sync = True
            if (
                reshard
                and self.is_unsharded
                and self.bucket_storage._reshard_after_backward
            ):
                self.reshard()
            # After resharding, so a deferred grad's upcast never coexists with
            # the unsharded params.
            self._set_unsharded_grad_dtypes(defer_upcast=False)
            return
        self.needs_sync = False
        grads: list[torch.Tensor | None] = []
        params: list[nn.Parameter] = []
        infos: list[ParamInfo] = []
        sharded_params: list[nn.Parameter] = []
        fsdp2_compatible = self.bucket_storage._fsdp2_compatible
        param_pairs = list(
            zip(self.bucket_params, self.unsharded_params or [], strict=False)
        )
        if fsdp2_compatible and param_pairs:
            param_pairs = [
                param_pairs[idx] for idx in self._fsdp2_reduce_scatter_order()
            ]
        for bucket_param, unsharded_param in param_pairs:
            # A param frozen since its grad was kept drops it, on every rank.
            grad, unsharded_param.grad = unsharded_param.grad, None
            if not unsharded_param.requires_grad:
                continue
            if grad is None and fsdp2_compatible:
                continue
            grads.append(grad)
            params.append(unsharded_param)
            infos.append(bucket_param.param_info)
            sharded_params.append(bucket_param.sharded_param)
        self._run_post_reduce_hook()
        if reshard and self.is_unsharded:
            self.reshard()
        if not grads:
            return
        # Promote over the real grads before the zeros exist, so zeros never
        # pick the reduce dtype.
        infos = _promote_reduce_dtype_over_grads(grads, infos)
        # Zeros fill missing grads after resharding, so they never coexist
        # with the unsharded params, in the dtype autograd would produce, as
        # FSDP2's unsharded_zero_grad_data does.
        grads = [
            torch.zeros(
                param.shape, dtype=param.grad_dtype or param.dtype, device=param.device
            )
            if grad is None
            else grad
            for grad, param in zip(grads, params, strict=True)
        ]
        # In the grad's dtype, which keeps autograd's upcast for these params,
        # before the copy-in casts to the reduce dtype, as FSDP2 redistributes a
        # Partial grad when it takes it.
        for grad, info in zip(grads, infos, strict=True):
            if info.partial_grad_group is not None:
                dist.all_reduce(grad, group=info.partial_grad_group)
        # After the zeros, which take the forward dtype while the upcast is
        # deferred. The grads are already taken, so this only resets grad_dtype.
        self._set_unsharded_grad_dtypes(defer_upcast=False)
        # Per-parameter grad dtypes meet in the bucket's reduce dtype in the
        # placement's copy-in, which casts as it copies, rather than one cast
        # kernel per param here.
        self.reduce_grads(grads, infos, sharded_params)

    def _fsdp2_reduce_scatter_order(self) -> list[int]:
        """FSDP2's reduce-scatter order: the params grouped by sharded grad
        dtype, in first-seen dtype order, each group in parameter order."""
        if self.fsdp2_reduce_scatter_order is None:
            indices_by_dtype: dict[torch.dtype | None, list[int]] = {}
            for idx, bucket_param in enumerate(self.bucket_params):
                indices_by_dtype.setdefault(
                    bucket_param.sharded_param.grad_dtype, []
                ).append(idx)
            self.fsdp2_reduce_scatter_order = [
                idx for indices in indices_by_dtype.values() for idx in indices
            ]
        return self.fsdp2_reduce_scatter_order

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    @_outside_selective_checkpoint
    def pre_forward_hook(
        self,
        mod: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if torch.compiler.is_compiling():
            self._pre_forward_compile()
            return None
        # As FSDP2's pre-forward, the unshard runs outside spmd_types' type
        # checker, since FlexShard's buffers carry no SPMD types; the trigger
        # wraps the typed inputs, so it stays checked.
        with dist._spmd_no_typecheck():
            if _in_backward():
                self.call_has_trigger.append(False)
                self._unshard_for_recompute()
                return None
            self._check_forward_allowed()
            if self.forward_index is None:
                self.forward_index = len(self.context.forward_order)
                self.context.forward_order.append(self)
            self.unshard()
            self._prefetch(forward=True)
        args, kwargs, has_trigger = self._register_input_grad_hook(args, kwargs)
        self.call_has_trigger.append(has_trigger)
        return args, kwargs

    @_outside_selective_checkpoint
    def group_pre_forward_hook(
        self,
        mod: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Pre-forward hook of each module of a bucket that hooks several, as
        FSDP2's ``_register_group_forward_hooks`` runs a group's: every call
        unshards, and the call that opens a forward pass of the group, by its
        first module or by one module on its own, gets the post-backward
        trigger."""
        if torch.compiler.is_compiling():
            self._pre_forward_compile()
            return None
        with dist._spmd_no_typecheck():
            if _in_backward():
                self._unshard_for_recompute()
                return None
            self._check_forward_allowed()
            if not self.modules_to_run:
                self.modules_to_run.update(self.group_modules)
            opens_pass = self.group_pass is None
            if opens_pass and self.forward_index is None:
                self.forward_index = len(self.context.forward_order)
                self.context.forward_order.append(self)
            self.unshard()
            if not opens_pass:
                return None
            self._prefetch(forward=True)
        forward_pass = GroupForwardPass()
        args, kwargs, forward_pass.has_trigger = self._register_input_grad_hook(
            args, kwargs, forward_pass
        )
        self.group_pass = forward_pass
        return args, kwargs

    def _unshard_for_recompute(self) -> None:
        """Unshard for an activation-checkpoint recompute in backward.

        The pre-backward hook usually re-gathered already; otherwise this
        consumes its prefetch. The recompute visits its buckets in forward
        order, but backward prefetched in reverse, so another bucket's
        prefetch is finished for that bucket rather than released and
        gathered again. A forward without grad (reentrant checkpointing) left
        no pre-backward hook to run the bucket's pre_backward_hook, so run it
        here, before the recomputed backward.
        """
        self.unshard(finish_other_prefetch=True)
        self._run_pre_backward_hook()

    def _check_forward_allowed(self) -> None:
        self.context.check_no_raised_backward()
        if self.context.pending_finalization is not None:
            raise RuntimeError(
                "FlexShard: wait on the finalize_backward() handle before the next "
                "forward."
            )

    def _register_input_grad_hook(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        forward_pass: GroupForwardPass | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any], bool]:
        """Run the post-backward trigger once this forward's input grads are
        computed. Return the args and kwargs to call the module with, and
        whether the trigger exists.

        The inputs are wrapped in an identity autograd function whose backward
        runs the trigger, as FSDP2's ``RegisterPostBackwardFunction`` wraps
        them. A multi-grad hook instead holds every input's ``grad_fn`` and is
        held by them, so it can keep the forward's graph alive after backward:
        - a hook on a leaf input, e.g. a detached chunk of hidden states,
          outlives the graph, and one on a view of it forms a reference cycle
          until the next garbage collection;
        - when one input is computed from another, e.g. an MoE's routing
          scores from its hidden states, the cycle runs through the autograd
          edge between them, which garbage collection can't see. The graph,
          and the leaf it starts from, e.g. a pipeline stage's received
          activation, then stay alive for good.

        Inputs that ``tree_map`` can't replace, inside dataclasses, get a
        multi-grad hook only if none is a leaf, and it's removed once it runs.
        Forwards without grad-requiring inputs, or with leaf ones inside
        dataclasses, get no trigger.
        """
        if not torch.is_grad_enabled():
            return args, kwargs, False
        inputs = list(
            {
                id(tensor): tensor
                for tensor in _tensor_leaves((args, kwargs))
                if tensor.requires_grad
            }.values()
        )
        if not inputs:
            return args, kwargs, False
        # Only the inputs tree_map reaches can be replaced by the wrapped ones.
        reachable = {
            id(leaf)
            for leaf in tree_leaves((args, kwargs))
            if isinstance(leaf, torch.Tensor)
        }
        if any(id(tensor) not in reachable for tensor in inputs):
            if any(tensor.grad_fn is None for tensor in inputs):
                return args, kwargs, False
            handles: list[RemovableHandle] = []

            def on_input_grads(grads: Any) -> None:
                for handle in handles:
                    handle.remove()
                self._input_grads_ready(forward_pass)

            # A hook creates no tensors for spmd_types' checker to type.
            with dist._spmd_no_typecheck():
                handles.append(
                    torch.autograd.graph.register_multi_grad_hook(
                        inputs, on_input_grads
                    )
                )
            return args, kwargs, True
        outputs = _InputGradsTrigger.apply(
            functools.partial(self._input_grads_ready, forward_pass), *inputs
        )
        wrapped = {
            id(tensor): output for tensor, output in zip(inputs, outputs, strict=True)
        }
        args, kwargs = tree_map_only(
            torch.Tensor, lambda tensor: wrapped.get(id(tensor), tensor), (args, kwargs)
        )
        return args, kwargs, True

    def _input_grads_ready(self, forward_pass: GroupForwardPass | None) -> None:
        if forward_pass is None or forward_pass.completed:
            self.on_input_grads()
        else:
            self.on_partial_input_grads(forward_pass)

    @_outside_selective_checkpoint
    def post_forward_hook(self, mod: nn.Module, args: Any, output: Any) -> None:
        if torch.compiler.is_compiling():
            self._swap_in_params(self.sharded_params)
            return
        has_trigger = self.call_has_trigger.pop() if self.call_has_trigger else False
        if _in_backward():
            # Activation-checkpoint recompute inside backward: this bucket's
            # backward still needs the params; post-backward reshards.
            return
        with dist._spmd_no_typecheck():
            self._post_forward(output, has_trigger)

    @_outside_selective_checkpoint
    def group_post_forward_hook(self, mod: nn.Module, args: Any, output: Any) -> None:
        """Post-forward hook of each module of a bucket that hooks several: the
        group's post-forward runs once the last module of the pass finishes. A
        module called on its own leaves the pass open, without post-forward,
        as in FSDP2's partial group forward."""
        if torch.compiler.is_compiling():
            self._swap_in_params(self.sharded_params)
            return
        if _in_backward() or mod not in self.modules_to_run:
            return
        self.modules_to_run.discard(mod)
        if not self.modules_to_run:
            with dist._spmd_no_typecheck():
                self.complete_group_pass(output)

    def complete_group_pass(self, output: Any) -> None:
        """Run the group's post-forward on ``output`` for its open forward
        pass: once the pass's last module finishes, or from the root module's
        post-forward for modules the forward skipped, as FSDP2's
        ``_force_complete_incomplete_states`` does."""
        forward_pass, self.group_pass = self.group_pass, None
        self.modules_to_run.clear()
        has_trigger = False
        if forward_pass is not None:
            forward_pass.completed = True
            has_trigger = forward_pass.has_trigger
        self._post_forward(output, has_trigger)

    def _post_forward(self, output: Any, has_trigger: bool) -> None:
        """Reshard after a forward call, or hook its outputs to re-gather in
        backward and count it for the post-backward trigger."""
        if self.post_forward_index is None:
            self.post_forward_index = len(self.context.post_forward_order)
            self.context.post_forward_order.append(self)
        tensors = _tensor_leaves(output)
        grad_outputs = [tensor for tensor in tensors if tensor.requires_grad]
        if (
            output is None
            or not torch.is_grad_enabled()
            or (tensors and not grad_outputs)
        ):
            # No backward will reach these params through this forward.
            self.reshard()
            return
        self.backward_calls += 1
        if not grad_outputs:
            # Outputs without visible tensors: no pre-backward hook can re-gather,
            # so keep the params until post-backward.
            return
        self.input_triggers += has_trigger
        for value in grad_outputs:
            # Hook the output's node, which backward reaches first: torch_remat
            # returns a view whose node starts the checkpoint recompute, which
            # replays the forward without its hooks. Also hook the node of a
            # view output's base: an in-place op on the view replaces the
            # view's node, which would drop a hook on it. The hook may run once
            # per node. Leaves (e.g. a returned param) get none: hooks on them
            # would persist.
            base = value._base
            nodes = {value.grad_fn, None if base is None else base.grad_fn}
            for node in nodes - {None}:
                node.register_prehook(self.pre_backward_hook)
        if self.bucket_storage._reshard_after_forward:
            persistent = {_storage_ptr(buffer) for buffer in self.persistent_buffers}
            # An output viewing the persistent storage would see it freed.
            if not any(_storage_ptr(value) in persistent for value in grad_outputs):
                self.reshard()

    def _pre_forward_compile(self) -> None:
        """Trace the bucket unshard and expose its outputs via ``_parameters``.

        Collectives run synchronously inside ``_BucketUnshard`` so Dynamo traces
        one all-gather (forward) and one reduce-scatter (backward) per bucket;
        graph passes may reorder them. The traced graph owns buffer lifetimes,
        so the persistent buffers are not used here.
        """
        if not self.bucket_storage._requires_gradient_sync:
            raise NotImplementedError(
                "FlexShard set_requires_gradient_sync(False) is eager-only; "
                "torch.compile reduce-scatters in the traced backward."
            )
        if (
            self.bucket_storage._pre_backward_hook is not None
            or self.bucket_storage._post_reduce_hook is not None
        ):
            raise NotImplementedError(
                "FlexShard BucketSpec pre_backward_hook and post_reduce_hook are "
                "eager-only; torch.compile does not use the persistent unsharded "
                "params they receive."
            )
        if self.bucket_storage._defer_post_backward:
            raise NotImplementedError(
                "FlexShard BucketSpec defer_post_backward is eager-only; "
                "torch.compile reduce-scatters in the traced backward."
            )
        if len(self.bucket_storage._hook_module_fqns or ()) > 1:
            raise NotImplementedError(
                "FlexShard buckets of patterns naming several modules are eager-only."
            )
        full_params = _BucketUnshard.apply(self, *self._local_shards(use_autograd=True))
        self._swap_in_params(list(full_params))


class _InputGradsTrigger(torch.autograd.Function):
    """Identity on a forward call's grad-requiring inputs whose backward runs
    ``callback`` once their grads are computed, as FSDP2's
    ``RegisterPostBackwardFunction`` runs its post-backward. Unlike hooks on
    the inputs, it holds nothing that refers back to the graph."""

    @staticmethod
    def forward(
        ctx: Any, callback: Callable[[], None], *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        ctx.callback = callback
        return inputs

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple[Any, ...]:
        ctx.callback()
        return (None, *grads)


if dist._is_spmd_types_available():
    import spmd_types

    # It runs on typed inputs under spmd_types' checker, as FSDP2 registers its
    # RegisterPostBackwardFunction.
    spmd_types.register_local_autograd_function(_InputGradsTrigger)


class _BucketUnshard(torch.autograd.Function):
    """Traced bucket unshard for the compile branch.

    Forward all-gathers the bucket and backward reduce-scatters its gradients
    with synchronous functional collectives, returning the local-shard grads
    through autograd. Graph passes own scheduling and buffer lifetimes.
    """

    @staticmethod
    def forward(
        ctx: Any,
        bucket: BucketRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.bucket = bucket
        full_params = (
            bucket.begin_unshard([shard.detach() for shard in local_shards])
            .finish()
            .full_params
        )
        frozen_params = [
            full_param
            for full_param, shard in zip(full_params, local_shards, strict=True)
            if not shard.requires_grad
        ]
        if frozen_params:
            ctx.mark_non_differentiable(*frozen_params)
        return tuple(full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        bucket: BucketRuntime = ctx.bucket
        input_grads: list[torch.Tensor | None] = [None] * len(full_param_grads)
        grads: list[torch.Tensor] = []
        infos: list[ParamInfo] = []
        indices: list[int] = []
        for idx, (grad, info) in enumerate(
            zip(full_param_grads, bucket.infos, strict=True)
        ):
            if grad is None:
                continue
            grads.append(grad)
            infos.append(info)
            indices.append(idx)
        if grads:
            # Frozen outputs still get zero grads, which eager never sees.
            trainable = [ctx.needs_input_grad[idx + 1] for idx in indices]
            infos = _promote_reduce_dtype_over_grads(grads, infos, trainable)
            # Compile does not declare output grad dtypes, so grads arrive in
            # each output's dtype; the reduce-scatter copy-in casts them to the
            # bucket's reduce dtype as it copies.
            sharded_grads = begin_reduce_grad(
                grads,
                infos,
                bucket.bucket_storage.reduce_mesh,
                bucket.bucket_storage.gradient_reduction,
                bucket.context.reduce_grad_stream,
                debug_fqn=bucket.debug_fqn,
            ).finish()
            # Inputs are the sharded params, so autograd casts each grad to
            # that leaf's grad_dtype before accumulating it.
            for idx, sharded_grad in zip(indices, sharded_grads, strict=True):
                input_grads[idx] = sharded_grad
        return (None, *input_grads)


def _promote_reduce_dtype_over_grads(
    grads: list[torch.Tensor | None],
    infos: list[ParamInfo],
    trainable: list[bool] | None = None,
) -> list[ParamInfo]:
    """Promote the bucket reduce dtype over this backward's grads.

    Only params requiring grad now count, which may differ from
    ``ParamInfo.requires_grad`` at wrap time; ``trainable`` marks them
    (default: all). Every returned info gets the promoted dtype.
    """
    if trainable is None:
        trainable = [True] * len(infos)
    kept = [
        (grad, info)
        for grad, info, keep in zip(grads, infos, trainable, strict=True)
        if keep
    ]
    if not kept:
        return infos
    # Only widens the wrap-time dtype, so it still covers params frozen since
    # wrap. Nothing checks that ranks agree: grads arriving in different
    # dtypes on different ranks make them reduce in different dtypes. A
    # missing grad counts in the unsharded dtype, in which grads normally
    # arrive.
    reduce_dtype = functools.reduce(
        torch.promote_types,
        (
            info.unsharded_grad_dtype
            or (info.unsharded_dtype if grad is None else grad.dtype)
            for grad, info in kept
        ),
        # Without trainable params at wrap, each info has its own dtype.
        kept[0][1].grad_reduce_dtype,
    )
    if all(info.grad_reduce_dtype == reduce_dtype for info in infos):
        return infos
    # ParamInfo is shared bucket metadata; this dtype holds for one backward.
    return [replace(info, bucket_reduce_dtype=reduce_dtype) for info in infos]


def _match_param_grad_layout(
    grad: torch.Tensor,
    param: nn.Parameter,
) -> torch.Tensor:
    """Return a grad tensor suitable for assignment to ``param.grad``."""
    dtype = param.grad_dtype or grad.dtype
    if grad.dtype != dtype or grad.device != param.device:
        grad = grad.to(device=param.device, dtype=dtype)
    if grad.layout != param.layout:
        raise RuntimeError(
            "FlexShard reduced gradient layout does not match the local "
            f"parameter layout: grad={grad.layout}, param={param.layout}"
        )
    if grad.layout == torch.strided and grad.stride() != param.stride():
        aligned = torch.empty_strided(
            tuple(param.shape),
            tuple(param.stride()),
            dtype=dtype,
            device=param.device,
        )
        aligned.copy_(grad)
        grad = aligned
    return grad


def _accumulate_sharded_grads(
    sharded_params: list[nn.Parameter],
    sharded_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Cast sharded grads to param grad dtype/layout and accumulate into .grad."""
    stored_grads: list[torch.Tensor] = []
    for param, grad in zip(sharded_params, sharded_grads, strict=True):
        grad = _match_param_grad_layout(grad, param)
        stored_grads.append(grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad
    return stored_grads


def _install_bucket_unshard_hooks(
    bucket_storages: list[ShardedBucketStorage],
) -> None:
    """Install each bucket's forward hooks (one collective per bucket), and
    state-dict hooks that reshard first, as FSDP2 does."""
    owner_buckets: dict[nn.Module, dict[int, BucketRuntime]] = {}
    group_buckets: list[BucketRuntime] = []
    module_positions = {
        module: position
        for position, module in enumerate(bucket_storages[0]._module.modules())
    }
    for bucket_storage in bucket_storages:
        if not bucket_storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if bucket_storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_runtime = BucketRuntime.from_bucket_storage(bucket_storage)
        # A bucket whose patterns name modules hooks those (BucketSpec.patterns).
        modules = (
            [
                _unwrap_checkpoint(bucket_storage._module.get_submodule(fqn))
                for fqn in bucket_storage._hook_module_fqns
            ]
            if bucket_storage._hook_module_fqns
            else [bucket_runtime.forward_hook_module()]
        )
        bucket_runtime.module_position = min(module_positions[m] for m in modules)
        if len(modules) == 1:
            pre_hook = bucket_runtime.pre_forward_hook
            post_hook = bucket_runtime.post_forward_hook
        else:
            bucket_runtime.group_modules = tuple(modules)
            pre_hook = bucket_runtime.group_pre_forward_hook
            post_hook = bucket_runtime.group_post_forward_hook
            group_buckets.append(bucket_runtime)
        for module in modules:
            # Prepended so earlier user pre-forward hooks see the unsharded params.
            module.register_forward_pre_hook(pre_hook, prepend=True, with_kwargs=True)
            module.register_forward_hook(post_hook, always_call=True)
        bucket_runtime.context.buckets.append(bucket_runtime)
        for bucket_param in bucket_runtime.bucket_params:
            owner_buckets.setdefault(bucket_param.param_owner.module, {})[
                id(bucket_runtime)
            ] = bucket_runtime

    if group_buckets:
        # After the buckets' own hooks on the root module, if any.
        bucket_storages[0]._module.register_forward_hook(
            functools.partial(_complete_group_passes, tuple(group_buckets))
        )
    for module, buckets in owner_buckets.items():
        reshard_hook = functools.partial(_reshard_buckets, tuple(buckets.values()))
        module.register_state_dict_pre_hook(reshard_hook)
        module._register_load_state_dict_pre_hook(reshard_hook)


def _complete_group_passes(
    buckets: tuple[BucketRuntime, ...], module: nn.Module, args: Any, output: Any
) -> None:
    """Root post-forward hook: complete the forward pass of each bucket of
    several modules that the forward ran only some of, e.g. a norm without the
    output projection it shares a bucket with, as FSDP2's root post-forward
    completes such groups."""
    if torch.compiler.is_compiling() or _in_backward():
        return
    with dist._spmd_no_typecheck():
        for bucket in buckets:
            if bucket.group_pass is not None:
                bucket.complete_group_pass(output)


def _unwrap_checkpoint(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def _reshard_buckets(
    buckets: tuple[BucketRuntime, ...], *args: Any, **kwargs: Any
) -> None:
    """State-dict pre-hook: swap the local shards back in first."""
    for bucket in buckets:
        if bucket.is_unsharded:
            bucket.reshard()
