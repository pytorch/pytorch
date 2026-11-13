# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The lifecycle of a sharded module, shared by ``fully_shard`` and
``flex_shard``.

A sharding call (one ``fully_shard`` or ``flex_shard`` call) puts a state on
its modules: its bucket groups, each of which unshards on the forward of its
modules, and the forward casts of its mixed precision policy. The first state
to run forward becomes the root: it finds the states in its module tree, and
joins all their buckets to one communication context, which schedules their
prefetches and reductions. ``FSDPState`` adds FSDP2's frontend rules.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.autograd.graph import _MultiHandle
from torch.distributed import _spmd_no_typecheck
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.utils import _apply_to_tensors

from . import module_ops
from .api import GradientReductionHandle
from .bucket_runtime import _in_backward, BucketCommContext, PartialBackward
from .utils import _cast_fp_tensor, TrainingState


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch.utils.hooks import RemovableHandle

    from .bucket_runtime import BucketRuntime
    from .bucket_storage import ShardedBucketStorage


logger = logging.getLogger("torch.distributed.fsdp.flex_shard")

_StateType = TypeVar("_StateType", bound="ShardedModuleState")


class BucketGroup:
    """Buckets that unshard together on the forward of ``modules``: one
    bucket, the runtime, as a ``fully_shard`` group or a ``flex_shard``
    bucket."""

    def __init__(
        self,
        runtime: BucketRuntime,
        modules: tuple[nn.Module, ...],
        *,
        named_by_modules: bool = True,
    ) -> None:
        self.bucket = runtime
        self.modules = modules
        if len(modules) > 1:
            runtime.group_modules = modules
        # The FQNs of the modules, for debug logs, once the root names them,
        # unless the frontend names the group
        self.module_fqn: str | None = None
        self.named_by_modules = named_by_modules
        # The most packed reduce-scatter inputs to keep alive; the root's
        # context keeps the largest of its groups' (FSDP2's
        # set_reduce_scatter_max_input_buffers)
        self.max_input_buffers = 1
        self._lazy_initialized = False

    @property
    def storage(self) -> ShardedBucketStorage:
        return self.bucket.bucket_storage

    def lazy_init(self) -> None:
        """Re-read parameters that modules replaced since construction (e.g.
        ``to_empty()`` on a submodule), once, before the group joins its root's
        context."""
        if self._lazy_initialized:
            return
        self._lazy_initialized = True
        homing = self.bucket.homing
        homing.reset_sharded_params()
        homing.init_extensions()
        homing.check_materialized(self.bucket.log_name)
        homing.refresh()

    def refresh(self) -> None:
        """Re-home parameters replaced since the last unshard, before an
        explicit unshard."""
        self.bucket.homing.refresh()

    def join(self, context: BucketCommContext, *, reshard_after_forward: bool) -> None:
        """Join the root module's context."""
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
        if self.named_by_modules:
            self.bucket.debug_fqn = self.module_fqn


class ModuleStateContext(Generic[_StateType]):
    """The state that the states under one root share."""

    def __init__(self) -> None:
        # All states in the root state's module tree, in ``named_modules()``
        # pre-order. The root post-forward completes incomplete group forwards
        # in ``reversed`` order so that within a nesting chain the deepest group
        # finishes first, matching the order ``_post_forward`` would unwind in
        # the non-partial case.
        self.all_states: list[_StateType] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: _StateType | None = None
        # The context of the states, which keeps is_last_backward as
        # BucketCommContext.wait_reductions_at_backward_end
        self.engine_context: BucketCommContext | None = None
        self._is_last_backward = True
        # Whether backward finalization is owned by the caller
        self.manual_backward_finalization: bool = False
        # Manual finalization mode selected when backward starts
        self.manual_backward_finalization_active: bool | None = None
        # An asynchronous explicit finalization must be waited on before this
        # tree starts more work.
        self.gradient_reduction_pending: bool = False
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None

    @property
    def is_last_backward(self) -> bool:
        """Whether to finalize backward in this backward's final callback."""
        return self._is_last_backward

    @is_last_backward.setter
    def is_last_backward(self, is_last_backward: bool) -> None:
        self._is_last_backward = is_last_backward
        if self.engine_context is not None:
            self.engine_context.wait_reductions_at_backward_end = is_last_backward


class ShardedModuleState(_State):
    """The state of one sharding call (see the module docstring)."""

    # Name used in error messages
    _state_name: str = "FlexShard"
    _logger: logging.Logger = logger

    def __init__(self) -> None:
        super().__init__()
        self._groups: list[BucketGroup] = []
        self._engine_context: BucketCommContext | None = None
        # The context share_comm_ctx gave this root, for its lazy init
        self._shared_engine_context: BucketCommContext | None = None
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx: ModuleStateContext = ModuleStateContext()
        self._training_state: TrainingState = TrainingState.IDLE
        # Whether the root keeps its groups unsharded after forward, since
        # training all-gathers them right away in backward: unless the user
        # set reshard_after_forward
        self._auto_reshard_after_forward: bool | None = True
        # The module whose forward is the root's, and each hooked module's
        # groups and input and output cast dtypes
        self._root_module: nn.Module | None = None
        self._module_groups: dict[nn.Module, list[BucketGroup]] = {}
        self._module_casts: dict[
            nn.Module, tuple[torch.dtype | None, torch.dtype | None]
        ] = {}
        self._hooked_modules: set[nn.Module] = set()
        self._hook_handles: list[RemovableHandle] = []

    @property
    def _initialized(self) -> bool:
        return hasattr(self, "_modules")

    def _get_state_for_module(self, module: nn.Module) -> ShardedModuleState | None:
        """The state of a module of the root's tree, if the same frontend
        made it."""
        state = _get_module_state(module)
        return state if type(state) is type(self) else None

    # Setup #
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        casts: tuple[torch.dtype | None, torch.dtype | None],
        auto_reshard_after_forward: bool,
        *,
        root_module: nn.Module | None = None,
        post_hook_always_call: bool | None = None,
    ) -> None:
        """Put the state on ``modules``, cast their forward inputs and outputs
        to ``casts``, and hook their forward and that of the modules of the
        groups (``add_group``). ``root_module`` is the module whose forward is
        the root's (default: the only module)."""
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._auto_reshard_after_forward = auto_reshard_after_forward
        self._root_module = root_module
        self._post_hook_always_call = (
            len(modules) > 1 if post_hook_always_call is None else post_hook_always_call
        )
        for module in modules:
            self._module_casts[module] = casts
        hooked = dict.fromkeys(modules)
        if root_module is not None:
            hooked[root_module] = None
        for group in self._groups:
            hooked.update(dict.fromkeys(group.modules))
        for module in hooked:
            self._hook(module)

    def _hook(self, module: nn.Module) -> None:
        """Run the state's forward hooks on ``module``, once."""
        if module in self._hooked_modules:
            return
        self._hooked_modules.add(module)
        self._module_casts.setdefault(module, (None, None))
        # The groups' hooks run per module and track the group's forward pass
        # themselves
        self._hook_handles.append(
            module.register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
        )
        # Hooks that unshard groups also run when the forward raises, so that
        # the groups reshard; with several modules, a group's pass can end in
        # any of them
        always_call = self._post_hook_always_call and (
            module in self._module_groups
            or (module in self._modules and len(self._modules) > 1)
        )
        self._hook_handles.append(
            module.register_forward_hook(
                self._post_forward, prepend=False, always_call=always_call
            )
        )
        handle = _MultiHandle(tuple(self._hook_handles))
        self._pre_forward_hook_handle: RemovableHandle | _MultiHandle = handle
        self._post_forward_hook_handle: RemovableHandle | _MultiHandle = handle

    def add_group(
        self,
        group: BucketGroup,
        casts: tuple[torch.dtype | None, torch.dtype | None] | None = None,
    ) -> None:
        """Add a group, which unshards on the forward of its modules, whose
        inputs and outputs it casts to ``casts`` if given."""
        self._groups.append(group)
        for module in group.modules:
            self._module_groups.setdefault(module, []).append(group)
            if (
                casts is not None
                and self._module_casts.setdefault(module, casts) != casts
            ):
                raise ValueError(
                    f"Buckets that hook module {type(module).__name__} must "
                    "cast its forward inputs and outputs alike, but cast to "
                    f"{self._module_casts[module]} and {casts}."
                )
            if self._initialized:
                self._hook(module)

    # Lazy init #
    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        self._logger.debug("%s::root_pre_forward", self._state_name)
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function(f"{self._state_name}::root_pre_forward"):
            # The all-gathers wait for the current stream when issued, so for
            # the optimizer; a post-optimizer event can make them wait for that
            # point instead.
            context = self._engine_context
            if context is not None:
                context.manual_backward_finalization = (
                    self._state_ctx.manual_backward_finalization
                )
                if (event := self._state_ctx.post_optim_event) is not None:
                    context.unshard_stream.wait_event(event)
                    self._state_ctx.post_optim_event = None
            args, kwargs = self._root_inputs(args, kwargs)
        return args, kwargs

    def _root_inputs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """The root's forward inputs, e.g. moved to its device."""
        return args, kwargs

    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. sharding has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        root_module = self._root_module
        if root_module is None:
            if len(self._modules) > 1:
                raise RuntimeError(
                    f"{self._state_name} requires a single root module but got "
                    f"{self._modules}"
                )
            root_module = self._modules[0]
        for module in root_module.modules():
            state = self._get_state_for_module(module)
            if (
                state is not None
                and state is not self
                and state._state_ctx.manual_backward_finalization
            ):
                raise RuntimeError(
                    "set_manual_backward_finalization must be called on the "
                    f"root {self._state_name} module"
                )
        self._is_root = True
        visited_states: set[ShardedModuleState] = set()
        for module_name, module in root_module.named_modules():
            if (state := self._get_state_for_module(module)) is None:
                continue
            if module is not root_module and state is not self:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        f"{self._state_name} state has already been lazily "
                        f"initialized for {module_name}\n{self._state_name} "
                        "requires running forward through the root module first"
                    )
                state._is_root = False
            # A single state can map to multiple modules (e.g.
            # fully_shard([mod_a, mod_b, mod_c])), so dedup here.
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        self._init_group_fqns(root_module)
        self._on_lazy_init(root_module)
        self._lazy_init_engine()

    def _on_lazy_init(self, root_module: nn.Module) -> None:
        """Frontend checks at the root's lazy init, before groups join."""

    def _init_group_fqns(self, root_module: nn.Module) -> None:
        """Sets the groups' module FQNs, for debug logs."""
        module_to_groups: dict[nn.Module, list[BucketGroup]] = {}
        for state in self._state_ctx.all_states:
            for group in state._groups:
                for module in group.modules:
                    module_to_groups.setdefault(module, []).append(group)
        for module_name, module in root_module.named_modules():
            for group in module_to_groups.get(module, ()):
                if not group.named_by_modules:
                    continue
                if group.module_fqn is None:
                    group.module_fqn = module_name
                else:
                    group.module_fqn += f", {module_name}"

    def _lazy_init_engine(self) -> None:
        """Join every state's groups to one context, as FSDP2 shares the
        root's comm context."""
        # The context share_comm_ctx gave, or the one the first group was built
        # in, which the others join
        context = self._shared_engine_context
        if context is None:
            groups = [g for s in self._state_ctx.all_states for g in s._groups]
            context = (
                groups[0].bucket.context
                if groups
                else BucketCommContext.for_device(self._device)
            )
        context.backward_start_callbacks.append(self._engine_backward_start)
        context.backward_end_callbacks.append(self._engine_backward_end)
        self._state_ctx.engine_context = context
        context.wait_reductions_at_backward_end = self._state_ctx.is_last_backward
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._engine_context = context
            groups = state._groups
            for group in groups:
                # For the root, do not reshard after forward since for
                # training, the parameters would be all-gathered right away
                reshard = group.storage._reshard_after_forward and not (
                    state is self and self._auto_reshard_after_forward
                )
                group.join(context, reshard_after_forward=reshard)
                group.bucket.retires_pending_reductions = (
                    state._retires_pending_reductions(group)
                )

    def _retires_pending_reductions(self, group: BucketGroup) -> bool:
        """Whether the group's reduce first retires the pending reductions
        over the context's cap (see ``BucketRuntime``): of the groups that
        unshard on the same modules, only the last, whose post-backward
        trigger wraps the others' so that it reduces first; the others'
        reductions then overlap with its."""
        later = self._groups[self._groups.index(group) + 1 :]
        return all(other.modules != group.modules for other in later)

    def _engine_backward_start(self) -> None:
        """Fix the manual backward finalization mode until the backward
        iteration ends, as FSDP2's pre-backward does."""
        state_ctx = self._state_ctx
        if state_ctx.manual_backward_finalization_active is None:
            state_ctx.manual_backward_finalization_active = (
                state_ctx.manual_backward_finalization
            )

    def _engine_backward_end(self) -> None:
        """End of a backward: reset the per-iteration state, as FSDP2's root
        post-backward callback does."""
        self._logger.debug("%s::root_post_backward", self._state_name)
        self._state_ctx.iter_forward_root = None
        # With manual finalization, finalize_backward ends the iteration
        if (
            self._state_ctx.is_last_backward
            and not self._state_ctx.manual_backward_finalization_active
        ):
            self._state_ctx.manual_backward_finalization_active = None
        for state in self._state_ctx.all_states:
            state._training_state = TrainingState.IDLE

    # Forward #
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if torch.compiler.is_compiling():
            # The buckets' hooks trace their unshards; the root's eager
            # bookkeeping stays out of the graph
            args, kwargs = self._cast_inputs(module, args, kwargs)
            for group in self._module_groups.get(module, ()):
                runtime = group.bucket
                hook = (
                    runtime.group_pre_forward_hook
                    if runtime.group_modules
                    else runtime.pre_forward_hook
                )
                if (result := hook(module, args, kwargs)) is not None:
                    args, kwargs = result
            return args, kwargs
        context = self._engine_context
        if self._state_ctx.gradient_reduction_pending or (
            context is not None and context.pending_finalization is not None
        ):
            raise RuntimeError(
                "The previous gradient reduction must be waited on before forward"
            )
        # In an activation checkpointing recompute, the buckets' hooks unshard
        if not _in_backward():
            with _spmd_no_typecheck():
                # With grouped ``fully_shard([a, b, ...])`` the pre-hook fires
                # per module; root setup runs on the first module's entry
                state_first_in_pass = self._training_state != TrainingState.FORWARD
                self._training_state = TrainingState.FORWARD
                if state_first_in_pass:
                    args, kwargs = self._root_pre_forward(module, args, kwargs)
        args, kwargs = self._cast_inputs(module, args, kwargs)
        # The buckets' hooks unshard outside spmd_types' type checker
        # themselves; the post-backward trigger they wrap the typed inputs in
        # stays checked
        for group in self._module_groups.get(module, ()):
            runtime = group.bucket
            hook = (
                runtime.group_pre_forward_hook
                if runtime.group_modules
                else runtime.pre_forward_hook
            )
            if (result := hook(module, args, kwargs)) is not None:
                args, kwargs = result
            self._after_group_pre_forward(group)
        return args, kwargs

    def _cast_inputs(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        input_dtype, _ = self._module_casts.get(module, (None, None))
        if input_dtype is None:
            return args, kwargs
        with torch.profiler.record_function(f"{self._state_name}::cast_forward_inputs"):
            cast_fn = functools.partial(_cast_fp_tensor, input_dtype)
            return _apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs)

    def _after_group_pre_forward(self, group: BucketGroup) -> None:
        """Called after a group's pre-forward unshard."""

    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        if torch.compiler.is_compiling():
            for group in self._module_groups.get(module, ()):
                runtime = group.bucket
                if runtime.group_modules:
                    runtime.group_post_forward_hook(module, input, output)
                else:
                    runtime.post_forward_hook(module, input, output)
            return self._cast_output(
                output, self._module_casts.get(module, (None, None))[1]
            )
        with _spmd_no_typecheck():
            for group in self._module_groups.get(module, ()):
                runtime = group.bucket
                if runtime.group_modules:
                    runtime.group_post_forward_hook(module, input, output)
                else:
                    runtime.post_forward_hook(module, input, output)
            _, output_dtype = self._module_casts.get(module, (None, None))
            if _in_backward():
                return self._cast_output(output, output_dtype)
            if not any(g.bucket.group_pass is not None for g in self._groups):
                self._training_state = TrainingState.IDLE
            if self._state_ctx.iter_forward_root is self and (
                self._root_module is None or module is self._root_module
            ):
                # Complete the forward of groups whose modules did not all run,
                # e.g. a decoder that leaves its output projection to the loss.
                # See ``all_states`` for why this iterates ``reversed``.
                for state in reversed(self._state_ctx.all_states):
                    if state is self and self._root_module is None:
                        continue
                    for group in state._groups:
                        if group.bucket.group_pass is not None:
                            group.bucket.complete_group_pass(output)
                            state._training_state = TrainingState.IDLE
                self._state_ctx.iter_forward_root = None
            if torch._C._are_functorch_transforms_active():
                # Under functorch, outputs that outer autograd differentiates can
                # report requires_grad=False, so the buckets hooked none of them
                output = _apply_to_tensors(
                    lambda t: _FunctorchPreBackward.apply(self, t)
                    if not t.requires_grad and (t.is_floating_point() or t.is_complex())
                    else t,
                    output,
                )
            return self._cast_output(output, output_dtype)

    def _cast_output(self, output: Any, dtype: torch.dtype | None) -> Any:
        if dtype is None:
            return output
        with torch.profiler.record_function(
            f"{self._state_name}::cast_forward_outputs"
        ):
            return _apply_to_tensors(functools.partial(_cast_fp_tensor, dtype), output)

    def _complete_grouped_forward(self, output: Any) -> Any:
        """See ``FSDPModule.complete_grouped_forward``."""
        with _spmd_no_typecheck():
            for group in self._groups:
                if group.bucket.group_pass is not None:
                    group.bucket.complete_group_pass(output)
                    self._training_state = TrainingState.IDLE
        return output

    # Backward #
    @contextmanager
    def _partial_backward(self, outer_backward_follows: bool) -> Iterator[None]:
        """See ``FSDPModule.partial_backward``."""
        from .module_ops import unshard

        context = self._engine_context
        if self._is_root is None or context is None:
            raise RuntimeError(
                f"{self._state_name} partial_backward needs the root forward to run first"
            )
        if context.partial_backward is not None:
            raise RuntimeError(f"{self._state_name} partial_backward cannot nest")
        groups = self._groups
        runtimes = tuple(group.bucket for group in groups)
        manual_finalization = self._state_ctx.manual_backward_finalization
        with _spmd_no_typecheck():
            for group in groups:
                group.lazy_init()
                group.refresh()
            unshard(runtimes, async_op=False)
        scope = context.partial_backward = PartialBackward(runtimes)
        try:
            yield
        except BaseException:
            # Reshard without collectives, which only some ranks might reach;
            # reset_iter_state discards the gradients
            context.partial_backward = None
            with _spmd_no_typecheck():
                for runtime in runtimes:
                    keep = scope.ran_backward and (
                        manual_finalization
                        or not runtime.bucket_storage._reshard_after_backward
                    )
                    if runtime.is_unsharded and not keep:
                        runtime.reshard()
            raise
        context.partial_backward = None
        with _spmd_no_typecheck():
            if not scope.ran_backward:
                for runtime in runtimes:
                    if runtime.is_unsharded:
                        runtime.reshard()
                return
            # The outer backward reduces a group shared with modules outside
            # this one, which get their gradients there
            shared = any(len(group.modules) > 1 for group in groups)
            if not manual_finalization and not (outer_backward_follows and shared):
                for runtime in runtimes:
                    runtime.post_backward()
            if not outer_backward_follows:
                context.end_backward(finish=True, manual=manual_finalization)

    def _functorch_pre_backward(self, grad: torch.Tensor) -> None:
        for group in self._groups:
            group.bucket.pre_backward_hook((grad,))

    # Explicit finalization and reset #
    @property
    def runtimes(self) -> list[BucketRuntime]:
        return [group.bucket for group in self._groups]

    @property
    def storages(self) -> list[ShardedBucketStorage]:
        return [group.storage for group in self._groups]

    def _set_reduce_scatter_max_input_buffers(self, max_input_buffers: int) -> None:
        # bool is an int subclass; reject it so True does not silently mean 1
        if isinstance(max_input_buffers, bool) or not isinstance(
            max_input_buffers, int
        ):
            raise TypeError(
                "max_input_buffers must be an int, got "
                f"{type(max_input_buffers).__name__}"
            )
        if max_input_buffers < 1:
            raise ValueError(
                f"max_input_buffers must be a positive int, got {max_input_buffers}"
            )
        for group in self._groups:
            group.max_input_buffers = max_input_buffers
        if (context := self._engine_context) is not None:
            context.max_pending_reduce_grads = max(
                (
                    g.max_input_buffers
                    for s in self._state_ctx.all_states
                    for g in s._groups
                ),
                default=1,
            )

    def _unshard(self, async_op: bool) -> list[BucketRuntime]:
        """Start (``async_op``) or finish the groups' unshards on the current
        stream, re-reading parameters replaced since (``BucketGroup.refresh``);
        return the buckets to wait on."""
        for group in self._groups:
            group.lazy_init()
            group.refresh()
        return module_ops.unshard(self.runtimes, async_op)

    def _set_manual_backward_finalization(self, enabled: bool) -> None:
        """Whether the caller finalizes backward (``finalize_backward``)
        instead of the end of each backward; set on the root."""
        if self._is_root is False:
            raise RuntimeError(
                "set_manual_backward_finalization must be called on the root "
                f"{self._state_name} module"
            )
        state_ctx = self._state_ctx
        if (
            enabled != state_ctx.manual_backward_finalization
            and torch._C._current_graph_task_id() != -1
        ):
            raise RuntimeError(
                "set_manual_backward_finalization cannot change mode during backward"
            )
        active_mode = state_ctx.manual_backward_finalization_active
        if active_mode is not None and enabled != active_mode:
            raise RuntimeError(
                "set_manual_backward_finalization cannot change mode after backward starts"
            )
        state_ctx.manual_backward_finalization = enabled
        # The next backward follows it even without a forward in between
        if (context := self._engine_context) is not None:
            context.manual_backward_finalization = enabled

    def _finalize_backward(self, async_op: bool) -> GradientReductionHandle | None:
        """Issue the reductions that backwards left and reshard; with
        ``async_op``, return a handle to wait for them instead of waiting."""
        if self._is_root is None:
            return GradientReductionHandle() if async_op else None
        self._validate_finalize_backward()
        self._start_finalize_backward()
        if not async_op:
            self._end_backward_iteration()
            return None
        self._state_ctx.gradient_reduction_pending = True
        return FinalizationHandle(self)

    def _validate_finalize_backward(self) -> None:
        state_ctx, context = self._state_ctx, self._engine_context
        name = self._state_name
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
        if not self._is_root:
            raise RuntimeError(
                f"finalize_backward must be called on the root {name} module"
            )
        if not state_ctx.manual_backward_finalization:
            raise RuntimeError(
                f"{name} finalize_backward requires manual backward finalization. "
                "Call set_manual_backward_finalization(True) before forward"
            )
        if context is None:
            return
        if torch._C._current_graph_task_id() != -1:
            raise RuntimeError(
                f"{name} finalize_backward must be called after backward completes"
            )
        groups = [g for s in state_ctx.all_states for g in s._groups]
        if partial := next(
            (g for g in groups if g.bucket.partial_reduce_output is not None), None
        ):
            raise RuntimeError(
                f"{name} manual backward finalization does not support partial "
                f"gradient reductions for parameter group "
                f"{partial.module_fqn or '<root>'!r}. "
                "Use automatic backward finalization for this configuration"
            )
        if no_all_reduce := next(
            (
                g
                for g in groups
                if g.bucket.needs_finish()
                and g.storage._requires_gradient_sync
                and g.storage._replicate_group is not None
                and not g.storage._requires_all_reduce
            ),
            None,
        ):
            raise RuntimeError(
                f"{name} manual backward finalization requires all-reduce for "
                f"parameter group {no_all_reduce.module_fqn or '<root>'!r}. Call "
                "set_requires_all_reduce(True) before finalization"
            )

    def _start_finalize_backward(self) -> None:
        """Issue the reductions backward left and reshard, without waiting."""
        if (context := self._engine_context) is not None:
            context.start_finalize_backward()

    def _end_backward_iteration(self) -> None:
        """Wait for the gradient reductions and reset per-iteration state."""
        if (context := self._engine_context) is not None:
            context.wait_and_clear_reduce_grad_states(debug_fqn=None)
            context.pending_finalization = None
        self._engine_backward_end()
        self._state_ctx.manual_backward_finalization_active = None
        self._state_ctx.gradient_reduction_pending = False

    def _reset_iter_state(self) -> None:
        """Order in-flight collectives, reshard, and drop in-flight and
        accumulated gradients."""
        if self._is_root is False:
            raise RuntimeError(
                f"reset_iter_state must be called on the root {self._state_name} module"
            )
        state_ctx, context = self._state_ctx, self._engine_context
        if (
            context is not None
            and (pending := context.pending_finalization) is not None
            and pending._state is not None
            and pending._state._state_ctx is not state_ctx
        ):
            raise RuntimeError(
                "reset_iter_state cannot reset a gradient reduction owned by "
                f"another {self._state_name} root sharing the communication context"
            )
        if context is not None:
            context.reset_iteration()
        for s in state_ctx.all_states:
            s._training_state = TrainingState.IDLE
            for param in (p for r in s.runtimes for p in r.sharded_params):
                param.grad = None
        state_ctx.iter_forward_root = None
        state_ctx.manual_backward_finalization_active = None
        state_ctx.gradient_reduction_pending = False


class FinalizationHandle(GradientReductionHandle):
    """Returned by ``finalize_backward(async_op=True)``."""

    def __init__(self, state: ShardedModuleState) -> None:
        self._state: ShardedModuleState | None = state
        # The engine context, which roots may share, holds one pending
        # finalization at a time
        state._engine_context.pending_finalization = self  # type: ignore[union-attr]

    def wait(self) -> None:
        if (state := self._state) is None:
            return
        if state._engine_context.pending_finalization is not self:  # type: ignore[union-attr]
            raise RuntimeError("This gradient reduction handle was invalidated")
        state._end_backward_iteration()
        self._state = None


def share_context(states: Sequence[ShardedModuleState]) -> None:
    """The roots share one context (streams, prefetches, pending
    reductions)."""
    first = states[0]
    context = first._engine_context or first._shared_engine_context
    if context is None:
        context = BucketCommContext.for_device(first._device)
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
            state._state_ctx.engine_context = context
        state._shared_engine_context = context


class _FunctorchPreBackward(torch.autograd.Function):
    """Runs the buckets' pre-backward hooks for an output under functorch
    transforms (e.g. vmap, jvp)."""

    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(state: ShardedModuleState, output: torch.Tensor) -> torch.Tensor:
        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        ctx.state = inputs[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
        ctx.state._functorch_pre_backward(grad)
        return None, grad

    @staticmethod
    # pyrefly: ignore [bad-override]
    def jvp(ctx: Any, state_tangent: Any, tangent: torch.Tensor | None) -> Any:
        # Keep the tangent's backward on the pre-backward path
        if tangent is None:
            return None
        return _FunctorchPreBackward.apply(ctx.state, tangent)
