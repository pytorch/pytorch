# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import logging
from collections.abc import Callable, Sequence
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import collect_grad_tensors
from torch.distributed.utils import _apply_to_tensors, _to_kwargs

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _cast_fp_tensor, _dynamo_disable, TrainingState
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup


if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

_StateType = TypeVar("_StateType", bound="FSDPState")


class FSDPStateContext(Generic[_StateType]):
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        # All FSDP states in the root state's module tree, in
        # ``named_modules()`` pre-order. ``_force_complete_incomplete_states``
        # iterates ``reversed`` so within a nesting chain the deepest
        # state's ``output_dtype`` cast applies first, matching the order
        # ``_post_forward`` would unwind in the non-partial case.
        self.all_states: list[_StateType] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: _StateType | None = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


class FSDPState(_State):
    # Name used in error messages; subclasses can override
    _state_name: str = "FSDP"

    def __init__(self) -> None:
        super().__init__()
        # Support multiple param groups for per-param mesh support.
        # Each group has params with the same mesh_info.
        self._fsdp_param_groups: list[FSDPParamGroup] = []
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._states_to_forward_prefetch: list[FSDPState] = []
        self._states_to_backward_prefetch: list[FSDPState] = []
        self._modules_to_run_forward: set[nn.Module] = set()
        # ``False`` when user set reshard_after_forward
        # through ``fully_shard`` or ``set_reshard_after_forward``
        self._auto_reshard_after_forward: bool | None = True

    def _get_state_for_module(self, module: nn.Module) -> "FSDPState | None":
        """Get the state for a module. Subclasses can override to use different state getters."""
        return _get_module_fsdp_state(module)

    @property
    def _fsdp_param_group(self) -> FSDPParamGroup | None:
        """
        Returns the param group for backward compatibility.
        This property is only valid when there is at most one param group.
        For per-param mesh support with multiple param groups, use
        ``_fsdp_param_groups`` instead.
        """
        if len(self._fsdp_param_groups) > 1:
            group_fqns = [g._module_fqn for g in self._fsdp_param_groups]
            raise AssertionError(
                f"Expected at most 1 param group for backward compatibility, "
                f"but got {len(self._fsdp_param_groups)} (fqns: {group_fqns}). "
                f"Use `_fsdp_param_groups` (plural) to access all param groups "
                f"when using per-param mesh via shard_placement_fn returning "
                f"ShardPlacementResult."
            )
        if self._fsdp_param_groups:
            return self._fsdp_param_groups[0]
        return None

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        self._auto_reshard_after_forward = auto_reshard_after_forward
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
                self._cast_output_dtype,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        logger.debug("FSDP::root_pre_forward")
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            if (event := self._state_ctx.post_optim_event) is not None:
                self._comm_ctx.all_gather_copy_in_stream.wait_event(event)
                self._comm_ctx.all_gather_stream.wait_event(event)
                self._state_ctx.post_optim_event = None
            else:
                current_stream = self._device_handle.current_stream()
                self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
                self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type in [
                "cuda",
                "hpu",
                "xpu",
                "mtia",
                torch._C._get_privateuse1_backend_name(),
            ]:
                with torch.profiler.record_function("FSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. FSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(
                f"{self._state_name} requires a single root module but got {self._modules}"
            )
        root_module = self._modules[0]
        visited_states: set[FSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := self._get_state_for_module(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        f"{self._state_name} state has already been lazily initialized for "
                        f"{module_name}\n{self._state_name} requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            # A single state can map to multiple modules (e.g.
            # fully_shard([mod_a, mod_b, mod_c])), so dedup here.
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        # For the root, do not reshard after forward since for training,
        # the parameters would be freed and all-gathered immediately
        if self._auto_reshard_after_forward:
            for fsdp_param_group in self._fsdp_param_groups:
                fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()
        self._validate_no_duplicate_params()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group.lazy_init()

    def _validate_no_duplicate_params(self) -> None:
        seen: set[int] = set()
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    if fsdp_param._orig_param_uid in seen:
                        raise ValueError(
                            f"Parameter '{fsdp_param._param_fqn}' is shared with a "
                            f"parameter already managed by another FSDP group. "
                            f"For shared/tied parameters, use "
                            f"fully_shard([module_a, module_b]) to place them in "
                            f"the same FSDP group."
                        )
                    seen.add(fsdp_param._orig_param_uid)

    def _init_shared_state(self) -> None:
        self._comm_ctx.lazy_init(self._device)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            num_groups = len(state._fsdp_param_groups)
            for i, fsdp_param_group in enumerate(state._fsdp_param_groups):
                fsdp_param_group.comm_ctx = self._comm_ctx
                fsdp_param_group._param_group_index = i
                fsdp_param_group._num_param_groups = num_groups

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        if not self._is_root:
            raise AssertionError("Expected _is_root to be True")
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, FSDPParam] = {}
        # Build a mapping from module to all its FSDPParamGroups (not just one)
        module_to_fsdp_param_groups: dict[nn.Module, list[FSDPParamGroup]] = {}
        for state in self._state_ctx.all_states:
            for fsdp_param_group in state._fsdp_param_groups:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    if module not in module_to_fsdp_param_groups:
                        module_to_fsdp_param_groups[module] = []
                    module_to_fsdp_param_groups[module].append(fsdp_param_group)
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_groups:
                # Set FQN for all param groups associated with this module
                for fsdp_param_group in module_to_fsdp_param_groups[module]:
                    module_fqn = fsdp_param_group._module_fqn
                    if module_fqn is None:
                        fsdp_param_group._module_fqn = module_name
                    else:
                        if not isinstance(module_fqn, str):
                            raise AssertionError(
                                f"Expected module_fqn to be str, got {type(module_fqn)}: {module_fqn}"
                            )
                        module_fqn += f", {module_name}"
                        fsdp_param_group._module_fqn = module_fqn

    @_dynamo_disable
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            # With nested FSDP and multiple forward passes before backward,
            # the params might have been resharded by a previous post_backward.
            # We need to ensure params are unsharded for AC recomputation.
            for fsdp_param_group in self._fsdp_param_groups:
                if not fsdp_param_group.is_unsharded:
                    fsdp_param_group.unshard()
                    fsdp_param_group.wait_for_unshard()
            return args, kwargs
        # With grouped ``fully_shard([a, b, ...])`` the pre-hook fires per
        # module (so ``cast_forward_inputs`` and ``fsdp_param_group.pre_forward``
        # run for each). Root setup and forward prefetch are one-shot, gated
        # on the first module's entry.
        state_first_in_pass = self._training_state != TrainingState.FORWARD
        self._training_state = TrainingState.FORWARD
        if state_first_in_pass:
            args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = (
                    _apply_to_tensors(cast_fn, args),
                    _apply_to_tensors(cast_fn, kwargs),
                )
        for fsdp_param_group in self._fsdp_param_groups:
            args, kwargs = fsdp_param_group.pre_forward(module, args, kwargs)
        if state_first_in_pass:
            for fsdp_state in self._states_to_forward_prefetch:
                # Forward order (not reversed) to match forward execution order;
                # contrast with reversed() in _pre_backward for backward order.
                for target_param_group in fsdp_state._fsdp_param_groups:
                    FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
        return args, kwargs

    @_dynamo_disable
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # When composing with module-hook-based activation checkpointing, the
        # post-backward hook is responsible for the reshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        for fsdp_param_group in self._fsdp_param_groups:
            output = fsdp_param_group.post_forward(module, input, output)
        output = self._register_pre_backward_hook(output)
        self._training_state = TrainingState.IDLE
        if self._state_ctx.iter_forward_root is self:
            output = self._force_complete_incomplete_states(output)
            if all_gather_state := self._comm_ctx.all_gather_state:
                # Free the last all-gather result if needed; refer to
                # [Note: Overlapping all-gather copy-in and all-gather]
                self._comm_ctx.all_gather_copy_in_stream.wait_event(
                    all_gather_state.event
                )
                self._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
                self._comm_ctx.all_gather_state = None  # free the all-gather result
            self._state_ctx.iter_forward_root = None
        return self._cast_output_dtype(output)

    def _cast_output_dtype(self, output: Any) -> Any:
        if self._mp_policy.output_dtype is None:
            return output
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            return _apply_to_tensors(
                functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                output,
            )

    def _force_complete_incomplete_states(self, output: Any) -> Any:
        # Complete post-forward for any state whose group forward did not
        # run all modules (e.g. chunked loss where model.forward skips
        # head). See ``all_states`` init for why we iterate ``reversed``.
        for state in reversed(self._state_ctx.all_states):
            if state is self or not state._modules_to_run_forward:
                continue
            logger.debug("FSDP::force_complete_post_forward")
            for fsdp_param_group in state._fsdp_param_groups:
                output = fsdp_param_group.post_forward(None, None, output)
            output = state._register_pre_backward_hook(output)
            state._training_state = TrainingState.IDLE
            output = state._cast_output_dtype(output)
            state._modules_to_run_forward.clear()
        return output

    @_dynamo_disable
    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        default_prefetch = len(self._states_to_backward_prefetch) == 0
        for fsdp_param_group in self._fsdp_param_groups:
            fsdp_param_group.pre_backward(default_prefetch)
        for fsdp_state in self._states_to_backward_prefetch:
            # Reverse so higher-indexed groups are prefetched first,
            # matching backward execution order (reverse of forward).
            for target_param_group in reversed(fsdp_state._fsdp_param_groups):
                FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
        return grad

    @_dynamo_disable
    def _root_post_backward_final_callback(self) -> None:
        logger.debug("FSDP::root_post_backward")
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            # Reset per-iteration state. With chunked loss, each standalone
            # per-chunk call repopulates an inner state's
            # ``_modules_to_run_forward`` (and claims ``iter_forward_root``)
            # but the group post-hook never empties it because not all
            # modules run; clear both here so the next iteration starts
            # fresh.
            self._state_ctx.iter_forward_root = None
            for state in self._state_ctx.all_states:
                state._modules_to_run_forward.clear()
                # Reverse so that the last param group (which gates the
                # reduce-scatter wait/clear) fires first, matching the
                # autograd backward order and preserving RS overlap for
                # per-param-mesh modules whose inputs lack gradients.
                for fsdp_param_group in reversed(state._fsdp_param_groups):
                    if fsdp_param_group._training_state != TrainingState.POST_BACKWARD:
                        # Run post-backward in case forward inputs did not require
                        # gradient so the autograd backward did not run
                        fsdp_param_group.post_backward()
                    fsdp_param_group._training_state = TrainingState.IDLE
                state._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    for fsdp_param_group in state._fsdp_param_groups:
                        fsdp_param_group.finalize_backward()
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
                # Catch the last module's RS states that no subsequent
                # module's group N-1 wait will clear.
                for rs_state in self._comm_ctx.reduce_scatter_states:
                    if rs_state.event is not None:
                        self._device_handle.current_stream().wait_event(rs_state.event)
                self._comm_ctx.reduce_scatter_states.clear()
            self._state_ctx.post_backward_final_callback_queued = False

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        # output is the forward return value — pass directly without wrapping
        # (unlike _register_post_backward_hook which wraps (args, kwargs))
        tensors = collect_grad_tensors(output)
        for t in tensors:
            t.register_hook(self._pre_backward)
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )

    def _reset_iter_state(self) -> None:
        # Iteration-wide recovery after a mid-forward or mid-backward
        # exception. Waits on in-flight collectives, reshards every param
        # group, and clears per-iteration trackers so the next forward can
        # start from a clean state. Any in-flight gradients (reduce-scatter
        # results, HSDP partial reduce outputs, grad-accum state) are
        # discarded: the failed iteration is treated as lost.
        if self._is_root is False:
            raise RuntimeError(
                "reset_iter_state must be called on the root FSDP module"
            )
        current_stream = self._device_handle.current_stream()
        if ag_state := self._comm_ctx.all_gather_state:
            if ag_state.event is not None:
                current_stream.wait_event(ag_state.event)
            self._comm_ctx.all_gather_state = None
        for rs_state in self._comm_ctx.reduce_scatter_states:
            if rs_state.event is not None:
                current_stream.wait_event(rs_state.event)
        self._comm_ctx.reduce_scatter_states.clear()
        self._comm_ctx.post_forward_order.clear()
        for state in self._state_ctx.all_states:
            state._modules_to_run_forward.clear()
            state._training_state = TrainingState.IDLE
            for fsdp_param_group in state._fsdp_param_groups:
                fsdp_param_group._reset_iter_state()
        self._state_ctx.iter_forward_root = None
        self._state_ctx.post_backward_final_callback_queued = False


def _get_module_fsdp_state(module: nn.Module) -> FSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
    modules_to_run: set[nn.Module],
    cast_output_dtype: Callable[[Any], Any],
) -> _MultiHandle:
    """
    Registers group forward pre and post-hooks. The pre-hook runs on every
    module pre-forward; downstream state gating ensures one-shot work
    (root setup, post_backward hook registration) fires once per group
    pass. The post-hook runs on every module's post-forward: on the
    partial path (group not yet complete) it applies only the
    ``mp_policy.output_dtype`` cast so standalone per-module callers
    observe the same output dtype semantics as the non-grouped case; on
    the last module it runs the full ``post_hook`` (which reshards,
    registers the pre-backward hook, and itself casts output_dtype). If
    a module never runs forward, the post-hook does not fire for it and
    ``_force_complete_incomplete_states`` finishes the group from the
    root's post-forward.
    """
    modules_set = set(modules)

    @_dynamo_disable
    @functools.wraps(pre_hook)
    def wrapped_pre_hook(*args: Any, **kwargs: Any):
        if len(modules_to_run) == 0:
            modules_to_run.update(modules_set)
        return pre_hook(*args, **kwargs)

    def get_wrapped_post_hook(module: nn.Module):
        @_dynamo_disable
        @functools.wraps(post_hook)
        def wrapped_post_hook(hook_module: nn.Module, input: Any, output: Any) -> Any:
            # Full path fires once, when this invocation completes the
            # group. Otherwise apply only the output_dtype cast so every
            # module in the group (including repeat invocations such as
            # per-chunk standalone head calls) produces output in the
            # mp_policy's output_dtype.
            if module in modules_to_run:
                modules_to_run.remove(module)
                if len(modules_to_run) == 0:
                    return post_hook(hook_module, input, output)
            return cast_output_dtype(output)

        return wrapped_post_hook

    pre_handles = [
        module.register_forward_pre_hook(
            wrapped_pre_hook, prepend=True, with_kwargs=True
        )
        for module in modules
    ]
    post_handles = [
        module.register_forward_hook(
            get_wrapped_post_hook(module), prepend=False, always_call=True
        )
        for module in modules
    ]
    return _MultiHandle(tuple(pre_handles + post_handles))
