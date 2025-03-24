# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import logging
from collections.abc import Sequence
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import (
    _cast_fp_tensor,
    compiled_autograd_enabled,
    detect_compiled_autograd,
    TrainingState,
)
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup


if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam


logger = logging.getLogger("torch.distributed.fsdp.fully_shard")


class FSDPStateContext:
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        # All FSDP states in the root state's module tree
        self.all_states: list[FSDPState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: Optional[FSDPState] = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: Optional[torch.Event] = None


def disable_if_config_true(func):
    @functools.wraps(func)
    def fsdp_hook_wrapper(*args, **kwargs):
        if torch._dynamo.config.skip_fsdp_hooks:
            return torch._dynamo.disable(func, recursive=True)(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return fsdp_hook_wrapper


class FSDPState(_State):
    def __init__(self) -> None:
        super().__init__()
        self._fsdp_param_group: Optional[FSDPParamGroup] = None
        self._is_root: Optional[bool] = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._states_to_forward_prefetch: list[FSDPState] = []
        self._states_to_backward_prefetch: list[FSDPState] = []
        self._modules_to_run_forward: set[nn.Module] = set()

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
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
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _root_pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        if not compiled_autograd_enabled():
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
            if self._device.type in ["cuda", "hpu", "xpu", "mtia"]:
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
                f"FSDP requires a single root module but got {self._modules}"
            )
        detect_compiled_autograd()
        root_module = self._modules[0]
        visited_states: set[FSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        "FSDP state has already been lazily initialized for "
                        f"{module_name}\nFSDP requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            self._state_ctx.all_states.append(state)
            visited_states.add(state)
        if self._fsdp_param_group:
            # For the root, do not reshard after forward since for training,
            # the parameters would be freed and all-gathered immediately
            self._fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            if state._fsdp_param_group:
                state._fsdp_param_group.lazy_init()

    def _init_shared_state(self) -> None:
        self._comm_ctx.lazy_init(self._device)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        assert self._is_root
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, FSDPParam] = {}
        module_to_fsdp_param_group: dict[nn.Module, FSDPParamGroup] = {}
        for state in self._state_ctx.all_states:
            if fsdp_param_group := state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    module_to_fsdp_param_group[module] = fsdp_param_group
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_group:
                module_fqn = module_to_fsdp_param_group[module]._module_fqn
                if module_fqn is None:
                    module_to_fsdp_param_group[module]._module_fqn = module_name
                else:
                    assert isinstance(module_fqn, str), f"{module_fqn}"
                    module_fqn += f", {module_name}"
                    module_to_fsdp_param_group[module]._module_fqn = module_fqn

    @disable_if_config_true
    def _pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # the pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        self._training_state = TrainingState.FORWARD
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        for fsdp_state in self._states_to_forward_prefetch:
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
        return args, kwargs

    @disable_if_config_true
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # When composing with module-hook-based activation checkpointing, the
        # post-backward hook is responsible for the reshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        if self._fsdp_param_group:
            output = self._fsdp_param_group.post_forward(module, input, output)
        output = self._register_pre_backward_hook(output)
        self._training_state = TrainingState.IDLE
        if self._state_ctx.iter_forward_root is self:
            if all_gather_state := self._comm_ctx.all_gather_state:
                # Free the last all-gather result if needed; refer to
                # [Note: Overlapping all-gather copy-in and all-gather]
                self._comm_ctx.all_gather_copy_in_stream.wait_event(
                    all_gather_state.event
                )
                self._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
                self._comm_ctx.all_gather_state = None  # free the all-gather result
            self._state_ctx.iter_forward_root = None
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                output = tree_map(
                    functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                    output,
                )
        return output

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        if self._fsdp_param_group:
            default_prefetch = len(self._states_to_backward_prefetch) == 0
            self._fsdp_param_group.pre_backward(default_prefetch)
        for fsdp_state in self._states_to_backward_prefetch:
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
        return grad

    def _root_post_backward_final_callback(self) -> None:
        if not compiled_autograd_enabled():
            logger.debug("FSDP::root_post_backward")
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            for state in self._state_ctx.all_states:
                fsdp_param_group = state._fsdp_param_group
                if (
                    fsdp_param_group
                    and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
                ):
                    # Run post-backward in case forward inputs did not require
                    # gradient so the autograd backward did not run
                    fsdp_param_group.post_backward()
                state._training_state = TrainingState.IDLE
                if fsdp_param_group:
                    fsdp_param_group._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
                if self._comm_ctx.reduce_scatter_state is not None:
                    self._device_handle.current_stream().wait_event(
                        self._comm_ctx.reduce_scatter_state.event
                    )
                    self._comm_ctx.reduce_scatter_state = None
            self._state_ctx.post_backward_final_callback_queued = False

    def _finalize_backward(self) -> None:
        if self._modules_to_run_forward:
            msg = (
                f"{len(self._modules_to_run_forward)} of the {len(self._modules)} "
                f"modules passed to fully_shard did not run forward before backward, "
                "which is error-prone since FSDP post-forward/pre-backward logic "
                "will not run for these modules. We recommend passing only modules "
                "that run forward together. Modules that did not run forward: "
                f"{list(self._modules_to_run_forward)}"
            )
            warning_once(logger, msg, stacklevel=2)
            # Clear since we want the next forward to run
            self._modules_to_run_forward.clear()
        if self._fsdp_param_group:
            self._fsdp_param_group.finalize_backward()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        flat_outputs, _ = tree_flatten(output)
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )


def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
    modules_to_run: set[nn.Module],
):
    """
    Registers group forward pre and post-hooks. The pre-hook runs upon the
    first module pre-forward, and the post-hook runs upon the last. If at least
    one module does not run forward, then the post-hook does not run.
    """
    modules_set = set(modules)

    @disable_if_config_true
    @functools.wraps(pre_hook)
    def wrapped_pre_hook(*args: Any, **kwargs: Any):
        if len(modules_to_run) == 0:  # first to run
            modules_to_run.update(modules_set)
            return pre_hook(*args, **kwargs)

    @disable_if_config_true
    def get_wrapped_post_hook(module: nn.Module):
        @functools.wraps(post_hook)
        def wrapped_post_hook(*args: Any, **kwargs: Any):
            modules_to_run.discard(module)
            if len(modules_to_run) == 0:
                return post_hook(*args, **kwargs)

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
