import functools

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook

from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _replace_module_state,
    _State,
)
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.hooks import RemovableHandle
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import _cast_fp_tensor, TrainingState
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup


class FSDPStateContext:
    """This has state shared across FSDP states."""

    def __init__(self):
        # All FSDP states in the root state's module tree
        self.all_states: List[FSDPState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward
        self.iter_forward_root: Optional[FSDPState] = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False


class FSDPState(_State):
    def __init__(self):
        super().__init__()
        self._fsdp_param_group: Optional[FSDPParamGroup] = None
        self._is_root: Optional[bool] = None  # root set during lazy init
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._pre_forward_hook_handle: Optional[RemovableHandle] = None
        self._post_forward_hook_handle: Optional[RemovableHandle] = None
        self._pre_backward_hook_handles: List[RemovableHandle] = []
        # Shared post-forward order for explicit backward prefetching
        self._post_forward_order: List[FSDPParamGroup] = []  # will cause ref cycles

    # Define a separate init since `__init__` is called in the contract
    def init(
        self, module: nn.Module, device: torch.device, mp_policy: MixedPrecisionPolicy
    ) -> None:
        _insert_module_state(module, self)
        self._modules: Tuple[nn.Module, ...] = (module,)
        self._device = device
        self._mp_policy = mp_policy
        self._pre_forward_hook_handle = module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _root_pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            current_stream = torch.cuda.current_stream()
            self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
            self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type == "cuda":
                with torch.profiler.record_function("FSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self) -> None:
        """
        Lazy initialization logically represents when all modules' parallelisms
        have finalized (e.g. FSDP has been applied to all desired modules).
        This means that we can determine root state. We identify the root by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        root_modules = self._modules
        visited_states: Set[FSDPState] = set()
        for root_module in root_modules:
            for module_name, module in root_module.named_modules():
                if (state := _get_module_fsdp_state(module)) is None:
                    continue
                if state in visited_states:
                    continue
                visited_states.add(state)
                if module is not root_module:
                    if state._is_root is not None:
                        raise RuntimeError(
                            "FSDP state has already been lazily initialized for "
                            f"{module_name}\nFSDP requires running forward through "
                            "the root module first"
                        )
                    state._is_root = False
                self._state_ctx.all_states.append(state)
                if state._fsdp_param_group:
                    state._fsdp_param_group.lazy_init()
        if self._fsdp_param_group:
            # For the root, do not reshard after forward since for training,
            # the parameters would be freed and all-gathered immediately
            self._fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()

    def _init_shared_state(self) -> None:
        self._comm_ctx.init()
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        assert self._is_root
        for root_module in self._modules:
            param_to_fsdp_param: Dict[nn.Parameter, FSDPParam] = {}
            module_to_fsdp_param_group: Dict[nn.Module, FSDPParamGroup] = {}
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
                    if (
                        fsdp_param_group := module_to_fsdp_param_group[module]
                    )._module_fqn:
                        fsdp_param_group._module_fqn += f", {module_name}"
                    else:
                        fsdp_param_group._module_fqn = module_name

    def _pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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
        return args, kwargs

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

    def _pre_backward(self, *unused: Any) -> None:
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        if self._fsdp_param_group:
            self._fsdp_param_group.pre_backward(*unused)

    def _root_post_backward_final_callback(self) -> None:
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            self._training_state = TrainingState.IDLE
            for state in self._state_ctx.all_states:
                state._training_state = TrainingState.IDLE
                if state._fsdp_param_group:
                    state._fsdp_param_group.finalize_backward()
            self._state_ctx.post_backward_final_callback_queued = False
            for handle in self._pre_backward_hook_handles:
                handle.remove()
            self._pre_backward_hook_handles.clear()
            self._comm_ctx.post_forward_order.clear()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output

        flat_outputs, _ = tree_flatten(output)
        tensors = tuple(t for t in flat_outputs if t.requires_grad)
        if tensors:
            handle = register_multi_grad_hook(tensors, self._pre_backward, mode="any")
            self._pre_backward_hook_handles.append(handle)
            if self._fsdp_param_group:
                self._fsdp_param_group.expected_backward_unshard_count += 1
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )

    @staticmethod
    def check_fusible(fsdp_states: Sequence["FSDPState"]) -> None:
        devices: Set[torch.device] = set()
        mp_policies: Set[MixedPrecisionPolicy] = set()
        fsdp_param_groups: List[FSDPParamGroup] = []
        for fsdp_state in fsdp_states:
            if len(fsdp_state._modules) > 1:
                raise NotImplementedError(
                    f"Fusing already-fused modules is not supported: {fsdp_state._modules}"
                )
            if fsdp_state._is_root is not None:
                raise NotImplementedError("Fusing after lazy init is not supported")
            devices.add(fsdp_state._device)
            mp_policies.add(fsdp_state._mp_policy)
            if fsdp_state._fsdp_param_group:
                fsdp_param_groups.append(fsdp_state._fsdp_param_group)
        prefix = "Cannot fuse with different "
        if len(devices) > 1:
            raise ValueError(prefix + f"devices: {devices}")
        if len(mp_policies) > 1:
            raise ValueError(prefix + f"mixed precision policies: {mp_policies}")
        FSDPParamGroup.check_fusible(fsdp_param_groups)

    @staticmethod
    def fuse(fsdp_states: Sequence["FSDPState"]) -> "FSDPState":
        FSDPState.check_fusible(fsdp_states)
        # Coalesce all parameter groups
        fsdp_param_groups = [
            state._fsdp_param_group
            for state in fsdp_states
            if state._fsdp_param_group is not None
        ]
        if fsdp_param_groups:
            new_fsdp_param_group = FSDPParamGroup.fuse(fsdp_param_groups)
        # Coalesce all modules to use the 1st module's state
        modules: List[nn.Module] = []
        for fsdp_state in fsdp_states:
            modules.extend(list(fsdp_state._modules))
        new_state = fsdp_states[0]
        if new_fsdp_param_group:
            new_state._fsdp_param_group = new_fsdp_param_group
        for module in modules[1:]:
            _replace_module_state(module, new_state)
        new_state._modules = tuple(modules)
        for fsdp_state in fsdp_states:
            if hook_handle := fsdp_state._pre_forward_hook_handle:
                hook_handle.remove()
            if hook_handle := fsdp_state._post_forward_hook_handle:
                hook_handle.remove()
        hook_handle = _register_group_forward_hooks(
            modules, new_state._pre_forward, new_state._post_forward
        )
        new_state._pre_forward_hook_handle = hook_handle
        new_state._post_forward_hook_handle = hook_handle
        # TODO: Add something to module repr to indicate fused.
        return new_state


def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None


class MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]

    def __init__(self, handles: Tuple[RemovableHandle, ...]):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def __getstate__(self):
        return self.handles

    def __setstate__(self, state):
        self.handles = state


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
):
    """
    Registers group forward pre and post-hooks. The pre-hook runs upon the
    first module pre-forward, and the post-hook runs upon the last. If at least
    one module does not run forward, then the post-hook does not run.
    """
    modules_set = set(modules)
    modules_to_run: Set[nn.Module] = set()

    @functools.wraps(pre_hook)
    def wrapped_pre_hook(*args: Any, **kwargs: Any):
        nonlocal modules_to_run
        if len(modules_to_run) == 0:  # first to run
            modules_to_run.update(modules_set)
            return pre_hook(*args, **kwargs)

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
            get_wrapped_post_hook(module), prepend=False, with_kwargs=False
        )
        for module in modules
    ]
    return MultiHandle(tuple(pre_handles + post_handles))
