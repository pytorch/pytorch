import functools
import logging
import weakref

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed.utils import _apply_to_tensors, _to_kwargs
from torch.utils._pytree import tree_map
from torch.utils.hooks import RemovableHandle

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import (
    _cast_floating_point_tensor,
    FSDP_ENABLE_LOGGING,
    TrainingState,
)
from ._fsdp_param import FSDPParam
from ._fsdp_param_group import FSDPParamGroup


log = logging.getLogger(__name__)
if not FSDP_ENABLE_LOGGING:
    log.disabled = True


class FSDPState(_State):
    """
    This class holds the stateful data needed by a single data parallel
    communication group (e.g. for FSDP). This may include a single parameter
    group (in ``_fsdp_param_group``) and some orchestration state like streams,
    hook handles, execution order, etc.

    NOTE: An instance of this class can manage *no* parameter group; this
    typically is only true of the root module's state.

    NOTE: With respect to managing state, this is analogous to the
    ``FullyShardedDataParallel`` module wrapper.
    """

    _default_stream: torch.cuda.Stream
    _all_gather_copy_in_stream: torch.cuda.Stream
    _all_gather_stream: torch.cuda.Stream
    _reduce_scatter_stream: torch.cuda.Stream

    def __init__(self):
        self._module: nn.Module = nn.ModuleList()  # for typing
        self._fsdp_param_group: Optional[FSDPParamGroup] = None
        self._mp_policy = MixedPrecisionPolicy()
        self._device = torch.device("cpu")
        self._is_root: Optional[bool] = None
        self._training_state: TrainingState = TrainingState.IDLE
        self._extra_repr_str: str = ""
        self._post_forward_order: List[weakref.ref[FSDPParamGroup]] = []
        self._pre_forward_hook_handle: Optional[RemovableHandle] = None
        self._post_forward_hook_handle: Optional[RemovableHandle] = None

        # Attributes only used on the root state:
        self._root_post_backward_final_callback_queued: Optional[bool] = None
        self._all_state_refs: List[weakref.ReferenceType[FSDPState]] = []
        self._wait_for_grad_sync = True

    def _root_pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ):
        self._lazy_init()
        if not self._is_root:
            return args, kwargs
        log.info("root pre-forward")
        with torch.profiler.record_function("FSDP::root_pre_forward"):
            self._training_state = TrainingState.FORWARD
            # Wait for optimizer computation before issuing all-gathers
            current_stream = torch.cuda.current_stream()
            self._all_gather_copy_in_stream.wait_stream(current_stream)
            self._all_gather_stream.wait_stream(current_stream)
            if self._device.type == "cuda":
                with torch.profiler.record_function("FSDP::inputs_to_gpu"):
                    # Assume that the current device is the correct device to which
                    # to copy inputs
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )
                args = args_tuple[0]
                kwargs = kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self):
        if self._is_root is not None:
            return  # no-op: already initialized
        log.info("_lazy_init")
        self._is_root = True
        root_module = self._module
        # Each module owns the reference to the state object
        for module in root_module.modules():
            state = _get_module_fsdp_state(module)
            if state is not None:
                if module is not root_module:
                    state._is_root = False
                self._all_state_refs.append(weakref.ref(state))
        if self._fsdp_param_group:
            # For the root, do not reshard after forward since in the typical
            # case, the parameters would be freed and all-gathered immediately
            self._fsdp_param_group.post_forward_mesh_info = None
        self._init_fqns()
        self._init_shared_state()

    def _init_shared_state(self) -> None:
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation
        high_priority = -1
        self._default_stream = torch.cuda.current_stream()
        self._all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
        self._all_gather_stream = torch.cuda.Stream(priority=high_priority)
        self._reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
        for state_ref in self._all_state_refs:
            state = state_ref()
            assert state is not None, "FSDPState deallocated"
            if state._fsdp_param_group:
                state._fsdp_param_group.default_stream = self._default_stream
                state._fsdp_param_group.all_gather_copy_in_stream = (
                    self._all_gather_copy_in_stream
                )
                state._fsdp_param_group.all_gather_stream = self._all_gather_stream
                state._fsdp_param_group.reduce_scatter_stream = (
                    self._reduce_scatter_stream
                )
                state._fsdp_param_group.post_forward_order = self._post_forward_order

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        if not self._is_root:
            return
        root_module = self._module
        param_to_fsdp_param: Dict[nn.Parameter, FSDPParam] = {}
        module_to_fsdp_param_group: Dict[nn.Module, FSDPParamGroup] = {}
        for state_ref in self._all_state_refs:
            state = state_ref()
            assert state is not None, "FSDPState deallocated"
            if state._fsdp_param_group:
                for fsdp_param in state._fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                module_to_fsdp_param_group[
                    state._fsdp_param_group.module
                ] = state._fsdp_param_group
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_group:
                module_to_fsdp_param_group[module]._module_fqn = module_name

    def _pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # the pre-backward hook is responsible for the unshard
        if self._in_pre_backward:
            return args, kwargs
        self._training_state = TrainingState.FORWARD
        self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_floating_point_tensor, self._mp_policy.param_dtype
                )
                args = tree_map(cast_fn, args)
                kwargs = tree_map(cast_fn, kwargs)
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        return args, kwargs

    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # When composing with module-hook-based activation checkpointing, the
        # post-backward hook is responsible for the reshard
        if self._in_pre_backward:
            return output
        if self._fsdp_param_group:
            output = self._fsdp_param_group.post_forward(module, input, output)
        self._training_state = TrainingState.IDLE
        output = self._register_pre_backward_hook(output)
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                output = tree_map(
                    functools.partial(
                        _cast_floating_point_tensor, self._mp_policy.output_dtype
                    ),
                    output,
                )
        return output

    def _pre_backward(self, *unused: Tuple[Any, ...]) -> None:
        self._training_state = TrainingState.PRE_BACKWARD
        if self._is_root and not self._root_post_backward_final_callback_queued:
            log.info("root pre-backward hook")
            self._register_root_post_backward_final_callback()
        if self._fsdp_param_group:
            self._fsdp_param_group._pre_backward(*unused)

    def _root_post_backward_final_callback(self) -> None:
        self._training_state = TrainingState.IDLE
        if not self._is_root:
            return
        log.info("root post-backward final callback")
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            for state_ref in self._all_state_refs:
                state = state_ref()
                assert state is not None, "FSDPState deallocated"
                state._training_state = TrainingState.IDLE
                if state._fsdp_param_group:
                    state._fsdp_param_group.finalize_backward()
            if not self._wait_for_grad_sync:
                return
            self._root_post_backward_final_callback_queued = False
            self._post_forward_order.clear()

    # Hook Registration #
    def _register_pre_forward_hook(self) -> None:
        if self._pre_forward_hook_handle is not None:
            self._pre_forward_hook_handle.remove()
        self._pre_forward_hook_handle = self._module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )

    def _register_post_forward_hook(self) -> None:
        if self._post_forward_hook_handle is not None:
            self._post_forward_hook_handle.remove()
        self._post_forward_hook_handle = self._module.register_forward_hook(
            self._post_forward, prepend=False
        )

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output

        def _register_hook(tensor: torch.Tensor):
            if tensor.requires_grad:
                tensor.register_hook(self._pre_backward)
                if self._fsdp_param_group:
                    self._fsdp_param_group.needs_pre_backward_unshard = True
            return tensor

        return _apply_to_tensors(_register_hook, output)

    def _register_root_post_backward_final_callback(self):
        if self._root_post_backward_final_callback_queued:
            return
        self._root_post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )

    # Utilities #
    def _set_reduce_scatter_grads(self, reduce_scatter_grads: bool) -> None:
        if self._fsdp_param_group:
            self._fsdp_param_group.set_reduce_scatter_grads(reduce_scatter_grads)

    def _set_all_reduce_grads(self, all_reduce_grads: bool) -> None:
        if self._fsdp_param_group:
            self._fsdp_param_group.set_all_reduce_grads(all_reduce_grads)

    def _set_wait_for_gradient_sync(self, wait_for_grad_sync: bool) -> None:
        self._wait_for_grad_sync = wait_for_grad_sync
        if self._fsdp_param_group:
            self._fsdp_param_group.set_wait_for_grad_sync(wait_for_grad_sync)

    @property
    def _in_pre_backward(self) -> bool:
        return self._training_state == TrainingState.PRE_BACKWARD


def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None
