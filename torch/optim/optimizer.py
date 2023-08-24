import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import ParamSpec, Self, TypeAlias

import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
    Indices,
    TensorListList,
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]

GlobalOptimizerPreHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], Optional[Tuple[Args, Kwargs]]]
GlobalOptimizerPostHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], None]

__all__ = ['Optimizer', 'register_optimizer_step_pre_hook', 'register_optimizer_step_post_hook']
_global_optimizer_pre_hooks: Dict[int, GlobalOptimizerPreHook] = OrderedDict()
_global_optimizer_post_hooks: Dict[int, GlobalOptimizerPostHook] = OrderedDict()
_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]

class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self) -> str:
        return "<required parameter>"

required = _RequiredParameter()


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        import torch._dynamo
        prev_grad = torch.is_grad_enabled()
        try:
            # Note on graph break below:
            # we need to graph break to ensure that aot respects the no_grad annotation.
            # This is important for perf because without this, functionalization will generate an epilogue
            # which updates the mutated parameters of the optimizer which is *not* visible to inductor, as a result,
            # inductor will allocate for every parameter in the model, which is horrible.
            # With this, aot correctly sees that this is an inference graph, and functionalization will generate
            # an epilogue which is appended to the graph, which *is* visible to inductor, as a result, inductor sees that
            # step is in place and is able to avoid the extra allocation.
            # In the future, we will either 1) continue to graph break on backward, so this graph break does not matter
            # or 2) have a fully fused forward and backward graph, which will have no_grad by default, and we can remove this
            # graph break to allow the fully fused fwd-bwd-optimizer graph to be compiled.
            # see https://github.com/pytorch/pytorch/issues/104053
            torch.set_grad_enabled(self.defaults['differentiable'])
            torch._dynamo.graph_break()
            ret = func(self, *args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret
    functools.update_wrapper(_use_grad, func)
    return _use_grad

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item()

def _stack_if_compiling(x):
    if not torch.jit.is_scripting() and is_compiling():
        return torch.stack(x)
    else:
        return x

def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)

# For any optimizer with a faster implementation, we attempt to default to the
# fastest + stablest whenever possible. For foreach, the requirements are to have
# native params all on CUDA. For fused, there's currently the additional requirement
# that the tensors' dtypes must be floating point. Neither alternative supports
# torch.jit.script nor differentiable, so we fall back to the single tensor
# implementation in those cases.
def _default_to_fused_or_foreach(params: List[torch.Tensor],
                                 differentiable: bool,
                                 use_fused: bool = False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None or (type(p) in _foreach_supported_types and
                      p.device.type in fused_supported_devices and
                      torch.is_floating_point(p)) for p in params
    )
    foreach = not fused and all(
        p is None or (type(p) in _foreach_supported_types and
                      p.device.type in foreach_supported_devices) for p in params
    )
    return fused, foreach


# Common doc strings among optimizers
_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)"""

_fused_doc = r"""fused (bool, optional): whether the fused implementation (CUDA only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation. Thus, if the user has not specified BOTH flags
              (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. For example, if the user specifies
              True for fused but nothing for foreach, we will run the fused implementation. If
              the user specifies False for foreach but nothing for fused (or False for fused but
              nothing for foreach), we will run the for-loop implementation. If the user specifies
              True for both foreach and fused, we will prioritize fused over foreach, as it is
              typically faster. We attempt to use the fastest, so the hierarchy goes fused ->
              foreach -> for-loop. HOWEVER, since the fused implementation is relatively new,
              we want to give it sufficient bake-in time, so we default to foreach and NOT
              fused when the user has not specified either flag."""

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)"""


def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    r"""Register a post hook common to all optimizers. The hook should have the following
    signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle

params_t: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

_P = ParamSpec("_P")
R = TypeVar("R")
T = TypeVar("T")


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    OptimizerPreHook: TypeAlias = Callable[[Self, Args, Kwargs], Optional[Tuple[Args, Kwargs]]]  # type: ignore[misc]
    OptimizerPostHook: TypeAlias = Callable[[Self, Args, Kwargs], None]  # type: ignore[misc]

    _optimizer_step_pre_hooks: Dict[int, OptimizerPreHook]
    _optimizer_step_post_hooks: Dict[int, OptimizerPostHook]
    _optimizer_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'
    _optimizer_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    _optimizer_load_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    _optimizer_load_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'


    @staticmethod
    def _verify_param_groups_align(groups, saved_groups) -> Dict[int, torch.Tensor]:
        """_verify_param_groups_align is a helper method that verifies the param groups of
        a saved checkpoint and the current optimizer are aligned and then returns an
        id map mapping param ID (integer) to a parameter.
        """

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        return dict(zip(chain.from_iterable(g['params'] for g in saved_groups),
                        chain.from_iterable(g['params'] for g in groups)))


    @staticmethod
    def _cast_state_to_match_params_hook(optimizer, state_dict) -> None:
        """The following hook is automatically registered to load_state_dict as a pre-hook. This processing
        used to live within load_state_dict, but, since the introduction of state_dict hooks, we've moved
        this to be its own hook to allow users flexibility to trigger pre-hook before OR after casting
        state to match the params' dtype and device.
        """

        # Validate the state_dict
        groups = optimizer.param_groups
        saved_groups = state_dict['param_groups']
        id_map = Optimizer._verify_param_groups_align(groups, saved_groups)

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                return Optimizer._process_value_according_to_param_policy(param, value, param_id, param_groups, key)
            elif isinstance(value, dict):
                return {k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k)
                        for k, v in value.items()}
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups)
                                   for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Cast state tensors to appropriate types in the state_dict
        for k in state_dict['state']:
            v = state_dict['state'][k]
            if k in id_map:
                param = id_map[k]
                state_dict['state'][k] = _cast(param, v, param_id=k, param_groups=saved_groups)


    def __init__(self, params: params_t, defaults: Dict[str, Any]) -> None:
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._load_state_dict_cast_handle = self.register_load_state_dict_pre_hook(
            Optimizer._cast_state_to_match_params_hook
        )
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

        self._patch_step_function()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(cast(dict, param_group))

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True

    def get_load_state_dict_cast_hook_handle(self) -> RemovableHandle:
        return self._load_state_dict_cast_handle

    def __getstate__(self) -> Dict[str, Any]:
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if '_optimizer_step_pre_hooks' not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if '_optimizer_step_post_hooks' not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        if '_optimizer_state_dict_pre_hooks' not in self.__dict__:
            self._optimizer_state_dict_pre_hooks = OrderedDict()
        if '_optimizer_state_dict_post_hooks' not in self.__dict__:
            self._optimizer_state_dict_post_hooks = OrderedDict()
        if '_optimizer_load_state_dict_pre_hooks' not in self.__dict__:
            self._optimizer_load_state_dict_pre_hooks = OrderedDict()
            self._load_state_dict_cast_handle = self.register_load_state_dict_pre_hook(
                Optimizer._cast_state_to_match_params_hook
            )
        if '_optimizer_load_state_dict_post_hooks' not in self.__dict__:
            self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault('differentiable', False)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string

    # Currently needed by Adam and AdamW
    def _cuda_graph_capture_health_check(self) -> None:
        # Note [torch.compile x capturable]
        # If we are compiling, we try to take the capturable path automatically by
        # setting the flag to True during tracing. Due to this, we skip all the checks
        # normally required for determining whether we can use CUDA graphs and
        # shunt the responsibility to torch.inductor. This saves time during tracing
        # since the checks are slow without sacrificing UX since inductor will warn
        # later if CUDA graphs cannot be enabled, e.g.,
        # https://github.com/pytorch/pytorch/blob/d3ba8901d8640eb16f88b2bfef9df7fa383d4b47/torch/_inductor/compile_fx.py#L390.
        # Thus, when compiling, inductor will determine if cudagraphs
        # can be enabled based on whether there is input mutation or CPU tensors.
        if not is_compiling() and torch.backends.cuda.is_built() and torch.cuda.is_available():
            capturing = torch.cuda.is_current_stream_capturing()

            if capturing and not all(group['capturable'] for group in self.param_groups):
                raise RuntimeError("Attempting CUDA graph capture of step() for an instance of " +
                                   self.__class__.__name__ +
                                   " but param_groups' capturable is False.")

            if (
                (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
                and all(group['capturable'] for group in self.param_groups)
                and (not capturing)
            ):
                warnings.warn(
                    "This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, "
                    "but step() is running without CUDA graph capture. If you never intend to graph-capture this "
                    "instance, capturable=True can impair performance, and you should set capturable=False.",
                    stacklevel=2,
                )
                self._warned_capturable_if_run_uncaptured = True

    def _optimizer_step_code(self) -> None:
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
        pass

    @staticmethod
    def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]:

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> R:
            self, *_ = args
            self = cast(Optimizer, self)
            profile_name = f"Optimizer.step#{self.__class__.__name__}.step"
            with torch.autograd.profiler.record_function(profile_name):
                # call optimizer step pre hooks
                for pre_hook in chain(_global_optimizer_pre_hooks.values(), self._optimizer_step_pre_hooks.values()):
                    result = pre_hook(self, args, kwargs)
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result  # type: ignore[assignment]
                        else:
                            raise RuntimeError(
                                f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
                            )

                out = func(*args, **kwargs)
                self._optimizer_step_code()

                # call optimizer step post hooks
                for post_hook in chain(self._optimizer_step_post_hooks.values(), _global_optimizer_post_hooks.values()):
                    post_hook(self, args, kwargs)

                return out

        return wrapper

    @staticmethod
    def _group_tensors_by_device_and_dtype(
        tensorlistlist: TensorListList,
        with_indices: bool = False,
    ) -> Union[
        Dict[Tuple[None, None], Tuple[TensorListList, Indices]],
        Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]],
    ]:
        """Groups a list of lists of tensors by device and dtype.
        Skips this step if we are compiling since this will occur during inductor lowering."""
        if is_compiling():
            return {(None, None): (tensorlistlist, list(range(len(tensorlistlist[0]))))}
        else:
            return _group_tensors_by_device_and_dtype(tensorlistlist, with_indices)

    def _patch_step_function(self) -> None:
        self._zero_grad_profile_name = f"Optimizer.zero_grad#{self.__class__.__name__}.zero_grad"
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)  # type: ignore[assignment]
            self.__class__.step.hooked = True  # type: ignore[attr-defined]

    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before
        optimizer step. It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_pre_hooks)
        self._optimizer_step_pre_hooks[handle.id] = hook
        return handle

    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.
        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_post_hooks)
        self._optimizer_step_post_hooks[handle.id] = hook
        return handle


    def register_state_dict_pre_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:
        r"""Register a state dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.
        The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
        The registered hook can be used to perform pre-processing before the ``state_dict``
        call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_pre_hooks)
        self._optimizer_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle


    def register_state_dict_post_hook(
        self,
        hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a state dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_post_hooks)
        self._optimizer_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_post_hooks.move_to_end(handle.id, last=False)
        return handle


    @torch._disable_dynamo
    def state_dict(self) -> StateDict:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                    }
                ]
            }

        """

        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)

        # Save order indices instead of Tensors
        param_mappings: Dict[int, int] = {}
        start_index = 0

        def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}

        state_dict = {
            'state': packed_state,
            'param_groups': param_groups,
        }

        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    @staticmethod
    def _process_value_according_to_param_policy(
        param: torch.Tensor,
        value: torch.Tensor,
        param_id: int,
        param_groups: List[Dict[Any, Any]],
        key: Hashable = None,
    ) -> torch.Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        assert param_groups is not None
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break

        if key == 'step':
            if capturable or fused:
                return value.to(dtype=torch.float32, device=param.device)
            else:
                return value
        else:
            if param.is_floating_point():
                return value.to(dtype=param.dtype, device=param.device)
            else:
                return value.to(device=param.device)


    def register_load_state_dict_pre_hook(
        self,
        hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        .. note::

            There is an automatically registered load_state_dict pre-hook which
            casts the state to match the params in dtype and device. Set ``prepend``
            to True in order to insert the your defined hook before this pre-processing,
            otherwise, leaving ``prepend`` as False will have your registered hook run
            after the casting has taken place.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_pre_hooks)
        self._optimizer_load_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle


    def register_load_state_dict_post_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:
        r"""Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_post_hooks)
        self._optimizer_load_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_post_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle


    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict['param_groups'])

        # Validate the state_dict
        groups = self.param_groups
        id_map = Optimizer._verify_param_groups_align(groups, saved_groups)

        # Copy state assigned to params
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]

        self.__setstate__({'state': state, 'param_groups': param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)


    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get('foreach', False) or self.defaults.get('fused', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: Optional[DefaultDict[torch.device, DefaultDict[torch.dtype, List[torch.Tensor]]]]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if (not foreach or p.grad.is_sparse):
                                p.grad.zero_()
                            else:
                                assert per_device_and_dtype_grads is not None
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                assert per_device_and_dtype_grads is not None
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    @torch._disable_dynamo
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not self.defaults.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(f"parameter group didn't specify a value of required optimization parameter {name}")
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set: Set[torch.Tensor] = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
