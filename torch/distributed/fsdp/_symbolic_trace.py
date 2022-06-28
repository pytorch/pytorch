import functools
import torch
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


__all__ = ["TracingConfig"]


@dataclass
class TracingConfig:
    """
    Configurations used in ``ParamExecOrderWrapPolicy`` for symbolic tracing of a model.
    tracer: An instance of ``torch.fx.Tracer`` that will be used to perform symbolic
        tracing. ``tracer`` is default to be ``torch.fx.Tracer()``, but can also be instance
        of some child class of ``torch.fx.Tracer``. For example, for hugginface transformer
        based models, one may want to use ``HFTracer``:
        https://github.com/huggingface/transformers/blob/6dd00f6bd49141ef4f26ed1d1c555bc0fe109ea8/src/transformers/utils/fx.py#L636

    concrete_args: Concrete arguments that should not be treated as ``torch.fx.Proxy``
        when tracing the forward function.
        ``concrete_args`` allows one to partially specialize the forward function,
        including removing control flow or data structures.
        ``concrete_args`` is also the parameter used in ``tracer.trace()``:
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.trace
    """
    tracer: torch.fx.Tracer = torch.fx.Tracer()
    concrete_args: Optional[Dict[str, Any]] = None


class _ExecutionInfo:
    """
    Contains the execution order information in the model forward pass.

    Attributes:
        current_module: record the module that is currently being traced.

        module_forward_order: a list of modules, where the ordering is based on
            when their forward function is called.

        param_exec_order: a list of parameters ordered based on their execution order.

        _traced_param_set: a set containing all parameters that have been traced.

        module_execution_info_dict: a dict that maps each module to a list of
            tuple containing a module and a list of named parameters.
            For a given module, each tuple:
            1. either contains this module and part of its ``named_parameters`` that will be executed together,
            2. or contains one of its child modules and all of the child module's ``named_parameters``.
            The list of tuple is ordered based on the parameter execution order.
    """
    def __init__(self, root_module: torch.nn.Module) -> None:
        self.current_module = root_module
        self.module_forward_order: List[torch.nn.Module] = [root_module]

        named_params_type = List[Tuple[str, torch.nn.Parameter]]
        self.module_execution_info_dict: Dict[
            torch.nn.Module,
            List[Tuple[torch.nn.Module, named_params_type]]
        ] = dict()
        self.module_execution_info_dict[root_module] = []

        self.param_exec_order: List[torch.nn.Parameter] = []
        self._traced_param_set: Set[torch.nn.Parameter] = set()


def _patched_create_proxy(
    create_proxy: Callable,
    execution_info: _ExecutionInfo,
    params_dict: Dict[str, torch.nn.Parameter],
    kind: str,
    target: torch.fx.node.Target,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: Optional[str] = None,
    type_expr: Optional[Any] = None
):
    """
    Override of ``Tracer.create_proxy`` (see
    https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.create_proxy).
    ``Tracer.create_proxy`` is called in symbolic tracing for each leaf function/method/module.
    This override intercepts the recording of each of these operations and
    update ``execution_info.module_execution_info_dict``.

    Args:
        create_proxy (Callable):
            The ``create_proxy`` function to be patched.
        execution_info (_ExecutionInfo):
            Used to repord the execution information.
        params_dict (Dict[str, torch.nn.Parameter]):
            A dict that maps each parameter name to the parameter
        kind, target, args, kwargs, name, type_expr: inputs to the ``create_proxy`` function.
    """
    proxy = create_proxy(kind, target, args, kwargs, name, type_expr)

    if kind in ["call_function", "call_method"]:
        if args is not None:
            named_params: List[Tuple[str, torch.nn.Parameter]] = []
            for arg in args:
                if isinstance(arg, torch.fx.Proxy) and arg.node.target in params_dict:
                    param = params_dict[arg.node.target]
                    named_params.append((arg.node.target, param))
                    if param not in execution_info._traced_param_set:
                        execution_info.param_exec_order.append(param)
                        execution_info._traced_param_set.add(param)
            if named_params:
                execution_info.module_execution_info_dict[execution_info.current_module].append(
                    (execution_info.current_module, named_params)
                )
    elif kind == "call_module":
        module = execution_info.current_module
        named_params_list = list(module.named_parameters())
        if named_params_list != []:
            execution_info.module_execution_info_dict[module].append(
                (module, named_params_list)
            )
        for (_, p) in named_params_list:
            if p not in execution_info._traced_param_set:
                execution_info.param_exec_order.append(p)
                execution_info._traced_param_set.add(p)
    return proxy


def _patched_call_module(
    call_module: Callable,
    execution_info: _ExecutionInfo,
    module: torch.nn.Module,
    forward: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Any:
    """
    Override of ``Tracer.call_module`` (see
    https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
    ``Tracer.call_module`` is called in symbolic tracing for each non-root module.
    This override intercepts the recording of each operation and
    update ``execution_info.module_forward_order`` and ``execution_info.module_execution_info_dict``.

    Args:
        call_module (Callable):
            The ``call_module`` function to be patched.
        execution_info (_ExecutionInfo):
            Used to repord the execution information.
        module, forward, args, kwargs: inputs to the ``call_module`` function.
    """
    execution_info.module_forward_order.append(module)
    named_params_list = list(module.named_parameters())
    if named_params_list != []:
        execution_info.module_execution_info_dict[execution_info.current_module].append(
            (module, list(module.named_parameters()))
        )
    # Stores away current_module for restoration later
    old_current_module = execution_info.current_module
    execution_info.current_module = module
    # Initialize execution_info.module_execution_info_dict[module]. Note that if
    # the forward of module is called multiple times, this will record
    # the execution info of the last forward pass.
    execution_info.module_execution_info_dict[module] = []
    # Delegates into the normal Tracer.call_module method
    output = call_module(module, forward, args, kwargs)
    # Restores current_module
    execution_info.current_module = old_current_module
    return output


def _patch_tracer(
    tracer: torch.fx.Tracer,
    root_module: torch.nn.Module
) -> _ExecutionInfo:
    """
    Patches the input tracer so that during ``tracer.trace()``, the forward order
    of all modules and the parameter execution information are recorded.
    ``root_module`` is the top-level module to be traced and should not contain
    any FSDP modules.
    """
    from .fully_sharded_data_parallel import FullyShardedDataParallel
    for module in root_module.modules():
        assert (
            not isinstance(module, FullyShardedDataParallel)
        ), "The input root_module of _patch_tracer should not contain FSDP modules"
    execution_info = _ExecutionInfo(root_module)
    tracer.call_module = functools.partial(
        _patched_call_module,
        tracer.call_module,
        execution_info
    )
    params_dict = dict(root_module.named_parameters())
    tracer.create_proxy = functools.partial(
        _patched_create_proxy,
        tracer.create_proxy,
        execution_info,
        params_dict
    )
    return execution_info
