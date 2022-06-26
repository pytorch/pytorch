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
    Configurations used in ParamExecOrderWrapPolicy for symbolic tracing of a model.
    tracer: An instance of torch.fx.Tracer that will be used to perform symbolic
        tracing. tracer is default to be torch.fx.Tracer(), but can also be instance
        of some child class of torch.fx.Tracer. For example, for hugginface transformer
        based models, one may want to use HFTracer:
        https://github.com/huggingface/transformers/blob/main/src/transformers/utils/fx.py#L636

    concrete_args: Concrete arguments that should not be treated as torch.fx.Proxy
        when tracing the forward function.
        concrete_args allows one to partially specialize the forward function,
        including removing control flow or data structures.
        concrete_args is also the parameter used in tracer.trace():
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.trace
    """
    tracer: torch.fx.Tracer = torch.fx.Tracer()
    concrete_args: Optional[Dict[str, Any]] = None


@dataclass
class _ExecutionUnitInfo:
    """
    Contains named_params in a module that will be executed together.
    """
    module: torch.nn.Module
    named_params: List[Tuple[str, torch.nn.Parameter]]


def patched_create_proxy(
    tracer: torch.fx.Tracer,
    create_proxy: Callable,
    kind: str,
    target: torch.fx.node.Target,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: Optional[str] = None,
    type_expr: Optional[Any] = None
):
    """
    Override of `Tracer.create_proxy` (see
    https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.create_proxy).
    Tracer.create_proxy is called in symbolic tracing for each leaf function/method/module.
    This override intercepts the recording of each of these operations and
    update tracer.module_execution_info_dict.
    """
    proxy = create_proxy(kind, target, args, kwargs, name, type_expr)

    assert hasattr(tracer, "current_module")
    assert hasattr(tracer, "module_execution_info_dict")
    assert hasattr(tracer, "param_exec_order")
    assert hasattr(tracer, "_traced_param_set")

    params_dict = dict(tracer.root.named_parameters())
    if kind in ["call_function", "call_method"]:
        if args is not None:
            named_params: List[Tuple[str, torch.nn.Parameter]] = []
            for arg in args:
                if isinstance(arg, torch.fx.Proxy) and arg.node.target in params_dict:
                    param = params_dict[arg.node.target]
                    named_params.append((arg.node.target, param))
                    if param not in tracer._traced_param_set:
                        tracer.param_exec_order.append(param)
                        tracer._traced_param_set.add(param)
            if named_params != []:
                tracer.module_execution_info_dict[tracer.current_module].append(
                    _ExecutionUnitInfo(module=tracer.current_module, named_params=named_params)
                )
    elif kind == "call_module":
        module = tracer.root.get_submodule(target)
        named_params_list = list(module.named_parameters())
        if named_params_list != []:
            tracer.module_execution_info_dict[module].append(
                _ExecutionUnitInfo(module=module, named_params=named_params_list)
            )
        for (_, p) in named_params_list:
            if p not in tracer._traced_param_set:
                tracer.param_exec_order.append(p)
                tracer._traced_param_set.add(p)
    return proxy


def patched_call_module(
    tracer: torch.fx.Tracer,
    call_module: Callable,
    module: torch.nn.Module,
    forward: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Any:
    """
    Override of Tracer.call_module (see
    https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
    Tracer.call_module is called in symbolic tracing for each non-root module.
    This override intercepts the recording of each operation and
    update tracer.module_forward_order and tracer.module_execution_info_dict.
    """
    assert hasattr(tracer, "module_forward_order")
    assert hasattr(tracer, "current_module")
    assert hasattr(tracer, "module_execution_info_dict")

    # Update tracer.module_forward_order
    tracer.module_forward_order.append(module)
    # Update tracer.module_execution_info_dict[tracer.current_module]
    named_params_list = list(module.named_parameters())
    if named_params_list != []:
        tracer.module_execution_info_dict[tracer.current_module].append(
            _ExecutionUnitInfo(
                module=module, named_params=list(module.named_parameters())
            )
        )
    # Stores away tracer.current_module for restoration later
    old_current_module = tracer.current_module
    tracer.current_module = module
    # Initialize tracer.module_execution_info_dict[module]. Note that if
    # the forward of module is called multiple times, this will record
    # the execution info of the last forward pass.
    tracer.module_execution_info_dict[module] = []
    # Delegates into the normal Tracer.call_module method
    output = call_module(module, forward, args, kwargs)
    # Restores tracer.current_module
    tracer.current_module = old_current_module
    return output


def _patch_tracer(tracer: Callable, root_module: torch.nn.Module) -> None:
    """
    Patches the input tracer so that during tracer.trace(), the forward order
    of all modules and the parameter execution information are recorded.
    root_module is the top-level module to be traced.

    The following attributes are added to tracer:
    tracer.current_module: record the module that is currently being traced.

    tracer.module_forward_order: a list of modules, where the ordering is based on
    when their forward function is called.

    tracer.param_exec_order: a list of parameters ordered based on their execution order.

    tracer._traced_param_set: a set containing all parameters that have been traced.

    tracer.module_execution_info_dict: a dict that maps each module to a list of
    _ExecutionUnitInfo.
    For a given module, each _ExecutionUnitInfo:
        1. either contains this module and part of its named_parameters that will be executed together,
        2. or contains one of its child modules and all of the child module's named_parameters.
    The list of _ExecutionUnitInfo is ordered based on the parameter execution order.
    """
    assert not hasattr(tracer, "current_module")
    assert not hasattr(tracer, "module_forward_order")
    assert not hasattr(tracer, "module_execution_info_dict")
    assert not hasattr(tracer, "param_exec_order")
    assert not hasattr(tracer, "_traced_param_set")

    tracer.current_module: torch.nn.Module = root_module
    tracer.module_forward_order: List[torch.nn.Module] = [root_module]

    tracer.module_execution_info_dict: Dict[torch.nn.Module, List[_ExecutionUnitInfo]] = dict()
    tracer.module_execution_info_dict[root_module] = []

    tracer.param_exec_order: List[torch.nn.Parameter] = []
    tracer._traced_param_set: Set[torch.nn.Parameter] = set()

    tracer.call_module = functools.partial(patched_call_module, tracer, tracer.call_module)
    tracer.create_proxy = functools.partial(patched_create_proxy, tracer, tracer.create_proxy)
