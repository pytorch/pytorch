import contextlib
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch


__all__ = ["TracingConfig"]


@dataclass
class TracingConfig:
    """
    Configurations used in ``ParamExecOrderWrapPolicy`` for symbolic tracing of
    a model.

    Args:
        tracer (torch.fx.Tracer): An instance of ``torch.fx.Tracer`` that will
            be used to perform symbolic tracing. ``tracer`` is default to be
            ``torch.fx.Tracer()``, but can also be instance of some child class
            of ``torch.fx.Tracer``. For example, one may want to use
            ``HFTracer`` for models in Transformers: .. _Transformers:
            https://huggingface.co/docs/transformers/index
        concrete_args (Optional[Dict[str, Any]]): Concrete arguments that should
            not be treated as ``torch.fx.Proxy`` when tracing the forward
            function. ``concrete_args`` allows one to partially specialize the
            forward function, including removing control flow or data
            structures. ``concrete_args`` is also the argument used in
            :meth:`~torch.fx.Tracer.trace`.
    """

    tracer: torch.fx.Tracer = torch.fx.Tracer()
    concrete_args: Optional[Dict[str, Any]] = None


@dataclass
class _ExecutionInfo:
    """
    Contains the execution order information in the model forward pass.

    Attributes:
        current_module: record the module that is currently being traced.

        module_forward_order: a list of modules, where the ordering is based on
            when their forward function is called. ``module_forward_order``
            includes the info of how many times a module is called + used to
            check the forward order in different iterations.

        param_exec_order: a list of parameters ordered based on their execution
        order.

        module_to_execution_infos: a dict that maps each module to a list of
            tuples each containing a module and a list of named parameters.
            ``module_execution_info_dict`` is used as the parameter execution
            order info. For a given module, each tuple: 1. either contains this
            module and part of its ``named_parameters`` that will be executed
            together, 2. or contains one of its child modules and all of the
            child module's ``named_parameters``. The list of tuples is ordered
            based on the parameter execution order.
    """

    current_module: torch.nn.Module
    module_forward_order: List[torch.nn.Module]
    module_to_execution_infos: Dict[
        torch.nn.Module,
        List[Tuple[torch.nn.Module, List[Tuple[str, torch.nn.Parameter]]]],
    ]
    param_exec_order: List[torch.nn.Parameter] = field(default_factory=list)


def _init_execution_info(root_module: torch.nn.Module) -> _ExecutionInfo:
    """
    Create an instance of _ExecutionInfo with initialization based on
    ``root_module``.

    Args:
        root_module (torch.nn.Module): the module to get the execution
        information via ``tracer.trace()`` inside ``_patch_tracer``.
    """
    return _ExecutionInfo(
        current_module=root_module,
        module_forward_order=[root_module],
        module_to_execution_infos={root_module: []},
    )


def _patched_create_proxy(
    create_proxy: Callable,
    execution_info: _ExecutionInfo,
    prefixed_param_name_to_param: Dict[str, torch.nn.Parameter],
    kind: str,
    target: torch.fx.node.Target,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    name: Optional[str] = None,
    type_expr: Optional[Any] = None,
    proxy_factory_fn: Callable[[torch.fx.Node], torch.fx.Proxy] = None,
) -> torch.fx.Proxy:
    """
    Override of :meth:`~torch.fx.Tracer.create_proxy`. ``Tracer.create_proxy``
    is called in symbolic tracing for each leaf function/method/module. This
    override intercepts the recording of each of these operations to update
    ``execution_info.module_to_execution_infos``.

    Args:
        create_proxy (Callable):
            The ``create_proxy`` function to be patched.
        execution_info (_ExecutionInfo):
            Used to record the execution information.
        prefixed_param_name_to_param (Dict[str, torch.nn.Parameter]):
            A dict that maps each prefixed parameter name to the parameter.
        kind (str):
            The type of the target method. One of 'call_function',
            'call_method', 'get_attr', 'call_module', 'placeholder', or
            'output'. The semantics of these opcodes are described in the
            ``torch.fx.Graph`` docstring. This is the input to ``create_proxy``.
        target (torch.fx.node.Target):
            Contains the string name of the method. This is the input to
            ``create_proxy``.
        args (Tuple[Any, ...]):
            Arguments of the method. This is the input to ``create_proxy``.
        kwargs (Dict[str, Any]):
            Keyword arguments of the method. This is the input to
            ``create_proxy``.
        name (Optional[str]):
            An optional string name for the ``Node`` created in
            ``create_proxy``. This is the input to ``create_proxy``.
        type_expr (Optional[Any]):
            An optional type annotation representing the Python type the output
            of a node will have. This is the input to ``create_proxy``.
        proxy_factory_fn (Callable[[torch.fx.Node], torch.fx.Proxy]):
            An alternative proxy constructor used in ``create_proxy``. This is
            the input to ``create_proxy``.
    """
    proxy = create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

    module = execution_info.current_module
    if kind in ["call_function", "call_method"]:
        if args is not None:
            named_params: List[Tuple[str, torch.nn.Parameter]] = []
            for arg in args:
                if isinstance(arg, torch.fx.Proxy) and arg.node.target in prefixed_param_name_to_param:
                    param = prefixed_param_name_to_param[arg.node.target]
                    named_params.append((arg.node.target, param))
                    if param not in set(execution_info.param_exec_order):
                        execution_info.param_exec_order.append(param)
            if named_params:
                execution_info.module_to_execution_infos[module].append((module, named_params))
    elif kind == "call_module":
        named_params = list(module.named_parameters())
        if named_params:
            execution_info.module_to_execution_infos[module].append(
                (module, named_params)
            )
        for (_, p) in named_params:
            if p not in set(execution_info.param_exec_order):
                execution_info.param_exec_order.append(p)
    return proxy


def _patched_call_module(
    call_module: Callable,
    execution_info: _ExecutionInfo,
    module: torch.nn.Module,
    forward: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """
    Override of :meth:`~torch.fx.Tracer.call_module`. ``Tracer.call_module`` is
    called in symbolic tracing for each non-root module. This override
    intercepts the recording of each operation to update
    ``execution_info.module_forward_order`` and
    ``execution_info.module_to_execution_infos``.

    Args:
        call_module (Callable):
            The ``call_module`` function to be patched.
        execution_info (_ExecutionInfo):
            Used to repord the execution information.
        module (torch.nn.Module):
            The module for which a call is being emitted.
        forward (Callable[..., Any]):
            The ``forward()`` method of the ``torch.nn.Module`` to be invoked.
        args (Tuple[Any, ...]):
            ``args`` of the module callsite.
        kwargs (Dict[str, Any]):
            ``kwargs`` of the module callsite.
    """
    execution_info.module_forward_order.append(module)
    named_params = list(module.named_parameters())
    if named_params:
        execution_info.module_to_execution_infos[execution_info.current_module].append(
            (module, list(module.named_parameters()))
        )
    # Stores away current_module for restoration later
    prev_current_module = execution_info.current_module
    execution_info.current_module = module
    # Note that if the forward of module is called multiple times, this will record
    # the execution info of the last forward pass.
    execution_info.module_to_execution_infos[module] = []
    output = call_module(module, forward, args, kwargs)
    execution_info.current_module = prev_current_module
    return output


@contextlib.contextmanager
def _patch_tracer(
    tracer: torch.fx.Tracer,
    root_module: torch.nn.Module,
    execution_info: _ExecutionInfo,
) -> Generator:
    """
    Within the context manager, patches the input tracer so that during
    ``tracer.trace()``, the forward order of all modules and the parameter
    execution information are recorded. The patches of the input tracer will be
    removed after the context manager exits.

    Args:
        tracer (torch.fx.Tracer): the input ``tracer`` whose member functions
            will be patched within the context manager.
        root_module (torch.nn.Module): the top-level module to be traced
            and should not contain any FSDP modules.
        execution_info (_ExecutionInfo): used to record the execution order
            information when performing ``tracer.trace()`` within the context
            manager.
    """
    original_call_module = tracer.call_module
    original_create_proxy = tracer.create_proxy

    tracer.call_module = functools.partial(
        _patched_call_module, original_call_module, execution_info
    )
    prefixed_param_name_to_param = dict(root_module.named_parameters())
    tracer.create_proxy = functools.partial(
        _patched_create_proxy, original_create_proxy, execution_info, prefixed_param_name_to_param
    )
    try:
        yield
    finally:
        tracer.call_module = original_call_module
        tracer.create_proxy = original_create_proxy
