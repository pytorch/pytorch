import functools
import torch
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from torch.storage import T


__all__ = [
    "TracingConfig",
]


@dataclass
class TracingConfig:
    tracer: Callable = torch.fx.Tracer()
    concrete_args: Optional[Dict[str, Any]] = None

@dataclass
class _ExecutionUnitInfo:
    module: torch.nn.Module
    named_params: List[Tuple[str, torch.nn.Parameter]]

    def __repr__(self):
        names = [n for (n, p) in self.named_params]
        return f"_ExecutionUnitInfo: {self.module}, {names}"


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
    Override of `Tracer.create_proxy`. This override intercepts the recording
    of every operation and stores away the current traced module's qualified
    name in `node_to_originating_module`
    """
    proxy = create_proxy(kind, target, args, kwargs, name, type_expr)
    params_dict = dict(tracer.root.named_parameters())
    if kind in ["call_function", "call_method"]:
        if args is not None:
            named_params: List[Tuple[str, torch.nn.Parameter]] = []
            for arg in args:
                if isinstance(arg, torch.fx.Proxy) and arg.node.target in params_dict:
                    param = params_dict[arg.node.target]
                    named_params.append((arg.node.target, param))
                    if param not in tracer._param_set:
                        tracer.param_exec_order.append(param)
                        tracer._param_set.add(param)
            if named_params != []:
                tracer.module_children_dict[tracer.parent_module].append(
                    _ExecutionUnitInfo(module=tracer.parent_module, named_params=named_params)
                )
    elif kind == "call_module":
        module = tracer.root.get_submodule(target)
        named_params_list = list(module.named_parameters())
        if named_params_list != []:
            tracer.module_children_dict[module].append(
                _ExecutionUnitInfo(module=module, named_params=named_params_list)
            )
        for (_, p) in named_params_list:
            if p not in tracer._param_set:
                tracer.param_exec_order.append(p)
                tracer._param_set.add(p)
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
    This override:
    1) Stores away the qualified name of the caller for restoration later
    2) Installs the qualified name of the caller in `current_module_qualified_name`
       for retrieval by `create_proxy`
    3) Delegates into the normal Tracer.call_module method
    4) Restores the caller's qualified name into current_module_qualified_name
    """
    tracer.module_forward_order.append(module)
    old_parent_module = tracer.parent_module
    named_params_list = list(module.named_parameters())
    if named_params_list != []:
        tracer.module_children_dict[old_parent_module].append(
            _ExecutionUnitInfo(
                module=module, named_params=list(module.named_parameters())
            )
        )
    try:
        tracer.parent_module = module
        # TODO (linjianma): each module only has one forward, so that if a module
        # is called multiple times, this will always override.
        tracer.module_children_dict[module] = []
        return call_module(module, forward, args, kwargs)
    finally:
        tracer.parent_module = old_parent_module


def _patch_tracer(tracer: Callable, root_module: torch.nn.Module) -> None:
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.

        The current qualified name of the Module being traced. The top-level
        module is signified by empty string. This is updated when entering
        call_module and restored when exiting call_module
        current_module_qualified_name : str = ''
        A map from FX Node to the qualname of the Module from which it
        originated. This is recorded by `create_proxy` when recording an
        operation
        node_to_originating_module : Dict[torch.fx.Node, str] = {}
    """
    tracer.parent_module = root_module
    # TODO (linjianma): this is used to track the model execution order
    tracer.module_forward_order = [root_module]
    # TODO (linjianma): explanation
    tracer.module_children_dict = dict()
    # TODO (linjianma):
    tracer.param_exec_order = []
    tracer._param_set = set()
    # TODO (linjianma):
    tracer.module_children_dict[root_module] = []

    tracer.call_module = functools.partial(patched_call_module, tracer, tracer.call_module)
    tracer.create_proxy = functools.partial(patched_create_proxy, tracer, tracer.create_proxy)
