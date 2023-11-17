import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class TracingConfig:
    """
    This represents a symbolic tracing configuration.

    Args:
        tracer (torch.fx.Tracer): An instance of :class:`torch.fx.Tracer` to
            use for symbolic tracing. The default value is the native
            :class:`torch.fx.Tracer` constructed with default arguments.
            However, the user may want to pass a different value such as the
            ``HFTracer`` for models in the HuggingFace Transformers_ library.
            .. _Transformers: https://huggingface.co/docs/transformers/index
        concrete_args (Optional[Dict[str, Any]]): Concrete arguments that
            should not be treated as ``torch.fx.Proxy`` when tracing the
            module ``forward()``. Passing ``concrete_args`` allows partially
            specializing the forward, e.g. to remove control flow or data
            structures. This ``concrete_args`` here is the same argument used
            in :meth:`~torch.fx.Tracer.trace`.
    """

    tracer: torch.fx.Tracer = torch.fx.Tracer()
    concrete_args: Optional[Dict[str, Any]] = None


class _ParamUsageInfo(NamedTuple):
    """
    This is used for ``_ExecutionInfo.module_to_param_usage_infos`` to record
    execution information. The ``dict`` maps modules to a list of these
    ``_ParamUsageInfo`` instances, where each instance represents a group of
    parameters used together.

    Specifically, for each module key in the ``dict``, each instance of this
    class represents either:
    (1) the module and some sublist of its ``named_parameters()`` used
    together in execution (see ``_patched_create_proxy()``), or
    (2) a submodule and all of ``submodule.named_parameters()`` (see
    ``_patched_call_module()``).

    Type (1) corresponds to directly using parameters in ops without calling
    ``forward()``, and type (2) corresponds to calling ``forward()``. The
    mapped-to lists in the ``dict`` follow the execution order.
    """

    module: nn.Module
    named_params: List[Tuple[str, nn.Parameter]]


class _ExecutionInfo:
    """
    This represents the execution order information from the forward pass.

    Attributes:
        curr_module (nn.Module): Current module being traced.
        module_forward_order (List[nn.Module]): The modules in (pre-)forward
            order, i.e. the order in which their ``forward()`` methods are
            called. Each call to a module's ``forward()`` corresponds to one
            element in the list.
        module_to_param_usage_infos (Dict[nn.Module, List[_ParamUsageInfo]]):
            Maps a module to a list of module execution infos. See
            :class:`_ParamUsageInfo` for details.
        param_forward_order (List[nn.Parameter]): The parameters in forward
            execution order, where only a parameter's first participation is
            included.
        visited_params (Set[nn.Parameter]): The parameters visited so far
            during the trace. This is only used during tracing for fast
            membership check. Invariant: The parameters in
            ``param_forward_order`` are exactly those in ``visited_params``.
    """

    def __init__(self, root_module: nn.Module) -> None:
        self.curr_module: nn.Module = root_module
        self.module_forward_order: List[nn.Module] = [root_module]
        self.module_to_param_usage_infos: Dict[nn.Module, List[_ParamUsageInfo]] = {
            root_module: []
        }
        self.param_forward_order: List[nn.Parameter] = []
        self.visited_params: Set[nn.Parameter] = set()


class _ExecOrderTracer:
    def __init__(self) -> None:
        self.exec_info: Optional[_ExecutionInfo] = None

    @contextmanager
    def patch_tracer(self, tracer: torch.fx.Tracer, root_module: nn.Module):
        self.exec_info = _ExecutionInfo(root_module)
        orig_call_module = tracer.call_module
        orig_create_proxy = tracer.create_proxy
        tracer.call_module = functools.partial(
            self._patched_call_module, orig_call_module, self.exec_info
        )
        fqn_to_param = dict(root_module.named_parameters())
        tracer.create_proxy = functools.partial(
            self._patched_create_proxy,
            orig_create_proxy,
            self.exec_info,
            fqn_to_param,
        )
        try:
            yield
        finally:
            tracer.call_module = orig_call_module
            tracer.create_proxy = orig_create_proxy

    def _patched_call_module(
        self,
        call_module: Callable,
        exec_info: _ExecutionInfo,
        # Below are the expected arguments to `call_module()`
        module: nn.Module,
        forward: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Overrides ``call_module`` to save execution information to
        ``exec_info``. Note that ``call_module`` is called during symbolic
        tracing for each non-root module.

        Args:
            call_module (Callable): Original ``call_module`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            module (nn.Module): Module corresponding to this ``call_module``.
            forward (Callable): ``forward()`` method of ``module`` to be called
                for this ``call_module``.
            args (Tuple[Any, ...]): Positional arguments for ``forward``.
            kwargs (Dict[str, Any]): Keyword arguments for ``forward``.

        Returns:
            Same return value as ``call_module``.
        """
        exec_info.module_forward_order.append(module)
        named_params = list(module.named_parameters())
        curr_module = exec_info.curr_module
        if named_params:
            assert (
                curr_module in exec_info.module_to_param_usage_infos
            ), "The current module should have already been processed by a patched `call_module`"
            exec_info.module_to_param_usage_infos[exec_info.curr_module].append(
                _ParamUsageInfo(module, named_params)
            )
        prev_curr_module = curr_module
        exec_info.curr_module = module
        exec_info.module_to_param_usage_infos[module] = []
        output = call_module(module, forward, args, kwargs)
        exec_info.curr_module = prev_curr_module
        return output

    def _patched_create_proxy(
        self,
        create_proxy: Callable,
        exec_info: _ExecutionInfo,
        fqn_to_param: Dict[str, nn.Parameter],
        # Below are the expected arguments to `create_proxy()`
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Optional[Callable[[torch.fx.Node], torch.fx.Proxy]] = None,
    ) -> torch.fx.Proxy:
        """
        Overrides ``create_proxy`` to save execution information to
        ``exec_info``. Note that ``create_proxy`` is called during symbolic
        tracing for each leaf function/method/module.

        Args:
            create_proxy (Callable): Original ``create_proxy`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            fqn_to_param (Dict[str, nn.Parameter]): ``dict`` version of the
                root module's ``named_parameters()`` with FQN as key and
                parameter as value.
            kind (str): Kind of the target method ('call_function',
                'call_method', 'get_attr', 'call_module', 'placeholder', or
                'output'). See :class:`torch.fx.Graph` for details. This is
                passed to ``create_proxy``.
            target (torch.fx.node.Target): Contains the string name of the
                function/method/module. This is passed to ``create_proxy``.
            args (Tuple[Any, ...]): Positional arguments for the function/
                method/module. This is passed to ``create_proxy``.
            kwargs (Dict[str, Any]): Keyword arguments for the function/method/
                module. This is passed to ``create_proxy``
            name (Optional[str]): An optional string name for the ``Node``
                created in ``create_proxy``. This is passed to
                ``create_proxy``.
            type_expr (Optional[Any]): An optional type annotation representing
                the Python type that the output of the node has. This is passed
                to ``create_proxy``.
            proxy_factory_fn (Callable[[torch.fx.Node], torch.fx.Proxy]):
                An alternative proxy constructor used in ``create_proxy``. This
                is passed to ``create_proxy``.

        Returns:
            torch.fx.Proxy: Created ``Node`` wrapped in a ``Proxy`` object.
        """
        proxy = create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )
        curr_module = exec_info.curr_module
        if kind in ("call_function", "call_method"):
            if args is not None:
                named_params: List[Tuple[str, nn.Parameter]] = []
                for arg in args:
                    if (
                        isinstance(arg, torch.fx.Proxy)
                        and arg.node.target in fqn_to_param
                    ):
                        param = fqn_to_param[arg.node.target]
                        named_params.append((arg.node.target, param))
                        if param not in exec_info.visited_params:
                            exec_info.visited_params.add(param)
                            exec_info.param_forward_order.append(param)
                if named_params:
                    exec_info.module_to_param_usage_infos[curr_module].append(
                        _ParamUsageInfo(curr_module, named_params)
                    )
        elif kind == "call_module":
            named_params = list(curr_module.named_parameters())
            if named_params:
                exec_info.module_to_param_usage_infos[curr_module].append(
                    _ParamUsageInfo(curr_module, named_params)
                )
            for _, param in named_params:
                if param not in exec_info.visited_params:
                    exec_info.visited_params.add(param)
                    exec_info.param_forward_order.append(param)
        return proxy
