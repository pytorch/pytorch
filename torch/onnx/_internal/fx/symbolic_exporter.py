from __future__ import annotations

import functools

import inspect

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import onnx

import torch
import torch.fx

from torch.onnx import _constants
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import exporter

# Functions directly wrapped to produce torch.fx.Proxy so that symbolic
# data can flow through those functions. Python functions (e.g., `torch.arange`)
# not defined by pybind11 in C++ do not go though Python dispatcher, so
# they are not automatically patched by FX's Python dispatcher.
# The list below means `torch.arange`, `torch.tensor`, and so on will be
# patched.
_TORCH_METHODS_TO_PATCH: Tuple[str, ...] = (
    "arange",
    "tensor",
    "finfo",
    "full",
    "empty",
)


class ModuleExpansionTracer(torch.fx._symbolic_trace.Tracer):
    """Tracer to create ONNX-exporting friendly FX graph.

    This tracer traces models into operators. That is,
    the traced graph mostly contains call_function nodes and
    has no call_module nodes. The call_module nodes
    are problematic to the use of make_fx(...) in ONNX
    exporter.
    """

    @_beartype.beartype
    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        # This returns False so that all sub-modules are considered as not leaves
        # and therefore expanded into operators in
        # torch.fx._symbolic_trace.Tracer.call_module.
        return False

    @_beartype.beartype
    def to_bool(self, obj: "torch.fx.Proxy") -> bool:
        # FIXME: This is a hack to tracing through if-else Python blocks.
        # It may generate incorrect ONNX graphs if the if-else block
        return False


@_beartype.beartype
def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    """
    This function move all placeholder nodes to the front of the graph node list.
    In torch.fx.Graph, placeholder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    graph = graph_module.graph
    placeholders = []
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)


@_beartype.beartype
def _replace_get_attr_with_placeholder(
    graph_module: torch.fx.GraphModule,
) -> Tuple[torch.Tensor, ...]:
    """
    Replace get_attr with placeholder.
    The parameters and buffers accessed by the original get_attr are returned;
    they are useful when creating random inputs for the modified graph_module.
    """
    graph = graph_module.graph
    replaced_attrs: List[torch.Tensor] = []
    for node in graph.nodes:
        if node.op == "get_attr":
            replaced_attr: Optional[torch.Tensor] = None
            # get_attr could retrieve either parameter or buffer, so
            # we need to try both.
            try:
                replaced_attr = graph_module.get_parameter(node.target)
            except AttributeError:
                # It's possible that model author use buffer instead of
                # parameter to store trainable weights. In this case,
                # 1. get_parameter will throw something like
                #    AttributeError: `bias` is not an nn.Parameter.
                # 2. get_buffer should work.
                replaced_attr = graph_module.get_buffer(node.target)

            # Reassign op type so that get_attr node becomes placeholder node.
            node.op = "placeholder"
            # The target name in placeholder must be a valid Python identifier.
            # Thus, we replace, e.g., "module.submodule.weight" with
            # "module_submodule_weight".
            node.target = node.target.replace(".", "_")
            # Default value is None. This is needed as long as the "graph_module"
            # has optional inputs. Assume the original forward signature is
            #  def forward(self, x, y=None)
            # and the replaced get_attr node has target "z". Then, the modified
            # signature should be
            #  def forward(self, x, y=None, z=None)
            # Without the following line, the signature will be
            #  def forward(self, x, y=None, z)
            # , which is not valid Python code.
            node.args = (None,)

            replaced_attrs.append(replaced_attr)

    return tuple(replaced_attrs)


@_beartype.beartype
def _trace_into_fx_graph_via_fx_symbolic_trace(
    module: torch.nn.Module,
    *args,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
) -> Tuple["torch.fx.GraphModule", Tuple[Any, ...]]:
    signature = inspect.signature(module.forward)

    # We hope the input kwargs will be mapped to bound.args after binding.
    # If not, we will raise an error.
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # After apply_defaults, all non keyword-only arguments are in bound.args.
    # Because below code do not support keyword-word arguments, bound.kwargs
    # must be empty.
    assert len(bound.kwargs) == 0, bound.kwargs

    # Create inputs to call symbolic trace (torch.fx.symbolic_trace)
    # Example content of concrete_args:
    #  concrete_args["x"] = torch.fx._symbolic_trace.PH
    #  concrete_args["b"] = 1
    # where "x" and "b" are argument names in "signature".
    concrete_args = {}
    for param_name, param_value in bound.arguments.items():
        if isinstance(param_value, torch.Tensor):
            # param_value can be, e.g., a real tensor or a fake tensor.
            # param_value is treated as substitutable tensor symbol (aka placeholder).
            concrete_args[param_name] = torch.fx._symbolic_trace.PH
        else:
            concrete_args[param_name] = param_value

    return (
        _module_expansion_symbolic_trace(module, concrete_args=concrete_args),
        bound.args,
    )


def _wrap_for_symbolic_trace(target: Callable) -> Tuple[Callable, Callable]:
    """This function wraps ```target`` for symbolic tracing.

    This function wraps ```target``` so that its wrapper produces
    torch.fx.Proxy in symbolic computation. The returned values are
    the wrapper and then the original function. Per `_TORCH_METHODS_TO_PATCH`,
    this function shall receive `torch.arange`, `torch.tensor`, etc. as inputs.
    """

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


@_beartype.beartype
def _module_expansion_symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """Trace a callable into FX graph.

    When "root" is torch.nn.Module, calls to its submodule (type: torch.nn.Module) will be
    expanded into operators (e.g., torch.matmul, torch.add, +, and -) to simplify graph
    structure.
    """
    # For functions doesn't support symbolic tracing, create wrappers
    # which produce symbolic results during tracing.
    patched_torch_methods = {
        target_name: _wrap_for_symbolic_trace(getattr(torch, target_name))
        for target_name in _TORCH_METHODS_TO_PATCH
    }

    # Set the symbolic-tracing friendly functions so that `tracer.trace` below
    # can work.
    for name, (wrapper, _) in patched_torch_methods.items():
        setattr(torch, name, wrapper)

    try:
        # Set up a tracer.
        tracer = ModuleExpansionTracer()
        # Trace the model.
        graph = tracer.trace(root, concrete_args)
        name = (
            root.__class__.__name__
            if isinstance(root, torch.nn.Module)
            else root.__name__
        )
        return torch.fx.GraphModule(tracer.root, graph, name)
    finally:
        # Revert the patches for symbolic tracing.
        for name, (_, wrapped) in patched_torch_methods.items():
            # wrapped is the original version of `torch.name`.
            setattr(torch, name, wrapped)


@_beartype.beartype
def export_without_parameters_and_buffers(
    module: torch.nn.Module,
    *args,
    decomposition_table: Optional[Dict[torch._ops.OpOverload, Callable]] = None,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
) -> Tuple[
    Union[onnx.ModelProto, bytes],
    torch.fx.GraphModule,
    Tuple[Any, ...],
    Tuple[Any, ...],
]:

    graph_module, bound_args = _trace_into_fx_graph_via_fx_symbolic_trace(
        module, *args, **kwargs
    )

    # Make sure all placeholder nodes are executed before get_attr nodes.
    # Otherwise, inputs can interleave with initializers in the final ModeoProto.graph.input.
    # Basically, we want
    #  ModeoProto.graph.input =
    #   [input_0, input_1, ..., input_n, weight_0, weight_1, ..., weight_m]
    # and we don't want
    #  ModeoProto.graph.input =
    #   [input_0, weight_0, input_1, weight_1, ..., input_n, weight_0, weight_1, ..., weight_m]
    _move_placeholder_to_front(graph_module)
    # To save memory, move get_attr to input so that the generated model doesn't
    # have weigh tensors. "replaced_attrs" are the list of replaced weight tensors.
    replaced_attrs = _replace_get_attr_with_placeholder(graph_module)
    # Move all newly created placeholder nodes to the front of the graph.
    _move_placeholder_to_front(graph_module)
    # Finalize the graph editing.
    graph_module.recompile()
    return (
        exporter._export(
            graph_module,
            (*bound_args, *replaced_attrs),
            opset_version=opset_version,
            decomposition_table=decomposition_table,
            use_binary_format=use_binary_format,
            op_level_debug=op_level_debug,
        ),
        graph_module,
        bound_args,
        replaced_attrs,
    )
