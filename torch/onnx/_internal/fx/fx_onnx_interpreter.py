# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building as onnxscript_graph_building,
)

import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    _pass,
    diagnostics,
    onnxfunction_dispatcher,
    op_validation,
    type_utils as fx_type_utils,
)
from torch.utils import _pytree


@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    return f"FX Node: {node.op}:{node.target}[name={node.name}]. "


@_beartype.beartype
def _fx_graph_to_onnx_message_formatter(
    fn: Callable,
    self,
    fx_graph_module: torch.fx.GraphModule,
    *args,
    **kwargs,
) -> str:
    return f"FX Graph: {fx_graph_module._get_name()}. "


def _location_from_fx_stack_trace(
    node_stack_trace: str,
) -> Optional[diagnostics.infra.Location]:
    """Extract location from FX node stack trace.

    TODO(bowbao): Create fx utils module and move this function there.

    Args:
        node_stack_trace: The stack trace of the FX node. Example:

            File "path/file.py", line 311, in <function>
                <code>
            |   File "path/file2.py", line 389, in <function>
                <code>

    Returns:
        location: The location of the FX node.
    """
    if "File" not in node_stack_trace:
        return None

    lines = node_stack_trace.strip().split("\n")
    idx = 0
    while idx < len(lines) and "File" not in lines[idx]:
        idx += 1
    if idx + 1 >= len(lines):
        return None

    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    matches = pattern.match(lines[idx].strip())
    if matches:
        uri = matches.group(1)
        line_number = int(matches.group(2))
        snippet = lines[idx + 1].strip()
        return diagnostics.infra.Location(uri=uri, line=line_number, snippet=snippet)
    return None


@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(
    fx_node_arg: fx_type_utils.Argument,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """

    onnx_tensor = fx_node_arg
    if isinstance(onnx_tensor, torch.fx.Node):
        # 1. fx_node_arg is a torch.fx.Node, which means
        #    fx_node_arg stands for the output of that torch.fx.Node.
        # 2. fx_node_arg (variable in torch.fx.Graph) is be mapped to
        #    torch.jit.Value, fx_name_to_onnxscript_value[fx_node_arg.name],
        #    in TorchScript graph.
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    elif isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node)
        and fx_type_utils.is_torch_symbolic_type(node.meta.get("val"))
        for node in onnx_tensor
    ):
        # This intends to handle dynamic axes. for example, if the input size of op.Expand
        # is dynamic, each dimension would be variable (i.e., sym variable in Pytorch
        # FX graph. Note that sym variable is mapped to tensor in ONNX Script world)
        # calculated by other operators.
        sequence_mixed_elements: List[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
                List[int],
            ]
        ] = []
        # onnx_tensor contains a list of scalars which could be one of
        #   - tensor with empty shape,
        #   - tensor with tensor with shape (1,),
        #   - torch.SymInt,
        #   - int
        #   - ...
        # They should all be promoted to tensor with shape (1,)
        # in order to call ONNX's Concat.
        for tensor in onnx_tensor:
            # Prepare `tensor` as input of ONNX's Concat.

            if isinstance(
                tensor, torch.fx.Node
            ) and fx_type_utils.is_torch_symbolic_type(tensor.meta.get("val")):
                # In this case, tensor is a torch.SymInt from Dynamo's perspective.
                # It might be mapped to tensor with shape () or (1,) in ONNX.
                element_value = fx_name_to_onnxscript_value[tensor.name]
                if isinstance(
                    element_value, onnxscript_graph_building.TorchScriptTensor
                ):
                    # All elements sequence_mixed_elements will be send to onnx's Concat
                    # as inputs. Therefore, they are required to have the same rank.
                    # Since tensors with rank=0 (i.e., scalar) cannot be concated, all
                    # scalars are promoted to tensors with shape (1,).
                    with onnxscript.evaluator.default_as(tracer):
                        element_value = onnxscript.opset18.Reshape(element_value, [1])  # type: ignore[arg-type, type-var]
                sequence_mixed_elements.append(element_value)
            elif isinstance(tensor, int):
                # NOTE: op.Concat doesn't support scalar, so we need to wrap it with
                # dim, and onnx-script will promote it to tensor(int64)
                sequence_mixed_elements.append([tensor])
            else:
                raise RuntimeError(
                    f"Unsupported type in sequence_mixed_elements: {type(tensor)}"
                )
        # Concat all the elements in the sequence.
        # shapes are mapped to tensors in ONNX graph (TorchScriptGraph),
        # so list of sym_ints is concatenated to a tensor before calling ONNX op.

        # For example:
        #    inputs: [[2], [4], fx.Node(SymIntA), [1], fx.Node(SymIntB)]
        #    outputs: op.Concat([op.Constant(2), op.Constant(4), TorchScriptTensor(A), op.Constant(1), TorchScriptTensor(B)])

        # onnx-script auto wraps python number with op.Constants,
        # so we don't need to specifically process them.
        with onnxscript.evaluator.default_as(tracer):
            output = onnxscript.opset18.Concat(*sequence_mixed_elements, axis=0)  # type: ignore[type-var]
        output.dtype = torch.int64  # type: ignore[union-attr]
        output.shape = [len(sequence_mixed_elements)]  # type: ignore[union-attr]
        return output
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) or node is None for node in onnx_tensor
    ):
        sequence_elements: List[
            Union[
                Optional[onnxscript_graph_building.TorchScriptTensor],
                Tuple[
                    onnxscript_graph_building.TorchScriptTensor,
                    ...,
                ],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(
                fx_name_to_onnxscript_value[tensor.name] if tensor is not None else None
            )
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(
            jit_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type()
        )
    # NOTE: if device is specified in kwargs (not consumed), it's free to ignored. But
    # if it's in args, we need to set it to string for dispatcher to match schema.
    if isinstance(onnx_tensor, torch.device):
        # torch.device is not supported by onnxscript (no op). We turn it into
        # a string.
        return str(onnx_tensor)
    # all other cases, we do nothing.
    return onnx_tensor


def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    filtered = {}
    for key, value in kwargs.items():
        if key in {
            "layout",
            "device",
            "requires_grad",
            "pin_memory",
            "memory_format",
            "implicit",
        }:
            continue
        if key == "dtype":
            if value is None:
                # We omit if dtype is not provided, because onnxscript handles the
                # default case.
                continue
            else:
                value = int(jit_type_utils.JitScalarType.from_dtype(value).onnx_type())
        filtered[key] = value
    return filtered


@_beartype.beartype
def _fill_tensor_shape_type(
    onnxscript_values: Union[
        onnxscript_graph_building.TorchScriptTensor,
        Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
    ],
    name: str,
    expected_values: Union[
        fx_type_utils.META_VALUE_TYPE,
        List[fx_type_utils.META_VALUE_TYPE],
        Tuple[Optional[fx_type_utils.META_VALUE_TYPE], ...],
    ],
):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""

    if isinstance(expected_values, (list, tuple)) and not isinstance(
        onnxscript_values, (list, tuple)
    ):
        # ex: aten::split - in onnx_dtype: seq(tensor)
        # onnxscript_values is a single tensor, but expected_values is a list of tensors.
        return

    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for i, (onnxscript_value, expected_value) in enumerate(
        zip(flat_onnxscript_values, flat_expected_values)
    ):
        if expected_value is None:
            # There is no shape/type from None.
            # NOTE: according to https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py,
            # None could be a valid value for return type, so we need to handle it.
            # e.g. the function: meta__scaled_dot_product_flash() in cpu mode.
            continue
        elif fx_type_utils.is_torch_symbolic_type(expected_value):
            # aten::sym_size output is a int, not a tensor, which stands
            # for the size of one dim. We treat it as 1-D tensor.
            onnxscript_value.dtype = fx_type_utils.from_sym_value_to_torch_dtype(
                expected_value
            )
            onnxscript_value.shape = torch.Size([1])
        elif isinstance(expected_value, (int, float, bool)):
            onnxscript_value.dtype = fx_type_utils.from_scalar_type_to_torch_dtype(
                type(expected_value)
            )
            onnxscript_value.shape = torch.Size([])
        elif isinstance(expected_value, complex):
            # From complex scalar to real representation
            onnxscript_value_to_torch_dtype = (
                fx_type_utils.from_scalar_type_to_torch_dtype(type(expected_value))
            )
            onnxscript_value.dtype = (
                fx_type_utils.from_complex_to_float(onnxscript_value_to_torch_dtype)
                if onnxscript_value_to_torch_dtype is not None
                else None
            )
            onnxscript_value.shape = torch.Size([2])
        elif fx_type_utils.is_torch_complex_dtype(expected_value.dtype):
            # Like torch.view_as_real, we flatten complex tensors to real tensors with
            # additional last dimension of 2
            onnxscript_value.shape = torch.Size((*expected_value.size(), 2))
            # complex64 -> float32, complex128 -> float64, etc.
            onnxscript_value.dtype = fx_type_utils.from_complex_to_float(
                expected_value.dtype
            )
            # Dispatcher needs to know the value is complex
            onnxscript_value.is_complex = True
        else:
            # We set node output sizes to be dynamic to continue the model conversion,
            # and inputs are also set to be dynamic in add_input().
            onnxscript_value.shape = expected_value.size()
            onnxscript_value.dtype = expected_value.dtype

        # naming
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


@_beartype.beartype
def _fill_in_default_kwargs(
    node: torch.fx.Node,
) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""

    # TODO: aten::sym_size has overload, but fx graph is using
    # overloadpacket for some reasons.
    # https://github.com/pytorch/pytorch/issues/97201
    # We manually assigned overload for aten::sym_size.
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args: List[fx_type_utils.Argument] = []
    complete_kwargs: Dict[str, fx_type_utils.Argument] = {}

    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                # Get default from schema.
                complete_kwargs[expected_arg.name] = expected_arg.default_value

    return complete_args, complete_kwargs


@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(
    complete_args: List[fx_type_utils.Argument],
    complete_kwargs: Dict[str, fx_type_utils.Argument],
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
) -> Tuple[
    Sequence[
        Optional[
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                str,
                int,
                float,
                bool,
                list,
                complex,
            ]
        ]
    ],
    Dict[str, fx_type_utils.Argument],
]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer)
        for arg in complete_args
    )
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    return onnxscript_args, onnxscript_kwargs


class FxOnnxInterpreter:
    """Stateless class to process FX graph Nodes and translate them into their ONNX counterparts.

    All FX nodes described by [FX Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) are supported.
    Similarly to [FX Interpreter pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter), each FX node
    must be implemented on its own method in this class.

    Each operator's implementation returns either an `onnxscript.OnnxFunction` or
    `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm. They can
    also raise RuntimeError: If there are no overloaded functions available for the given FX node.

    TODO: Convert methods to @staticmethod when the diagnostic system supports it
          DO NOT ADD NEW ATTRIBUTES TO THIS CLASS!
    """

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        # THIS SHOULD BE THE ONLY STATE IN THIS CLASS (constraint from diagnosticS API)
        # TODO: Diagnostics API should be revised to get rid of this attribute.
        # DO NOT add other class-level attributes.
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_node_to_onnx,
        diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
    )
    def run_node(
        self,
        node,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        """Execute a single FX node to produce its ONNX counterpart.

        Args:
            node: The FX node to be translated.
            fx_graph_module: The FX graph module containing the node.
            onnxfunction_dispatcher: The dispatcher to find the best matched ONNX op.
            op_level_debug (bool): Whether to enable op level debug.
            onnxscript_graph: The ONNX graph to be populated.
            onnxscript_tracer: The tracer to trace the ONNX graph.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNX Script value.

        Raises:
            RuntimeError: When a node.op is not supported.
        """
        # Record stack trace of node in diagnostic.
        node_stack_trace = node.stack_trace
        if node_stack_trace:
            diagnostic = self.diagnostic_context.inflight_diagnostic(
                rule=diagnostics.rules.fx_node_to_onnx
            )
            with diagnostic.log_section(logging.INFO, "PyTorch source information"):
                diagnostic.info("```\n%s\n```", node_stack_trace)
            location = _location_from_fx_stack_trace(node_stack_trace)
            if location is not None:
                diagnostic.with_location(location)

        if node.op == "placeholder":
            self.placeholder(node, onnxscript_graph, fx_name_to_onnxscript_value)
        elif node.op == "get_attr":
            self.get_attr(
                node,
                onnxscript_graph,
                fx_name_to_onnxscript_value,
                fx_graph_module,
            )
        elif node.op == "call_function":
            self.call_function(
                node,
                onnxscript_tracer,
                fx_name_to_onnxscript_value,
                onnxfunction_dispatcher,
                op_level_debug,
                fx_graph_module,
            )
        elif node.op == "call_method":
            self.call_method(node)
        elif node.op == "call_module":
            self.call_module(
                node,
                onnxscript_graph,
                fx_name_to_onnxscript_value,
                onnxscript_tracer,
                fx_graph_module,
                onnxfunction_dispatcher,
                op_level_debug,
            )
        elif node.op == "output":
            self.output(node, onnxscript_graph, fx_name_to_onnxscript_value)
        else:
            raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_graph_to_onnx,
        diagnostic_message_formatter=_fx_graph_to_onnx_message_formatter,
    )
    def run(
        self,
        fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        parent_onnxscript_graph: Optional[
            onnxscript_graph_building.TorchScriptGraph
        ] = None,
    ) -> onnxscript_graph_building.TorchScriptGraph:
        """Analyze all FX nodes and trigger their ONNX translation.

        Args:
            fx_graph_module: FX graph module to be translated.
            onnxfunction_dispatcher: ONNX function dispatcher.
            op_level_debug: Whether to enable op-level debug.
            parent_onnxscript_graph: The parent TorchScript graph. Must be provided if
                `fx_graph_module` is a submodule. If not provided,
                `fx_graph_module` is assumed to be the root module.
        """
        diagnostic = self.diagnostic_context.inflight_diagnostic()
        with diagnostic.log_section(logging.DEBUG, "FX Graph:"):
            diagnostic.debug(
                "```\n%s\n```",
                diagnostics.LazyString(fx_graph_module.print_readable, False),
            )

        if parent_onnxscript_graph is not None:
            # If parent_onnxscript_graph is provided, we assume fx_graph_module is a
            # submodule representing a forward call of an nn.Module.
            # Compose package and version where the nn.Module is defined as domain name
            # for the local function.

            onnx_meta: Optional[_pass.GraphModuleOnnxMeta] = fx_graph_module.meta.get(
                "onnx"
            )
            if onnx_meta is None:
                raise RuntimeError(
                    f"ONNX meta is not found in submodule {fx_graph_module._get_name()}. "
                    f"Only submodules produced by `Modularize` pass is supported in ONNX export."
                )

            onnx_domain = onnx_meta.package_info.to_onnx_domain_string()
        else:
            # Leave as default domain name for the root module.
            onnx_domain = None

        onnxscript_graph = onnxscript_graph_building.TorchScriptGraph(
            parent_onnxscript_graph, domain_name=onnx_domain
        )
        onnxscript_tracer = onnxscript_graph_building.TorchScriptTracingEvaluator(
            onnxscript_graph
        )
        # In the following loop, a TorchScript graph is created to
        # represent the input FX graph with ONNX symbols (e.g., onnx::add).
        # To connect the values to nodes in the TorchScript graph, we maintain
        # fx_name_to_onnxscript_value. Basically, we want to translate
        #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
        # to
        #   fx_name_to_onnxscript_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_onnxscript_value[fx_tensor_y.name]
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ] = {}

        # TODO: Fix FakeTensorMode limitation asap
        # We want to pass list of ints and floats to TorchScript graph correctly
        # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
        # receive FakeTensor and results runtime error. In addition, TorchScript-based
        # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
        # with FakeTensorMode.
        with torch.utils._mode_utils.no_dispatch():
            # node_fixed_shape is only used on op_level_debug purpose.
            for node in fx_graph_module.graph.nodes:
                self.run_node(
                    node,
                    fx_graph_module,
                    onnxfunction_dispatcher,
                    op_level_debug,
                    onnxscript_graph,
                    onnxscript_tracer,
                    fx_name_to_onnxscript_value,
                )

        with diagnostic.log_section(logging.DEBUG, "ONNX Graph:"):
            diagnostic.debug("```\n%s\n```", onnxscript_graph.torch_graph)

        return onnxscript_graph

    @_beartype.beartype
    def placeholder(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        # Input of graph.
        # The node.meta["val"] is generated by FakeTensorProp.
        # NOTE: add_input() intends to create nodes with shape/type
        fake_tensor = node.meta.get("val", None)
        # NOTE: During the tracing, when inputs are constants, they are represented
        # by nodes with node.meta['val'] being None (nn.Module to dynamo_export)
        # or nodes with node.meta['val'] being a builtin value (ExportedProgram to dynamo_export).
        # Nonethless, the nodes are not consumed by others, so we don't need to
        # create a TorchScriptTensor for them.
        if fake_tensor is None or isinstance(fake_tensor, (int, float, bool, str)):
            output = onnxscript_graph.add_input(
                input_name=None,
            )
        elif isinstance(fake_tensor, torch.Tensor):
            # NOTE: ONNX doesn't support tensor of complex64/complex128, so we
            # convert them to float32/float64 with real representation.
            if fx_type_utils.is_torch_complex_dtype(fake_tensor.dtype):
                fake_tensor = torch.view_as_real(fake_tensor.resolve_conj())
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=fake_tensor.shape,
                dtype=fake_tensor.dtype,
            )

        elif fx_type_utils.is_torch_symbolic_type(fake_tensor):
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=torch.Size([]),
                dtype=fx_type_utils.from_sym_value_to_torch_dtype(fake_tensor),
            )
        else:
            raise RuntimeError(
                f"Unsupported type(node.meta['val']) for placeholder: {type(fake_tensor)}"
            )
        assert (
            output is not None
        ), f"Node creates None with target={node.target} and name={node.name}"

        assert isinstance(output, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(output, onnxscript.tensor.Tensor)

        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def call_function(
        self,
        node: torch.fx.Node,
        onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
        fx_graph_module: torch.fx.GraphModule,
    ):
        # aten ops and other stateless functions.
        if node.target == operator.getitem and isinstance(
            fx_name_to_onnxscript_value[node.args[0].name], tuple  # type: ignore[union-attr,index]
        ):
            onnx_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]  # type: ignore[union-attr,index]
            index = node.args[1]
            output = onnx_tensor_tuple[index]  # type: ignore[index]
            assert (
                output is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            assert isinstance(
                output, (onnxscript_graph_building.TorchScriptTensor, tuple)
            ), type(output)

            fx_name_to_onnxscript_value[node.name] = output
            return

        # Map FX inputs to ONNX inputs and fill optional inputs with default values.
        # torch_args and torch_kwargs are for op-level validation
        fx_args, fx_kwargs = _fill_in_default_kwargs(node)

        onnx_args, onnx_kwargs = _wrap_fx_args_as_onnxscript_args(
            fx_args,
            fx_kwargs,
            fx_name_to_onnxscript_value,
            onnxscript_tracer,
        )
        # Dispatch to ONNX op through OpShema. The input argument dtypes are compared to
        # function signature in OpSchema, and find the best matched overload.
        symbolic_fn = onnxfunction_dispatcher.dispatch(
            node=node,
            onnx_args=onnx_args,
            onnx_kwargs=onnx_kwargs,
            diagnostic_context=self.diagnostic_context,
        )
        with onnxscript.evaluator.default_as(onnxscript_tracer):
            output: Union[  # type: ignore[no-redef]
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ] = symbolic_fn(*onnx_args, **onnx_kwargs)
        assert (
            output is not None
        ), f"Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}"
        # Assign type and shape from fx graph.
        _fill_tensor_shape_type(output, node.name, node.meta["val"])
        # One fx node could produce multiple outputs (e.g., tuple of tensors); in
        # that case, v is a tuple of TorchScriptTensors.
        assert isinstance(
            output, (onnxscript_graph_building.TorchScriptTensor, tuple)
        ), type(output)
        # NOTE(titaiwang): We bypass two kinds of ops as it's not meaningful to
        # validate them with op level debug.
        # 1. aten::sym_size: The op is simply get item from a list of tensors.
        # 2. BuiltinFunction: It doesn't supported tensor
        if (
            op_level_debug
            and node.target != torch.ops.aten.sym_size
            and not isinstance(node.target, types.BuiltinFunctionType)
        ):
            op_validation.validate_op_between_ort_torch(
                self.diagnostic_context,
                node,
                symbolic_fn,
                fx_args,
                fx_kwargs,
                fx_graph_module,
            )
        fx_name_to_onnxscript_value[node.name] = output

    @_beartype.beartype
    def output(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
    ):
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)

    @_beartype.beartype
    def call_method(self, node: torch.fx.Node):
        # TODO(wechi): Support call_method.
        raise RuntimeError("call_method is not supported yet.")

    @_beartype.beartype
    def call_module(
        self,
        node: torch.fx.Node,
        parent_onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        tracer: onnxscript_graph_building.TorchScriptTracingEvaluator,
        root_fx_graph_module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
        op_level_debug: bool,
    ) -> None:
        """Export a fx.GraphModule submodule to ONNXScript graph.

        The export process specifically targets `call_module` nodes that are created by
        the exporter's `Modularize` pass. Each `call_module` node has an associated fx.GraphModule
        by `node.target` underneath the root fx.GraphModule. These `call_module` nodes are exported as ONNX
        function nodes. The related `sub_module` is then exported as an ONNX model local function,
        which is represented by another `TorchScriptGraph`. This `TorchScriptGraph` sets the current
        `onnxscript_graph` as its parent.

        Args:
            node: The call_module node in the FX graph that represents the submodule call.
            parent_onnxscript_graph: The parent ONNXScript graph to which the ONNX function and
                function node belong.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNXScript value.
            tracer: The tracer used to trace the ONNXScript graph.
            root_fx_graph_module: The root FX module.
            onnxfunction_dispatcher: The dispatcher.
            op_level_debug: Whether to enable op-level debug.
        """
        assert isinstance(
            node.target, str
        ), f"node.target must be a str, not {type(node.target)} for node {node}."

        sub_module = root_fx_graph_module.get_submodule(node.target)

        assert isinstance(
            sub_module, torch.fx.GraphModule
        ), f"sub_module must be a torch.fx.GraphModule, not {type(sub_module)} for node {node}."

        sub_onnxscript_graph = self.run(
            sub_module, onnxfunction_dispatcher, op_level_debug, parent_onnxscript_graph
        )

        onnx_args, _ = _wrap_fx_args_as_onnxscript_args(
            list(node.args), {}, fx_name_to_onnxscript_value, tracer
        )

        # TODO: We may want to consider other naming styles. The goal is to be stable and
        # unique such that it can be easily identified in case of kernel substitution.
        # Example for current style is combination of qualified module class name and
        # module attribute name: `torch_nn_modules_conv_Conv2d_conv1`.
        # Other naming styles such as qualified module class name made unique can also
        # be considered.
        unique_module_name = f"{sub_module._get_name()}_{node.target}"

        outputs: Union[  # type: ignore[no-redef]
            onnxscript_graph_building.TorchScriptTensor,
            Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
        ] = parent_onnxscript_graph.add_module_call(
            unique_module_name, sub_onnxscript_graph, onnx_args
        )

        assert isinstance(
            outputs, (onnxscript_graph_building.TorchScriptTensor, tuple)
        ), f"Unexpected outputs type {type(outputs)} for node {node}."

        _fill_tensor_shape_type(outputs, node.name, node.meta["val"])
        fx_name_to_onnxscript_value[node.name] = outputs

        # Skip op_level_validation for call_module. Subgraph nodes are validated individually.

    @_beartype.beartype
    def get_attr(
        self,
        node: torch.fx.Node,
        onnxscript_graph: onnxscript_graph_building.TorchScriptGraph,
        fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_building.TorchScriptTensor,
                Tuple[onnxscript_graph_building.TorchScriptTensor, ...],
            ],
        ],
        fx_graph_module: torch.fx.GraphModule,
    ):
        assert isinstance(node.target, str), f"node.target {node.target} is not a str."
        attr_tensor = getattr(fx_graph_module, node.target)
        assert isinstance(attr_tensor, torch.Tensor), f"{attr_tensor} is not a tensor."

        # Parameter/buffer name cannot contain "."
        # Revert from "/" to restore namespace formatting.
        input_ = onnxscript_graph.add_initializer(
            name=node.target.replace("/", "."),
            value=attr_tensor,
        )

        assert isinstance(input_, onnxscript_graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        fx_name_to_onnxscript_value[node.name] = input_
