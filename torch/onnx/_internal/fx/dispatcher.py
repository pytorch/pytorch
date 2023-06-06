"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import inspect

import operator
import re
import types

import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from onnxscript import evaluator, opset18  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building as onnxscript_graph_builder,
)

import torch
import torch._ops
import torch.fx
from torch._subclasses import fake_tensor
from torch.onnx import _constants, _type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, op_validation, registration
from torch.utils import _pytree

if TYPE_CHECKING:
    import onnx.defs  # type: ignore[import]
    import onnxscript  # type: ignore[import]


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.


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
def _fill_in_default_kwargs(
    node: torch.fx.Node,
) -> Tuple[List[_type_utils.Argument], Dict[str, _type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""

    # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
    # overloadpacket for some reasons.
    # https://github.com/pytorch/pytorch/issues/97201
    # We manually assigned overload for aten::sym_size.
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args: List[_type_utils.Argument] = []
    complete_kwargs: Dict[str, _type_utils.Argument] = {}

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
def _retrieve_or_adapt_input_to_graph_set(
    fx_node_arg: _type_utils.Argument,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_builder.TorchScriptTensor,
            Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_builder.TorchScriptTracingEvaluator,
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
    if isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node) and isinstance(node.meta["val"], torch.SymInt)
        for node in onnx_tensor
    ):
        # This intends to handle dynamic axes. for example, if the input size of op.Expand
        # is dynamic, each dimension would be variable (i.e., sym variable in Pytorch
        # FX graph. Note that sym variable is mapped to tensor in ONNX Script world)
        # calculated by other operators.
        sequence_mixed_elements: List[
            Union[onnxscript_graph_builder.TorchScriptTensor, List[int]]
        ] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and isinstance(
                tensor.meta["val"], torch.SymInt
            ):
                sequence_mixed_elements.append(fx_name_to_onnxscript_value[tensor.name])
            elif isinstance(tensor, int):
                # NOTE: op.Concat doesn't support scalar, so we need to wrap it with
                # dim, and onnx-script will promote it to tensot(int64)
                sequence_mixed_elements.append([tensor])
        # Concat all the elements in the sequence.
        # shapes are mapped to tensors in ONNX graph (TorchScriptGraph),
        # so list of sym_ints is concatenated to a tensor before calling ONNX op.

        # For example:
        #    inputs: [[2], [4], fx.Node(SymIntA), [1], fx.Node(SymIntB)]
        #    outputs: op.Concat([op.Constant(2), op.Constant(4), TorchScriptTensor(A), op.Constant(1), TorchScriptTensor(B)])

        # onnx-script auto wraps python number with op.Constants,
        # so we don't need to specifically process them.
        with evaluator.default_as(tracer):
            output = opset18.Concat(*sequence_mixed_elements, axis=0)
        output.dtype = torch.int64
        output.shape = [len(sequence_mixed_elements)]
        return output
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) for node in onnx_tensor
    ):
        sequence_elements: List[
            Union[
                onnxscript_graph_builder.TorchScriptTensor,
                Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscript_value[tensor.name])
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())

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
                filtered["dtype"] = int(
                    _type_utils.JitScalarType.from_dtype(value).onnx_type()
                )
            continue
        filtered[key] = value
    return filtered


@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(
    complete_args: List[_type_utils.Argument],
    complete_kwargs: Dict[str, _type_utils.Argument],
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            onnxscript_graph_builder.TorchScriptTensor,
            Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
        ],
    ],
    tracer: onnxscript_graph_builder.TorchScriptTracingEvaluator,
) -> Tuple[
    Sequence[
        Optional[
            Union[
                onnxscript_graph_builder.TorchScriptTensor, str, int, float, bool, list
            ]
        ]
    ],
    Dict[str, _type_utils.Argument],
]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer)
        for arg in complete_args
    )
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    return onnxscript_args, onnxscript_kwargs


@_beartype.beartype
def _fill_tensor_shape_type(
    onnxscript_values: Union[
        onnxscript_graph_builder.TorchScriptTensor,
        Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
    ],
    name: str,
    expected_values: Union[
        fake_tensor.FakeTensor,
        torch.SymInt,
        torch.SymFloat,
        List[fake_tensor.FakeTensor],
        Tuple[fake_tensor.FakeTensor, ...],
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
        # aten::sym_size output is a int, not a tensor, which stands
        # for the size of one dim. We treat it as 0-D tensor.
        # TODO(titaiwang): set shape?
        if isinstance(expected_value, torch.SymInt):
            onnxscript_value.dtype = torch.int64
        elif isinstance(expected_value, torch.SymFloat):
            onnxscript_value.dtype = torch.float32
        else:
            # We set node output sizes to be dynamic to continue the model conversion,
            # and inputs are also set to be dynamic in add_input().
            onnxscript_value.shape = tuple(
                [dim if isinstance(dim, int) else None for dim in expected_value.size()]
            )
            onnxscript_value.dtype = expected_value.dtype
        # naming
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable,
    self: "torch.onnx._internal.fx.dispatcher.OnnxDispatcher",
    graph_module,
    onnxscript_graph,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    return f"FX Node: {node.op}:{node.target}[name={node.name}]. "


@runtime_checkable
class _TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


class OnnxDispatcher:
    """
    The OnnxDispatcher class finds the nearest matched function for a given aten operation
    in the FX exporter. It uses the torch.ops name to find the function. If not found,
    it falls back to default. Then, it finds the nearest match among all overloaded
    functions. If the types match, it selects the function. Otherwise, it finds the
    nearest one with matching score mechanism.

    method: dispatch

    Steps for overloaded function dispatch:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists.
        b. If not found, check ATen overload=default.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
              the potential wronly annotated dtypes and attributes matching, we use
              nearest match to find the best function once the aten name is targeted.
              The nearest match `doesn't guarantee` a correct match, and a warning message
              will be sent to SARIF.
    """

    def __init__(
        self,
        registry: registration.OnnxRegistry,
        diagnostic_context: diagnostics.DiagnosticContext,
        opset_version: int = 18,
        op_level_debug: bool = False,
    ):
        """Initialize the Dispatcher.

        Args:
            registry: The registration registry.
            opset_version: The model opset version.
        """
        self._registry = registry
        self._opset_version = opset_version
        # In the main loop, a TorchScript graph is augmented to
        # represent the input FX graph with ONNX symbols (e.g., onnx::add).
        # To connect the values to nodes in the TorchScript graph, we maintain
        # fx_name_to_onnxscript_value. Basically, we want to translate
        #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
        # to
        #   fx_name_to_onnxscript_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_onnxscript_value[fx_tensor_y.name]
        self.fx_name_to_onnxscript_value: Dict[
            str,
            Union[
                onnxscript_graph_builder.TorchScriptTensor,
                Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
            ],
        ] = {}
        self.tracer = None
        self.op_level_debug = op_level_debug
        self.diagnostic_context = diagnostic_context

    @property
    def opset_version(self) -> int:
        """Get the model opset version."""
        return self._opset_version

    @property
    def registry(self) -> registration.OnnxRegistry:
        """Get the registration registry."""
        return self._registry

    @_beartype.beartype
    @diagnostics.diagnose_call(
        diagnostics.rules.fx_node_to_onnx,
        diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
    )
    def dispatch(self, graph_module, onnxscript_graph, node: torch.fx.Node):
        """Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.

        Args:
            node: The TorchFX node to dispatch the function for.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.

        Raises:
            RuntimeError: If there are no overloaded functions available for the given FX node.
        """

        # TODO: Refactor where/how onnxscript_graph is store/passed in
        if self.tracer is None:
            self.tracer = onnxscript_graph_builder.TorchScriptTracingEvaluator(
                onnxscript_graph
            )

        # Record stack trace of node in diagnostic.
        node_stack_trace = node.stack_trace
        if node_stack_trace:
            diagnostic = self.diagnostic_context.inflight_diagnostic(
                rule=diagnostics.rules.fx_node_to_onnx
            )
            diagnostic.with_additional_message(
                f"### PyTorch source information\n```\n{node_stack_trace}\n```"
            )
            location = _location_from_fx_stack_trace(node_stack_trace)
            if location is not None:
                diagnostic.with_location(location)

        if node.op == "placeholder":
            # Input of graph.
            # The node.meta["val"] is generated by FakeTensorProp.
            # NOTE: add_input() intends to create nodes with shape/type
            fake_tensor = node.meta.get("val", None)
            if fake_tensor is None:
                output = onnxscript_graph.add_input(
                    input_name=None,
                )
            else:
                output = onnxscript_graph.add_input(
                    input_name=node.name,
                    shape=fake_tensor.shape,
                    dtype=fake_tensor.dtype,
                )
            assert (
                output is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            assert isinstance(output, onnxscript_graph_builder.TorchScriptTensor)
            import onnxscript

            assert isinstance(output, onnxscript.tensor.Tensor)

            self.fx_name_to_onnxscript_value[node.name] = output
        elif node.op == "call_function":
            # aten ops and other stateless functions.
            if node.target == operator.getitem and isinstance(
                self.fx_name_to_onnxscript_value[node.args[0].name], tuple  # type: ignore[union-attr]
            ):
                onnx_tensor_tuple = self.fx_name_to_onnxscript_value[node.args[0].name]  # type: ignore[union-attr]
                index = node.args[1]
                output = onnx_tensor_tuple[index]  # type: ignore[index]
                assert (
                    output is not None
                ), f"Node creates None with target={node.target} and name={node.name}"
                assert isinstance(
                    output, (onnxscript_graph_builder.TorchScriptTensor, tuple)
                ), type(output)

                self.fx_name_to_onnxscript_value[node.name] = output
                return

            # Map FX inputs to ONNX inputs and fill optional inputs with default values.
            # torch_args and torch_kwargs are for op-level validation
            complete_args, complete_kwargs = _fill_in_default_kwargs(node)
            onnx_args, onnx_kwargs = _wrap_fx_args_as_onnxscript_args(
                complete_args,
                complete_kwargs,
                self.fx_name_to_onnxscript_value,
                self.tracer,
            )

            aten_name = self._get_aten_name(node, self.diagnostic_context)
            # If there are no overloaded functions available for the given FX node, raise an
            # unsupported error
            function_overloads = self._get_function_overloads(
                node, aten_name, self.diagnostic_context
            )
            # If there are overloaded functions available, we will find one that perfect or
            # nearest matches the given arguments and keyword arguments
            symbolic_fn = self._find_the_perfect_or_nearest_match_onnxfunction(
                node,
                aten_name,
                function_overloads,
                onnx_args,
                onnx_kwargs,
                self.diagnostic_context,
            )

            with evaluator.default_as(self.tracer):  # type: ignore[arg-type]
                output: Union[  # type: ignore[no-redef]
                    onnxscript_graph_builder.TorchScriptTensor,
                    Tuple[onnxscript_graph_builder.TorchScriptTensor, ...],
                ] = symbolic_fn(*onnx_args, **onnx_kwargs)
            assert (
                output is not None
            ), f"Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}"
            # Assign type and shape from fx graph.
            _fill_tensor_shape_type(output, node.name, node.meta["val"])
            # One fx node could produce multiple outputs (e.g., tuple of tensors); in
            # that case, v is a tuple of TorchScriptTensors.
            assert isinstance(
                output, (onnxscript_graph_builder.TorchScriptTensor, tuple)
            ), type(output)
            # NOTE(titaiwang): We bypass two kinds of ops as it's not meaningful to
            # validate them with op level debug.
            # 1. aten::sym_size: The op is simply get item from a list of tensors.
            # 2. BuiltinFunction: It doesn't supported tensor
            if (
                self.op_level_debug
                and node.target != torch.ops.aten.sym_size
                and not isinstance(node.target, types.BuiltinFunctionType)
            ):
                (
                    node_with_fixed_shape_args,
                    node_with_fixed_shape_kwargs,
                ) = _fill_in_default_kwargs(node)
                try:
                    torch_args, torch_kwargs = op_validation.wrap_fx_args_as_torch_args(
                        node_with_fixed_shape_args, node_with_fixed_shape_kwargs
                    )
                except ValueError as value_error:
                    warnings.warn(
                        f"\nFound unsupported input types on PyTorch Op {node.target} with "
                        f"ValueError: \n{value_error}.\n"
                    )
                    diagnostic = self.diagnostic_context.inflight_diagnostic()
                    diagnostic.with_additional_message(
                        f"### Op level debug fails due to unsupported input types\n"
                        f"{diagnostics.decorator.format_exception_in_markdown(value_error)}"
                    )
                    diagnostic.level = diagnostics.levels.ERROR
                else:
                    op_validation.validate_op_between_ort_torch(
                        self.diagnostic_context,
                        node,
                        symbolic_fn,
                        torch_args,
                        torch_kwargs,
                    )
            self.fx_name_to_onnxscript_value[node.name] = output
        elif node.op == "output":
            if isinstance(node.args[0], torch.fx.Node):
                onnx_tensor_or_tensor_tuple = self.fx_name_to_onnxscript_value[
                    node.args[0].name
                ]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
            else:
                # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
                # tensor, etc), we flatten the collection and register each element as output.
                flat_args, _ = _pytree.tree_flatten(node.args[0])
                for arg in flat_args:
                    assert isinstance(
                        arg, torch.fx.Node
                    ), f"arg must be a torch.fx.Node, not {type(arg)}"
                    onnx_tensor_or_tensor_tuple = self.fx_name_to_onnxscript_value[
                        arg.name
                    ]
                    onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        elif node.op == "call_method":
            # TODO(wechi): Support call_method.
            raise RuntimeError("call_method is not supported yet.")
        elif node.op == "call_module":
            # TODO(wechi): Support call_module.
            raise RuntimeError("call_module is not supported yet.")
        elif node.op == "get_attr":
            current_attr = graph_module
            sub_attr_names = node.target.split(".")  # type: ignore[union-attr]
            # If node.targe is "conv.weight", the following loop first
            # assigns fx_module_with_metadata.conv to current_attr, and then
            # fx_module_with_metadata.conv.weight to current_attr.
            while sub_attr_names:
                sub_attr_name = sub_attr_names.pop(0)
                if not hasattr(current_attr, sub_attr_name):
                    raise AttributeError(
                        f"Attribute {sub_attr_name} is not found in {current_attr}."
                    )
                current_attr = getattr(current_attr, sub_attr_name)

            input_ = onnxscript_graph.add_initializer(node.name, current_attr)

            assert isinstance(input_, onnxscript_graph_builder.TorchScriptTensor)
            import onnxscript

            assert isinstance(input_, onnxscript.tensor.Tensor)
            self.fx_name_to_onnxscript_value[node.name] = input_
            # FIXME: Refactor logic getting 'current_attr'.
            assert isinstance(current_attr, torch.Tensor)
        else:
            # TODO(wechi): Support get_attr, call_module, call_method.
            raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")

    @_beartype.beartype
    def _find_the_perfect_or_nearest_match_onnxfunction(
        self,
        node: torch.fx.Node,
        aten_name: str,
        function_overloads: Set[
            Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
        ],
        onnx_args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        onnx_kwargs: Dict[str, _type_utils.Argument],
        diagnostic_context: diagnostics.DiagnosticContext,
    ):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments."""
        overload_match_ranking: Dict[
            Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction], int
        ] = {}
        # TODO(justinchuby): Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        for overload_func in function_overloads:
            function_opschema = _OpSchemaWrapper(overload_func.op_schema)
            if function_opschema.perfect_match_inputs(onnx_args, onnx_kwargs):
                # If the perfect match is found, return the function
                return overload_func
            # Record the match score for the nearest match if it's not the perfect match
            overload_match_ranking[overload_func] = function_opschema.match_score

        # TODO(titaiwang): Change inflight_diagnostic to a new rule. Do we need to handle
        # the special case where the same scores happended?
        # If the perfect match is not found, find the nearest match
        warnings.warn(
            f"A perfect matched Opchema is not found in torchlib for {aten_name}, but \n"
            f"a nearest match is found. Please check the ONNX output carefully. \n",
        )
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.WARNING,
            f"Cannot find a perfect match of symbolic overload for {aten_name}, "
            f"which should be registered under {node.target}. But a nearest match is found.",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        return max(overload_match_ranking, key=overload_match_ranking.get)  # type: ignore[arg-type]

    @_beartype.beartype
    def _get_aten_name(
        self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext
    ) -> str:
        """Get the aten name from the target."""
        if node.target == operator.getitem:
            return "aten::getitem"
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            # aten::sym_size is the only OverloadPacket that we support.
            # schema: aten::sym_size(Tensor self, int dim) -> Tensor
            if node.target != torch.ops.aten.sym_size:
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                    diagnostics.rules.no_symbolic_function_for_call_function,
                    diagnostics.levels.ERROR,
                    f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!",
                    unsupported_fx_node=node,
                )
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            aten_op_default = node.target.default
            return aten_op_default.name()  # type: ignore[attr-defined]
        # import pdb; pdb.set_trace()
        if _symint_symfloat_builtin_to_exporter_key_table(node.target) is not None:
            # Make sure it's symint/symfloat consuming builtin ops.
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not isinstance(
                        node_arg.meta["val"], (torch.SymInt, torch.SymFloat)
                    )
                ):
                    # TODO: reduce number of explicit initializations.
                    # TODO: Log location, stack.
                    diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
                        diagnostics.rules.no_symbolic_function_for_call_function,
                        diagnostics.levels.ERROR,
                        f"Unsupported node arg: {node_arg} (type {type(node_arg)}) with builtin function: {node.target},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!",
                        unsupported_fx_node=node,
                    )
                    diagnostic_context.log(diagnostic)
                    raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            aten_op = _symint_symfloat_builtin_to_exporter_key_table(node.target)
            return aten_op.name()  # type: ignore[attr-defined]
        if isinstance(node.target, torch._ops.OpOverload):
            return node.target.name()

        # Unexpected target, raise error.
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Unknown call_function target: {node.target}",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

    @_beartype.beartype
    def _get_function_overloads(
        self,
        node: torch.fx.Node,
        aten_name: str,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Set[Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]]:
        """Get the function overloads from the registry."""
        function_group = None

        if self.registry.is_registered_op(aten_name, self.opset_version):
            function_group = self.registry.get_function_group(aten_name)

        # Fall back to overloadpacket name: eg: aten.add.Tensor -> aten::add
        elif hasattr(node.target, "overloadpacket") and self.registry.is_registered_op(
            node.target.overloadpacket._qualified_op_name,  # type: ignore[union-attr]
            self.opset_version,
        ):
            function_group = self.registry.get_function_group(
                node.target.overloadpacket._qualified_op_name  # type: ignore[union-attr]
            )

        if function_group is not None:
            # TODO(titaiwang): dispatch opset version.
            dispatched_version = _dispatch_opset_version(
                self.opset_version, function_group.support_opset()
            )
            if dispatched_version is not None:
                function_overloads = function_group.get(dispatched_version)
                if function_overloads is not None:
                    return function_overloads

        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(
            diagnostics.rules.no_symbolic_function_for_call_function,
            diagnostics.levels.ERROR,
            f"Cannot find symbolic function for {aten_name}, "
            f"which should be registered under {node.target}.",
            unsupported_fx_node=node,
        )
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)


@_beartype.beartype
def _symint_symfloat_builtin_to_exporter_key_table(
    target,
) -> Optional[torch._ops.OpOverload]:
    """Maps builtin ops to exporter key table."""

    _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE: Dict[
        Union[Callable[..., Any], str], torch._ops.OpOverload
    ] = {
        operator.mul: torch.ops.aten.mul.default,  # type: ignore[has-type]
        operator.add: torch.ops.aten.add.default,  # type: ignore[has-type]
        operator.pow: torch.ops.aten.pow.int,  # type: ignore[has-type]
        operator.sub: torch.ops.aten.sub.default,  # type: ignore[has-type]
    }
    return _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE.get(target)


@_beartype.beartype
def _dispatch_opset_version(
    target: int, registered_opsets: Collection[int]
) -> Optional[int]:
    """Finds the registered opset given a target opset version and the available opsets.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    if not registered_opsets:
        return None

    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    if target >= _constants.ONNX_BASE_OPSET:
        # Always look down toward opset 1 when the target is >= ONNX_BASE_OPSET (opset 9).
        # When a custom op is register at opset 1, we want to be able to discover it as a
        # fallback for all opsets >= ONNX_BASE_OPSET.
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9. This is the legacy behavior to support opset 7 and opset 8.
    # for caffe2 support. We search up toward opset 9.
    for version in reversed(descending_registered_versions):
        # Count back up until _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


class _OpSchemaWrapper:
    """
    The OpSchemaWrapper class is a wrapper for ONNX OpSchema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types.

    There are three types of ONNX overloads in torchlib:

    1. Different types: Caused by the difference between the ONNX spec and PyTorch.The
        matching system finds the correct one.

        ```python
        @torch_op("aten::mul")
        def aten_mul(self: TReal, other: TReal) -> TReal:
            ...

        @torch_op("aten::mul")
        def aten_mul_bool(self: BOOL, other: BOOL) -> BOOL:
            ...
    ```

    2. Optional dim: caused by unsupported op.OptionalHasElement (will support on opset
        version == 20). dim could be "None"

        ```python
        @torch_op("aten::argmax", trace_only=True)
        def aten_argmax(
            self: TrealOrUInt8, dim: Optional[int] = None, keepdim: bool = False
        ) -> TrealOrUInt8:
            ...

        @torch_op("aten::argmax", private=True)
        def _aten_argmax_dim(self: TrealOrUInt8, dim: int, keepdim: bool = False) -> TrealOrUInt8:
            ...
        ```

        This case is impossible to differentiate, as they both might have dim in kwargs, so
        in this case, please make sure you turn the one with `dim: int` to private function.

    3. Optional dtype: dtype could be "unprovided". The difference from 2 is that dtype
        would not be None.

        ```python
        @torch_op("aten::new_full")
        def aten_new_full(self: TTensor, size: INT64, fill_value: TTensor) -> TTensor:
            ...

        @torch_op("aten::new_full")
        def aten_new_full_dtype(self: TTensor, size: INT64, fill_value: TTensor, dtype: int) -> TTensor:
            ...
        ```

        Depends on dtype is provided or not, matching system will dispatch the ATen op to
        the correct one.

    Attributes:
        schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.

    """

    def __init__(self, op_schema: onnx.defs.OpSchema):
        """Initialize the OpSchemaWrapper.

        Args:
            op_schema: The ONNX OpSchema.
        """
        self.schema = op_schema
        self.type_constraints = {
            # "T": {"tensor(int64)"}
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in op_schema.type_constraints
        }
        # FIXME(titaiwang): Need AttributeProto to support get default_value.
        # TODO(titaiwang): attribut type is not checked.
        self.attributes = set(op_schema.attributes)
        self._matching_score: int = 0

    @property
    def match_score(self) -> int:
        """The matching score of the OpSchemaWrapper.

        Returns:
            The matching score of the OpSchemaWrapper.
        """
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(
        self,
        args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        kwargs: Dict[str, _type_utils.Argument],
    ) -> bool:
        """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        # TODO(titaiwang): Currently the functions in torchlib are manully annotated,
        # so there are quite a few functions that wrongly annotated or strctly annotated.
        # The matching system relax the match while we fix them in the future.
        self._record_matching_score(args, kwargs)

        # TODO: Refine the logic for it to be more robust
        # TODO: Handle attributes
        if len(args) != len(self.schema.inputs):
            return False
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types isn't in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are not compatible
                return False
        if set(kwargs) != self.attributes:
            # If the attributes of the OpSchema and the kwargs don't match,
            # we know the function and the input are not compatible
            return False
        return True

    @_beartype.beartype
    def _record_matching_score(
        self,
        args: Sequence[Optional[Union[_TensorLike, str, int, float, bool, list]]],
        kwargs: Dict[str, _type_utils.Argument],
    ):
        """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        How the matchsing score is calculated:
        1. score += 1 if one input type is in the type constraints.
        2. score -= 1 if one kwarg is not symmetrically the same.

        Limitations:
        1. An Overload is punished if it doesn't have `default` attributes.
        2. None/NoeType/[] could result in zero matches, and the same score of overloads,
            which will be recorded in SARIF.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """

        # If they have different length of arguments, the score would be lower to those
        # functions which have the same length of arguments.
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                # If torch_input_compatible_types is in allowed_types
                # of this input defined in the OpSchema, we know the function
                # and the input are compatible
                self._matching_score += 1
        # The penalty is applied to those functions which have different attributes.
        diff = self.attributes.symmetric_difference(set(kwargs))
        self._matching_score -= len(diff)


@_beartype.beartype
def _find_onnx_data_type(
    torch_input: Optional[Union[_TensorLike, str, int, float, bool, list, tuple]]
) -> Set[str]:
    """Convert inputs data type from torch acceptable dtype to the compatible onnx dtype string."""
    if isinstance(torch_input, _TensorLike) and torch_input.dtype is not None:
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            torch_input.dtype
        ]
    if isinstance(torch_input, (int, float, bool, str)):
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            type(torch_input)
        ]
    if isinstance(torch_input, (list, tuple)) and torch_input:  # [Tensor, Tensor]
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any(isinstance(input, _TensorLike) for input in torch_input):
            # NOTE: Any Tensor involved in a list would make it a seq(tensor(onnx_type))
            return {f"seq({dtype})" for dtype in set_dtype}
        else:
            # constant list of non-tensor type
            return set_dtype
    if (
        torch_input is None
        or (isinstance(torch_input, _TensorLike) and torch_input.dtype is None)
        or (isinstance(torch_input, (list, tuple)) and not torch_input)
    ):
        # NOTE: None, No dtype, and empty list are edge cases, we allow it to be any type to relax the type check
        # seq(tensor) also goes to here, as it is not supported in torchscript, and it would be None in this case.
        return set()

    raise RuntimeError(f"Unknown input type from input: {torch_input}")
