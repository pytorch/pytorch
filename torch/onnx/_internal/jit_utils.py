"""Utilities for manipulating the torch.Graph object and the torchscript."""

# TODO(justinchuby): Move more of the symbolic helper functions here and expose
# them to the user.

import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype

_ATTR_PATTERN = re.compile("^(.+)_(([ifstgz])|(ty))$")
_SKIP_NODE_ATTRIBUTES = {"inplace", "aten"}


@dataclasses.dataclass
class GraphContext:
    """Extra context for symbolic functions with all methods from torch.Graph.

    NOTE: This class is not meant for external consumption. Please do not depend on
    it outside of torch.onnx as the interface may evolve.

    Attributes:
        graph: The _C.Graph being constructed.
        block: The current _C.Block being constructed.
        opset: The opset version.
        original_node: Current node that is being converted from.
        params_dict: Mapping from graph initializer name to IValue.
        env: Mapping from Torch domain graph Value to ONNX domain graph Value.
    """

    graph: _C.Graph
    block: _C.Block
    opset: int
    original_node: _C.Node
    params_dict: Dict[str, "_C.IValue"]
    env: Dict[_C.Value, _C.Value]

    # Relay methods from _C.Graph for compatibility with symbolic functions that expect
    # a _C.Graph
    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    @_beartype.beartype
    def op(
        self,
        opname: str,
        *raw_args: Union[torch.Tensor, _C.Value],
        outputs: int = 1,
        **kwargs,
    ):
        """Creates an ONNX operator "opname", taking "raw_args" as inputs and "kwargs" as attributes.

        The set of operators and the inputs/attributes they take
        is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

        Args:
            opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
                with a namespace, e.g., `aten::add`.
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        # FIXME(justinchuby): Add the return type back once we know how to handle mypy
        return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)

    @_beartype.beartype
    def aten_op(self, operator: str, *args, overload_name: str = "", **kwargs):
        """Generates an ONNX ATen op node.

        This function is for backward compatibility with the old symbolic functions.
        """
        return self.op(
            "aten::ATen",
            *args,
            operator_s=operator,
            overload_name_s=overload_name,
            **kwargs,
        )


@_beartype.beartype
def add_op_with_blocks(
    graph_context: GraphContext,
    opname: str,
    *inputs: _C.Value,
    outputs: int = 1,
    n_blocks: int = 1,
    **attributes,
) -> Tuple[Any, Tuple[GraphContext, ...], _C.Node]:
    """Creates an ONNX operator "opname", taking inputs and attributes.

    Args:
        graph_context: The context for the current graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        inputs: The inputs to the operator.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        n_blocks: The number of sub-blocks to create in the node.
        attributes: The attributes of the ONNX operator.

    Returns:
        A tuple of (output_values, new_contexts, node) where:
            output_values: ONe or more output value of this operator
                (see the `outputs` keyword argument for multi-return nodes).
            new_contexts: A tuple of new graph contexts for each sub-block.
            node: The node representing the operator.
    """

    output_values = graph_context.op(opname, *inputs, outputs=outputs, **attributes)
    if isinstance(output_values, Sequence):
        node = output_values[0].node()
    else:
        node = output_values.node()

    new_contexts = []
    for _ in range(n_blocks):
        new_block = node.addBlock()
        # Create shallow copy of the graph context and update the block
        new_context = dataclasses.replace(graph_context, block=new_block)
        new_contexts.append(new_context)

    return output_values, tuple(new_contexts), node


@_beartype.beartype
def _add_op(
    graph_context: GraphContext,
    opname: str,
    *args: Union[torch.Tensor, _C.Value],
    outputs: int = 1,
    **kwargs,
):
    """Creates an ONNX operator "opname", taking "args" as inputs and attributes "kwargs".

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Args:
        g: The Torch Graph or Block.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        kwargs: The attributes of the ONNX operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).

    Returns:
        (Union[_C.Value, Tuple[_C.Value, ...]])
        The value representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    inputs = [_const_if_tensor(graph_context, arg) for arg in args]
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in kwargs.items() if v is not None}

    if "::" not in opname:
        opname = "onnx::" + opname

    node = _create_node(
        graph_context.block,
        opname,
        inputs,
        attributes,
        params_dict=graph_context.params_dict,
        opset_version=graph_context.opset,
        n_outputs=outputs,
        shape_inference=GLOBALS.onnx_shape_inference,
    )

    if outputs == 1:
        return node.output()
    return tuple(node.outputs())


@_beartype.beartype
def _const_if_tensor(graph_context: GraphContext, arg):
    if arg is None:
        return arg
    if isinstance(arg, _C.Value):
        return arg

    return _add_op(graph_context, "onnx::Constant", value_z=arg)


def _create_node(
    graph_or_block: Union[_C.Graph, _C.Block],
    domain_op: str,
    inputs: Sequence,
    attributes: dict,
    params_dict: dict,
    opset_version: int,
    n_outputs: int,
    shape_inference: bool = True,
) -> _C.Node:
    """Creates an node 'domain_op', taking inputs and attributes."""
    if isinstance(graph_or_block, _C.Graph):
        graph = graph_or_block
        node = graph.create(domain_op, inputs, n_outputs)
        node = graph.insertNode(node)
    elif isinstance(graph_or_block, _C.Block):
        block = graph_or_block
        node = block.addNode(domain_op, inputs)

        # Block does not have create defined, so we need to add outputs manually
        if n_outputs > 1:
            for _ in range(1, n_outputs):
                node.addOutput()

    node_ouputs = tuple(node.outputs())
    assert len(node_ouputs) == n_outputs

    aten = domain_op.startswith("aten::")

    # Add all attributes
    for key, value in sorted(attributes.items()):
        if key in _SKIP_NODE_ATTRIBUTES:
            continue
        _add_attribute(node, key, value, aten=aten)
    if shape_inference:
        _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    return node


@_beartype.beartype
def _is_onnx_list(value):
    return (
        not isinstance(value, torch._six.string_classes)
        and not isinstance(value, torch.Tensor)
        and isinstance(value, Iterable)
    )


@_beartype.beartype
def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


@_beartype.beartype
def _is_caffe2_aten_fallback() -> bool:
    return (
        GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    )


@_beartype.beartype
def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    m = _ATTR_PATTERN.match(key)
    if m is None:
        raise ValueError(
            f"Invalid attribute specifier '{key}' names "
            "must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
        )
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"

    if aten and _is_caffe2_aten_fallback():
        if isinstance(value, torch.Tensor):
            # Caffe2 proto does not support tensor attribute.
            if value.numel() > 1:
                raise ValueError("Should not pass tensor attribute")
            value = _scalar(value)
            if isinstance(value, float):
                kind = "f"
            else:
                kind = "i"
    return getattr(node, f"{kind}_")(name, value)


# TODO: Expose this to user when migrating symbolic helper functions to here.
@_beartype.beartype
def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


@_beartype.beartype
def get_device_from_value(value: _C.Value) -> Optional[torch.device]:
    if not _is_tensor(value):
        return None
    tensor_type = typing.cast(_C.TensorType, value.type())
    return tensor_type.device()
