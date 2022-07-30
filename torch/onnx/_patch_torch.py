"""Importing this patches torch._C classes to add ONNX conveniences."""
import re
from typing import Any, Iterable, Tuple, Union

import torch
from torch import _C
import torch._C._onnx as _C_onnx
from torch.onnx._globals import GLOBALS


# TODO(#78694): Refactor the patching process to make it more transparent to users.
def _graph_op(
    g: torch._C.Graph,
    opname: str,
    *raw_args: torch._C.Value,
    outputs: int = 1,
    **kwargs,
) -> Union[torch._C.Value, Tuple[torch._C.Value, ...]]:
    r"""Creates an ONNX operator "opname", taking "args" as inputs and attributes "kwargs".

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Args:
        g: The Torch graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`. TODO(justinchu): Update examples to correct ones.
        raw_args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
        kwargs: The attributes of the ONNX operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).

    Returns:
        The node representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def const_if_tensor(arg):
        if arg is None:
            return arg
        elif isinstance(arg, torch._C.Value):
            return arg
        else:
            return g.op("Constant", value_z=arg)  # type: ignore[attr-defined]

    args = [const_if_tensor(arg) for arg in raw_args]
    n = g.insertNode(_new_node(g, opname, outputs, *args, **kwargs))  # type: ignore[attr-defined]

    # Import utils to get _params_dict because it is a global that is accessed by c++ code
    from torch.onnx import utils

    if GLOBALS.onnx_shape_inference:
        torch._C._jit_pass_onnx_node_shape_type_inference(
            n, utils._params_dict, GLOBALS.export_onnx_opset_version
        )

    if outputs == 1:
        return n.output()
    return tuple(n.outputs())


# Generate an ONNX ATen op node.
def _aten_op(g, operator: str, *args, overload_name: str = "", **kwargs):
    kwargs["aten"] = True
    return g.op(
        "ATen", *args, operator_s=operator, overload_name_s=overload_name, **kwargs
    )


def _block_op(b: _C.Block, opname: str, *args, **kwargs):
    if "::" in opname:
        aten = False
        ns_opname = opname
    else:
        aten = kwargs.pop("aten", False)
        ns = "aten" if aten else "onnx"
        ns_opname = ns + "::" + opname
    n = b.addNode(ns_opname, list(args))
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    if len(list(n.outputs())) == 1:
        return n.output()
    return tuple(o for o in n.outputs())


def _new_node(g: torch._C.Graph, opname: str, outputs, *args, **kwargs):
    if "::" in opname:
        aten = False
        ns_opname = opname
    else:
        aten = kwargs.pop("aten", False)
        ns = "aten" if aten else "onnx"
        ns_opname = ns + "::" + opname
    n = g.create(ns_opname, args, outputs)  # type: ignore[attr-defined]
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    return n


_attr_pattern = re.compile("^(.+)_(([ifstgz])|(ty))$")


def _is_onnx_list(value):
    return (
        not isinstance(value, torch._six.string_classes)
        and not isinstance(value, torch.Tensor)
        and isinstance(value, Iterable)
    )


def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _is_caffe2_aten_fallback():
    return (
        GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    )


def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    m = _attr_pattern.match(key)
    if m is None:
        raise IndexError(
            f"Invalid attribute specifier '{key}' names "
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
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
    return getattr(node, kind + "_")(name, value)


torch._C.Graph.op = _graph_op  # type: ignore[attr-defined]
torch._C.Graph.at = _aten_op  # type: ignore[attr-defined]
torch._C.Block.op = _block_op  # type: ignore[attr-defined]
