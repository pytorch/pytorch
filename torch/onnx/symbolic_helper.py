from __future__ import annotations

import functools
import inspect
import sys
import typing
import warnings
from typing import Any, Callable, List, NoReturn, Optional, Sequence, Set, Tuple, Union

from typing_extensions import Literal

import torch
import torch._C._onnx as _C_onnx
from torch import _C

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
from torch.onnx import (  # noqa: F401
    _constants,
    _deprecation,
    _patch_torch,
    _type_utils,
    errors,
)
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number

__all__ = [
    "args_have_same_dtype",
    "cast_pytorch_to_onnx",
    "check_training_mode",
    "dequantize_helper",
    "is_caffe2_aten_fallback",
    "is_complex_value",
    "parse_args",
    "pytorch_name_to_type",
    "quantize_helper",
    "quantized_args",
    "requantize_bias_helper",
    "scalar_name_to_pytorch",
    "scalar_type_to_onnx",
    "scalar_type_to_pytorch_type",
]

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

_ValueDescriptor = Literal[
    "v",
    "i",
    "is",
    "f",
    "fs",
    "b",
    "s",
    "t",
    "none",
]


@_beartype.beartype
def _parse_arg(
    value,
    desc: _ValueDescriptor,
    arg_name: Optional[str] = None,
    node_name: Optional[str] = None,
):
    if desc == "none":
        return value
    if desc == "v" or not _is_value(value):
        return value

    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == "onnx::Constant":
        node_val = _node_get(node, "value")
        if desc == "i":
            return int(node_val)
        elif desc == "f":
            return float(node_val)
        elif desc == "b":
            return bool(node_val)
        elif desc == "s":
            return str(node_val)
        elif desc == "t":
            return node_val
        elif desc == "is":
            return [int(v) for v in node_val]
        elif desc == "fs":
            return [float(v) for v in node_val]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not understand the Constant node '{node}' "
                f"specified with descriptor '{desc}'.",
                value,
            )
    elif node.kind() == "prim::ListConstruct":
        if desc == "is":
            for v in node.inputs():
                element_node = v.node()
                if element_node.kind() != "onnx::Constant":
                    raise errors.SymbolicValueError(
                        f"Failed to export a node '{element_node}' "
                        f"(in list node {node}) "
                        f"because it is not constant. "
                        f"Please try to make things (e.g. kernel sizes) static if possible.",
                        value,
                    )
            return [int(_node_get(v.node(), "value")) for v in value.node().inputs()]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not know how to unpack the ListConstruct node that "
                f"is not a list of integers: '{node}'",
                value,
            )

    if arg_name is None or node_name is None:
        raise errors.SymbolicValueError(
            f"Expected node type 'onnx::Constant', got '{node.kind()}'.",
            value,
        )

    raise errors.SymbolicValueError(
        "Expected node type 'onnx::Constant' "
        f"for argument '{arg_name}' of node '{node_name}', got '{node.kind()}'.",
        value,
    )


@_beartype.beartype
def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    assert isinstance(node, _C.Node)
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


@_beartype.beartype
def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
    return value.node().kind() == "onnx::Constant"


@_beartype.beartype
def _maybe_get_const(
    value: Optional[Union[_C.Value, torch.Tensor, Number, Sequence]],
    descriptor: _ValueDescriptor,
):
    # NOTE: prim::Constant at this stage usually means something not compatible in ONNX,
    # otherwise it'd be converted to onnx::Constant
    # TODO(justinchuby): Replace insinstance with _is_value once we figure out mypy
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value


@_beartype.beartype
def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, "t")
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


@_beartype.beartype
def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected a constant value of the '{arg_name}' argument, "
            f"got '{value}'",
            value,
        )
    return _parse_arg(value, desc)


@_beartype.beartype
def _unpack_list(list_value: _C.Value) -> List[_C.Value]:
    list_node = list_value.node()
    if list_node.kind() != "prim::ListConstruct":
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type prim::ListConstruct, "
            f"got '{list_node}'.",
            list_value,
        )
    return list(list_node.inputs())


@_beartype.beartype
def _unpack_tuple(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type 'prim::TupleConstruct', "
            f"got '{tuple_node.kind()}'.",
            tuple_value,
        )
    return tuple(tuple_node.inputs())


@_beartype.beartype
def _unpack_quantized_tensor(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    """Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.
    Args:
        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.
    Returns:
        A tuple of tensor, scale, zero_point, and optionally axis.
    """
    tuple_node = tuple_value.node()
    # A quantized tensor is represented as tuple of the form (tensor, scale, zero_point, <axis>)
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected the output of `{tuple_node}` to be a quantized "
            f"tensor. Is this likely due to missing support for quantized "
            f"`{tuple_node.kind()}`. Please create an issue on {_constants.PYTORCH_GITHUB_ISSUES_URL}",
            tuple_value,
        )
    unpacked = tuple(tuple_node.inputs())
    assert len(unpacked) == 3 or len(unpacked) == 4
    return unpacked


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
@_beartype.beartype
def _is_packed_list(list_value: _C.Value) -> bool:
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


@_beartype.beartype
def parse_args(*arg_descriptors: _ValueDescriptor):
    """A decorator which converts args from torch._C.Value to built-in types.

    For example:

    ```
    @parse_args('v', 'i', 'fs')
    foo(g, a, b, c):
        assert isinstance(a, torch._C.Value)
        assert isinstance(b, int)
        assert isinstance(c, list)
        assert isinstance(c[0], float)
    ```

    Args:
        arg_descriptors: list of str, where each element is
            a string that specifies the type to convert to. Valid descriptors:
            "v": no conversion, keep torch._C.Value.
            "i": int
            "is": list of int
            "f": float
            "fs": list of float
            "b": bool
            "s": str
            "t": torch.Tensor
            "none": the variable is unused
    """

    def decorator(fn):
        fn._arg_descriptors = arg_descriptors

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            # some args may be optional, so the length may be smaller
            FILE_BUG_MSG = (
                "If you believe this is not due to custom symbolic implementation within your code or "
                "an external library, please file an issue at "
                "https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug."
            )
            assert len(arg_descriptors) >= len(args), (
                f"A mismatch between the number of arguments ({len(args)}) and "
                f"their descriptors ({len(arg_descriptors)}) was found at symbolic function '{fn.__name__}'. "
                f"{FILE_BUG_MSG}"
            )

            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                # FIXME(justinchuby): Avoid catching Exception.
                # Catch a more specific exception instead.
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[assignment]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # only support _outputs in kwargs
            assert len(kwargs) <= 1, (
                f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single "
                f"key/value entry. "
                f"{FILE_BUG_MSG}"
            )

            if len(kwargs) == 1:
                assert "_outputs" in kwargs, (
                    f"Symbolic function {fn.__name__}'s '**kwargs' can only contain "
                    f"'_outputs' key at '**kwargs'. "
                    f"{FILE_BUG_MSG}"
                )
            return fn(g, *args, **kwargs)

        return wrapper

    return decorator


@_beartype.beartype
def quantized_args(
    *arg_q_descriptors: bool,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
):
    """A decorator which extends support for quantized version of the base operator.
    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            nonlocal scale
            nonlocal zero_point
            if scale is not None:
                _scale = g.op("Constant", value_t=torch.tensor(scale))
            else:
                _scale = None
            if zero_point is not None:
                _zero_point = g.op("Constant", value_t=torch.tensor(zero_point))
            else:
                _zero_point = None

            # Support variable length arguments by marking unspecified ones as non-quantized
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (
                len(args) - len(arg_q_descriptors)
            )
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))

            # Run regular symbolic function if none of the argument is QTensor.
            if not any(
                (descriptor and _is_value(arg) and _is_tuple_construct(arg))
                for descriptor, arg in descriptor_args
            ):
                return fn(g, *args, **kwargs)

            # Dequantize arguments that are quantized
            non_quantized_args = []
            for descriptor, arg in descriptor_args:
                if descriptor and _is_value(arg) and _is_tuple_construct(arg):
                    # Quantized arg is a tuple of (value, scale, zero_point)
                    dequantized_arg, arg_scale, arg_zero_point, _ = dequantize_helper(
                        g, arg
                    )
                    non_quantized_args.append(dequantized_arg)
                    # Set scale and zero_point to the first quantized input if not already set
                    if _scale is None:
                        _scale = arg_scale
                    if _zero_point is None:
                        _zero_point = arg_zero_point
                else:
                    # Non-quantized arg
                    non_quantized_args.append(arg)
            # TODO(justinchuby): Only single output is supported for now. We may want to
            # support multiple outputs in the future.
            output = fn(g, *non_quantized_args, **kwargs)

            assert _scale is not None, "Bug: Scale must be set for quantized operator"
            assert (
                _zero_point is not None
            ), "Bug: Zero point must be set for quantized operator"

            return quantize_helper(g, output, _scale, _zero_point)

        return wrapper

    return decorator


@_beartype.beartype
def _scalar(x: Any) -> Optional[Number]:
    """Convert a scalar tensor into a Python value."""
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    return None


@_beartype.beartype
def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, _C.Value):
        return self

    scalar_type = _type_utils.JitScalarType.from_value(
        tensor, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        ty = scalar_type.scalar_name().lower()
        return getattr(self, ty)()
    return self


@_beartype.beartype
def _is_none(x: _C.Value) -> bool:
    return x.node().mustBeNone()


@_beartype.beartype
def _is_value(x: Any) -> bool:
    return isinstance(x, _C.Value)


@_beartype.beartype
def _is_constant(value: Any) -> bool:
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


@_beartype.beartype
def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


# Note: _C.JitType is not exposed to Python and cannot be checked in runtime.
def _as_list_type(jit_type: _C.JitType) -> Optional[_C.ListType]:
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None


@_beartype.beartype
def _is_list(x: _C.Value) -> bool:
    return _as_list_type(x.type()) is not None


@_beartype.beartype
def _is_tensor_list(x: _C.Value) -> bool:
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    return isinstance(x_type.getElementType(), _C.TensorType)


@_beartype.beartype
def _is_scalar_list(x: _C.Value) -> bool:
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    scalar_type = _type_utils.JitScalarType.from_value(x)
    return scalar_type.onnx_compatible()


@_beartype.beartype
def _is_tuple_construct(x: _C.Value) -> bool:
    return x.node().kind() == "prim::TupleConstruct"


@_beartype.beartype
def is_complex_value(x: _C.Value) -> bool:
    assert _is_value(x)
    return _type_utils.JitScalarType.from_value(
        x, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.COMPLEX32,
        _type_utils.JitScalarType.COMPLEX64,
        _type_utils.JitScalarType.COMPLEX128,
    }


@_beartype.beartype
def is_caffe2_aten_fallback() -> bool:
    return (
        GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    )


@_beartype.beartype
def _get_tensor_rank(x: _C.Value) -> Optional[int]:
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()


@_beartype.beartype
def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool = True):
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    if allow_nonstatic:
        # Each individual symbol is returned as None.
        # e.g. [1, "a", "b"] -> [1, None, None]
        return x_type.varyingSizes()
    # returns None, if exists any symbol in sizes.
    # e.g. [1, "a", "b"] -> None
    return x_type.sizes()


@_beartype.beartype
def _get_tensor_dim_size(x: _C.Value, dim: int) -> Optional[int]:
    sizes = _get_tensor_sizes(x)
    return sizes[dim] if sizes else None


@_beartype.beartype
def _get_dim_for_cross(x: _C.Value, dim: Optional[int]):
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        assert tensor_rank is not None
        return dim + tensor_rank
    # If dim is not given, it defaults to the first dimension found with the size 3
    if dim is None:
        sizes = _get_tensor_sizes(x)
        assert sizes is not None
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim


@_beartype.beartype
def _unimplemented(op: str, msg: str, value: Optional[_C.Value] = None) -> None:
    # For BC reasons, the behavior for Caffe2 does not raise exception for unimplemented operators
    if _C_onnx._CAFFE2_ATEN_FALLBACK:
        warnings.warn(f"ONNX export failed on {op} because {msg} not supported")
    elif GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}", value)


@_beartype.beartype
def _onnx_unsupported(op_name: str, value: Optional[_C.Value] = None) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of operator {op_name}. "
        f"Please feel free to request support or submit a pull request "
        f"on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}"
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


@_beartype.beartype
def _onnx_opset_unsupported(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    value: Optional[_C.Value] = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in opset {current_opset}. "
        f"Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


@_beartype.beartype
def _onnx_opset_unsupported_detailed(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    reason: str,
    value: Optional[_C.Value] = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in "
        f"opset {current_opset}. {reason}. Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


@_beartype.beartype
def _block_list_in_opset(name: str):
    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(
            f"ONNX export failed on {name}, which is not implemented for opset "
            f"{GLOBALS.export_onnx_opset_version}. "
            "Try exporting with other opset versions."
        )

    return symbolic_fn


@_beartype.beartype
def _try_get_scalar_type(*args) -> Optional[_type_utils.JitScalarType]:
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(
            arg, _type_utils.JitScalarType.UNDEFINED
        )
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None


@_beartype.beartype
def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape=True):
    index_const = _maybe_get_scalar(index)
    index_dim = _get_tensor_rank(index)
    if not _is_value(index_const):
        # Index is a constant scalar. Make it a size 1 constant tensor.
        index = g.op("Constant", value_t=torch.LongTensor([index_const]))
    elif index_dim is not None and apply_reshape:
        if index_dim == 0:
            # Index is a scalar. Reshape it to a size 1 tensor.
            index = _reshape_helper(
                g, index, g.op("Constant", value_t=torch.LongTensor([1]))
            )

    index_scalar_type = _type_utils.JitScalarType.from_value(
        index, _type_utils.JitScalarType.UNDEFINED
    )
    if index_scalar_type not in {
        _type_utils.JitScalarType.INT64,
        _type_utils.JitScalarType.INT,
    }:
        index = g.op("Cast", index, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op("Gather", self, index, axis_i=dim)


@_beartype.beartype
def _slice_helper(
    g: jit_utils.GraphContext,
    input,
    axes,
    starts,
    ends,
    steps=None,
    dynamic_slice=False,
):
    if g.opset <= 9:
        from torch.onnx.symbolic_opset9 import _slice as _slice9

        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx.symbolic_opset10 import _slice as _slice10

        return _slice10(g, input, axes, starts, ends, steps, dynamic_slice)


@_beartype.beartype
def _is_fp(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    }


@_beartype.beartype
def _is_bool(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {_type_utils.JitScalarType.BOOL}


@_beartype.beartype
def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    """Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    """
    assert not isinstance(scalar, torch.Tensor)
    if isinstance(scalar, float):
        return g.op("Constant", value_t=torch.tensor(scalar, dtype=torch.double))
    return g.op("Constant", value_t=torch.tensor(scalar))


@_beartype.beartype
def _sort_helper(g: jit_utils.GraphContext, input, dim, decending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op(
        "Gather",
        shape_,
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
    )
    if g.opset <= 10:
        if not decending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=decending, outputs=2
        )


@_beartype.beartype
def _topk_helper(
    g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None
):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op("Constant", value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op("Cast", k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if g.opset <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2
        )


@_beartype.beartype
def _lt_helper(g: jit_utils.GraphContext, input, other):
    if g.opset <= 8:
        from torch.onnx.symbolic_opset8 import lt as _lt8

        return _lt8(g, input, other)
    else:
        from torch.onnx.symbolic_opset9 import lt as _lt9

        return _lt9(g, input, other)


@_beartype.beartype
def _interpolate_warning(interpolate_mode):
    onnx_op = (
        "onnx:Resize" if GLOBALS.export_onnx_opset_version >= 10 else "onnx:Upsample"
    )
    warnings.warn(
        "You are trying to export the model with "
        + onnx_op
        + " for ONNX opset version "
        "" + str(GLOBALS.export_onnx_opset_version) + ". "
        "This operator might cause results to not match the expected results by PyTorch.\n"
        "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
        "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
        "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
        "We recommend using opset 11 and above for models using this operator."
    )


@_beartype.beartype
def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Unsqueeze", input, axes)
        return g.op("Unsqueeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Unsqueeze with dynamic axes.", input
        )
    return g.op("Unsqueeze", input, axes_i[0])


@_beartype.beartype
def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Squeeze", input, axes)
        return g.op("Squeeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Squeeze with dynamic axes.", input
        )
    axes_t = axes_i[0]
    axes_rank = _get_tensor_rank(axes_t)
    assert axes_rank is not None
    if axes_rank > 1:
        raise errors.SymbolicValueError(
            "For Squeeze axses as input, the axes rank must be one in ONNX spec.", input
        )
    elif axes_rank == 0:
        # The axes is a scalar. Unsqueeze it to a rank 1 tensor.
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        return g.op("Squeeze", input, axes_t)
    return g.op("Squeeze", input, axes_t)


@_beartype.beartype
def _reducesum_helper(
    g: jit_utils.GraphContext,
    input,
    axes_i=None,
    keepdims_i=1,
    noop_with_empty_axes_i=0,
):
    keepdims_i = _maybe_get_const(keepdims_i, "i")
    if g.opset >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op(
                    "Constant", value_t=torch.tensor(axes_i, dtype=torch.long)
                )
            return g.op(
                "ReduceSum",
                input,
                axes_i,
                keepdims_i=keepdims_i,
                noop_with_empty_axes_i=noop_with_empty_axes_i,
            )
        return g.op(
            "ReduceSum",
            input,
            keepdims_i=keepdims_i,
            noop_with_empty_axes_i=noop_with_empty_axes_i,
        )
    else:
        return g.op("ReduceSum", input, axes_i=axes_i, keepdims_i=keepdims_i)


@_beartype.beartype
def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    output_size = _maybe_get_const(output_size, "is")
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        divisor = _slice_helper(
            g, g.op("Shape", input), axes=[0], ends=[sys.maxsize], starts=[offset]
        )
        divisor = g.op("Cast", divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scale_dims = g.op("Div", dividend, divisor)
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [
            1.0
            if i < 2
            else float(output_size[-(dim - i)])
            / float(input.type().sizes()[-(dim - i)])
            for i in range(0, dim)
        ]
        scales = g.op(
            "Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32)
        )
    return scales


@_beartype.beartype
def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    available_scales = _maybe_get_const(scales[0], "fs") != -1 and not _is_none(
        scales[0]
    )

    if not available_scales:
        return None

    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op(
        "Constant", value_t=torch.tensor(_maybe_get_const(scales[0], "fs"))
    )
    scales = g.op("Concat", offsets, scales_list, axis_i=0)
    return scales


@_beartype.beartype
def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    if mode == "nearest":
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


@_beartype.beartype
def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (
        scale_factor_rank is not None and scale_factor_rank > 0
    ):
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op(
            "Cast", scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT
        )
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


@_beartype.beartype
def _interpolate_get_scales_and_mode(
    g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    _interpolate_warning(mode)

    align_corners = _maybe_get_const(align_corners, "b")
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    if not input.type().dim():
        return _unimplemented("interpolate", "missing input shape")
    dim = input.type().dim()

    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, "t").dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented(
            "interpolate", "Both size and scales are None in __interpolate"
        )
    return scale_factor, mode


@_beartype.beartype
def _argmin_argmax_helper(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
    op_name: str,
):
    def op_wrapper(input, axis_i, keepdims_i):
        if g.opset >= 12:
            return g.op(
                op_name,
                input,
                axis_i=axis_i,
                keepdims_i=keepdims_i,
                select_last_index_i=False,
            )
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)

    if _is_none(dim):
        flattened = _reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = g.op("Shape", input)
            input_shape_shape = g.op("Shape", input_shape)
            new_shape = g.op(
                "ConstantOfShape",
                input_shape_shape,
                value_t=torch.tensor([1], dtype=torch.int64),
            )
            output = g.op("Reshape", output, new_shape)
        return output

    dim = _parse_arg(dim, "i")
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)


@_beartype.beartype
def _interpolate_helper(name, dim, interpolate_mode):
    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = (
            "asymmetric"
            if interpolate_mode == "nearest"
            else "align_corners"
            if align_corners
            else "half_pixel"
        )

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            output_size = g.op(
                "Cast", output_size, to_i=_C_onnx.TensorProtoDataType.INT64
            )
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )
                empty_scales = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"

    return symbolic_fn


@_beartype.beartype
def __interpolate_helper(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    align_corners = _maybe_get_const(align_corners, "b")
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = (
        "asymmetric"
        if mode == "nearest"
        else "align_corners"
        if align_corners
        else "half_pixel"
    )

    if not _is_none(size):
        input_size = g.op("Shape", input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        # in some cases size is not a packed list but size is a scalar
        # We need to also verify that (_maybe_get_const(size, "t").dim() == 0)
        # but this information is not always available. Try to get the dim,
        # and if not assume that it is not a scalar.
        try:
            is_scalar = not _is_packed_list(size) and (
                _maybe_get_const(size, "t").dim() == 0
            )
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn(
                    "Cannot verify if the output_size is a scalar "
                    "while exporting interpolate. Assuming that it is not a scalar."
                )

        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented(
                    "interpolate (with a scalar output_size)",
                    "missing input shape (try giving an array of output_size values)",
                )
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op("Concat", *size, axis_i=0)
        size = g.op("Cast", size, to_i=_C_onnx.TensorProtoDataType.INT64)
        size = g.op("Concat", input_size, size, axis_i=0)

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op(
                "Constant", value_t=torch.tensor([], dtype=torch.float32)
            )

        return g.op(
            "Resize",
            input,
            empty_roi,
            empty_scales,
            size,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )
    else:  # if not _is_none(scales)
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented("interpolate (with scales)", "missing input shape")

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op(
            "Resize",
            input,
            empty_roi,
            scales,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )  # only valid when mode="nearest"


@_beartype.beartype
def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if g.opset < 11:
        from torch.onnx.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx.symbolic_opset11 import unbind  # type: ignore[no-redef]
    else:
        from torch.onnx.symbolic_opset13 import unbind  # type: ignore[no-redef]
    return unbind(g, self, dim, _outputs)


@_beartype.beartype
def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore[no-redef]
    return scatter(g, self, dim, index, src)


@_beartype.beartype
def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if g.opset <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx.symbolic_opset13 import split

        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]


@_beartype.beartype
def _arange_cast_helper(
    g: jit_utils.GraphContext, end, start=None, step=None, dtype=None
) -> Tuple[
    _type_utils.JitScalarType,
    Optional[_C.Value],
    Optional[_C.Value],
    Optional[_C.Value],
]:
    def _is_all_integral(scalars):
        for scalar in scalars:
            scalar_type = _type_utils.JitScalarType.from_value(
                scalar, _type_utils.JitScalarType.UNDEFINED
            )
            if (
                scalar_type != _type_utils.JitScalarType.INT64
                and scalar_type != _type_utils.JitScalarType.UNDEFINED
            ):
                return False
        return True

    # This logic is based on torch.arange docs. If "dtype" is provided,
    # infer input types from dtype. If not, then check if any of start, stop,
    # or step are floating point, and infer the type from get_default.
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            scalar_type = _type_utils.JitScalarType.INT64
        else:
            scalar_type = _type_utils.JitScalarType.from_dtype(
                torch.get_default_dtype()
            )
    else:
        assert isinstance(dtype, int)
        # TODO(justinchuby): Check if dtype is indeed a int.
        scalar_type = _type_utils.JitScalarType(dtype)

    start = g.op("Cast", start, to_i=scalar_type.onnx_type()) if start else None
    end = g.op("Cast", end, to_i=scalar_type.onnx_type()) if end else None
    step = g.op("Cast", step, to_i=scalar_type.onnx_type()) if step else None
    return scalar_type, end, start, step


@_beartype.beartype
def _arange_helper(g: jit_utils.GraphContext, *args):
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import arange
    else:
        from torch.onnx.symbolic_opset11 import arange  # type: ignore[no-redef]
    return arange(g, *args)


@_beartype.beartype
def _size_helper(g: jit_utils.GraphContext, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx.symbolic_opset9 import select

    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


@_beartype.beartype
def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 3. expand value as well.
    # 4. apply onnx::scatter.

    from torch.onnx.symbolic_opset9 import expand

    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore[no-redef]

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accesible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, "i")
    unsqueezed_index = _unsqueeze_helper(
        g, index, [i for i in range(self_dim) if i != dim_value]
    )
    expanded_index_shape = scatter(
        g, g.op("Shape", self), 0, _unsqueeze_helper(g, dim, [0]), g.op("Shape", index)
    )
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return expanded_index_shape, expanded_index


# By default, when any value in the 'shape' input is equal to zero
# the corresponding dimension value is copied from the input tensor dynamically.
# allowzero=1 indicates that if any value in the 'shape' input is set to zero,
# the zero value is honored, similar to NumPy.
# allowzero=1 is only supported for opset version >= 14.
@_beartype.beartype
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0, copy=False):
    shape = _maybe_get_const(shape, "is")
    copy = _maybe_get_const(copy, "b")
    assert not copy, repr(copy)
    if not _is_value(shape):
        shape = g.op("Constant", value_t=torch.LongTensor(shape))
    if g.opset <= 13:
        if allowzero == 1:
            _onnx_opset_unsupported(
                "Reshape with allowzero=1", GLOBALS.export_onnx_opset_version, 14, input
            )
        return g.op("Reshape", input, shape)
    else:
        return g.op("Reshape", input, shape, allowzero_i=allowzero)


@_beartype.beartype
def _batchnorm_helper(
    g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var
):
    from torch.onnx.symbolic_opset9 import _var_mean

    batch_size = _get_tensor_dim_size(input, 0)
    channel_size = _get_tensor_dim_size(input, 1)

    if weight is None or _is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or _is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)
    # If track_running_stats is set to False batch statistics are instead used during evaluation time
    if (
        running_mean is None
        or _is_none(running_mean)
        or running_var is None
        or _is_none(running_var)
    ):
        assert batch_size is not None and channel_size is not None
        reshape_in = _reshape_helper(
            g,
            input,
            g.op(
                "Constant",
                value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64),
            ),
        )
        trans_in = g.op("Transpose", reshape_in, perm_i=[0, 2, 1])
        running_var, running_mean = _var_mean(
            g,
            trans_in,
            g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64)),
            False,
            False,
        )
    return weight, bias, running_mean, running_var


@_beartype.beartype
def _avgpool_helper(
    tuple_fn: Callable[[Any], Sequence[int]],
    padding: Union[int, Sequence[int]],
    kernel_size,
    stride,
    divisor_override,
    name,
) -> Tuple[int, ...]:
    if divisor_override and divisor_override.node().kind() != "prim::Constant":
        _unimplemented(name, "divisor_override")
    return tuple(tuple_fn(padding))


@_beartype.beartype
def check_training_mode(op_train_mode: int, op_name: str) -> None:
    """Warns the user if the model's training mode and the export mode do not agree."""
    if GLOBALS.training_mode == _C_onnx.TrainingMode.PRESERVE:
        return

    if op_train_mode:
        op_mode_enum = _C_onnx.TrainingMode.TRAINING
    else:
        op_mode_enum = _C_onnx.TrainingMode.EVAL
    if op_mode_enum == GLOBALS.training_mode:
        # The modes agree. Do nothing
        return

    op_mode_text = f"train={bool(op_train_mode)}"
    # Setting the model mode could result in op_mode != GLOBALS.training_mode
    # if the model is a FuncModule. In this case we warn the user of
    # the state and export depending on op_mode
    # This is to support use-cases of fixing certain layer weights
    # in training.
    warnings.warn(
        f"ONNX export mode is set to {GLOBALS.training_mode}, but operator '{op_name}' "
        f"is set to {op_mode_text}. Exporting with {op_mode_text}."
    )


@_beartype.beartype
def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim):
    input_size = g.op("Shape", input)
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))]
    if end_dim < dim - 1:
        slice3 = _slice_helper(
            g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim]
        )
        slices = [
            slice1,
            g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
            slice3,
        ]

    final_shape = g.op("Concat", *slices, axis_i=0)
    from torch.onnx.symbolic_opset9 import _reshape_from_tensor

    return _reshape_from_tensor(g, input, final_shape)


@_beartype.beartype
def _is_split_static(split_size_or_sizes, _outputs):
    if _outputs is None:
        return False
    if (
        _is_value(split_size_or_sizes)
        and split_size_or_sizes.node().kind() != "onnx::Constant"
    ):
        return False
    return True


@_beartype.beartype
def _optional_input_placeholder_tensor(g):
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


@_beartype.beartype
def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    rank = _get_tensor_rank(self)
    if rank is not None and any(
        [_get_tensor_dim_size(self, i) == 0 for i in range(rank)]
    ):
        # If input tensor is empty, according to ONNX ReduceSum definition,
        # set keepdims=1 so that the resulted tensor has the same rank as the input.
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)


@_beartype.beartype
def dequantize_helper(
    g: jit_utils.GraphContext,
    qtensor: _C.Value,
    qdtype: Optional[_C_onnx.TensorProtoDataType] = None,
) -> Tuple[_C.Value, _C.Value, _C.Value, Optional[_C.Value]]:
    """Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)
            for per tensor quantization, or
            (quantized_tensor, scale, zero_point, axis) for per channel quantization,
            representing the quantized tensor.
        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the
            data type of quantized tensor. It must be either
            torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.
    """
    unpacked_qtensors = _unpack_quantized_tensor(qtensor)
    tensor, scale, zero_point = unpacked_qtensors[:3]
    axis = unpacked_qtensors[3] if len(unpacked_qtensors) >= 4 else None
    axis_i = _get_const(axis, "i", "axis")
    input_qdtype = _type_utils.JitScalarType.from_value(tensor)
    if qdtype is None:
        if input_qdtype is not None:
            qdtype = input_qdtype.onnx_type()
        else:
            qdtype = _C_onnx.TensorProtoDataType.UINT8
    value = g.op("Cast", tensor, to_i=qdtype)
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    zero_point = g.op("Cast", zero_point, to_i=qdtype)

    if axis_i is not None and GLOBALS.export_onnx_opset_version < 13:
        _onnx_opset_unsupported_detailed(
            "DequantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            qtensor,
        )

    return (
        g.op("DequantizeLinear", value, scale, zero_point, axis_i=axis_i),
        scale,
        zero_point,
        axis,
    )


@_beartype.beartype
def quantize_helper(
    g: jit_utils.GraphContext,
    tensor: _C.Value,
    scale: _C.Value,
    zero_point: _C.Value,
    axis: Optional[_C.Value] = None,
) -> _C.Value:
    """Appends to graph `g` ONNX nodes that quantizes `tensor` based on `scale`, `zero_point` and `axis`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        tensor: torch._C.Value, representing the tensor to be quantized.
        scale: torch._C.Value, quantized scale.
        zero_point: torch._C.Value, quantized zero point.
        axis: Optional[torch._C.Value] default None, if None, represents per tensor quantization.
            Otherwise, represents per channel quantization, along given axis.

    Returns:
        A TupleConstruct storing information of the quantized tensor.
    """
    if (
        axis is not None
        and not _is_none(axis)
        and GLOBALS.export_onnx_opset_version < 13
    ):
        _onnx_opset_unsupported_detailed(
            "QuantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            tensor,
        )

    assert scale is not None
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    assert zero_point is not None
    if _type_utils.JitScalarType.from_value(
        zero_point, _type_utils.JitScalarType.UNDEFINED
    ) not in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
    }:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    output = g.op(
        "QuantizeLinear",
        tensor,
        scale,
        zero_point,
        axis_i=_get_const(axis, "i", "axis"),
    )
    args = [output, scale, zero_point]
    if axis is not None and not _is_none(axis):
        args.append(axis)
    return g.op("prim::TupleConstruct", *args)


@_beartype.beartype
def requantize_bias_helper(
    g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None
):
    """In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.
    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.
    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using
    regular operators.
    """
    bias_scale = g.op("Mul", weight_scale, input_scale)
    bias_scale_shape = g.op("Shape", bias_scale)
    bias_zero_point = g.op(
        "ConstantOfShape", bias_scale_shape, value_t=torch.tensor([0], dtype=torch.int)
    )
    q_bias = g.op(
        "Cast", g.op("Div", bias, bias_scale), to_i=_C_onnx.TensorProtoDataType.INT32
    )
    axis_args = []
    if axis is not None and not _is_none(axis):
        axis_args.append(axis)
    return g.op("prim::TupleConstruct", q_bias, bias_scale, bias_zero_point, *axis_args)


@_beartype.beartype
def args_have_same_dtype(args):
    assert args
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    has_same_dtype = all(
        _type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args
    )
    return has_same_dtype


# TODO(justinchuby): Delete these setters, users should set the vars directly.
@_deprecation.deprecated(
    "1.13",
    "1.14",
    "remove its usage and avoid setting internal variables directly",
)
def _set_opset_version(opset_version: int):
    GLOBALS.export_onnx_opset_version = opset_version


@_deprecation.deprecated(
    "1.13",
    "1.14",
    "remove its usage and avoid setting internal variables directly",
)
def _set_operator_export_type(operator_export_type):
    GLOBALS.operator_export_type = operator_export_type


# This function is for debug use only.
# onnx_shape_inference = True by default.
@_deprecation.deprecated(
    "1.13",
    "1.14",
    "remove its usage and avoid setting internal variables directly",
)
def _set_onnx_shape_inference(onnx_shape_inference: bool):
    GLOBALS.onnx_shape_inference = onnx_shape_inference


# Deprecated. Internally use _type_utils.ScalarType
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    "Byte": _C_onnx.TensorProtoDataType.UINT8,
    "Char": _C_onnx.TensorProtoDataType.INT8,
    "Double": _C_onnx.TensorProtoDataType.DOUBLE,
    "Float": _C_onnx.TensorProtoDataType.FLOAT,
    "Half": _C_onnx.TensorProtoDataType.FLOAT16,
    "Int": _C_onnx.TensorProtoDataType.INT32,
    "Long": _C_onnx.TensorProtoDataType.INT64,
    "Short": _C_onnx.TensorProtoDataType.INT16,
    "Bool": _C_onnx.TensorProtoDataType.BOOL,
    "ComplexFloat": _C_onnx.TensorProtoDataType.COMPLEX64,
    "ComplexDouble": _C_onnx.TensorProtoDataType.COMPLEX128,
    "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
    "Undefined": _C_onnx.TensorProtoDataType.UNDEFINED,
}

# Deprecated. Internally use _type_utils.ScalarType
scalar_name_to_pytorch = {
    "uint8_t": "Byte",
    "int8_t": "Char",
    "double": "Double",
    "float": "Float",
    "half": "Half",
    "int": "Int",
    "int64_t": "Long",
    "int16_t": "Short",
    "bool": "Bool",
    "complex64": "ComplexFloat",
    "complex128": "ComplexDouble",
    "qint8": "QInt8",
    "quint8": "QUInt8",
    "qint32": "QInt32",
    "bfloat16": "BFloat16",
}


# Deprecated. Internally use _type_utils.ScalarType
# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    torch.complex32,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
    torch.qint8,  # 12
    torch.quint8,  # 13
    torch.qint32,  # 14
    torch.bfloat16,  # 15
]

# Deprecated. Internally use _type_utils.ScalarType
# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
pytorch_name_to_type = {
    "Byte": torch.uint8,
    "Char": torch.int8,
    "Double": torch.double,
    "Float": torch.float,
    "Half": torch.half,
    "Int": torch.int,
    "Long": torch.int64,
    "Short": torch.short,
    "Bool": torch.bool,
    "ComplexFloat": torch.complex64,
    "ComplexDouble": torch.complex128,
    "QInt8": torch.qint8,
    "QUInt8": torch.quint8,
    "QInt32": torch.qint32,
    "BFloat16": torch.bfloat16,
}


# Deprecated. Internally use _type_utils.ScalarType
scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],  # 0
    cast_pytorch_to_onnx["Char"],  # 1
    cast_pytorch_to_onnx["Short"],  # 2
    cast_pytorch_to_onnx["Int"],  # 3
    cast_pytorch_to_onnx["Long"],  # 4
    cast_pytorch_to_onnx["Half"],  # 5
    cast_pytorch_to_onnx["Float"],  # 6
    cast_pytorch_to_onnx["Double"],  # 7
    cast_pytorch_to_onnx["Undefined"],  # 8
    cast_pytorch_to_onnx["ComplexFloat"],  # 9
    cast_pytorch_to_onnx["ComplexDouble"],  # 10
    cast_pytorch_to_onnx["Bool"],  # 11
    cast_pytorch_to_onnx["Char"],  # 12
    cast_pytorch_to_onnx["Byte"],  # 13
    cast_pytorch_to_onnx["Int"],  # 14
    cast_pytorch_to_onnx["BFloat16"],  # 15
]

# Global set to store the list of quantized operators in the network.
# This is currently only used in the conversion of quantized ops from PT -> C2 via ONNX.
_quantized_ops: Set[int] = set()
