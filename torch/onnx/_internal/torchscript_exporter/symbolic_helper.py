# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "_apply_params",
    "_arange_cast_helper",
    "_arange_helper",
    "_argmin_argmax_helper",
    "_as_list_type",
    "_avgpool_helper",
    "_batchnorm_helper",
    "_block_list_in_opset",
    "_embedding_bag_helper",
    "_flatten_helper",
    "_generate_wrapped_number",
    "_get_const",
    "_get_dim_for_cross",
    "_get_interpolate_attributes",
    "_get_tensor_dim_size",
    "_get_tensor_rank",
    "_get_tensor_sizes",
    "_handle_reduce_dim_none",
    "_if_scalar_type_as",
    "_index_fill_reshape_helper",
    "_interpolate_get_scales_and_mode",
    "_interpolate_get_scales_if_available",
    "_interpolate_get_scales",
    "_interpolate_helper",
    "_interpolate_size_to_scales",
    "_interpolate_warning",
    "_is_bool",
    "_is_constant",
    "_is_fp",
    "_is_list",
    "_is_none",
    "_is_onnx_constant",
    "_is_packed_list",
    "_is_scalar_list",
    "_is_split_static",
    "_is_tensor_list",
    "_is_tensor",
    "_is_tuple_construct",
    "_is_value",
    "_linalg_vector_norm_helper",
    "_lt_helper",
    "_max_helper",
    "_maybe_cast_reduce_op_input",
    "_maybe_cast_to_type",
    "_maybe_get_const",
    "_maybe_get_scalar",
    "_min_helper",
    "_node_get",
    "_numel_helper",
    "_onnx_opset_unsupported_detailed",
    "_onnx_opset_unsupported",
    "_onnx_unsupported",
    "_op_with_optional_float_cast",
    "_optional_input_placeholder_tensor",
    "_overload_by_arg_count",
    "_parse_arg",
    "_reduce_op_symbolic_helper",
    "_reduce_with_dtype_helper",
    "_reducesum_helper",
    "_repeat_interleave_single_value_repeat_helper",
    "_repeat_interleave_split_helper",
    "_reshape_helper",
    "_scalar",
    "_scatter_helper",
    "_select_helper",
    "_size_helper",
    "_slice_helper",
    "_sort_helper",
    "_squeeze_helper",
    "_topk_helper",
    "_try_get_scalar_type",
    "_type_promote_from_values",
    "_unbind_helper",
    "_unimplemented",
    "_unpack_list",
    "_unpack_quantized_tensor",
    "_unpack_tuple",
    "_unsqueeze_helper",
    "_var_mean_helper",
    "args_have_same_dtype",
    "cast_pytorch_to_onnx",
    "check_training_mode",
    "dequantize_helper",
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

import functools
import inspect
import math
import sys
import typing
import warnings
from typing import (
    Any,
    Concatenate as _Concatenate,
    Literal,
    NoReturn,
    TypeVar as _TypeVar,
)
from typing_extensions import ParamSpec as _ParamSpec

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import _type_utils, jit_utils, utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.types import Number

_T = _TypeVar("_T")
_U = _TypeVar("_U")
_P = _ParamSpec("_P")

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


def _parse_arg(
    value,
    desc: _ValueDescriptor,
    arg_name: str | None = None,
    node_name: str | None = None,
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


def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    assert isinstance(node, _C.Node)
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
    return value.node().kind() == "onnx::Constant"


def _maybe_get_const(
    value: _C.Value | torch.Tensor | Number | Sequence | None,
    descriptor: _ValueDescriptor,
):
    # NOTE: prim::Constant at this stage usually means something not compatible in ONNX,
    # otherwise it'd be converted to onnx::Constant
    # TODO(justinchuby): Replace insinstance with _is_value once we figure out mypy
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, "t")
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected a constant value of the '{arg_name}' argument, "
            f"got '{value}'",
            value,
        )
    return _parse_arg(value, desc)


def _unpack_list(list_value: _C.Value) -> list[_C.Value]:
    list_node = list_value.node()
    if list_node.kind() != "prim::ListConstruct":
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type prim::ListConstruct, got '{list_node}'.",
            list_value,
        )
    return list(list_node.inputs())


def _unpack_tuple(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type 'prim::TupleConstruct', "
            f"got '{tuple_node.kind()}'.",
            tuple_value,
        )
    return tuple(tuple_node.inputs())


def _unpack_quantized_tensor(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
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
def _is_packed_list(list_value: Any) -> bool:
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(
    *arg_descriptors: _ValueDescriptor,
) -> Callable[[Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]]:
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

    def decorator(
        fn: Callable[_Concatenate[_U, _P], _T],
    ) -> Callable[_Concatenate[_U, _P], _T]:
        fn._arg_descriptors = arg_descriptors  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(g: _U, *args: _P.args, **kwargs: _P.kwargs) -> _T:
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
            # pyrefly: ignore [bad-assignment]
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[method-assign]
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


def quantized_args(
    *arg_q_descriptors: bool,
    scale: float | None = None,
    zero_point: int | None = None,
    quantize_output: bool = True,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
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
        quantize_output: If True, quantize the output of the base operator. Default is True
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

            def _is_arg_quantized(descriptor, arg):
                return descriptor and _is_value(arg) and _is_tuple_construct(arg)

            # Run regular symbolic function if none of the argument is QTensor.
            is_quantized: list[bool] = []
            for descriptor, arg in descriptor_args:
                # ListConstruct
                if _is_packed_list(arg):
                    is_quantized.extend(
                        _is_arg_quantized(descriptor, arg_input)
                        for arg_input in arg.node().inputs()
                    )
                else:
                    is_quantized.append(_is_arg_quantized(descriptor, arg))

            if not any(is_quantized):
                return fn(g, *args, **kwargs)

            # Dequantize arguments that are quantized
            non_quantized_args = []
            for descriptor, arg in descriptor_args:
                if _is_arg_quantized(descriptor, arg):
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
                # ListConstruct
                elif _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        if _is_arg_quantized(descriptor, arg_input):
                            # Quantized arg is a tuple of (value, scale, zero_point)
                            (
                                dequantized_arg,
                                arg_scale,
                                arg_zero_point,
                                _,
                            ) = dequantize_helper(g, arg_input)
                            # Set scale and zero_point to the first quantized input if not already set
                            if _scale is None:
                                _scale = arg_scale
                            if _zero_point is None:
                                _zero_point = arg_zero_point
                            arg_input.replaceAllUsesWith(dequantized_arg)
                    non_quantized_args.append(arg)
                else:
                    # Non-quantized arg
                    non_quantized_args.append(arg)
            # TODO(justinchuby): Only single output is supported for now. We may want to
            # support multiple outputs in the future.
            output = fn(g, *non_quantized_args, **kwargs)

            assert _scale is not None, "Bug: Scale must be set for quantized operator"
            assert _zero_point is not None, (
                "Bug: Zero point must be set for quantized operator"
            )

            if quantize_output:
                return quantize_helper(g, output, _scale, _zero_point)
            return output

        return wrapper

    return decorator


def _scalar(x: Any) -> Number | None:
    """Convert a scalar tensor into a Python value."""
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    return None


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


def _is_none(x: Any) -> bool:
    return x is None or (x.node().mustBeNone() if isinstance(x, _C.Value) else False)


def _is_value(x: Any) -> bool:
    return isinstance(x, _C.Value)


def _is_constant(value: Any) -> bool:
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


# Note: _C.JitType is not exposed to Python and cannot be checked in runtime.
def _as_list_type(jit_type: _C.JitType) -> _C.ListType | None:
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None


def _is_list(x: _C.Value) -> bool:
    return _as_list_type(x.type()) is not None


def _is_tensor_list(x: _C.Value) -> bool:
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    return isinstance(x_type.getElementType(), _C.TensorType)


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


def _is_tuple_construct(x: _C.Value) -> bool:
    return x.node().kind() == "prim::TupleConstruct"


def is_complex_value(x: _C.Value) -> bool:
    assert _is_value(x)
    return _type_utils.JitScalarType.from_value(
        x, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.COMPLEX32,
        _type_utils.JitScalarType.COMPLEX64,
        _type_utils.JitScalarType.COMPLEX128,
    }


def _get_tensor_rank(x: _C.Value) -> int | None:
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()


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


def _get_tensor_dim_size(x: _C.Value, dim: int) -> int | None:
    sizes = _get_tensor_sizes(x)
    return sizes[dim] if sizes else None


def _get_dim_for_cross(x: _C.Value, dim: int | None):
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


def _unimplemented(op: str, msg: str, value: _C.Value | None = None) -> None:
    # For BC reasons, the behavior for Caffe2 does not raise exception for unimplemented operators
    if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}", value)


def _onnx_unsupported(op_name: str, value: _C.Value | None = None) -> NoReturn:
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


def _onnx_opset_unsupported(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    value: _C.Value | None = None,
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


def _onnx_opset_unsupported_detailed(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    reason: str,
    value: _C.Value | None = None,
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


def _block_list_in_opset(name: str):
    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(
            f"ONNX export failed on {name}, which is not implemented for opset "
            f"{GLOBALS.export_onnx_opset_version}. "
            "Try exporting with other opset versions."
        )

    return symbolic_fn


def _try_get_scalar_type(*args) -> _type_utils.JitScalarType | None:
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(
            arg, _type_utils.JitScalarType.UNDEFINED
        )
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None


def _type_promote_from_values(*args) -> _type_utils.JitScalarType:
    undef = _type_utils.JitScalarType.UNDEFINED
    jit_types = [_try_get_scalar_type(arg) for arg in args]
    if len(jit_types) == 0:
        return undef
    if len(jit_types) == 1:
        return jit_types[0]  # type: ignore[return-value]
    new_dtype = jit_types[0].dtype()  # type: ignore[union-attr]
    for t in jit_types:
        new_dtype = torch.promote_types(new_dtype, t.dtype())  # type: ignore[union-attr]
    return _type_utils.JitScalarType.from_dtype(new_dtype)


def _maybe_cast_to_type(
    g: jit_utils.GraphContext, value, jit_type: _type_utils.JitScalarType
):
    if (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        != jit_type
    ):
        return g.op(
            "Cast",
            value,
            to_i=jit_type.onnx_type(),
        )
    return value


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


def _slice_helper(
    g: jit_utils.GraphContext,
    input,
    axes,
    starts,
    ends,
    steps=None,
):
    if g.opset <= 9:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
            _slice as _slice9,
        )

        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (
            _slice as _slice10,
        )

        return _slice10(g, input, axes, starts, ends, steps)


def _is_fp(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    }


def _is_bool(value) -> bool:
    return (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.BOOL
    )


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


def _sort_helper(g: jit_utils.GraphContext, input, dim, descending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op(
        "Gather",
        shape_,
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
    )
    if g.opset <= 10:
        if not descending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=descending, outputs=2
        )


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


def _lt_helper(g: jit_utils.GraphContext, input, other):
    if g.opset <= 8:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset8 import lt as _lt8

        return _lt8(g, input, other)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import lt as _lt9

        return _lt9(g, input, other)


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
        "We recommend using opset 11 and above for models using this operator.",
        stacklevel=2,
    )


def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if len(axes_i) == 0:
        # unnecessary unsqueeze if axes length==0
        return input
    elif _is_constant(axes_i[0]):
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
            for i in range(dim)
        ]
        scales = g.op(
            "Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32)
        )
    return scales


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


def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    if mode == "nearest":
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


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
                    "while exporting interpolate. Assuming that it is not a scalar.",
                    stacklevel=2,
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


def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if g.opset < 11:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            unbind,  # type: ignore[no-redef]
        )
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import (
            unbind,  # type: ignore[no-redef]
        )
    return unbind(g, self, dim, _outputs)


def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            scatter,  # type: ignore[no-redef]
        )
    return scatter(g, self, dim, index, src)


def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if g.opset <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import split

        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]


def _repeat_interleave_single_value_repeat_helper(
    g: jit_utils.GraphContext, self, repeats, dim
):
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
        flatten,
        unsqueeze,
    )

    if not _is_tensor(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))

    const_repeats: bool = _is_constant(repeats)
    reps = _maybe_get_const(repeats, "t")

    # Convert 'repeats' to 1-d if it is 0-d.
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op("Reshape", repeats, g.op("Constant", value_t=torch.tensor([1])))

    # Create a new dim of size 1, then expand it to be 'repeats' long, and finally collapse it.
    unsqueezed = unsqueeze(g, self, dim + 1)

    # repeats_per_dim is 1 for all dims except for the new unsqueezed dim, where it has value 'repeats'.
    if const_repeats:
        # 'Repeats' is a constant, 'repeats_per_dim' can be a constant.
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)  # type: ignore[arg-type]
        onehot[dim + 1] = reps
        repeats_per_dim = g.op("Constant", value_t=onehot)
    else:
        # 'Repeats' is a variable, 'repeats_per_dim' cannot be a constant.
        onehot = g.op(
            "OneHot",
            unsqueeze(g, dim + 1, 0),  # indices, must be >= 1-dimensional
            g.op(
                "Constant", value_t=torch.tensor(_get_tensor_rank(unsqueezed))
            ),  # depth
            g.op(
                "Concat", g.op("Constant", value_t=torch.tensor([1])), repeats, axis_i=0
            ),  # on/off values
        )
        repeats_per_dim = flatten(g, onehot, 0, 1)

    tiled = g.op("Tile", unsqueezed, repeats_per_dim)
    return flatten(g, tiled, dim, dim + 1)


def _arange_cast_helper(
    g: jit_utils.GraphContext, end, start=None, step=None, dtype=None
) -> tuple[
    _type_utils.JitScalarType,
    _C.Value | None,
    _C.Value | None,
    _C.Value | None,
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


def _arange_helper(g: jit_utils.GraphContext, *args):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import arange
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            arange,  # type: ignore[no-redef]
        )
    return arange(g, *args)


def _size_helper(g: jit_utils.GraphContext, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import select

    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 3. expand value as well.
    # 4. apply onnx::scatter.

    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import expand

    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            scatter,  # type: ignore[no-redef]
        )

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accessible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, "i")
    if dim_value < 0:
        dim_value += self_dim
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
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0):
    shape = _maybe_get_const(shape, "is")
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


def _batchnorm_helper(
    g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var
):
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import _var_mean

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


def _avgpool_helper(
    tuple_fn: Callable[[Any], Sequence[int]],
    padding: int | Sequence[int],
    kernel_size,
    stride,
    divisor_override,
    name,
) -> tuple[int, ...]:
    if divisor_override and divisor_override.node().kind() != "prim::Constant":
        _unimplemented(name, "divisor_override")
    return tuple(tuple_fn(padding))


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
        f"is set to {op_mode_text}. Exporting with {op_mode_text}.",
        stacklevel=2,
    )


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
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
        _reshape_from_tensor,
    )

    return _reshape_from_tensor(g, input, final_shape)


def _is_split_static(split_size_or_sizes, _outputs):
    if _outputs is None:
        return False
    if (
        _is_value(split_size_or_sizes)
        and split_size_or_sizes.node().kind() != "onnx::Constant"
    ):
        return False
    return True


def _optional_input_placeholder_tensor(g):
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    rank = _get_tensor_rank(self)
    if rank is not None and any(
        _get_tensor_dim_size(self, i) == 0 for i in range(rank)
    ):
        # If input tensor is empty, according to ONNX ReduceSum definition,
        # set keepdims=1 so that the resulted tensor has the same rank as the input.
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)


def dequantize_helper(
    g: jit_utils.GraphContext,
    qtensor: _C.Value,
    qdtype: _C_onnx.TensorProtoDataType | None = None,
) -> tuple[_C.Value, _C.Value, _C.Value, _C.Value | None]:
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


def quantize_helper(
    g: jit_utils.GraphContext,
    tensor: _C.Value,
    scale: _C.Value,
    zero_point: _C.Value,
    axis: _C.Value | None = None,
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


def args_have_same_dtype(args):
    assert args
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    has_same_dtype = all(
        _type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args
    )
    return has_same_dtype


def _op_with_optional_float_cast(g: jit_utils.GraphContext, op_name, *args, **kwargs):
    """Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, torch.onnx.JitScalarType.FLOAT by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    """
    opset_before = kwargs.pop("opset_before", None)
    target_float_t = kwargs.pop("target_float_t", _type_utils.JitScalarType.FLOAT)

    inputs = list(args)
    dtype_0 = _type_utils.JitScalarType.from_value(inputs[0])

    require_cast = not _is_fp(inputs[0]) and (
        opset_before is None or GLOBALS.export_onnx_opset_version < opset_before
    )

    if require_cast:
        for input in inputs:
            if input.isCompleteTensor():
                input_scalar_type = _type_utils.JitScalarType.from_value(input)
                if input_scalar_type != dtype_0:
                    raise errors.SymbolicValueError(
                        f"Inputs of {op_name} must have same dtype."
                        f"Got {dtype_0.scalar_name()} and {input_scalar_type.scalar_name()}",
                        input,
                    )
        for i, input in enumerate(inputs):
            if input.isCompleteTensor() and not _is_fp(input):
                inputs[i] = g.op(
                    "Cast",
                    input,
                    to_i=target_float_t.onnx_type(),
                )

    self = g.op(op_name, *inputs, **kwargs)

    if require_cast:
        self = g.op("Cast", self, to_i=dtype_0.onnx_type())

    return self


def _maybe_cast_reduce_op_input(g: jit_utils.GraphContext, self):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        # This check only covers traced modules where dtype is present
        # pytorch reduce-ops cast all other integral types to int64
        if not _is_fp(self) and scalar_type != _type_utils.JitScalarType.INT64:
            self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.INT64)
    return self


def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply


def _reduce_op_symbolic_helper(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None or dim == ():
            # Dim can be 0, which will cause (not dim) == True. So we don't want to do
            # (not dim)
            # all-reduce path
            return _handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            # dim-reduce path
            keepdim = _get_const(keepdim, "i", "keepdim")
            if g.opset < 18:
                desc = "is" if allow_multi_dim_support else "i"
                dim = _get_const(dim, desc, "dim")
                dim_list = dim if allow_multi_dim_support else [dim]
                return g.op(onnx_op_name, self, axes_i=dim_list, keepdims_i=keepdim)
            else:
                if _is_value(dim):
                    axes = dim
                else:
                    if allow_multi_dim_support:
                        axes = g.op(
                            "Constant", value_t=torch.tensor(dim, dtype=torch.long)
                        )
                    else:
                        axes = g.op(
                            "Constant", value_t=torch.tensor([dim], dtype=torch.long)
                        )
                return g.op(onnx_op_name, self, axes, keepdims_i=keepdim)

    return symbolic


def _overload_by_arg_count(fn):
    @functools.wraps(fn)
    def wrapper(g, *args):
        overloads = fn(g, *args)
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        return _unimplemented(f"aten::{fn.__name__}", f"with {len(args)} arguments")

    return wrapper


def _reduce_with_dtype_helper(
    onnx_op: str, name: str, allow_multi_dim_support: bool = True
):
    symbolic = _reduce_op_symbolic_helper(
        onnx_op, allow_multi_dim_support=allow_multi_dim_support
    )

    @_overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @quantized_args(True)
        @parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = _get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            result = symbolic(g, self)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        dim_desc = "is" if allow_multi_dim_support else "i"

        @quantized_args(True)
        @parse_args("v", dim_desc, "i", "none")  # type: ignore[arg-type]
        def reduce_dim(g, self, dim, keepdim, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = _get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            result = symbolic(g, self, dim, keepdim)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        return reduce_nodim, reduce_dim

    return reduce


def _max_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    # torch.max(input, other)
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Max", self, dim_or_y, opset_before=12)
    # torch.max(input, dim, keepdim)
    else:
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        if g.opset < 18:
            max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            max = g.op("ReduceMax", self, axes, keepdims_i=keepdim)
        indices = g.op("ArgMax", self, axis_i=dim, keepdims_i=keepdim)
        return max, indices


def _min_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    # torch.min(input, other)
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Min", self, dim_or_y, opset_before=12)
    # torch.min(input, dim, keepdim)
    else:
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        if g.opset < 18:
            min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            min = g.op("ReduceMin", self, axes, keepdims_i=keepdim)
        indices = g.op("ArgMin", self, axis_i=dim, keepdims_i=keepdim)
        return min, indices


def _numel_helper(g: jit_utils.GraphContext, self):
    shape = g.op("Shape", self)
    return g.op("ReduceProd", shape, keepdims_i=0)


@parse_args("v", "is", "i", "i")
def _var_mean_helper(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    if g.opset < 18:
        if dim is None:
            mean = g.op("ReduceMean", input, keepdims_i=0)
            t_mean = mean
            num_elements = _numel_helper(g, input)
        else:
            mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
            t_mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=1)
            redudced_dims = g.op("Shape", input)
            # dim could contain one or multiple dimensions
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        sub_v = g.op("Sub", input, t_mean)
        sqr_sub = g.op("Mul", sub_v, sub_v)
        keepdim_mean = 0 if dim is None else keepdim
        var = g.op("ReduceMean", sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
        # Correct bias in calculating variance, by dividing it over (N - correction) instead on N
        if correction is None:
            correction = 1
        if correction != 0:
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            mul = g.op("Mul", var, num_elements)
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        return var, mean
    else:
        axes = None
        if dim is None:
            mean = g.op("ReduceMean", input, keepdims_i=0)
            t_mean = mean
            num_elements = _numel_helper(g, input)
        else:
            axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
            mean = g.op("ReduceMean", input, axes, keepdims_i=keepdim)
            t_mean = g.op("ReduceMean", input, axes, keepdims_i=1)
            redudced_dims = g.op("Shape", input)
            # dim could contain one or multiple dimensions
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        sub_v = g.op("Sub", input, t_mean)
        sqr_sub = g.op("Mul", sub_v, sub_v)
        keepdim_mean = 0 if dim is None else keepdim
        if axes is None:
            var = g.op("ReduceMean", sqr_sub, keepdims_i=keepdim_mean)
        else:
            var = g.op("ReduceMean", sqr_sub, axes, keepdims_i=keepdim_mean)
        # Correct bias in calculating variance, by dividing it over (N - correction) instead on N
        if correction is None:
            correction = 1
        if correction != 0:
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            mul = g.op("Mul", var, num_elements)
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        return var, mean


def _embedding_bag_helper(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    if scale_grad_by_freq and GLOBALS.export_training:
        return _onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    zero = g.op("Constant", value_t=torch.tensor([0]))

    indices_len = _unsqueeze_helper(
        g,
        _size_helper(g, indices, g.op("Constant", value_t=torch.tensor(0))),
        [0],
    )
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # Offsets holds the starting index position of each bag. So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row. Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output, so we can avoid creating a sequence and inserting elements in.
    offsets_starts = _slice_helper(
        g, offsets, axes=[0], starts=[0], ends=[sys.maxsize], steps=[1]
    )
    offsets_ends = _slice_helper(
        g, offsets, axes=[0], starts=[1], ends=[sys.maxsize], steps=[1]
    )

    loop_len = _size_helper(g, offsets_ends, g.op("Constant", value_t=torch.tensor(0)))

    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    loop_block = loop_context.block

    # FIXME(justinchuby): We need to handle what happens when we call b.op on a node return
    block_input_iter = utils._add_input_to_block(loop_block)
    utils._add_input_to_block(loop_block)

    indices_start = loop_context.op(
        "Gather", offsets_starts, block_input_iter, axis_i=0
    )
    indices_end = loop_context.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    indices_start = _unsqueeze_helper(loop_context, indices_start, [0])
    indices_end = _unsqueeze_helper(loop_context, indices_end, [0])

    indices_row = loop_context.op("Slice", indices, indices_start, indices_end, zero)
    embeddings = loop_context.op("Gather", embedding_matrix, indices_row, axis_i=0)
    if not _is_none(per_sample_weights):
        per_sample_weights_row = loop_context.op(
            "Slice", per_sample_weights, indices_start, indices_end, zero
        )
        per_sample_weights_row = _unsqueeze_helper(
            loop_context, per_sample_weights_row, [1]
        )
        embeddings = loop_context.op("Mul", embeddings, per_sample_weights_row)
    if mode == 0:
        embeddings = _reducesum_helper(
            loop_context, embeddings, axes_i=[0], keepdims_i=0
        )
    elif mode == 1:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMean", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMean", embeddings, axes, keepdims_i=0)
    else:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMax", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMax", embeddings, axes, keepdims_i=0)

    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, embeddings)

    # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
    # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    return loop.node().output(), None, None, None


def _linalg_vector_norm_helper(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype: torch._C.Value,
):
    axes = None
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    if _is_none(dim):
        self = _reshape_helper(g, self, [-1])
        keepdim = False
    elif g.opset >= 18:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))

    if ord == math.inf:
        if g.opset < 18:
            result = g.op(
                "ReduceMax", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:
            if axes is None:
                result = g.op("ReduceMax", g.op("Abs", self), keepdims_i=keepdim)
            else:
                result = g.op("ReduceMax", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == -math.inf:
        if g.opset < 18:
            result = g.op(
                "ReduceMin", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:
            if axes is None:
                result = g.op("ReduceMin", g.op("Abs", self), keepdims_i=keepdim)
            else:
                result = g.op("ReduceMin", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == 0:
        if g.opset < 11:
            return _onnx_opset_unsupported_detailed(
                "linalg_vector_norm", 9, 11, "ord=0 not supported", self
            )
        else:
            if dim is None:
                self = _reshape_helper(
                    g,
                    self,
                    g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
                )
                keepdim = False

            cond_op = g.op(
                "Not",
                g.op("Equal", self, g.op("Constant", value_t=torch.LongTensor([0]))),
            )
            cond_op = g.op(
                "Cast",
                cond_op,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )
            return _reducesum_helper(g, cond_op, axes_i=dim, keepdims_i=keepdim)
    elif ord == 1:
        if g.opset < 18:
            result = _reduce_op_symbolic_helper("ReduceL1")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:
            if axes is None:
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, keepdim=keepdim
                )
            else:
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, axes, keepdim=keepdim
                )
    elif ord == 2:
        if g.opset < 18:
            result = _reduce_op_symbolic_helper("ReduceL2")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:
            if axes is None:
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, keepdim=keepdim
                )
            else:
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, axes, keepdim=keepdim
                )
    else:
        ord_op = g.op("Constant", value_t=torch.tensor(ord, dtype=torch.float32))
        result = _reducesum_helper(
            g, g.op("Pow", g.op("Abs", self), ord_op), axes_i=dim, keepdims_i=keepdim
        )
        result = g.op(
            "Pow",
            result,
            g.op(
                "Div",
                g.op("Constant", value_t=torch.tensor(1, dtype=torch.float32)),
                ord_op,
            ),
        )

    if not _is_none(dtype):
        dtype = _get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())  # type: ignore[arg-type]
    return result


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
_quantized_ops: set[int] = set()
