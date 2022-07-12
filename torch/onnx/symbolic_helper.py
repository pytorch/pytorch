from __future__ import annotations

import enum
import functools
import inspect
import sys
import warnings
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C

# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
from torch.onnx import _patch_torch  # noqa: F401
from torch.onnx._globals import GLOBALS

# Note [Edit Symbolic Files]
# EDITING THIS FILE AND SYMBOLIC_OPSET<VERSION> FILES? READ THIS FIRST!
#
# - Module-level functions are called to convert the corresponding op in the `aten` domain.
#   E.g. symbolic_opset9.foo is called to convert aten::foo.
#   Symbolic functions for other domains are staticmethods in classes named after the domain.
#   E.g. symbolic_opset9.Prim.ConstantChunk is called to convert prim::ConstantChunk.
# - Parameter names must *exactly* match the names in
#   aten/src/ATen/native/native_functions.yaml, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by
#   `_jit_pass_onnx_remove_inplace_ops_for_onnx`, and
#   transparently dispatched to their non inplace versions in
#   "run_symbolic_function".   See Note [Export inplace]
#
# ----------------------------------------------------------------------------------
# A note on Tensor types
# ----------------------------------------------------------------------------------
#
# In general, we should avoid depending on the type of Tensor Values contained
# within the trace graph. However, this is sometimes unavoidable (due to ONNX
# spec requirements, etc). The TensorType object has accessors for these properties
# that return the property if it is statically known and return nullopt otherwise.
#
# In general, we should prefer to rely on the least specific information possible.
# For example, not relying on tensor properties at all is better than relying
# on the number of dimensions which is better than relying on
# concrete shapes. Doing so will make the export symbolics
# more robust to different graphs.
#
# ----------------------------------------------------------------------------------
# Extra context for symbolic functions
# ----------------------------------------------------------------------------------
#
# In general, symbolic functions only require inputs and attributes to
# the original node. In rare circumstances, extra context may be required.
# For example, symbolic function for `prim::Loop` needs access to the subblock of
# the original node.
# A symbolic function that has a first arg (before the Graph object) with the
# type annotation of torch.onnx.SymbolicContext will be called with that additional context.
# During export, it is populated from `utils._run_symbolic_function`
# to contain the context for each node being converted.

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------


def _parse_arg(value, desc, arg_name=None, node_name=None):
    if desc == "none":
        return value
    if desc == "v" or not _is_value(value):
        return value
    if value.node().mustBeNone():
        return None
    if value.node().kind() == "onnx::Constant":
        tval = value.node()["value"]
        if desc == "i":
            return int(tval)
        elif desc == "f":
            return float(tval)
        elif desc == "b":
            return bool(tval)
        elif desc == "s":
            return str(tval)
        elif desc == "t":
            return tval
        elif desc == "is":
            return [int(v) for v in tval]
        elif desc == "fs":
            return [float(v) for v in tval]
        else:
            raise RuntimeError("ONNX symbolic doesn't know to interpret Constant node")
    elif value.node().kind() == "prim::ListConstruct":
        if desc == "is":
            for v in value.node().inputs():
                if v.node().kind() != "onnx::Constant":
                    raise RuntimeError(
                        "Failed to export an ONNX attribute '"
                        + v.node().kind()
                        + "', since it's not constant, please try to make "
                        "things (e.g., kernel size) static if possible"
                    )
            return [int(v.node()["value"]) for v in value.node().inputs()]
        else:
            raise RuntimeError(
                "ONNX symbolic doesn't know to interpret ListConstruct node"
            )

    if arg_name is None or node_name is None:
        raise RuntimeError(
            "Expected node type 'onnx::Constant', got '{}'.".format(value.node().kind())
        )
    else:
        raise RuntimeError(
            "Expected node type 'onnx::Constant' "
            "for argument '{}' of node '{}', got '{}'.".format(
                arg_name, node_name, value.node().kind()
            )
        )


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == "onnx::Constant":
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, "t")
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise RuntimeError(
            "ONNX symbolic expected a constant value of the {} argument, got `{}`".format(
                arg_name, value
            )
        )
    return _parse_arg(value, desc)


def _unpack_list(list_value: _C.Value) -> List[_C.Value]:
    list_node = list_value.node()
    assert list_node.kind() == "prim::ListConstruct"
    return list(list_node.inputs())


def _unpack_tuple(tuple_value):
    tuple_node = tuple_value.node()
    if tuple_node.kind() != "prim::TupleConstruct":
        raise RuntimeError(
            "ONNX symbolic expected node type `prim::TupleConstruct`, got `{}`".format(
                tuple_node
            )
        )
    return list(tuple_node.inputs())


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
def _is_packed_list(list_value):
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(*arg_descriptors):
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
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[assignment]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # only support _outputs in kwargs
            assert len(kwargs) <= 1, (
                f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single key/value entry. "
                f"{FILE_BUG_MSG}"
            )

            if len(kwargs) == 1:
                assert "_outputs" in kwargs, (
                    f"Symbolic function {fn.__name__}'s '**kwargs' can only contain '_outputs' key at '**kwargs'. "
                    f"{FILE_BUG_MSG}"
                )
            return fn(g, *args, **kwargs)

        return wrapper

    return decorator


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
        fn._scale = scale
        fn._zero_point = zero_point

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            _scale = fn._scale
            if _scale is not None:
                _scale = g.op("Constant", value_t=torch.tensor(_scale))
            _zero_point = fn._zero_point
            if _zero_point is not None:
                _zero_point = g.op("Constant", value_t=torch.tensor(_zero_point))

            # Support variable length arguments by marking unspecified ones as non-quantized
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (
                len(args) - len(arg_q_descriptors)
            )
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))
            # Run regular symbolic function if none of the argument is QTensor.
            if not any(
                (descriptor and arg.node().kind() == "prim::TupleConstruct")
                for descriptor, arg in descriptor_args
            ):
                return fn(g, *args, **kwargs)

            dequantized_args = []
            for descriptor, arg in descriptor_args:
                if descriptor:
                    dequantized_arg, scale, zero_point, _ = dequantize_helper(g, arg)
                    dequantized_args.append(dequantized_arg)
                    if _scale is None:
                        _scale = scale
                    if _zero_point is None:
                        _zero_point = zero_point
                else:
                    dequantized_args.append(arg)
            # TODO(justinchuby): Only single output is supported for now. We may want to
            # support multiple outputs in the future.
            output = fn(g, *dequantized_args, **kwargs)

            return quantize_helper(g, output, _scale, _zero_point)

        return wrapper

    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g: _C.Graph, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, _C.Value):
        return self

    scalar_type = tensor.type().scalarType()
    if scalar_type:
        ty = scalar_type.lower()
        return getattr(self, ty)()

    return self


def _is_none(x):
    return x.node().mustBeNone()


def _is_value(x):
    return isinstance(x, _C.Value)


def _is_constant(value):
    return not _is_value(value) or value.node().kind() in (
        "onnx::Constant",
        "prim::Constant",
    )


def _is_tensor(x):
    return x.type().isSubtypeOf(_C.TensorType.get())


def _is_list(x):
    return isinstance(x.type(), _C.ListType)


def _is_tensor_list(x):
    return _is_list(x) and isinstance(x.type().getElementType(), _C.TensorType)


def _is_scalar_list(x):
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
    element_type = str(x.type().getElementType())
    return (
        _is_list(x)
        and element_type in scalar_name_to_pytorch.keys()
        and (scalar_name_to_pytorch[element_type] in cast_pytorch_to_onnx.keys())
    )


def is_caffe2_aten_fallback():
    return (
        GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    )


def _get_tensor_rank(x):
    if not _is_tensor(x) or x.type() is None:
        return None
    return x.type().dim()


def _get_tensor_sizes(x, allow_nonstatic=True):
    if not _is_tensor(x) or x.type() is None:
        return None
    if allow_nonstatic:
        # Each individual symbol is returned as None.
        # e.g. [1, "a", "b"] -> [1, None, None]
        return x.type().varyingSizes()
    # returns None, if exists any symbol in sizes.
    # e.g. [1, "a", "b"] -> None
    return x.type().sizes()


def _get_tensor_dim_size(x, dim):
    try:
        sizes = _get_tensor_sizes(x)
        return sizes[dim]
    except Exception:
        pass
    return None


def _get_dim_for_cross(input, dim):
    if dim == -1:
        return dim + _get_tensor_rank(input)
    # If dim is not given, it defaults to the first dimension found with the size 3
    if dim is None:
        sizes = _get_tensor_sizes(input)
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim


def _unimplemented(op, msg):
    # For BC reasons, the behavior for Caffe2 does not raise exception for unimplemented operators
    if _C_onnx._CAFFE2_ATEN_FALLBACK:
        warnings.warn(
            "ONNX export failed on " + op + " because " + msg + " not supported"
        )
    elif GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}")


def _onnx_unsupported(op_name):
    raise RuntimeError(
        "Unsupported: ONNX export of operator {}. "
        "Please feel free to request support or submit a pull request on PyTorch GitHub.".format(
            op_name
        )
    )


def _onnx_opset_unsupported(op_name, current_opset, supported_opset):
    raise RuntimeError(
        "Unsupported: ONNX export of {} in "
        "opset {}. Please try opset version {}.".format(
            op_name, current_opset, supported_opset
        )
    )


def _onnx_opset_unsupported_detailed(op_name, current_opset, supported_opset, reason):
    raise RuntimeError(
        "Unsupported: ONNX export of {} in "
        "opset {}. {}. Please try opset version {}.".format(
            op_name, current_opset, reason, supported_opset
        )
    )


def _block_list_in_opset(name):
    def symbolic_fn(*args, **kwargs):
        raise RuntimeError(
            "ONNX export failed on {}, which is not implemented for opset {}. "
            "Try exporting with other opset versions.".format(
                name, GLOBALS.export_onnx_opset_version
            )
        )

    return symbolic_fn


def _try_get_scalar_type(*args):
    for arg in args:
        try:
            return arg.type().scalarType()
        except RuntimeError:
            pass
    return None


def _select_helper(g, self, dim, index, apply_reshape=True):
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

    index_scalar_type = index.type().scalarType()
    if index_scalar_type is None or index_scalar_type not in ["Long", "Int"]:
        index = g.op("Cast", index, to_i=cast_pytorch_to_onnx["Long"])
    return g.op("Gather", self, index, axis_i=dim)


def _slice_helper(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if GLOBALS.export_onnx_opset_version <= 9:
        from torch.onnx.symbolic_opset9 import _slice as _slice9

        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx.symbolic_opset10 import _slice as _slice10

        return _slice10(g, input, axes, starts, ends, steps, dynamic_slice)


def _is_fp(value):
    if value:
        if isinstance(value, torch.Tensor):
            return value.dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            )
        else:
            type = value.type().scalarType()
            if type is None:
                warnings.warn(
                    "Type cannot be inferred, which might cause exported graph to produce incorrect results."
                )
            return type in ("Float", "Double", "Half", "BFloat16")
    return False


def _generate_wrapped_number(g, scalar):
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


def _sort_helper(g, input, dim, decending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op(
        "Gather",
        shape_,
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
    )
    if GLOBALS.export_onnx_opset_version <= 10:
        if not decending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=decending, outputs=2
        )


def _topk_helper(g, input, k, dim, largest=True, sorted=False, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op("Constant", value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != "Long":
            k = g.op("Cast", k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if GLOBALS.export_onnx_opset_version <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2
        )


def _lt_helper(g, input, other):
    if GLOBALS.export_onnx_opset_version <= 8:
        from torch.onnx.symbolic_opset8 import lt as _lt8

        return _lt8(g, input, other)
    else:
        from torch.onnx.symbolic_opset9 import lt as _lt9

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
        "We recommend using opset 11 and above for models using this operator."
    )


def _unsqueeze_helper(g, input, axes_i):
    if _is_constant(axes_i[0]):
        if GLOBALS.export_onnx_opset_version >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Unsqueeze", input, axes)
        return g.op("Unsqueeze", input, axes_i=axes_i)
    # Tensor type
    if GLOBALS.export_onnx_opset_version < 13:
        raise ValueError(
            f"Opset version must be >= 13 for Unsqueeze with dynamic axes. {input.node().sourceRange()}"
        )
    return g.op("Unsqueeze", input, axes_i[0])


def _squeeze_helper(g, input, axes_i):
    if _is_constant(axes_i[0]):
        if GLOBALS.export_onnx_opset_version >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Squeeze", input, axes)
        return g.op("Squeeze", input, axes_i=axes_i)
    # Tensor type
    if GLOBALS.export_onnx_opset_version < 13:
        raise ValueError(
            f"Opset version must be >= 13 for Squeeze with dynamic axes. {input.node().sourceRange()}"
        )
    axes_t = axes_i[0]
    axes_rank = _get_tensor_rank(axes_t)
    if axes_rank > 1:
        raise ValueError(
            "For Squeeze axses as input, the axes rank must be one in ONNX spec."
        )
    elif axes_rank == 0:
        # The axes is a scalar. Unsqueeze it to a rank 1 tensor.
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        return g.op("Squeeze", input, axes_t)
    return g.op("Squeeze", input, axes_t)


def _reducesum_helper(g, input, axes_i=None, keepdims_i=1, noop_with_empty_axes_i=0):
    keepdims_i = _maybe_get_const(keepdims_i, "i")
    if GLOBALS.export_onnx_opset_version >= 13:
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


def _interpolate_size_to_scales(g, input, output_size, dim):
    output_size = _maybe_get_const(output_size, "is")
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=cast_pytorch_to_onnx["Float"])
        divisor = _slice_helper(
            g, g.op("Shape", input), axes=[0], ends=[sys.maxsize], starts=[offset]
        )
        divisor = g.op("Cast", divisor, to_i=cast_pytorch_to_onnx["Float"])
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


def _interpolate_get_scales_if_available(g, scales):
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


def _get_interpolate_attributes(g, mode, args):
    if mode == "nearest":
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


def _interpolate_get_scales(g, scale_factor, dim):
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (
        scale_factor_rank is not None and scale_factor_rank > 0
    ):
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op("Cast", scale_factor, to_i=cast_pytorch_to_onnx["Float"])
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


def _interpolate_get_scales_and_mode(g, input, size, scale_factor, mode, align_corners):
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
            else "pytorch_half_pixel"
        )

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            output_size = g.op("Cast", output_size, to_i=cast_pytorch_to_onnx["Long"])
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if GLOBALS.export_onnx_opset_version >= 13:
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
            if GLOBALS.export_onnx_opset_version >= 13:
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
    g, input, size, scale_factor, mode, align_corners, recompute_scale_factor
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
        else "pytorch_half_pixel"
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
                (_maybe_get_const(size, "t").dim() == 0)
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
        size = g.op("Cast", size, to_i=cast_pytorch_to_onnx["Long"])
        size = g.op("Concat", input_size, size, axis_i=0)

        if GLOBALS.export_onnx_opset_version >= 13:
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

        if GLOBALS.export_onnx_opset_version >= 13:
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


def _unbind_helper(g, self, dim, _outputs):
    if GLOBALS.export_onnx_opset_version < 11:
        from torch.onnx.symbolic_opset9 import unbind
    elif GLOBALS.export_onnx_opset_version <= 12:
        from torch.onnx.symbolic_opset11 import unbind  # type: ignore[no-redef]
    else:
        from torch.onnx.symbolic_opset13 import unbind  # type: ignore[no-redef]
    return unbind(g, self, dim, _outputs)


def _scatter_helper(g, self, dim, index, src):
    if GLOBALS.export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore[no-redef]
    return scatter(g, self, dim, index, src)


def _repeat_interleave_split_helper(g, self, reps, dim):
    if GLOBALS.export_onnx_opset_version <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx.symbolic_opset13 import split

        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]


def _arange_cast_helper(g, end, start=None, step=None, dtype=None):
    def _is_all_integral(scalars):
        for scalar in scalars:
            try:
                if scalar.type().scalarType() != "Long":
                    return False
            except Exception:
                pass
        return True

    # This logic is based on torch.arange docs. If "dtype" is provided,
    # infer input types from dtype. If not, then check if any of start, stop,
    # or step are floating point, and infer the type from get_default.
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            type = scalar_type_to_pytorch_type.index(torch.int64)
        else:
            type = scalar_type_to_pytorch_type.index(torch.get_default_dtype())
    else:
        type = dtype

    start = g.op("Cast", start, to_i=scalar_type_to_onnx[type]) if start else None
    end = g.op("Cast", end, to_i=scalar_type_to_onnx[type]) if end else None
    step = g.op("Cast", step, to_i=scalar_type_to_onnx[type]) if step else None
    return type, end, start, step


def _arange_helper(g, *args):
    if GLOBALS.export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import arange
    else:
        from torch.onnx.symbolic_opset11 import arange  # type: ignore[no-redef]
    return arange(g, *args)


def _size_helper(g, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx.symbolic_opset9 import select

    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


def _index_fill_reshape_helper(g, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 3. expand value as well.
    # 4. apply onnx::scatter.

    from torch.onnx.symbolic_opset9 import expand

    if GLOBALS.export_onnx_opset_version <= 10:
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
def _reshape_helper(g, input, shape, allowzero=0):
    shape = _maybe_get_const(shape, "is")
    if not _is_value(shape):
        shape = g.op("Constant", value_t=torch.LongTensor(shape))
    if GLOBALS.export_onnx_opset_version <= 13:
        if allowzero == 1:
            raise _onnx_opset_unsupported(
                "Reshape with allowzero=1", GLOBALS.export_onnx_opset_version, 14
            )
        return g.op("Reshape", input, shape)
    else:
        return g.op("Reshape", input, shape, allowzero_i=allowzero)


def _batchnorm_helper(g, input, weight, bias, running_mean, running_var):
    from torch.onnx.symbolic_opset9 import _var_mean

    batch_size = _get_tensor_dim_size(input, 0)
    channel_size = _get_tensor_dim_size(input, 1)

    if weight is None or _is_none(weight):
        if channel_size is None:
            raise RuntimeError(
                "Unsupported: ONNX export of batch_norm for unknown " "channel size."
            )
        weight_value = torch.tensor([1.0] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor"
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or _is_none(bias):
        if channel_size is None:
            raise RuntimeError(
                "Unsupported: ONNX export of batch_norm for unknown " "channel size."
            )
        bias_value = torch.tensor([0.0] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor"
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
    padding: Union[int, Sequence[int]],
    kernel_size,
    stride,
    divisor_override,
    name,
) -> Tuple[int, ...]:
    if divisor_override and divisor_override.node().kind() != "prim::Constant":
        _unimplemented(name, "divisor_override")
    return tuple(tuple_fn(padding))


def check_training_mode(op_train_mode, op_name):
    op_train_mode = True if op_train_mode == 1 else False
    if GLOBALS.training_mode is not None and op_train_mode != GLOBALS.training_mode:
        op_mode = "training " if op_train_mode else "inference"
        training_mode = "training " if GLOBALS.training_mode else "inference"
        # setting the model mode could result in op_mode != _flags.training_mode
        # if the model is a FuncModule. In this case we warn the user of
        # the state and export depending on op_mode
        # This is to support use-cases of fixing certain layer weights
        # in training.
        warnings.warn(
            "ONNX export mode is set to "
            + training_mode
            + " mode, but operator "
            + op_name
            + " is set to "
            + op_mode
            + " mode. The operators will be exported in "
            + op_mode
            + ", as specified by the functional operator."
        )


def _flatten_helper(g, input, start_dim, end_dim, dim):
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


def _handle_reduce_dim_none(g, self, op_name):
    rank = _get_tensor_rank(self)
    if rank is not None and any(
        [_get_tensor_dim_size(self, i) == 0 for i in range(rank)]
    ):
        # If input tensor is empty, according to ONNX ReduceSum definition,
        # set keepdims=1 so that the resulted tensor has the same rank as the input.
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)


def dequantize_helper(
    g,
    qtensor: _C.Value,
    qdtype: Optional[torch.onnx.TensorProtoDataType] = None,
) -> Tuple[_C.Value, _C.Value, _C.Value, Optional[_C.Value]]:
    """Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point) for per tensor quantization,
          or (quantized_tensor, scale, zero_point, axis) for per channel quantization.
          Representing the quantized tensor.
        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the data type of quantized tensor.
          It must be either torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.
    """
    unpacked_qtensors = _unpack_tuple(qtensor)
    tensor, scale, zero_point = unpacked_qtensors[:3]
    axis = unpacked_qtensors[3] if len(unpacked_qtensors) >= 4 else None
    axis_i = _get_const(axis, "i", "axis")
    input_qdtype = cast_pytorch_to_onnx[tensor.type().scalarType()]
    if qdtype is None:
        if input_qdtype is not None:
            qdtype = input_qdtype
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
        )

    return (
        g.op("DequantizeLinear", value, scale, zero_point, axis_i=axis_i),
        scale,
        zero_point,
        axis,
    )


def quantize_helper(
    g,
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
        )

    assert scale is not None
    if scale.type().scalarType() != "Float":  # type: ignore[attr-defined]
        # TODO(justinchuby): Remove type ignore after #81112 is checked in.
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    assert zero_point is not None
    if zero_point.type().scalarType() not in ("Byte", "Char"):  # type: ignore[attr-defined]
        # TODO(justinchuby): Remove type ignore after #81112 is checked in.
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


def requantize_bias_helper(g, bias, input_scale, weight_scale, axis=None):
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
    base_dtype = args[0].type().scalarType()
    has_same_dtype = all(elem.type().scalarType() == base_dtype for elem in args)
    return has_same_dtype


# TODO(justinchuby): Delete these setters, users should set the vars directly.
def _set_opset_version(opset_version: int):
    GLOBALS.export_onnx_opset_version = opset_version


def _set_operator_export_type(operator_export_type):
    GLOBALS.operator_export_type = operator_export_type


def _set_training_mode(training_mode):
    GLOBALS.training_mode = training_mode


# This function is for debug use only.
# onnx_shape_inference = False by default.
def _set_onnx_shape_inference(onnx_shape_inference: bool):
    GLOBALS.onnx_shape_inference = onnx_shape_inference


# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute "UINT8"
#
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


class ScalarType(enum.IntEnum):
    """A human-readable name for a key into scalar_type_to_pytorch_type."""

    UINT8 = 0
    INT8 = enum.auto()
    SHORT = enum.auto()
    INT = enum.auto()
    INT64 = enum.auto()
    HALF = enum.auto()
    FLOAT = enum.auto()
    DOUBLE = enum.auto()
    COMPLEX32 = enum.auto()
    COMPLEX64 = enum.auto()
    COMPLEX128 = enum.auto()
    BOOL = enum.auto()
    QINT8 = enum.auto()
    QUINT8 = enum.auto()
    QINT32 = enum.auto()
    BFLOAT16 = enum.auto()


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


def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


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
