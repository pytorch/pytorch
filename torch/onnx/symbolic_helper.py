
import torch
import warnings
import inspect
from sys import maxsize as maxsize
from typing import Set

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from functools import wraps
from torch._C import OptionalType


# Note [Edit Symbolic Files]
# EDITING THIS FILE AND SYMBOLIC_OPSET<VERSION> FILES? READ THIS FIRST!
#
# - These files is ONLY for ATen operators (e.g., operators that show up in the
#   trace as aten::blah).  If you need to special case a primitive operator,
#   look at _run_symbolic_function
# - Parameter ordering does NOT necessarily match what is in VariableType.cpp;
#   tensors are always first, then non-tensor arguments.
# - Parameter names must *exactly* match the names in VariableType.cpp, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by the trailing underscore, and
#   transparently dispatched to their non inplace versions in
#   'run_symbolic_function'.   See Note [Export inplace]
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

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

# Save some builtins as locals, because we'll shadow them below
_sum = sum


def _parse_arg(value, desc, arg_name=None, node_name=None):
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    if value.node().mustBeNone():
        return None
    if value.node().kind() == 'onnx::Constant':
        tval = value.node()['value']
        if desc == 'i':
            return int(tval)
        elif desc == 'f':
            return float(tval)
        elif desc == 'b':
            return bool(tval)
        elif desc == 's':
            return str(tval)
        elif desc == 't':
            return tval
        elif desc == 'is':
            return [int(v) for v in tval]
        elif desc == 'fs':
            return [float(v) for v in tval]
        else:
            raise RuntimeError("ONNX symbolic doesn't know to interpret Constant node")
    elif value.node().kind() == 'prim::ListConstruct':
        if desc == 'is':
            for v in value.node().inputs():
                if v.node().kind() != 'onnx::Constant':
                    raise RuntimeError("Failed to export an ONNX attribute '" + v.node().kind() +
                                       "', since it's not constant, please try to make "
                                       "things (e.g., kernel size) static if possible")
            return [int(v.node()['value']) for v in value.node().inputs()]
        else:
            raise RuntimeError("ONNX symbolic doesn't know to interpret ListConstruct node")

    if arg_name is None or node_name is None:
        raise RuntimeError("Expected node type 'onnx::Constant', got '{}'.".format(value.node().kind()))
    else:
        raise RuntimeError("Expected node type 'onnx::Constant' "
                           "for argument '{}' of node '{}', got '{}'.".format(arg_name, node_name, value.node().kind()))


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == 'onnx::Constant':
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if _is_value(value) and value.node().kind() not in ('onnx::Constant', 'prim::Constant'):
        raise RuntimeError("ONNX symbolic expected a constant value of the {} argument, got `{}`".format(arg_name, value))
    return _parse_arg(value, desc)


def _unpack_list(list_value):
    list_node = list_value.node()
    assert list_node.kind() == "prim::ListConstruct"
    return list(list_node.inputs())


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
def _is_packed_list(list_value):
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(*arg_descriptors):
    def decorator(fn):
        fn._arg_descriptors = arg_descriptors

        def wrapper(g, *args, **kwargs):
            # some args may be optional, so the length may be smaller
            assert len(arg_descriptors) >= len(args)
            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                arg_names = [None] * len(args)  # type: ignore
                fn_name = None  # type: ignore
            args = [_parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore
                    for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)]  # type: ignore
            # only support _outputs in kwargs
            assert len(kwargs) <= 1
            if len(kwargs) == 1:
                assert '_outputs' in kwargs
            return fn(g, *args, **kwargs)
        # In Python 2 functools.wraps chokes on partially applied functions, so we need this as a workaround
        try:
            wrapper = wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper
    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self

    scalar_type = tensor.type().scalarType()
    if scalar_type:
        ty = scalar_type.lower()
        return getattr(self, ty)()

    return self


def _is_none(x):
    return x.node().mustBeNone()

def _is_value(x):
    return isinstance(x, torch._C.Value)

def _is_tensor(x):
    return x.type().isSubtypeOf(torch._C.TensorType.get())

def _is_tensor_list(x):
    return isinstance(x.type(), torch._C.ListType) and isinstance(x.type().getElementType(), torch._C.TensorType)

def _get_tensor_rank(x):
    if not _is_tensor(x) or x.type() is None:
        return None
    return x.type().dim()

def _get_tensor_sizes(x, allow_nonstatic=True):
    if not _is_tensor(x) or x.type() is None:
        return None
    if allow_nonstatic:
        # Each individual symbol is returned as None.
        # e.g. [1, 'a', 'b'] -> [1, None, None]
        return x.type().varyingSizes()
    # returns None, if exists any symbol in sizes.
    # e.g. [1, 'a', 'b'] -> None
    return x.type().sizes()

def _get_tensor_dim_size(x, dim):
    try:
        sizes = _get_tensor_sizes(x)
        return sizes[dim]
    except Exception:
        pass
    return None

def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def _onnx_unsupported(op_name):
    raise RuntimeError('Unsupported: ONNX export of operator {}. '
                       'Please open a bug to request ONNX export support for the missing operator.'.format(op_name))


def _onnx_opset_unsupported(op_name, current_opset, supported_opset):
    raise RuntimeError('Unsupported: ONNX export of {} in '
                       'opset {}. Please try opset version {}.'.format(op_name, current_opset, supported_opset))

def _onnx_opset_unsupported_detailed(op_name, current_opset, supported_opset, reason):
    raise RuntimeError('Unsupported: ONNX export of {} in '
                       'opset {}. {}. Please try opset version {}.'.format(op_name, current_opset, reason, supported_opset))


def _block_list_in_opset(name):
    def symbolic_fn(*args, **kwargs):
        raise RuntimeError("ONNX export failed on {}, which is not implemented for opset {}. "
                           "Try exporting with other opset versions."
                           .format(name, _export_onnx_opset_version))
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
            index = g.op("Reshape", index, g.op("Constant", value_t=torch.LongTensor([1])))

    index_scalar_type = index.type().scalarType()
    if index_scalar_type is None or index_scalar_type not in ['Long', 'Int']:
        index = g.op("Cast", index, to_i=cast_pytorch_to_onnx["Long"])
    return g.op("Gather", self, index, axis_i=dim)


def _slice_helper(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if _export_onnx_opset_version <= 9:
        from torch.onnx.symbolic_opset9 import _slice as _slice9
        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx.symbolic_opset10 import _slice as _slice10
        return _slice10(g, input, axes, starts, ends, steps, dynamic_slice)

def _hardtanh_helper(g, input, min_val, max_val):
    if _export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import hardtanh
        return hardtanh(g, input, min_val, max_val)
    else:
        from torch.onnx.symbolic_opset11 import hardtanh  # type: ignore[no-redef]
        return hardtanh(g, input, min_val, max_val)

def _is_fp(value):
    if value:
        if isinstance(value, torch.Tensor):
            type = value.dtype
            return (type == 'torch.float32') or (type == 'torch.float64') or (type == 'torch.float16')
        else:
            type = value.type().scalarType()
            if type is None:
                warnings.warn("Type cannot be inferred, which might cause exported graph to produce incorrect results.")
            return (type == 'Float') or (type == 'Double') or (type == 'Half')
    return False


def _sort_helper(g, input, dim, decending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op("Gather", shape_, g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)))
    if _export_onnx_opset_version <= 10:
        if not decending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op("TopK", input, dim_size_, axis_i=dim, largest_i=decending, outputs=2)


def _topk_helper(g, input, k, dim, largest=True, sorted=False, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = g.op("Reshape", k, g.op("Constant", value_t=torch.tensor([1])))
    if _export_onnx_opset_version <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op("TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2)


def _interpolate_warning(interpolate_mode):
    onnx_op = "onnx:Resize" if _export_onnx_opset_version >= 10 else "onnx:Upsample"
    warnings.warn("You are trying to export the model with " + onnx_op + " for ONNX opset version "
                  "" + str(_export_onnx_opset_version) + ". "
                  "This operator might cause results to not match the expected results by PyTorch.\n"
                  "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
                  "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
                  "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
                  "We recommend using opset 11 and above for models using this operator. ")

def _unsqueeze_helper(g, input, axes_i):
    if _export_onnx_opset_version >= 13:
        axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
        return g.op("Unsqueeze", input, axes)
    else:
        return g.op("Unsqueeze", input, axes_i=axes_i)

def _squeeze_helper(g, input, axes_i):
    if _export_onnx_opset_version >= 13:
        axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
        return g.op("Squeeze", input, axes)
    else:
        return g.op("Squeeze", input, axes_i=axes_i)

def _reducesum_helper(g, input, axes_i=None, keepdims_i=1, noop_with_empty_axes_i=0):
    keepdims_i = _maybe_get_const(keepdims_i, 'i')
    if _export_onnx_opset_version >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("ReduceSum", input, axes_i, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
        return g.op("ReduceSum", input, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
    else:
        return g.op("ReduceSum", input, axes_i=axes_i, keepdims_i=keepdims_i)

def _interpolate_size_to_scales(g, input, output_size, dim):
    output_size = _maybe_get_const(output_size, 'is')
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=cast_pytorch_to_onnx["Float"])
        divisor = _slice_helper(g, g.op("Shape", input), axes=[0], ends=[maxsize], starts=[offset])
        divisor = g.op("Cast", divisor, to_i=cast_pytorch_to_onnx["Float"])
        scale_dims = g.op("Div", dividend, divisor)
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [1. if i < 2 else
                           float(output_size[-(dim - i)]) / float(input.type().sizes()[-(dim - i)])
                           for i in range(0, dim)]
        scales = g.op("Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32))
    return scales


def _interpolate_get_scales_if_available(g, scales):
    available_scales = _maybe_get_const(scales[0], 'fs') != -1 and not _is_none(scales[0])

    if not available_scales:
        return None

    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op("Constant", value_t=torch.tensor(_maybe_get_const(scales[0], 'fs')))
    scales = g.op("Concat", offsets, scales_list, axis_i=0)
    return scales


def _get_interpolate_attributes(g, mode, args):
    if mode == 'nearest':
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
    if isinstance(scale_factor.type(), torch._C.ListType) or (scale_factor_rank is not None and scale_factor_rank > 0):
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op("Cast", scale_factor, to_i=cast_pytorch_to_onnx["Float"])
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


def _interpolate_get_scales_and_mode(g, input, size, scale_factor, mode , align_corners):
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    _interpolate_warning(mode)

    align_corners = _maybe_get_const(align_corners, 'b')
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    if not input.type().dim():
        return _unimplemented("interpolate", "missing input shape")
    dim = input.type().dim()

    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = ((_maybe_get_const(size, 't').dim() == 0))
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented("interpolate", "Both size and scales are None in __interpolate")
    return scale_factor, mode


def _interpolate_helper(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = "asymmetric" if interpolate_mode == "nearest" \
            else "align_corners" if align_corners else "pytorch_half_pixel"

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
            output_size = g.op("Cast", output_size, to_i=cast_pytorch_to_onnx['Long'])
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if _export_onnx_opset_version >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
                empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

            return g.op("Resize",
                        input,
                        empty_roi,
                        empty_scales,
                        output_size,
                        coordinate_transformation_mode_s=coordinate_transformation_mode,
                        cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                        mode_s=interpolate_mode,  # nearest, linear, or cubic
                        nearest_mode_s="floor")  # only valid when mode="nearest"
        else:
            if _export_onnx_opset_version >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

            return g.op("Resize",
                        input,
                        empty_roi,
                        scales,
                        coordinate_transformation_mode_s=coordinate_transformation_mode,
                        cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                        mode_s=interpolate_mode,  # nearest, linear, or cubic
                        nearest_mode_s="floor")  # only valid when mode="nearest"
    return symbolic_fn


def __interpolate_helper(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    align_corners = _maybe_get_const(align_corners, 'b')
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = "asymmetric" if mode == "nearest" \
        else "align_corners" if align_corners else "pytorch_half_pixel"

    if not _is_none(size) :
        input_size = g.op("Shape", input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        # in some cases size is not a packed list but size is a scalar
        # We need to also verify that (_maybe_get_const(size, 't').dim() == 0)
        # but this information is not always available. Try to get the dim,
        # and if not assume that it is not a scalar.
        try:
            is_scalar = not _is_packed_list(size) and ((_maybe_get_const(size, 't').dim() == 0))
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn("Cannot verify if the output_size is a scalar "
                              "while exporting interpolate. Assuming that it is not a scalar.")

        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented("interpolate (with a scalar output_size)",
                                      "missing input shape (try giving an array of output_size values)")
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op("Concat", *size, axis_i=0)
        size = g.op("Cast", size, to_i=cast_pytorch_to_onnx['Long'])
        size = g.op("Concat", input_size, size, axis_i=0)

        if _export_onnx_opset_version >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        return g.op("Resize",
                    input,
                    empty_roi,
                    empty_scales,
                    size,
                    coordinate_transformation_mode_s=coordinate_transformation_mode,
                    cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                    mode_s=mode,  # nearest, linear, or cubic
                    nearest_mode_s="floor")
    else:  # if not _is_none(scales)
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented("interpolate (with scales)", "missing input shape")

        if _export_onnx_opset_version >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op("Resize",
                    input,
                    empty_roi,
                    scales,
                    coordinate_transformation_mode_s=coordinate_transformation_mode,
                    cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                    mode_s=mode,  # nearest, linear, or cubic
                    nearest_mode_s="floor")  # only valid when mode="nearest"


def _unbind_helper(g, self, dim, _outputs):
    if _export_onnx_opset_version < 11:
        from torch.onnx.symbolic_opset9 import unbind
    else:
        from torch.onnx.symbolic_opset11 import unbind  # type: ignore[no-redef]
    return unbind(g, self, dim, _outputs)


def _scatter_helper(g, self, dim, index, src):
    if _export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore
    return scatter(g, self, dim, index, src)


def _arange_cast_helper(g, end, start=None, step=None, dtype=None):
    def _is_all_integral(scalars):
        for scalar in scalars:
            try:
                if scalar.type().scalarType() != 'Long':
                    return False
            except Exception:
                pass
        return True

    # This logic is based on torch.arange docs. If 'dtype' is provided,
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
    if _export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accesible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, 'i')
    unsqueezed_index = _unsqueeze_helper(g, index, [i for i in range(self_dim) if i != dim_value])
    expanded_index_shape = scatter(g, g.op("Shape", self), 0,
                                   _unsqueeze_helper(g, dim, [0]), g.op("Shape", index))
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return expanded_index_shape, expanded_index


def _avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name):
    if divisor_override and divisor_override.node().kind() != 'prim::Constant':
        return _unimplemented(name, "divisor_override")
    if not stride:
        stride = kernel_size
    padding = tuple(tuple_fn(padding))
    return padding

def assert_training_mode(op_mode, op_name):
    global _training_mode
    op_mode = True if op_mode == 1 else False
    if op_mode != _training_mode:
        op_mode = "training " if op_mode else "inference"
        training_mode = "training " if _training_mode else "inference"
        # setting the model mode could result in op_mode != _training_mode
        # if the model is a FuncModule. In this case we warn the user of
        # the state and export depending on training_mode
        warnings.warn("ONNX export mode is set to " + training_mode +
                      " mode, but operator " + op_name + " is set to " +
                      op_mode + " mode. The model will be exported in " +
                      training_mode + ", as specified by the export mode.")

def _flatten_helper(g, input, start_dim, end_dim, dim):
    input_size = g.op("Shape", input)
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))]
    if end_dim < dim - 1:
        slice3 = _slice_helper(g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim])
        slices = [slice1, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)), slice3]

    final_shape = g.op("Concat", *slices, axis_i=0)
    from torch.onnx.symbolic_opset9 import _reshape_from_tensor
    return _reshape_from_tensor(g, input, final_shape)

def _is_split_static(split_size_or_sizes, _outputs):
    if _outputs is None:
        return False
    if _is_value(split_size_or_sizes) and split_size_or_sizes.node().kind() != 'onnx::Constant':
        return False
    return True

def _optional_input_placeholder_tensor(g):
    n = g.op("prim::Constant")
    n.setType(OptionalType.ofTensor())
    return n


# ---------------------------------------------------------------------
# ONNX operator version
# ---------------------------------------------------------------------

# READ ME BEFORE EDITING _default_onnx_opset_version:
#
# The variable below controls which ONNX operator set version we are
# targeting. THIS VARIABLE HAS SEMANTIC EFFECT! Say a breaking
# change occurred in version 8. As long as this variable < 8, you can
# export models targeting the old behavior. However, if you bump
# this variable to 8 or later, the breaking change will take into effect:
# you MUST adjust any symbolic affected by breaking changes. The ONNX
# spec publishes a *comprehensive* list of BC-breaking changes for every
# operator revision at:
#
#   https://github.com/onnx/onnx/blob/master/docs/Changelog.md
#
# Please be sure to go through and check all of our implementations here before
# increasing this number. This includes symbolic definitions NOT in this
# file, so grep for "OpName" (with quotes)
#
# Besides, opset_version can be specified in the invocation of export()
# and export_to_pretty_string(), and _export_onnx_opset_version will be set
# and the symbolic functions should check it to determine the behavior
# of the exporter.


_default_onnx_opset_version = 9
_onnx_main_opset = 13
_onnx_stable_opsets = [7, 8, 9, 10, 11, 12]
_export_onnx_opset_version = _default_onnx_opset_version


def _set_opset_version(opset_version):
    global _export_onnx_opset_version
    if opset_version == _default_onnx_opset_version:
        _export_onnx_opset_version = opset_version
        return
    if opset_version in _onnx_stable_opsets + [_onnx_main_opset]:
        _export_onnx_opset_version = opset_version
        return
    raise ValueError("Unsupported ONNX opset version: " + str(opset_version))

_operator_export_type = None
def _set_operator_export_type(operator_export_type):
    global _operator_export_type
    _operator_export_type = operator_export_type

_training_mode = None
def _set_training_mode(training_mode):
    global _training_mode
    _training_mode = training_mode

_onnx_shape_inference = False
def _set_onnx_shape_inference(onnx_shape_inference):
    global _onnx_shape_inference
    _onnx_shape_inference = onnx_shape_inference


# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute 'UINT8'
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
    'Bool': torch.onnx.TensorProtoDataType.BOOL,
    'ComplexFloat': torch.onnx.TensorProtoDataType.COMPLEX64,
    'ComplexDouble': torch.onnx.TensorProtoDataType.COMPLEX128,
    'Undefined': torch.onnx.TensorProtoDataType.UNDEFINED,
}

scalar_name_to_pytorch = {
    'uint8_t': 'Byte',
    'int8_t': 'Char',
    'double': 'Double',
    'float': 'Float',
    'half': 'Half',
    'int': 'Int',
    'int64_t': 'Long',
    'int16_t': 'Short',
    'bool': 'Bool',
    'complex64': 'ComplexFloat',
    'complex128': 'ComplexDouble'
}


# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/da7468853ae322252270bbb58032668bd21b7457/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,        # 0
    torch.int8,         # 1
    torch.short,        # 2
    torch.int,          # 3
    torch.int64,        # 4
    torch.half,         # 5
    torch.float,        # 6
    torch.double,       # 7
    torch.complex32,    # 8
    torch.complex64,    # 9
    torch.complex128,   # 10
    torch.bool,         # 11
]

def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
    cast_pytorch_to_onnx["Undefined"],
    cast_pytorch_to_onnx["ComplexFloat"],
    cast_pytorch_to_onnx["ComplexDouble"],
    cast_pytorch_to_onnx["Bool"],
]

# Global set to store the list of quantized operators in the network.
# This is currently only used in the conversion of quantized ops from PT -> C2 via ONNX.
_quantized_ops: Set[int] = set()
