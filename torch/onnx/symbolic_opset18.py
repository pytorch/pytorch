"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
    Split
"""

import functools
from typing import Sequence

from torch import _C
from torch.onnx import (
    errors,
    symbolic_helper,
    symbolic_opset9 as opset9,
    utils,
)
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["col2im"]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
@_beartype.beartype
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
):
    # convert [i0, i1, ..., in] into [i0, i0, i1, i1, ..., in, in]
    adjusted_padding = []
    for pad in padding:
        for _ in range(2):
            adjusted_padding.append(pad)

    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
    if not adjusted_padding:
        adjusted_padding = [0, 0] * num_dimensional_axis

    if not dilation:
        dilation = [1] * num_dimensional_axis

    if not stride:
        stride = [1] * num_dimensional_axis

    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        pads_i=adjusted_padding,
        strides_i=stride,
    )

import torch
import torch._C._onnx as _C_onnx
from typing import Callable, List, Optional, Sequence, Tuple, Union
import math
from torch.onnx._globals import GLOBALS
import sys

@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def _logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    if dim is None:
        return g.op("ReduceLogSumExp", input, keepdims_i=0)
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceLogSumExp", input, axes, keepdims_i=keepdim)

@_beartype.beartype
def _reduce_op_symbolic(onnx_op_name):
    @_beartype.beartype
    def symbolic(g, self, dim=None, keepdim=None):
        self = opset9._maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return symbolic_helper._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            keepdim = symbolic_helper._get_const(keepdim, "i", "keepdim")
            return g.op(onnx_op_name, self, dim, keepdims_i=keepdim)

    return symbolic

@_onnx_symbolic("aten::norm")
@symbolic_helper.parse_args("v", "t", "is", "i", "v")
@_beartype.beartype
def _norm(g: jit_utils.GraphContext, self, p, dim, keepdim, dtype=None):
    if p == 1:
        f = _reduce_op_symbolic("ReduceL1")
    elif p == 2:
        f = _reduce_op_symbolic("ReduceL2")
    else:
        raise errors.SymbolicValueError(
            "ONNX export only p-norms with p of 1 or 2", self
        )
    result = f(g, self, dim, keepdim=keepdim)
    if dtype is not None:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=symbolic_helper._type_utils.JitScalarType(dtype).onnx_type())
    return result

@_onnx_symbolic("aten::linalg_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def _linalg_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
    ord_value = None
    if dim is None:
        if symbolic_helper._is_none(ord):
            self = symbolic_helper._reshape_helper(g, self, [-1])
            ord = g.op("Constant", value_t=torch.LongTensor([2]))
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "dim", "Input rank must be known at export time.", self
            )
        if self_dim == 1:
            ord_value = symbolic_helper._parse_arg(ord, "f")
        else:
            dim = [0, 1]
    else:
        if len(dim) == 1:
            if symbolic_helper._is_none(ord):
                ord = g.op("Constant", value_t=torch.LongTensor([2]))
            ord_value = symbolic_helper._parse_arg(ord, "f")
    if ord_value:
        return _linalg_vector_norm(g, self, ord_value, dim, keepdim, dtype)
    return _linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)

@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
@_beartype.beartype
def _linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    if symbolic_helper._is_none(dim):
        self = symbolic_helper._reshape_helper(g, self, [-1])
        keepdim = False
        axes = None
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))

    if ord == math.inf:
        if axes is None:
            result = g.op("ReduceMax", g.op("Abs", self), keepdims_i=keepdim)
        else:
            result = g.op("ReduceMax", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == -math.inf:
        if axes is None:
            result = g.op("ReduceMin", g.op("Abs", self), keepdims_i=keepdim)
        else:
            result = g.op("ReduceMin", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == 0:
        if dim is None:
            self = symbolic_helper._reshape_helper(
                g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
            )
            keepdim = False

        cond_op = g.op(
            "Not", g.op("Equal", self, g.op("Constant", value_t=torch.LongTensor([0])))
        )
        cond_op = g.op(
            "Cast",
            cond_op,
            to_i=symbolic_helper._type_utils.JitScalarType.from_value(self).onnx_type(),
        )
        return symbolic_helper._reducesum_helper(
            g, cond_op, axes_i=dim, keepdims_i=keepdim
        )
    elif ord == 1:
        if axes is None:
            result = _reduce_op_symbolic("ReduceL1")(g, self, keepdim=keepdim)
        else:
            result = _reduce_op_symbolic("ReduceL1")(g, self, axes, keepdim=keepdim)
    elif ord == 2:
        if axes is None:
            result = _reduce_op_symbolic("ReduceL2")(g, self, keepdim=keepdim)
        else:
            result = _reduce_op_symbolic("ReduceL2")(g, self, axes, keepdim=keepdim)
    else:
        ord_op = g.op("Constant", value_t=torch.tensor(ord, dtype=torch.float32))
        result = symbolic_helper._reducesum_helper(
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

    if not symbolic_helper._is_none(dtype):
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())  # type: ignore[arg-type]
    return result

@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def _linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: List[int],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # Conditions based on https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html
    ord_value = symbolic_helper._parse_arg(ord, "s")
    if ord_value == "fro":
        return _frobenius_norm(g, self, dim, keepdim)
    elif ord_value == "nuc":
        return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==nuc", self)
    else:
        ord_value = symbolic_helper._parse_arg(ord, "f")
        if ord_value is None:
            return _frobenius_norm(g, self, dim, keepdim)
        if ord_value == 2 or ord_value == -2:
            # ord = 2/-2 unimplemented due to lack of operators
            # used to calculate singular values
            return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==2", self)
        # Wrap the dim vector to handle negative dim values
        self_dim = symbolic_helper._get_tensor_rank(self)
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "linalg.matrix_norm", "Input rank must be known at export time.", self
            )
        # Common implementation for cases with
        # ord = 1/-1 and ord = inf/-inf
        if dim[0] < 0:
            dim[0] += self_dim
        if dim[1] < 0:
            dim[1] += self_dim

        if ord_value == math.inf or ord_value == -math.inf:
            dim[0], dim[1] = dim[1], dim[0]
        if dim[1] > dim[0] and not keepdim:
            dim[1] -= 1
        sum = symbolic_helper._reducesum_helper(
            g, g.op("Abs", self), axes_i=[dim[0]], keepdims_i=keepdim
        )
        if ord_value > 0:
            result, indices = opset9.max(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        else:
            result, indices = opset9.min(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        return result

@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def _frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    dim_val = symbolic_helper._maybe_get_const(dim, "is")
    if not symbolic_helper._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)
    sqr = g.op("Mul", self, self)
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)

def _numel(g: jit_utils.GraphContext, self):
    shape = g.op("Shape", self)
    return g.op("ReduceProd", shape, keepdims_i=0)

@symbolic_helper.parse_args("v", "is", "i", "i")
@_beartype.beartype
def _var_mean(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    if dim is None:
        mean = g.op("ReduceMean", input, keepdims_i=0)
        t_mean = mean
        num_elements = _numel(g, input)
    else:
        axes = g.op(
            "Constant", value_t=torch.tensor(dim, dtype=torch.long)
        )

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
    if dim is None:
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


@_onnx_symbolic("aten::std")
@_beartype.beartype
def _std(g: jit_utils.GraphContext, input, *args):
    var, _ = var_mean(g, input, *args)
    return g.op("Sqrt", var)


@_onnx_symbolic("aten::var")
@_beartype.beartype
def _var(g: jit_utils.GraphContext, input, *args):
    var, _ = var_mean(g, input, *args)
    return var


@_onnx_symbolic("aten::var_mean")
@_beartype.beartype
def var_mean(g: jit_utils.GraphContext, input, *args):
    # var_mean (and all variance-related functions) has multiple signatures, so need to manually figure
    # out the correct arguments:
    # aten::var_mean(Tensor self, bool unbiased)
    # aten::var_mean(Tensor self, int[1] dim, bool unbiased, bool keepdim=False)
    # aten::var_mean(Tensor self, int[1]? dim=None, *, int? correction=None, bool keepdim=False)
    if len(args) == 1:
        return _var_mean(g, input, None, args[0], None)
    else:
        return _var_mean(g, input, *args)


@_onnx_symbolic("aten::std_mean")
@_beartype.beartype
def _std_mean(g: jit_utils.GraphContext, input, *args):
    var, mean = var_mean(g, input, *args)
    return g.op("Sqrt", var), mean

def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply

# @_onnx_symbolic("aten::prod", decorate=[_apply_params("ReduceProd", "prod")],)
@_beartype.beartype
def _reduce_with_dtype(onnx_op: str, name: str):
    symbolic = _reduce_op_symbolic(onnx_op)

    @opset9.overload_by_arg_count
    @_beartype.beartype
    def reduce(g, *args, **kwargs):
        @symbolic_helper.parse_args("v", "none")
        @_beartype.beartype
        def reduce_nodim(g, self, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = symbolic_helper._type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            result = symbolic(g, self)
            if dtype_onnx is not None:
                result_dtype_onnx = symbolic_helper._type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        @symbolic_helper.parse_args("v", "v", "i", "none")
        @_beartype.beartype
        def reduce_dim(g, self, dim, keepdim, dtype):
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = symbolic_helper._type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            result = symbolic(g, self, dim, keepdim)
            if dtype_onnx is not None:
                result_dtype_onnx = symbolic_helper._type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        return reduce_nodim, reduce_dim

    return reduce

@_onnx_symbolic("aten::amax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def _amax(g: jit_utils.GraphContext, self, dim, keepdim):
    if dim is None:
        return g.op("ReduceMax", self, keepdims_i=keepdim)
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceMax", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::amin")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def _amin(g: jit_utils.GraphContext, self, dim, keepdim):
    if dim is None:
        return g.op("ReduceMin", self, keepdims_i=keepdim)
    else:
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceMin", self, axes, keepdims_i=keepdim)


@_onnx_symbolic("aten::aminmax")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def _aminmax(g: jit_utils.GraphContext, self, dim, keepdim):
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        return g.op("ReduceMin", self, axes, keepdims_i=keepdim), g.op("ReduceMax", self, axes, keepdims_i=keepdim)
    else:
        return g.op("ReduceMin", self, keepdims_i=keepdim), g.op(
            "ReduceMax", self, keepdims_i=keepdim
        )

@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def _glu(g: jit_utils.GraphContext, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", input, axis_i=dim, num_outputs_i=2, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))

@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
@_beartype.beartype
def _embedding_bag(
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
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    zero = g.op("Constant", value_t=torch.tensor([0]))

    indices_len = symbolic_helper._unsqueeze_helper(
        g,
        symbolic_helper._size_helper(
            g, indices, g.op("Constant", value_t=torch.tensor(0))
        ),
        [0],
    )
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # Offsets holds the starting index position of each bag. So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row. Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output, so we can avoid creating a sequence and inserting elements in.
    offsets_starts = symbolic_helper._slice_helper(
        g, offsets, axes=[0], starts=[0], ends=[sys.maxsize], steps=[1]
    )
    offsets_ends = symbolic_helper._slice_helper(
        g, offsets, axes=[0], starts=[1], ends=[sys.maxsize], steps=[1]
    )

    loop_len = symbolic_helper._size_helper(
        g, offsets_ends, g.op("Constant", value_t=torch.tensor(0))
    )

    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    loop_block = loop_context.block

    # FIXME(justinchuby): We need to handle what happens when we call b.op on a node return
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)

    indices_start = loop_context.op(
        "Gather", offsets_starts, block_input_iter, axis_i=0
    )
    indices_end = loop_context.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    indices_start = symbolic_helper._unsqueeze_helper(loop_context, indices_start, [0])
    indices_end = symbolic_helper._unsqueeze_helper(loop_context, indices_end, [0])

    indices_row = loop_context.op("Slice", indices, indices_start, indices_end, zero)
    embeddings = loop_context.op("Gather", embedding_matrix, indices_row, axis_i=0)
    if not symbolic_helper._is_none(per_sample_weights):
        per_sample_weights_row = loop_context.op(
            "Slice", per_sample_weights, indices_start, indices_end, zero
        )
        per_sample_weights_row = symbolic_helper._unsqueeze_helper(
            loop_context, per_sample_weights_row, [1]
        )
        embeddings = loop_context.op("Mul", embeddings, per_sample_weights_row)
    if mode == 0:
        embeddings = symbolic_helper._reducesum_helper(
            loop_context, embeddings, axes_i=[0], keepdims_i=0
        )
    elif mode == 1:
        axes = loop_context.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        embeddings = loop_context.op("ReduceMean", embeddings, axes, keepdims_i=0)
    else:
        axes = loop_context.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        embeddings = loop_context.op("ReduceMax", embeddings, axes, keepdims_i=0)

    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, embeddings)

    # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
    # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    return loop.node().output(), None, None, None
