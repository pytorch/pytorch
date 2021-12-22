# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 13
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.symbolic_opset9 import (overload_by_arg_count, _maybe_cast_reduce_op_input,
                                        nonzero, expand, zeros, ones, size)
from torch.onnx.symbolic_opset11 import unsqueeze
from torch.onnx.utils import _add_block, _add_input_to_block, _add_output_to_block


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 13


@parse_args("v", "i", "none")
def softmax(g, input, dim, dtype=None):
    softmax = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])

    return softmax


@parse_args("v", "i", "none")
def log_softmax(g, input, dim, dtype=None):
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        return_op = g.op("Cast", return_op, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return return_op


@parse_args("v", "v", "i")
def frobenius_norm(g, self, dim=None, keepdim=False):
    dim_val = sym_help._maybe_get_const(dim, "is")
    if not sym_help._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)
    sqr = g.op("Mul", self, self)
    sumsqr = sym_help._reducesum_helper(g, sqr, dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)


@parse_args("v", "v", "i", "i")
def split(g, self, split_size_or_sizes, dim, _outputs=None):
    if not sym_help._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if sym_help._is_packed_list(split_size_or_sizes) and \
                len(sym_help._unpack_list(split_size_or_sizes)) == _outputs:
            split_sizes = [sym_help._unsqueeze_helper(g, v, [0]) for v in sym_help._unpack_list(split_size_or_sizes)]

            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            for i in range(_outputs):
                end = g.op("Add", start, split_sizes[i])  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        return [g.op("SequenceAt", split_out, g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)))
                for i in range(_outputs)]

    split_val = split_size_or_sizes.node()["value"]
    if split_val.dim() > 0:
        return g.op("Split", self, split_size_or_sizes, axis_i=dim, outputs=_outputs)
    split_size = sym_help._get_const(split_size_or_sizes, "i", "split_size")

    size = sym_help._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            raise RuntimeError("Unknown dimension size not supported")
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    splits = g.op("Constant", value_t=torch.tensor(splits))
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


def split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


def unsafe_split(g, self, split_size_or_sizes, dim, _outputs=None):
    return split(g, self, split_size_or_sizes, dim, _outputs)


def unsafe_split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@parse_args("v", "i", "i")
def unbind(g, self, dim=0, _outputs=None):
    if _outputs is None:
        return g.op("SplitToSequence",
                    self,
                    g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
                    axis_i=dim, keepdims_i=0)

    splits = g.op("Constant", value_t=torch.tensor([1] * _outputs))
    outputs = g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [g.op("Squeeze", out, g.op("Constant", value_t=torch.tensor([dim]))) for out in outputs]
    return squeezed_outputs


# Emitted from `torch.nonzero(x, as_tuple=True)`
def nonzero_numpy(g, input, _outputs=None):
    return unbind(g, nonzero(g, input), 1, _outputs=_outputs)


@parse_args("v", "v", "v", "i")
def where(g, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if condition.type().scalarType() != "Bool":
        condition = g.op("Cast", condition, to_i=sym_help.cast_pytorch_to_onnx["Bool"])
    if self is None:
        condition = nonzero(g, condition)
        return sym_help._unbind_helper(g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs)
    return g.op("Where", condition, self, other)

@parse_args("v", "v", "v", "i", "i", "i")
def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))

    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx["Byte"])
    else:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx["Char"])
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
        scale, zero_point, axis_i=axis)

def _reduce_op_symbolic(onnx_op_name):
    def symbolic(g, self, dim=None, keepdim=None):
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return sym_help._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            keepdim = sym_help._get_const(keepdim, "i", "keepdim")
            return g.op(onnx_op_name, self, dim, keepdims_i=keepdim)
    return symbolic

def _reduce_with_dtype(onnx_op, name):
    symbolic = _reduce_op_symbolic(onnx_op)

    @overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = sym_help._get_const(dtype, "i", "dtype")
                self = g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype")
            return symbolic(g, self)

        @parse_args("v", "v", "i", "none")
        def reduce_dim(g, self, dim, keepdim, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = sym_help._get_const(dtype, "i", "dtype")
                self = g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype")
            return symbolic(g, self, dim, keepdim)
        return reduce_nodim, reduce_dim
    return reduce

sum = _reduce_with_dtype("ReduceSum", "sum")

@parse_args("v", "i", "i", "i")
def unsafe_chunk(g, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return g.op("SplitToSequence",
                    self,
                    g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
                    axis_i=dim, keepdims_i=0)

    size = sym_help._get_tensor_dim_size(self, dim)
    if size is None:
        return _unimplemented("unsafe_chunk", "unknown dimension size")
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)

    # TODO: So far we don"t have a module using this method. We"ll keep
    # this as a constant unless we see a request of dynamics in any
    # user's modules.
    splits = g.op("Constant", value_t=torch.tensor(splits, dtype=torch.long))
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)

def repeat_interleave(g, self, repeats, dim=None, output_size=None):
    input = self
    final_dim = dim
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if sym_help._is_none(dim):
        input = sym_help._reshape_helper(g, self, g.op("Constant", value_t=torch.tensor([-1])))
        dim = 0
    else:
        dim = sym_help._maybe_get_scalar(dim)

    repeats_dim = sym_help._get_tensor_rank(repeats)
    repeats_sizes = sym_help._get_tensor_sizes(repeats)
    input_sizes = sym_help._get_tensor_sizes(input)
    if repeats_dim is None:
        raise RuntimeError("Unsupported: ONNX export of repeat_interleave for unknown "
                           "repeats rank.")
    if repeats_sizes is None:
        raise RuntimeError("Unsupported: ONNX export of repeat_interleave for unknown "
                           "repeats size.")
    if input_sizes is None:
        raise RuntimeError("Unsupported: ONNX export of repeat_interleave for unknown "
                           "input size.")
    # Handle cases where dim is negative
    if dim < 0:
        dim += len(input_sizes)

    output_sizes = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            output_sizes[idx], input_sizes[idx] = 0, -1
    print(output_sizes, input_sizes)

    cond_dynamic_repeats = (repeats_dim == 1 and repeats_sizes[0] is None)
    # If input size is dynamic or repeats vector is dynamic
    if output_sizes[dim] == 0 or cond_dynamic_repeats:
        reps = sym_help._size_helper(g, input, dim)
        reps = unsqueeze(g, reps, 0)
        # Check if repeats vector is a single integer value
        # or a single dimension tensor with non-dynamic values
        if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
            if not sym_help._is_tensor(repeats):
                repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
            repeats = g.op("Expand", repeats, reps)
        # Check if repeats is dynamic
        # As repeats is dynamic, we use a where node as a substitute for the if statement
        # If repests_dim = 1, expand repeats otherwise use original tensor
        elif cond_dynamic_repeats:
            repeat_dim = sym_help._size_helper(g, repeats, g.op("Constant", value_t=torch.LongTensor([0])))
            repeat_cond = g.op("Equal", repeat_dim, g.op("Constant", value_t=torch.LongTensor([1])))
            repeats = where(g, repeat_cond, g.op("Expand", repeats, reps), repeats)
    # There are cases when the repeats are 1-d tensor with multiple repeats, but dim
    # provided along one of the dynamic axes provided. A simple example would be
    # input.shape -> [1, 1, *] where * represents the dynamic axes, and dim = 2
    # Now, repeat interleaving can be performed in pytorch when the value of * matches
    # with the number of elements in repeat, for example if * -> 2, number of repeats
    # should be 2 as well.
    else:
        return torch.onnx.symbolic_opset9.repeat_interleave(g, self, repeats, final_dim)

    reps_like = g.op("ConstantOfShape", g.op("Shape", repeats),
                     value_t=torch.tensor([1], dtype=torch.long))
    r_splits = split(g, repeats, reps_like, 0)
    i_splits = split(g, input, reps_like, dim)

    output_sizes[dim], input_sizes[dim] = -1, 1

    # Create a loop to iterate over each value along the dimension
    # and perform individual interleaving using the repeats tensor
    # Loop is of the following pattern
    # input (trip_count, cond)
    #   int trip_count = ...;
    #   bool cond = ...;
    #   for (int i=0; i < trip_count && cond; ++i) {
    #     cond = ...;
    #   }

    # Loop conditions
    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=9)
    loop_len = reps

    # Create an empty sequence to store final expansions
    final_splits = g.op("SequenceEmpty")
    loop = g.op("Loop", loop_len, loop_condition, final_splits)

    # Loop inputs
    loop_block = _add_block(loop.node())
    block_input_iter = _add_input_to_block(loop_block)
    cond = _add_input_to_block(loop_block)
    final_splits = _add_input_to_block(loop_block)

    r_split = loop_block.op("SequenceAt", r_splits, block_input_iter)
    i_split = loop_block.op("SequenceAt", i_splits, block_input_iter)

    i_split = unsqueeze(loop_block, i_split, dim + 1)
    r_concat = [loop_block.op("Constant", value_t=torch.LongTensor(input_sizes[:dim + 1])),
                r_split,
                loop_block.op("Constant", value_t=torch.LongTensor(input_sizes[dim + 1:]))]
    r_concat = loop_block.op("Concat", *r_concat, axis_i=0)
    i_split = expand(loop_block, i_split, r_concat, None)
    i_split = sym_help._reshape_helper(loop_block, i_split,
                                       g.op("Constant", value_t=torch.LongTensor(output_sizes)))
    final_splits = loop_block.op("SequenceInsert", final_splits, i_split)

    # Loop outputs
    cond_out = loop_block.op("Cast", loop_condition, to_i=9)
    _add_output_to_block(loop_block, cond_out)
    _add_output_to_block(loop_block, final_splits)

    loop_out = loop.node().output()
    loop_out = g.op("ConcatFromSequence", loop_out, axis_i=dim)
    return loop_out


@parse_args("v", "i", "i", "i")
def diagonal(g, self, offset, dim1, dim2):
    dim1_size = size(g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim1])))
    dim2_size = size(g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim2])))

    # Create appropriate mask
    mask_shape = g.op("Concat", dim1_size, dim2_size, axis_i=0)
    mask = zeros(g, mask_shape, None, None, None)
    mask = g.op("EyeLike", mask, k_i=offset)

    # dim1 and dim2 appended as a dimension at the end of the shape
    rank = sym_help._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes.remove(dim1)
        axes.remove(dim2)
        self = g.op("Transpose", self, perm_i=axes + [dim1, dim2])
    else:
        return _unimplemented("diagonal", "unknown input rank")

    # Multiply input and mask to calculate values along diagonal
    # The mask consists of one values where diagonal values are to be calculated
    # For example:
    # [[1.1, 1.2, 1.3],   *    [[1, 0, 0]   =   [[1.1, 0, 0],
    #  [2.1, 2.2, 2.3],         [0, 1, 0]        [0, 2.2, 0],
    #  [3.1, 3.2, 3.3]]         [0, 0, 1]]       [0, 0, 3.3]]
    result = g.op("Mul", self, mask)
    result = sym_help._reducesum_helper(g, result, axes_i=[-1], keepdims_i=0)

    # Calculate gather indices based on offset and dims
    # If offset is greater than zero, set offset to zero as this aids in
    # calculation of selection window
    offset_op = g.op("Constant", value_t=torch.LongTensor([offset]))
    if offset >= 0:
        diag_size = g.op("Max", g.op("Min", dim1_size, g.op("Sub", dim2_size, offset_op)),
                         g.op("Constant", value_t=torch.LongTensor([0])))
        offset = 0
    else:
        diag_size = g.op("Max", g.op("Min", g.op("Add", dim1_size, offset_op), dim2_size),
                         g.op("Constant", value_t=torch.LongTensor([0])))
    diag_size = g.op("Concat", diag_size, axis_i=0)

    # Calculate which diagonal values to select
    # For example, in cases with offsets:
    # [[0, 1.1, 0]
    #  [0, 0, 2.2]]
    # we need to select the last two columns, so we create a tensor
    # with all columns that are to be selected
    # So in this example, it is [1, 2]
    select_window_ones_fill = ones(g, diag_size, 4, None, None)
    select_window = g.op("CumSum", select_window_ones_fill, g.op("Constant", value_t=torch.LongTensor([0])))
    select_window = g.op("Add", select_window, g.op("Constant", value_t=torch.LongTensor([abs(offset) - 1])))

    gather_shape = [size(g, result,
                         dim=g.op("Constant", value_t=torch.LongTensor([axis]))) for axis in list(range(rank))[:-2]]
    gather_shape.append(diag_size)
    gather_shape = g.op("Concat", *gather_shape, axis_i=0)
    gather_indices = zeros(g, gather_shape, 4, None, None)

    # There might be cases where offset value is greater than number of rows/columns
    # and might cause the diagonal to overrun and as a result of this, diag_size would be zero.
    # For example, if
    #       offset = 9, dim1_size = 2 (columns), dim2_size = 4 (rows)
    #       diag_size = max(min(2, (4-9)), 0) = 0, based on calculation above
    # Cases with diagonal overrun always result in diag_size = max(0, -ve value) = 0
    # In cases without diagonal overrun, we select the appropriate rows/columns along which we
    # are calculating diagonal values. In cases with diagonal overrun, we return a tensor which has
    # the dimension of the row/column where overrun occurred as 0-dim, as we are essentially
    # returning an empty tensor
    overrun_cond = g.op("Not", g.op("Equal", diag_size, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))))
    if_op = g.op("If", overrun_cond)
    if_node = if_op.node()

    if_block = _add_block(if_node)
    gather_indices_if_block = if_block.op("Add", gather_indices, select_window)
    gather_indices_if_block = sym_help._unsqueeze_helper(if_block, gather_indices_if_block, [rank - 1])
    final_non_overrun_ = if_block.op("GatherND", result, gather_indices_if_block, batch_dims_i=rank - 2)
    _add_output_to_block(if_block, final_non_overrun_)

    else_block = _add_block(if_node)
    final_overrun_ = zeros(else_block, gather_shape, 6, None, None)
    _add_output_to_block(else_block, final_overrun_)
    return if_op
