# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 13
import functools

import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset11 as opset11,
    symbolic_opset9 as opset9,
    utils,
)
from torch.onnx._internal import _beartype, jit_utils, registration


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=13)


def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply


@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    softmax = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )

    return softmax


@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    return return_op


@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    dim_val = symbolic_helper._maybe_get_const(dim, "is")
    if not symbolic_helper._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)
    sqr = g.op("Mul", self, self)
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, dim, keepdims_i=keepdim)
    return g.op("Sqrt", sumsqr)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if (
            symbolic_helper._is_packed_list(split_size_or_sizes)
            and len(symbolic_helper._unpack_list(split_size_or_sizes)) == _outputs
        ):
            split_sizes = [
                symbolic_helper._unsqueeze_helper(g, v, [0])
                for v in symbolic_helper._unpack_list(split_size_or_sizes)
            ]

            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            for i in range(_outputs):
                end = g.op(
                    "Add", start, split_sizes[i]
                )  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        return [
            g.op(
                "SequenceAt",
                split_out,
                g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
            )
            for i in range(_outputs)
        ]

    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    if split_val.dim() > 0:
        return g.op("Split", self, split_size_or_sizes, axis_i=dim, outputs=_outputs)
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            raise errors.SymbolicValueError(
                "Unknown dimension size not supported", self
            )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    splits = g.op("Constant", value_t=torch.tensor(splits))
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::split_with_sizes")
@_beartype.beartype
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unsafe_split")
@_beartype.beartype
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::unsafe_split_with_sizes")
@_beartype.beartype
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::tensor_split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
def tensor_split(
    g: jit_utils.GraphContext, self, indices_or_sections, dim, _outputs=None
):
    axis = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    axis = opset11.unsqueeze(g, axis, 0)
    const_1 = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))

    if symbolic_helper._is_split_static(indices_or_sections, _outputs):
        split_val = symbolic_helper._node_get(indices_or_sections.node(), "value")

        if split_val.dim() > 0:
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            res = []
            assert _outputs is not None
            for i in range(_outputs - 1):
                end = g.op(
                    "Gather",
                    indices_or_sections,
                    g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
                    axis_i=0,
                )
                res.append(g.op("Slice", self, start, end, axis))
                start = end

            end = symbolic_helper._size_helper(g, self, axis)
            res.append(g.op("Slice", self, start, end, axis))
            return res

        split_size = symbolic_helper._get_const(
            indices_or_sections, "i", "indices_or_sections"
        )

        size = symbolic_helper._get_tensor_dim_size(self, dim)
        if size is None:
            if _outputs is not None:
                size = split_size * _outputs
            else:
                raise errors.SymbolicValueError(
                    "Unknown dimension size not supported", self
                )

        min_split_size = size // split_size
        num_splits_one_extra = size % split_size

        splits = num_splits_one_extra * [min_split_size + 1]
        leftover = (split_size - num_splits_one_extra) * [min_split_size]

        splits = g.op(
            "Constant", value_t=torch.tensor(splits + leftover, dtype=torch.long)
        )
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)

    if (
        symbolic_helper._is_tensor(indices_or_sections)
        and symbolic_helper._get_tensor_rank(indices_or_sections) == 1
    ):
        loop_len = symbolic_helper._size_helper(
            g, indices_or_sections, g.op("Constant", value_t=torch.tensor(0))
        )
        loop_len = opset11.unsqueeze(g, loop_len, 0)
        loop_condition = g.op("Cast", const_1, to_i=_C_onnx.TensorProtoDataType.BOOL)

        # To make the first slice in the below loop work,
        # we pad a zero to the first position so that it will be the initial start of slice.
        padding_0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        indices_or_sections = g.op("Concat", padding_0, indices_or_sections, axis_i=0)

        final_splits = g.op("SequenceEmpty")
        # Loop inputs
        loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
            g, "Loop", loop_len, loop_condition, final_splits, outputs=1, n_blocks=1
        )

        loop_block = loop_context.block
        block_input_iter = utils._add_input_to_block(loop_block)
        cond = utils._add_input_to_block(loop_block)
        final_splits = utils._add_input_to_block(loop_block)

        start = loop_context.op(
            "Gather", indices_or_sections, block_input_iter, axis_i=0
        )
        end = loop_context.op(
            "Gather",
            indices_or_sections,
            loop_context.op("Add", block_input_iter, const_1),
            axis_i=0,
        )

        slice = loop_context.op("Slice", self, start, end, axis)
        final_splits = loop_context.op("SequenceInsert", final_splits, slice)

        # Loop outputs
        cond_out = loop_context.op("Identity", loop_condition)
        utils._add_output_to_block(loop_block, cond_out)
        utils._add_output_to_block(loop_block, final_splits)

        loop_out = loop.node().output()
        start = g.op(
            "Gather",
            indices_or_sections,
            g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long)),
            axis_i=0,
        )
        start = opset11.unsqueeze(g, start, 0)
        end = symbolic_helper._size_helper(g, self, axis)

        last_slice = g.op("Slice", self, start, end, axis)

        return g.op("SequenceInsert", loop_out, last_slice)

    else:  # scalar tensor
        dim_size = symbolic_helper._size_helper(g, self, axis)
        min_split_size = g.op("Div", dim_size, indices_or_sections)
        min_split_size_plus_1 = g.op(
            "Add",
            min_split_size,
            const_1,
        )
        num_splits_one_extra = g.op("Mod", dim_size, indices_or_sections)
        splits = g.op("Tile", min_split_size_plus_1, num_splits_one_extra)
        leftover = g.op(
            "Tile",
            min_split_size,
            g.op(
                "Sub",
                opset11.unsqueeze(g, indices_or_sections, 0),
                num_splits_one_extra,
            ),
        )

        splits = g.op("Concat", splits, leftover, axis_i=0)
        if _outputs is None:
            return g.op("SplitToSequence", self, splits, axis_i=dim)
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    splits = g.op("Constant", value_t=torch.tensor([1] * _outputs))
    outputs = g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [
        g.op("Squeeze", out, g.op("Constant", value_t=torch.tensor([dim])))
        for out in outputs
    ]
    return squeezed_outputs


@_onnx_symbolic("aten::nonzero_numpy")
# Emitted from `torch.nonzero(x, as_tuple=True)`
@_beartype.beartype
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    return unbind(g, opset9.nonzero(g, input), 1, _outputs=_outputs)


@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
@_beartype.beartype
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if not symbolic_helper._is_bool(condition):
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    if self is None:
        condition = opset9.nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    return g.op("Where", condition, self, other)


@_onnx_symbolic("aten::fake_quantize_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
@_beartype.beartype
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
@_beartype.beartype
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point)


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


@_onnx_symbolic(
    "aten::sum",
    decorate=[_apply_params("ReduceSum", "sum")],
)
@_beartype.beartype
def _reduce_with_dtype(onnx_op, name):
    symbolic = _reduce_op_symbolic(onnx_op)

    @opset9.overload_by_arg_count
    @_beartype.beartype
    def reduce(g, *args, **kwargs):
        @symbolic_helper.parse_args("v", "none")
        @_beartype.beartype
        def reduce_nodim(g, self, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                self = g.op(
                    "Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type()
                )
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            return symbolic(g, self)

        @symbolic_helper.parse_args("v", "v", "i", "none")
        @_beartype.beartype
        def reduce_dim(g, self, dim, keepdim, dtype):
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                self = g.op(
                    "Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type()
                )
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            return symbolic(g, self, dim, keepdim)

        return reduce_nodim, reduce_dim

    return reduce


@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented("unsafe_chunk", "unknown dimension size")
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


@_onnx_symbolic("aten::repeat_interleave")
@_beartype.beartype
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=None, output_size=None
):
    input = self
    final_dim = dim
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if symbolic_helper._is_none(dim):
        input = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = 0
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)

    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    input_sizes = symbolic_helper._get_tensor_sizes(input)
    if repeats_dim is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats rank.",
            self,
        )
    if repeats_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size.",
            self,
        )
    if input_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size.",
            self,
        )
    # Handle cases where dim is negative
    if dim < 0:
        dim += len(input_sizes)

    output_sizes = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            output_sizes[idx], input_sizes[idx] = 0, -1

    cond_dynamic_repeats = repeats_dim == 1 and repeats_sizes[0] is None
    # If input size is dynamic or repeats vector is dynamic
    if output_sizes[dim] == 0 or cond_dynamic_repeats:
        reps = symbolic_helper._size_helper(g, input, dim)
        reps = opset11.unsqueeze(g, reps, 0)
        # Check if repeats vector is a single integer value
        # or a single dimension tensor with non-dynamic values
        if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
            if not symbolic_helper._is_tensor(repeats):
                repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
            repeats = g.op("Expand", repeats, reps)
        # Check if repeats is dynamic
        # As repeats is dynamic, we use a where node as a substitute for the if statement
        # If repests_dim = 1, expand repeats otherwise use original tensor
        elif cond_dynamic_repeats:
            repeat_dim = symbolic_helper._size_helper(
                g, repeats, g.op("Constant", value_t=torch.LongTensor([0]))
            )
            repeat_cond = g.op(
                "Equal", repeat_dim, g.op("Constant", value_t=torch.LongTensor([1]))
            )
            repeats = where(g, repeat_cond, g.op("Expand", repeats, reps), repeats)
    # There are cases when the repeats are 1-d tensor with multiple repeats, but dim
    # provided along one of the dynamic axes provided. A simple example would be
    # input.shape -> [1, 1, *] where * represents the dynamic axes, and dim = 2
    # Now, repeat interleaving can be performed in pytorch when the value of * matches
    # with the number of elements in repeat, for example if * -> 2, number of repeats
    # should be 2 as well.
    else:
        return opset9.repeat_interleave(g, self, repeats, final_dim)

    reps_like = g.op(
        "ConstantOfShape",
        g.op("Shape", repeats),
        value_t=torch.tensor([1], dtype=torch.long),
    )
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
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    loop_len = reps

    # Create an empty sequence to store final expansions
    final_splits = g.op("SequenceEmpty")

    # Loop inputs
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, final_splits, n_blocks=1
    )

    loop_block = loop_context.block
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)
    final_splits = utils._add_input_to_block(loop_block)

    r_split = loop_context.op("SequenceAt", r_splits, block_input_iter)
    i_split = loop_context.op("SequenceAt", i_splits, block_input_iter)

    i_split = opset11.unsqueeze(loop_context, i_split, dim + 1)
    r_concat = [
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[: dim + 1])),
        r_split,
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[dim + 1 :])),
    ]
    r_concat = loop_context.op("Concat", *r_concat, axis_i=0)
    i_split = opset9.expand(loop_context, i_split, r_concat, None)
    i_split = symbolic_helper._reshape_helper(
        loop_context, i_split, g.op("Constant", value_t=torch.LongTensor(output_sizes))
    )
    final_splits = loop_context.op("SequenceInsert", final_splits, i_split)

    # Loop outputs
    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, final_splits)

    loop_out = loop.node().output()
    loop_out = g.op("ConcatFromSequence", loop_out, axis_i=dim)
    return loop_out


@_onnx_symbolic("aten::diagonal")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def diagonal(g: jit_utils.GraphContext, self, offset, dim1, dim2):
    dim1_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim1]))
    )
    dim2_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim2]))
    )

    # Create appropriate mask
    mask_shape = g.op("Concat", dim1_size, dim2_size, axis_i=0)
    mask = opset9.zeros(g, mask_shape, None, None, None)
    mask = g.op("EyeLike", mask, k_i=offset)

    # dim1 and dim2 appended as a dimension at the end of the shape
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes.remove(dim1)
        axes.remove(dim2)
        self = g.op("Transpose", self, perm_i=axes + [dim1, dim2])
    else:
        return symbolic_helper._unimplemented("diagonal", "unknown input rank")

    # Multiply input and mask to calculate values along diagonal
    # The mask consists of one values where diagonal values are to be calculated
    # For example:
    # [[1.1, 1.2, 1.3],   *    [[1, 0, 0]   =   [[1.1, 0, 0],
    #  [2.1, 2.2, 2.3],         [0, 1, 0]        [0, 2.2, 0],
    #  [3.1, 3.2, 3.3]]         [0, 0, 1]]       [0, 0, 3.3]]
    result = g.op("Mul", self, mask)
    result = symbolic_helper._reducesum_helper(g, result, axes_i=[-1], keepdims_i=0)

    # Calculate gather indices based on offset and dims
    # If offset is greater than zero, set offset to zero as this aids in
    # calculation of selection window
    offset_op = g.op("Constant", value_t=torch.LongTensor([offset]))
    if offset >= 0:
        diag_size = g.op(
            "Max",
            g.op("Min", dim1_size, g.op("Sub", dim2_size, offset_op)),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
        offset = 0
    else:
        diag_size = g.op(
            "Max",
            g.op("Min", g.op("Add", dim1_size, offset_op), dim2_size),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
    diag_size = g.op("Concat", diag_size, axis_i=0)

    # Calculate which diagonal values to select
    # For example, in cases with offsets:
    # [[0, 1.1, 0]
    #  [0, 0, 2.2]]
    # we need to select the last two columns, so we create a tensor
    # with all columns that are to be selected
    # So in this example, it is [1, 2]
    select_window_ones_fill = opset9.ones(g, diag_size, 4, None, None)
    select_window = g.op(
        "CumSum",
        select_window_ones_fill,
        g.op("Constant", value_t=torch.LongTensor([0])),
    )
    select_window = g.op(
        "Add",
        select_window,
        g.op("Constant", value_t=torch.LongTensor([abs(offset) - 1])),
    )

    gather_shape = [
        opset9.size(g, result, dim=g.op("Constant", value_t=torch.LongTensor([axis])))
        for axis in list(range(rank))[:-2]
    ]
    gather_shape.append(diag_size)
    gather_shape = g.op("Concat", *gather_shape, axis_i=0)
    gather_indices = opset9.zeros(g, gather_shape, 4, None, None)

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
    overrun_cond = g.op(
        "Not",
        g.op(
            "Equal",
            diag_size,
            g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64)),
        ),
    )

    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", overrun_cond, n_blocks=2
    )

    gather_indices_if_block = if_context.op("Add", gather_indices, select_window)
    gather_indices_if_block = symbolic_helper._unsqueeze_helper(
        if_context, gather_indices_if_block, [rank - 1]
    )
    final_non_overrun = if_context.op(
        "GatherND", result, gather_indices_if_block, batch_dims_i=rank - 2
    )
    final_overrun = opset9.zeros(else_context, gather_shape, 6, None, None)
    utils._add_output_to_block(if_context.block, final_non_overrun)
    utils._add_output_to_block(else_context.block, final_overrun)
    return if_op


# Quantized ops


@_onnx_symbolic("quantized::linear")
@_beartype.beartype
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.linear(g, input, weight, bias)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d")
@_beartype.beartype
def quantized_conv2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::conv2d_relu")
@_beartype.beartype
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
