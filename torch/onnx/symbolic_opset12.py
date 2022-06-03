import sys
import warnings

import torch
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_opset9 as opset9
from torch.onnx import utils

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12


def einsum_helper(g, equation, tensors):
    if not tensors:
        raise RuntimeError("Einsum inputs are empty.")
    # ONNX does not support bool for Einsum inputs.
    if tensors[0].type().scalarType() == "Bool":
        tensors = [
            g.op("Cast", tensor, to_i=symbolic_helper.cast_pytorch_to_onnx["Long"])
            for tensor in tensors
        ]
        return g.op(
            "Cast",
            g.op("Einsum", *tensors, equation_s=equation),
            to_i=symbolic_helper.cast_pytorch_to_onnx["Bool"],
        )
    else:
        return g.op("Einsum", *tensors, equation_s=equation)


@symbolic_helper.parse_args("s", "v")
def einsum(g, equation, tensor_list):
    tensors = symbolic_helper._unpack_list(tensor_list)
    return einsum_helper(g, equation, tensors)


@symbolic_helper.parse_args("v", "v")
def outer(g, input, other):
    # make sure to cast other to self's type
    if other.type().scalarType() != input.type().scalarType():
        other = g.op(
            "Cast",
            other,
            to_i=symbolic_helper.cast_pytorch_to_onnx[input.type().scalarType()],
        )
    return einsum_helper(g, "i,j->ij", [input, other])


@symbolic_helper.parse_args("v", "f", "i")
def dropout(g, input, p, train):
    symbolic_helper.check_training_mode(train, "dropout")
    # in eval mode, dropout is non-op - if the node's train param is set to False, dropout is non-op
    if not train:
        return input
    warnings.warn(
        "Dropout is a training op and should not be exported in inference mode. "
        "For inference, make sure to call eval() on the model and to export it with param training=False."
    )
    p = g.op("Constant", value_t=torch.tensor(p))
    t = g.op("Constant", value_t=torch.tensor(True))
    r, _ = g.op("Dropout", input, p, t, outputs=2)
    return r


def nll_loss(g, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    # in onnx NegativeLogLikelihoodLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return nllloss


def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


def nll_loss_nd(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


def cross_entropy_loss(
    g, self, target, weight, reduction, ignore_index, label_smoothing
):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    label_smoothing = symbolic_helper._maybe_get_const(label_smoothing, "f")
    if label_smoothing > 0.0:
        raise RuntimeError("Unsupported: ONNX does not support label_smoothing")

    # in onnx SoftmaxCrossEntropyLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    if weight.node().mustBeNone():
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    return celoss


@symbolic_helper.parse_args("v", "v", "v", "v", "i")
def binary_cross_entropy_with_logits(g, input, target, weight, pos_weight, reduction):
    p = g.op("Constant", value_t=torch.tensor([1]))
    sig_x = opset9.sigmoid(g, input)
    log_sig_x = opset9.log(g, sig_x)
    sub_1_x = opset9.sub(g, p, sig_x)
    sub_1_y = opset9.sub(g, p, target)
    log_1_x = opset9.log(g, sub_1_x)
    if pos_weight is None or symbolic_helper._is_none(pos_weight):
        output = opset9.neg(
            g,
            opset9.add(
                g, opset9.mul(g, target, log_sig_x), opset9.mul(g, sub_1_y, log_1_x)
            ),
        )
    else:
        output = opset9.neg(
            g,
            opset9.add(
                g,
                opset9.mul(g, opset9.mul(g, target, log_sig_x), pos_weight),
                opset9.mul(g, sub_1_y, log_1_x),
            ),
        )

    if weight is not None and not symbolic_helper._is_none(weight):
        output = opset9.mul(g, weight, output)

    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        return g.op("ReduceSum", output, keepdims_i=0)
    else:
        return symbolic_helper._onnx_unsupported(
            "binary_cross_entropy_with_logits with reduction other than none, mean, or sum"
        )


def celu(g, self, alpha):
    alpha = symbolic_helper._maybe_get_const(alpha, "f")
    # if the input is of type double cast it to float
    if self.type().scalarType() == "Double":
        self = g.op("Cast", self, to_i=symbolic_helper.cast_pytorch_to_onnx["Float"])
        out = g.op("Celu", self, alpha_f=alpha)
        return g.op("Cast", out, to_i=symbolic_helper.cast_pytorch_to_onnx["Double"])

    return g.op("Celu", self, alpha_f=alpha)


def argmax(g, input, dim, keepdim):
    if symbolic_helper._is_none(dim):
        flattened = symbolic_helper._reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        return g.op(
            "ArgMax", flattened, axis_i=0, keepdims_i=False, select_last_index_i=False
        )
    else:
        dim = symbolic_helper._parse_arg(dim, "i")
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
        return g.op(
            "ArgMax", input, axis_i=dim, keepdims_i=keepdim, select_last_index_i=False
        )


def argmin(g, input, dim, keepdim):
    if symbolic_helper._is_none(dim):
        flattened = symbolic_helper._reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        return g.op(
            "ArgMin", flattened, axis_i=0, keepdims_i=False, select_last_index_i=False
        )
    else:
        dim = symbolic_helper._parse_arg(dim, "i")
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
        return g.op(
            "ArgMin", input, axis_i=dim, keepdims_i=keepdim, select_last_index_i=False
        )


def pow(g, self, exponent):
    return g.op("Pow", self, exponent)


def ge(g, input, other):
    return g.op("GreaterOrEqual", input, other)


def le(g, input, other):
    return g.op("LessOrEqual", input, other)


@symbolic_helper.parse_args("v", "i", "v", "v")
def unfold(g, input, dimension, size, step):
    const_size = symbolic_helper._maybe_get_const(size, "i")
    const_step = symbolic_helper._maybe_get_const(step, "i")
    if not symbolic_helper._is_value(const_size) and not symbolic_helper._is_value(
        const_step
    ):
        return opset9.unfold(g, input, dimension, const_size, const_step)
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("unfold", input, dimension_i=dimension, size_i=size, step_i=step)

    sizedim = symbolic_helper._get_tensor_dim_size(input, dimension)
    if sizedim is not None:
        low_start = g.op("Constant", value_t=torch.tensor(0))
        low_end = g.op("Constant", value_t=torch.tensor(sizedim))
        hi_end = g.op("Constant", value_t=torch.tensor(sizedim + 1))
        low_indices = g.op("Range", low_start, low_end, step)
        hi_indices = g.op("Range", size, hi_end, step)

        low_size = symbolic_helper._size_helper(
            g, low_indices, g.op("Constant", value_t=torch.tensor(0))
        )
        hi_size = symbolic_helper._size_helper(
            g, hi_indices, g.op("Constant", value_t=torch.tensor(0))
        )

        ndim = symbolic_helper._get_tensor_rank(input)
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))

        unsqueeze_list = []
        loop_condition = g.op("Constant", value_t=torch.tensor(1))
        loop_condition = g.op("Cast", loop_condition, to_i=9)
        loop_len = g.op("Min", low_size, hi_size)
        loop = g.op("Loop", loop_len, loop_condition)

        loop_block = utils._add_block(loop.node())
        block_input_iter = utils._add_input_to_block(loop_block)
        cond = utils._add_input_to_block(loop_block)

        starts = loop_block.op("Gather", low_indices, block_input_iter)
        ends = loop_block.op("Gather", hi_indices, block_input_iter)
        axes = loop_block.op("Constant", value_t=torch.tensor([2]))
        starts = symbolic_helper._unsqueeze_helper(loop_block, starts, [0])
        ends = symbolic_helper._unsqueeze_helper(loop_block, ends, [0])
        stack = loop_block.op("Slice", input, starts, ends, axes)

        unsqueeze = symbolic_helper._unsqueeze_helper(
            loop_block, loop_block.op("Transpose", stack, perm_i=perm), [dimension]
        )
        unsqueeze_list.append(unsqueeze)
        concat = loop_block.op("Concat", *unsqueeze_list, axis_i=0)

        cond_out = loop_block.op("Cast", loop_condition, to_i=9)
        utils._add_output_to_block(loop_block, cond_out)
        utils._add_output_to_block(loop_block, concat)

        loop_output = loop.node().output()
        perm = [0, 1, 2, 3, 4]
        perm[0], perm[dimension + 1] = perm[dimension + 1], perm[0]
        transpose = g.op("Transpose", loop_output, perm_i=perm)
        squeeze = symbolic_helper._squeeze_helper(g, transpose, [0])

        return squeeze
    else:
        return symbolic_helper._unimplemented("Unfold", "input size not accessible")


@symbolic_helper.parse_args("v", "v", "is", "is", "v")
def tensordot(g, input_a, input_b, dims_a, dims_b, out=None):
    if out is not None:
        symbolic_helper._unimplemented(
            "Tensordot", "Out parameter is not supported for tensordot."
        )

    dim_count_a = symbolic_helper._get_tensor_rank(input_a)
    if dim_count_a is None:
        raise RuntimeError(
            "Unsupported: ONNX export of tensordot for tensor(input_a) of unknown rank."
        )

    dim_count_b = symbolic_helper._get_tensor_rank(input_b)
    if dim_count_b is None:
        raise RuntimeError(
            "Unsupported: ONNX export of tensordot for tensor(input_b) of unknown rank."
        )

    dims_a = [
        (dims_a[i] + dim_count_a) if (dims_a[i] < 0) else dims_a[i]
        for i in range(len(dims_a))
    ]
    dims_b = [
        (dims_b[i] + dim_count_b) if (dims_b[i] < 0) else dims_b[i]
        for i in range(len(dims_b))
    ]

    left_dims_a = [i for i in range(dim_count_a) if (i not in dims_a)]
    left_dims_b = [i for i in range(dim_count_b) if (i not in dims_b)]

    new_input_a = opset9.permute(g, input_a, left_dims_a + dims_a)
    new_input_b = opset9.permute(g, input_b, dims_b + left_dims_b)

    input_shape = g.op("Shape", new_input_a)
    left_sizes_a = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(left_dims_a)]
    )
    shape_sizes = [
        left_sizes_a,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", output_a)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    input_shape = g.op("Shape", new_input_b)
    left_sizes_b = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[len(dims_b)], ends=[sys.maxsize]
    )
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(dims_b)]
    )
    shape_sizes = [
        slices,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    input_shape = g.op("Shape", output_b)
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    output = einsum(g, "ij,jk->ik", g.op("prim::ListConstruct", *[output_a, output_b]))

    shape_sizes = [left_sizes_a, left_sizes_b]
    return opset9._reshape_from_tensor(g, output, shape_sizes)
