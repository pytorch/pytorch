import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _unimplemented
from torch.onnx.utils import _add_block, _add_input_to_block, _add_output_to_block


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Einsum", *tensors, equation_s=equation)

@parse_args('v', 'v')
def outer(g, input, other):
    # make sure to cast other to self's type
    if other.type().scalarType() != input.type().scalarType():
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[input.type().scalarType()])
    return g.op("Einsum", input, other, equation_s='i,j->ij')

@parse_args('v', 'f', 'i')
def dropout(g, input, p, train):
    sym_help.assert_training_mode(train, "dropout")
    # in eval mode, dropout is non-op - if the node's train param is set to False, dropout is non-op
    if not sym_help._training_mode:
        return input

    p = g.op("Constant", value_t=torch.tensor(p))
    t = g.op("Constant", value_t=torch.tensor(True))
    r, _ = g.op("Dropout", input, p, t, outputs=2)
    return r


def nll_loss(g, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]

    # in onnx NegativeLogLikelihoodLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = sym_help._maybe_get_const(ignore_index, 'i')
    if weight.node().mustBeNone():
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction, ignore_index_i=ignore_index)
    else:
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction, ignore_index_i=ignore_index)

    return nllloss


def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


def nll_loss_nd(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


def cross_entropy_loss(g, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]

    # in onnx SoftmaxCrossEntropyLoss specification, ignore_index is optional without default value.
    # therefore we need to set ignore_index attribute even if it is not specified (e.g. ignore_index=-100).
    ignore_index = sym_help._maybe_get_const(ignore_index, 'i')
    if weight.node().mustBeNone():
        celoss = g.op("SoftmaxCrossEntropyLoss", self, target, reduction_s=reduction, ignore_index_i=ignore_index)
    else:
        celoss = g.op("SoftmaxCrossEntropyLoss", self, target, weight, reduction_s=reduction, ignore_index_i=ignore_index)

    return celoss


@parse_args('v', 'v', 'v', 'v', 'i')
def binary_cross_entropy_with_logits(g, input, target, weight, pos_weight, reduction):
    from torch.onnx.symbolic_opset9 import sigmoid, log, sub, neg, mul, add
    p = g.op("Constant", value_t=torch.tensor([1]))
    sig_x = sigmoid(g, input)
    log_sig_x = log(g, sig_x)
    sub_1_x = sub(g, p, sig_x)
    sub_1_y = sub(g, p, target)
    log_1_x = log(g, sub_1_x)
    if pos_weight is None or sym_help._is_none(pos_weight):
        output = neg(g, add(g, mul(g, target, log_sig_x), mul(g, sub_1_y, log_1_x)))
    else:
        output = neg(g, add(g, mul(g, mul(g, target, log_sig_x), pos_weight), mul(g, sub_1_y, log_1_x)))

    if weight is not None and not sym_help._is_none(weight):
        output = mul(g, weight, output)

    reduction = sym_help._maybe_get_const(reduction, 'i')
    if reduction == 0:
        return output
    elif reduction == 1:
        return g.op("ReduceMean", output)
    elif reduction == 2:
        return g.op("ReduceSum", output)
    else:
        return sym_help._onnx_unsupported("binary_cross_entropy_with_logits with reduction other than none, mean, or sum")


def celu(g, self, alpha):
    alpha = sym_help._maybe_get_const(alpha, 'f')
    # if the input is of type double cast it to float
    if self.type().scalarType() == 'Double':
        self = g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx['Float'])
        out = g.op("Celu", self, alpha_f=alpha)
        return g.op("Cast", out, to_i=sym_help.cast_pytorch_to_onnx['Double'])

    return g.op("Celu", self, alpha_f=alpha)


def argmax(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        from torch.onnx.symbolic_opset9 import reshape
        flattened = reshape(g, input, g.op("Constant", value_t=torch.tensor([-1])))
        return g.op('ArgMax', flattened, axis_i=0, keepdims_i=False, select_last_index_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMax', input, axis_i=dim, keepdims_i=keepdim, select_last_index_i=False)


def argmin(g, input, dim, keepdim):
    if sym_help._is_none(dim):
        from torch.onnx.symbolic_opset9 import reshape
        flattened = reshape(g, input, g.op("Constant", value_t=torch.tensor([-1])))
        return g.op('ArgMin', flattened, axis_i=0, keepdims_i=False, select_last_index_i=False)
    else:
        dim = _parse_arg(dim, 'i')
        keepdim = _parse_arg(keepdim, 'i')
        return g.op('ArgMin', input, axis_i=dim, keepdims_i=keepdim, select_last_index_i=False)


def pow(g, self, exponent):
    return g.op("Pow", self, exponent)

def ge(g, input, other):
    return g.op('GreaterOrEqual', input, other)

def le(g, input, other):
    return g.op('LessOrEqual', input, other)

@parse_args('v', 'i', 'v', 'v')
def unfold(g, input, dimension, size, step):
    const_size = sym_help._maybe_get_const(size, 'i')
    const_step = sym_help._maybe_get_const(step, 'i')
    if not sym_help._is_value(const_size) and not sym_help._is_value(const_step):
        from torch.onnx.symbolic_opset9 import unfold as _unfold
        return _unfold(g, input, dimension, const_size, const_step)
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", input, operator_s="unfold", dimension_i=dimension, size_i=size, step_i=step)

    sizedim = sym_help._get_tensor_dim_size(input, dimension)
    if sizedim is not None:
        low_start = g.op("Constant", value_t=torch.tensor(0))
        low_end = g.op("Constant", value_t=torch.tensor(sizedim))
        hi_end = g.op("Constant", value_t=torch.tensor(sizedim + 1))
        low_indices = g.op("Range", low_start, low_end, step)
        hi_indices = g.op("Range", size, hi_end, step)

        low_size = sym_help._size_helper(g, low_indices, g.op("Constant", value_t=torch.tensor(0)))
        hi_size = sym_help._size_helper(g, hi_indices, g.op("Constant", value_t=torch.tensor(0)))

        ndim = sym_help._get_tensor_rank(input)
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))

        unsqueeze_list = []
        loop_condition = g.op("Constant", value_t=torch.tensor(1))
        loop_condition = g.op("Cast", loop_condition, to_i=9)
        loop_len = g.op("Min", low_size, hi_size)
        loop = g.op("Loop", loop_len, loop_condition)

        loop_block = _add_block(loop.node())
        block_input_iter = _add_input_to_block(loop_block)
        cond = _add_input_to_block(loop_block)

        starts = loop_block.op("Gather", low_indices, block_input_iter)
        ends = loop_block.op("Gather", hi_indices, block_input_iter)
        axes = loop_block.op("Constant", value_t=torch.tensor([2]))
        starts = sym_help._unsqueeze_helper(loop_block, starts, [0])
        ends = sym_help._unsqueeze_helper(loop_block, ends, [0])
        stack = loop_block.op("Slice", input, starts, ends, axes)

        unsqueeze = sym_help._unsqueeze_helper(loop_block, loop_block.op("Transpose", stack, perm_i=perm), [dimension])
        unsqueeze_list.append(unsqueeze)
        concat = loop_block.op("Concat", *unsqueeze_list, axis_i=0)

        cond_out = loop_block.op("Cast", loop_condition, to_i=9)
        _add_output_to_block(loop_block, cond_out)
        _add_output_to_block(loop_block, concat)

        loop_output = loop.node().output()
        perm = [0, 1, 2, 3, 4]
        perm[0], perm[dimension + 1] = perm[dimension + 1], perm[0]
        transpose = g.op("Transpose", loop_output, perm_i=perm)
        squeeze = sym_help._squeeze_helper(g, transpose, [0])

        return squeeze
    else:
        return _unimplemented("Unfold", "input size not accessible")
