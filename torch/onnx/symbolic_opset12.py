
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Einsum", *tensors, equation_s=equation)


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
