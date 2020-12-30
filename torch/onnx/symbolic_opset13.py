# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 13
from torch.onnx.symbolic_helper import _block_list_in_opset
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args

block_listed_operators = ['embedding_bag']

for block_listed_op in block_listed_operators:
    vars()[block_listed_op] = _block_list_in_opset(block_listed_op)


@parse_args('v', 'i', 'none')
def softmax(g, input, dim, dtype=None):
    softmax = g.op('Softmax', input, axis_i=dim)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        softmax = g.op("Cast", softmax, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])

    return softmax


@parse_args('v', 'i', 'none')
def log_softmax(g, input, dim, dtype=None):
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        return_op = g.op("Cast", return_op, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return return_op


@parse_args('v', 'v', 'i')
def frobenius_norm(g, self, dim=None, keepdim=False):
    dim_val = sym_help._maybe_get_const(dim, 'is')
    if not sym_help._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)
    sqr = g.op('Mul', self, self)
    sumsqr = g.op('ReduceSum', sqr, dim, keepdims_i=keepdim)
    return g.op('Sqrt', sumsqr)


@parse_args('v', 'v', 'i', 'i')
def split(g, self, split_size_or_sizes, dim, _outputs=None):
    if not sym_help._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if sym_help._is_packed_list(split_size_or_sizes) and \
                len(sym_help._unpack_list(split_size_or_sizes)) == _outputs:
            split_sizes = [g.op("Unsqueeze", v, g.op("Constant", value_t=torch.tensor([0])))
                           for v in sym_help._unpack_list(split_size_or_sizes)]
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

    split_val = split_size_or_sizes.node()['value']
    if split_val.dim() > 0:
        return g.op("Split", self, split_size_or_sizes, axis_i=dim, outputs=_outputs)
    split_size = sym_help._get_const(split_size_or_sizes, 'i', 'split_size')

    size = self.type().sizes()[dim]
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


@parse_args('v', 'i', 'i')
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


def glu(g, input, dim):
    first, second = g.op('Split', input, dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))
