from torch.onnx.symbolic_helper import _block_list_in_opset
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.utils import _add_block, _add_input_to_block, _add_output_to_block
from sys import maxsize

block_listed_operators = [embedding_bag]

def _reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True):
    symbolic = torch.onnx.symbolic_opset9._reduce_op_symbolic(onnx_op, allow_multi_dim_support=allow_multi_dim_support)

    @torch.onnx.symbolic_opset9.overload_by_arg_count
    def reduce(g, *args, **kwargs):
        @parse_args('v', 'none')
        def reduce_nodim(g, self, dtype):
            if dtype.node().kind() != 'prim::Constant':
                return _unimplemented(name, "dtype")
            return symbolic(g, self)

        dim_desc = 'is' if allow_multi_dim_support else 'i'

        @parse_args('v', dim_desc, 'i', 'none')
        def reduce_dim(g, self, dim, keepdim, dtype):
            if dtype.node().kind() != 'prim::Constant':
                return _unimplemented(name, "dtype")
            return symbolic(g, self, dim, keepdim)
        return reduce_nodim, reduce_dim
    return reduce


sum = _reduce_with_dtype('ReduceSum', 'sum')
mean = _reduce_with_dtype('ReduceMean', 'mean')


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
