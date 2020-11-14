import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg

from functools import wraps

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 13


def _reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True):
    def symbolic(g, self, dim=None, keepdim=None):
        from torch.onnx.symbolic_opset9 import _maybe_cast_reduce_op_input
        self = _maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # all-reduce path
            return g.op(onnx_op_name, self, keepdims_i=0)
        else:
            # dim-reduce path
            desc = 'is' if allow_multi_dim_support else 'i'
            dim, keepdim = sym_help._get_const(dim, desc, 'dim'), sym_help._get_const(keepdim, 'i', 'keepdim')
            dim_list = dim if allow_multi_dim_support else [dim]
            dim_list = g.op("Constant", value_t=torch.tensor(dim_list, dtype=torch.long))
            return g.op(onnx_op_name, self, dim_list, keepdims_i=keepdim)
    return symbolic



def overload_by_arg_count(fn):
    @wraps(fn)
    def wrapper(g, *args):
        overloads = fn(g, *args)
        last_exception = None
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        raise NotImplementedError("Unknown aten::{} signature".format(fn.__name__))
    return wrapper


def _reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True):
    symbolic = _reduce_op_symbolic(onnx_op, allow_multi_dim_support=allow_multi_dim_support)

    @overload_by_arg_count
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