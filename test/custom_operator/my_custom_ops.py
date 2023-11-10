import torch
import torch._custom_ops as library
from model import get_custom_op_library_path

torch.ops.load_library(get_custom_op_library_path())


@library.impl_abstract("custom::nonzero")
def nonzero_abstract(x):
    n = x.dim()
    ctx = library.get_ctx()
    nnz = ctx.create_unbacked_symint()
    shape = [nnz, n]
    return x.new_empty(shape, dtype=torch.long)
