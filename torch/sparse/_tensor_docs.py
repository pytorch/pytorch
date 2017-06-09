"""Adds docstrings to sparse Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.SparseFloatTensorBase._linear_indices,
           """
_linear_indices() -> LongTensor

Return the linearized indices of a sparse tensor.  This indices
tensor is one dimensional.

Examples:
    >>> i = torch.LongTensor([[1, 2],
    >>>                       [2, 1]])
    >>> x = torch.Tensor([1, 2])
    >>> s = torch.sparse.FloatTensor(i, x, torch.Size([2,4]))
    >>> s._linear_indices()
     6
     9
    [torch.LongTensor of size 2]
""")
