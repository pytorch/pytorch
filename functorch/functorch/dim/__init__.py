import torch
from typing import Union, Sequence
import inspect
import dis
from .tree_map import tree_flatten, tree_map
from .wrap_type import wrap_type
from functorch._C import dim as _C
_C._patch_tensor_class()
dims, DimList, dimlists = _C.dims, _C.DimList, _C.dimlists

class DimensionMismatchError(Exception):
    pass

class DimensionBindError(Exception):
    pass

from . import op_properties

# use dict to avoid writing C++ bindings for set
pointwise = {t: True for t in op_properties.pointwise}

use_c = True
if not use_c:
    from . import reference

class _Tensor:
    # fast path around slow wrapping/unwrapping logic for simply queries used
    # by the implementation...


    @property
    def dims(self):
        return tuple(d for d in self._levels if isinstance(d, Dim))

    def dim(self):
        return self.ndim

    if use_c:
        __torch_function__ = classmethod(_C.__torch_function__)
        expand = _C._instancemethod(_C.expand)
    else:
        __torch_function__ = reference.__torch_function__
        expand = reference.expand

    index = _C._instancemethod(_C.index)

    def __repr__(self):
        tensor, levels, ndim = self._tensor, self._levels, self.ndim
        return f'{tensor}\nwith dims={tuple(l + ndim if isinstance(l, int) else l for l in levels)} sizes={tuple(tensor.size())}'


TensorLike = (_Tensor, torch.Tensor)

class Dim(_C.Dim, _Tensor):
    # note that _C.Dim comes before tensor because we want the Dim API for things like size to take precendence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__


class Tensor(_Tensor, _C.Tensor):
    if not use_c:
        from_batched = staticmethod(_C.Tensor_from_batched)
    from_positional = staticmethod(_C.Tensor_from_positional)
    sum = _C._instancemethod(_C.Tensor_sum)


def cat(tensors, dim, new_dim):
    n = dims()
    return stack(tensors, n, dim).index([n, dim], new_dim)

if use_c:
    _wrap = _C._wrap

    def _def(name, *args, **kwargs):
        orig = getattr(torch.Tensor, name)
        setattr(_Tensor, name, _C._instancemethod(_wrap(orig, *args, **kwargs)))

    t__getitem__ = _C._instancemethod(_C.__getitem__)
    stack = _C.stack
    split = _C._instancemethod(_C.split)
else:
    _wrap, _def = reference._wrap, reference._def
    t__getitem__ = reference.t__getitem__
    stack = reference.stack
    split = reference.split

# note: there is no python reference
t__setitem__ = _C._instancemethod(_C.__setitem__)
# this is patched in the C API because otherwise torch.Tensor will
# no longer be considered a sequence and things will break
# torch.Tensor.__getitem__ = t__getitem__

_Tensor.__getitem__ = t__getitem__
# torch.Tensor.__setitem__ = t__setitem__
_Tensor.__setitem__ = t__setitem__

torch.Tensor.split = split
_Tensor.split = split
torch.Tensor.expand = _C._instancemethod(_C.expand)
torch.Tensor.index = _C._instancemethod(_C.index)
wrap_type(use_c, _Tensor, torch.Tensor, _Tensor.__torch_function__)
del _Tensor.ndim

if use_c:
    _Tensor.permute = _Tensor.order = _C._instancemethod(_C.order)
else:
    _Tensor.permute = _Tensor.order = reference.positional

_def('mean')
_def('sum')
_def('all')
_def('amax')
_def('amin')
_def('aminmax')
_def('any')
_def('count_nonzero')
_def('logsumexp')
_def('nanmean')
_def('nansum')
_def('prod')
_def('std', keepdim_offset=2)
_def('var', keepdim_offset=2)
_def('max', single_dim=True)
_def('min', single_dim=True)
_def('argmax', single_dim=True)
_def('argmin', single_dim=True)
_def('kthvalue', single_dim=True)
_def('median', single_dim=True)
_def('nanmedian', single_dim=True)
_def('mode', single_dim=True)
_def('sort', reduce=False)
_def('argsort', reduce=False)
_def('unbind', single_dim=True)
_def('chunk', dim_offset=1, reduce=False)
_def('cummax', single_dim=True, reduce=False)
_def('cummin', single_dim=True, reduce=False)
_def('cumprod', single_dim=True, reduce=False)
_def('cumprod_', single_dim=True, reduce=False)
_def('cumsum', single_dim=True, reduce=False)
_def('cumsum_', single_dim=True, reduce=False)
_def('logcumsumexp', single_dim=True, reduce=False)
_def('renorm', dim_offset=1, single_dim=True, reduce=False)
_def('softmax', single_dim=True, reduce=False)
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# stuff to handle in the future, because they require special
# binding logic for dims
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (new dimes on output)
# should these all be subsumed by inplace indexing?
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
