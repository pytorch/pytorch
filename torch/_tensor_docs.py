"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr
from ._torch_docs import parse_kwargs


def add_docstr_all(method, docstr):
    add_docstr(getattr(torch._C._TensorBase, method), docstr)

new_common_args = parse_kwargs("""
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same :class:`torch.dtype` as this tensor.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, same :class:`torch.device` as this tensor.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
""")

add_docstr_all('new_tensor',
               r"""
new_tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor

Returns a new Tensor with :attr:`data` as the tensor data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

.. warning::

    :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.from_numpy`.

Args:
    data (array_like): The returned Tensor copies :attr:`data`.
    {dtype}
    {device}
    {requires_grad}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.int8)
    >>> data = [[0, 1], [2, 3]]
    >>> tensor.new_tensor(data)
    tensor([[ 0,  1],
            [ 2,  3]], dtype=torch.int8)

""".format(**new_common_args))

add_docstr_all('new_full',
               r"""
new_full(size, fill_value, dtype=None, device=None, requires_grad=False) -> Tensor

Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    fill_value (scalar): the number to fill the output tensor with.
    {dtype}
    {device}
    {requires_grad}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.float64)
    >>> tensor.new_full((3, 4), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)

""".format(**new_common_args))

add_docstr_all('new_empty',
               r"""
new_empty(size, dtype=None, device=None, requires_grad=False) -> Tensor

Returns a Tensor of size :attr:`size` filled with uninitialized data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    {dtype}
    {device}
    {requires_grad}

Example::

    >>> tensor = torch.ones(())
    >>> tensor.new_empty((2, 3))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

""".format(**new_common_args))

add_docstr_all('new_ones',
               r"""
new_ones(size, dtype=None, device=None, requires_grad=False) -> Tensor

Returns a Tensor of size :attr:`size` filled with ``1``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    {dtype}
    {device}
    {requires_grad}

Example::

    >>> tensor = torch.tensor((), dtype=torch.int32)
    >>> tensor.new_ones((2, 3))
    tensor([[ 1,  1,  1],
            [ 1,  1,  1]], dtype=torch.int32)

""".format(**new_common_args))

add_docstr_all('new_zeros',
               r"""
new_zeros(size, dtype=None, device=None, requires_grad=False) -> Tensor

Returns a Tensor of size :attr:`size` filled with ``0``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    {dtype}
    {device}
    {requires_grad}

Example::

    >>> tensor = torch.tensor((), dtype=torch.float64)
    >>> tensor.new_ones((2, 3))
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]], dtype=torch.float64)

""".format(**new_common_args))

add_docstr_all('abs',
               r"""
abs() -> Tensor

See :func:`torch.abs`
""")

add_docstr_all('abs_',
               r"""
abs_() -> Tensor

In-place version of :meth:`~Tensor.abs`
""")

add_docstr_all('acos',
               r"""
acos() -> Tensor

See :func:`torch.acos`
""")

add_docstr_all('acos_',
               r"""
acos_() -> Tensor

In-place version of :meth:`~Tensor.acos`
""")

add_docstr_all('add',
               r"""
add(value) -> Tensor

See :func:`torch.add`
""")

add_docstr_all('add_',
               r"""
add_(value) -> Tensor

In-place version of :meth:`~Tensor.add`
""")

add_docstr_all('addbmm',
               r"""
addbmm(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

See :func:`torch.addbmm`
""")

add_docstr_all('addbmm_',
               r"""
addbmm_(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

In-place version of :meth:`~Tensor.addbmm`
""")

add_docstr_all('addcdiv',
               r"""
addcdiv(value=1, tensor1, tensor2) -> Tensor

See :func:`torch.addcdiv`
""")

add_docstr_all('addcdiv_',
               r"""
addcdiv_(value=1, tensor1, tensor2) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""")

add_docstr_all('addcmul',
               r"""
addcmul(value=1, tensor1, tensor2) -> Tensor

See :func:`torch.addcmul`
""")

add_docstr_all('addcmul_',
               r"""
addcmul_(value=1, tensor1, tensor2) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""")

add_docstr_all('addmm',
               r"""
addmm(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

See :func:`torch.addmm`
""")

add_docstr_all('addmm_',
               r"""
addmm_(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""")

add_docstr_all('addmv',
               r"""
addmv(beta=1, tensor, alpha=1, mat, vec) -> Tensor

See :func:`torch.addmv`
""")

add_docstr_all('addmv_',
               r"""
addmv_(beta=1, tensor, alpha=1, mat, vec) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""")

add_docstr_all('addr',
               r"""
addr(beta=1, alpha=1, vec1, vec2) -> Tensor

See :func:`torch.addr`
""")

add_docstr_all('addr_',
               r"""
addr_(beta=1, alpha=1, vec1, vec2) -> Tensor

In-place version of :meth:`~Tensor.addr`
""")

add_docstr_all('all',
               r"""
all() -> bool

Returns ``True`` if all elements in the tensor are non-zero, ``False`` otherwise.
""")

add_docstr_all('any',
               r"""
any() -> bool

Returns ``True`` if any elements in the tensor are non-zero, ``False`` otherwise.
""")

add_docstr_all('apply_',
               r"""
apply_(callable) -> Tensor

Applies the function :attr:`callable` to each element in the tensor, replacing
each element with the value returned by :attr:`callable`.

.. note::

    This function only works with CPU tensors and should not be used in code
    sections that require high performance.
""")

add_docstr_all('asin', r"""
asin() -> Tensor

See :func:`torch.asin`
""")

add_docstr_all('asin_',
               r"""
asin_() -> Tensor

In-place version of :meth:`~Tensor.asin`
""")

add_docstr_all('atan',
               r"""
atan() -> Tensor

See :func:`torch.atan`
""")

add_docstr_all('atan2',
               r"""
atan2(other) -> Tensor

See :func:`torch.atan2`
""")

add_docstr_all('atan2_',
               r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""")

add_docstr_all('atan_',
               r"""
atan_() -> Tensor

In-place version of :meth:`~Tensor.atan`
""")

add_docstr_all('baddbmm',
               r"""
baddbmm(beta=1, alpha=1, batch1, batch2) -> Tensor

See :func:`torch.baddbmm`
""")

add_docstr_all('baddbmm_',
               r"""
baddbmm_(beta=1, alpha=1, batch1, batch2) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""")

add_docstr_all('bernoulli',
               r"""
bernoulli() -> Tensor

See :func:`torch.bernoulli`
""")

add_docstr_all('bernoulli_',
               r"""
bernoulli_() -> Tensor

In-place version of :meth:`~Tensor.bernoulli`
""")

add_docstr_all('bmm',
               r"""
bmm(batch2) -> Tensor

See :func:`torch.bmm`
""")

add_docstr_all('btrifact_with_info',
               r"""
btrifact_with_info(pivot=True) -> (Tensor, Tensor, Tensor)

See :func:`torch.btrifact_with_info`
""")

add_docstr_all('cauchy_',
               r"""
cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

.. math::

    f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - median)^2 + \sigma^2}
""")

add_docstr_all('ceil',
               r"""
ceil() -> Tensor

See :func:`torch.ceil`
""")

add_docstr_all('ceil_',
               r"""
ceil_() -> Tensor

In-place version of :meth:`~Tensor.ceil`
""")

add_docstr_all('clamp',
               r"""
clamp(min, max) -> Tensor

See :func:`torch.clamp`
""")

add_docstr_all('clamp_',
               r"""
clamp_(min, max) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""")

add_docstr_all('clone',
               r"""
clone() -> Tensor

Returns a copy of the :attr:`self` tensor. The copy has the same size and data
type as :attr:`self`.
""")

add_docstr_all('contiguous',
               r"""
contiguous() -> Tensor

Returns a contiguous tensor containing the same data as :attr:`self` tensor. If
:attr:`self` tensor is contiguous, this function returns the :attr:`self`
tensor.
""")

add_docstr_all('copy_',
               r"""
copy_(src, non_blocking=False) -> Tensor

Copies the elements from :attr:`src` into :attr:`self` tensor and returns
:attr:`self`.

The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
with the :attr:`self` tensor. It may be of a different data type or reside on a
different device.

Args:
    src (Tensor): the source tensor to copy from
    non_blocking (bool): if ``True`` and this copy is between CPU and GPU,
        the copy may occur asynchronously with respect to the host. For other
        cases, this argument has no effect.
""")

add_docstr_all('cos',
               r"""
cos() -> Tensor

See :func:`torch.cos`
""")

add_docstr_all('cos_',
               r"""
cos_() -> Tensor

In-place version of :meth:`~Tensor.cos`
""")

add_docstr_all('cosh',
               r"""
cosh() -> Tensor

See :func:`torch.cosh`
""")

add_docstr_all('cosh_',
               r"""
cosh_() -> Tensor

In-place version of :meth:`~Tensor.cosh`
""")

add_docstr_all('cross',
               r"""
cross(other, dim=-1) -> Tensor

See :func:`torch.cross`
""")

add_docstr_all('cuda',
               r"""
cuda(device=None, non_blocking=False) -> Tensor

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`): The destination GPU device.
        Defaults to the current CUDA device.
    non_blocking (bool): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
""")

add_docstr_all('cumprod',
               r"""
cumprod(dim) -> Tensor

See :func:`torch.cumprod`
""")

add_docstr_all('cumsum',
               r"""
cumsum(dim) -> Tensor

See :func:`torch.cumsum`
""")

add_docstr_all('data_ptr',
               r"""
data_ptr() -> int

Returns the address of the first element of :attr:`self` tensor.
""")

add_docstr_all('diag',
               r"""
diag(diagonal=0) -> Tensor

See :func:`torch.diag`
""")

add_docstr_all('dim',
               r"""
dim() -> int

Returns the number of dimensions of :attr:`self` tensor.
""")

add_docstr_all('dist',
               r"""
dist(other, p=2) -> Tensor

See :func:`torch.dist`
""")

add_docstr_all('div',
               r"""
div(value) -> Tensor

See :func:`torch.div`
""")

add_docstr_all('div_',
               r"""
div_(value) -> Tensor

In-place version of :meth:`~Tensor.div`
""")

add_docstr_all('dot',
               r"""
dot(tensor2) -> Tensor

See :func:`torch.dot`
""")

add_docstr_all('eig',
               r"""
eig(eigenvectors=False) -> (Tensor, Tensor)

See :func:`torch.eig`
""")

add_docstr_all('element_size',
               r"""
element_size() -> int

Returns the size in bytes of an individual element.

Example::

    >>> torch.tensor([]).element_size()
    4
    >>> torch.tensor([], dtype=torch.uint8).element_size()
    1

""")

add_docstr_all('eq',
               r"""
eq(other) -> Tensor

See :func:`torch.eq`
""")

add_docstr_all('eq_',
               r"""
eq_(other) -> Tensor

In-place version of :meth:`~Tensor.eq`
""")

add_docstr_all('equal',
               r"""
equal(other) -> bool

See :func:`torch.equal`
""")

add_docstr_all('erf',
               r"""
erf() -> Tensor

See :func:`torch.erf`
""")

add_docstr_all('erfinv',
               r"""
erfinv() -> Tensor

See :func:`torch.erfinv`
""")

add_docstr_all('exp',
               r"""
exp() -> Tensor

See :func:`torch.exp`
""")

add_docstr_all('exp_',
               r"""
exp_() -> Tensor

In-place version of :meth:`~Tensor.exp`
""")

add_docstr_all('expm1',
               r"""
expm1() -> Tensor

See :func:`torch.expm1`
""")

add_docstr_all('expm1_',
               r"""
expm1_() -> Tensor

In-place version of :meth:`~Tensor.expm1`
""")

add_docstr_all('exponential_',
               r"""
exponential_(lambd=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the exponential distribution:

.. math::

    f(x) = \lambda e^{-\lambda x}
""")

add_docstr_all('fill_',
               r"""
fill_(value) -> Tensor

Fills :attr:`self` tensor with the specified value.
""")

add_docstr_all('floor',
               r"""
floor() -> Tensor

See :func:`torch.floor`
""")

add_docstr_all('floor_',
               r"""
floor_() -> Tensor

In-place version of :meth:`~Tensor.floor`
""")

add_docstr_all('fmod',
               r"""
fmod(divisor) -> Tensor

See :func:`torch.fmod`
""")

add_docstr_all('fmod_',
               r"""
fmod_(divisor) -> Tensor

In-place version of :meth:`~Tensor.fmod`
""")

add_docstr_all('frac',
               r"""
frac() -> Tensor

See :func:`torch.frac`
""")

add_docstr_all('frac_',
               r"""
frac_() -> Tensor

In-place version of :meth:`~Tensor.frac`
""")

add_docstr_all('gather',
               r"""
gather(dim, index) -> Tensor

See :func:`torch.gather`
""")

add_docstr_all('ge',
               r"""
ge(other) -> Tensor

See :func:`torch.ge`
""")

add_docstr_all('ge_',
               r"""
ge_(other) -> Tensor

In-place version of :meth:`~Tensor.ge`
""")

add_docstr_all('gels',
               r"""
gels(A) -> Tensor

See :func:`torch.gels`
""")

add_docstr_all('geometric_',
               r"""
geometric_(p, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the geometric distribution:

.. math::

    f(X=k) = (1 - p)^{k - 1} p

""")

add_docstr_all('geqrf',
               r"""
geqrf() -> (Tensor, Tensor)

See :func:`torch.geqrf`
""")

add_docstr_all('ger',
               r"""
ger(vec2) -> Tensor

See :func:`torch.ger`
""")

add_docstr_all('gesv',
               r"""
gesv(A) -> Tensor, Tensor

See :func:`torch.gesv`
""")

add_docstr_all('gt',
               r"""
gt(other) -> Tensor

See :func:`torch.gt`
""")

add_docstr_all('gt_',
               r"""
gt_(other) -> Tensor

In-place version of :meth:`~Tensor.gt`
""")

add_docstr_all('histc',
               r"""
histc(bins=100, min=0, max=0) -> Tensor

See :func:`torch.histc`
""")

add_docstr_all('index',
               r"""
index(m) -> Tensor

Selects elements from :attr:`self` tensor using a binary mask or along a given
dimension. The expression ``tensor.index(m)`` is equivalent to ``tensor[m]``.

Args:
    m (int or ByteTensor or slice): the dimension or mask used to select elements
""")

add_docstr_all('index_add_',
               r"""
index_add_(dim, index, tensor) -> Tensor

Accumulate the elements of :attr:`tensor` into the :attr:`self` tensor by adding
to the indices in the order given in :attr:`index`. For example, if ``dim == 0``
and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is added to the
``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`tensor` to select from
    tensor (Tensor): the tensor containing values to add

Example::

    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
""")

add_docstr_all('index_copy_',
               r"""
index_copy_(dim, index, tensor) -> Tensor

Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
the indices in the order given in :attr:`index`. For example, if ``dim == 0``
and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`tensor` to select from
    tensor (Tensor): the tensor containing values to copy

Example::

    >>> x = torch.zeros(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_copy_(0, index, t)
    tensor([[ 1.,  2.,  3.],
            [ 0.,  0.,  0.],
            [ 7.,  8.,  9.],
            [ 0.,  0.,  0.],
            [ 4.,  5.,  6.]])
""")

add_docstr_all('index_fill_',
               r"""
index_fill_(dim, index, val) -> Tensor

Fills the elements of the :attr:`self` tensor with value :attr:`val` by
selecting the indices in the order given in :attr:`index`.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`self` tensor to fill in
    val (float): the value to fill with

Example::
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    tensor([[-1.,  2., -1.],
            [-1.,  5., -1.],
            [-1.,  8., -1.]])
""")

add_docstr_all('index_put_',
               r"""
index_put_(indices, value) -> Tensor

Puts values from the tensor :attr:`value` into the tensor :attr:`self` using
the indices specified in :attr:`indices` (which is a tuple of Tensors). The
expression ``tensor.index_put_(indices, value)`` is equivalent to
``tensor[indices] = value``. Returns :attr:`self`.

Args:
    indices (tuple of LongTensor): tensors used to index into `self`.
    value (Tensor): tensor of same dtype as `self`.
""")

add_docstr_all('index_select',
               r"""
index_select(dim, index) -> Tensor

See :func:`torch.index_select`
""")

add_docstr_all('inverse',
               r"""
inverse() -> Tensor

See :func:`torch.inverse`
""")

add_docstr_all('is_contiguous',
               r"""
is_contiguous() -> bool

Returns True if :attr:`self` tensor is contiguous in memory in C order.
""")

add_docstr_all('is_set_to',
               r"""
is_set_to(tensor) -> bool

Returns True if this object refers to the same ``THTensor`` object from the
Torch C API as the given tensor.
""")

add_docstr_all('item', r"""
item() -> number

Returns the value of this tensor as a standard Python number. This only works
for tensors with one element.

This operation is not differentiable.

Example::

    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0

""")

add_docstr_all('kthvalue',
               r"""
kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.kthvalue`
""")

add_docstr_all('le',
               r"""
le(other) -> Tensor

See :func:`torch.le`
""")

add_docstr_all('le_',
               r"""
le_(other) -> Tensor

In-place version of :meth:`~Tensor.le`
""")

add_docstr_all('lerp',
               r"""
lerp(start, end, weight) -> Tensor

See :func:`torch.lerp`
""")

add_docstr_all('lerp_',
               r"""
lerp_(start, end, weight) -> Tensor

In-place version of :meth:`~Tensor.lerp`
""")

add_docstr_all('log',
               r"""
log() -> Tensor

See :func:`torch.log`
""")

add_docstr_all('log_', r"""
log_() -> Tensor

In-place version of :meth:`~Tensor.log`
""")

add_docstr_all('log10',
               r"""
log10() -> Tensor

See :func:`torch.log10`
""")

add_docstr_all('log10_',
               r"""
log10_() -> Tensor

In-place version of :meth:`~Tensor.log10`
""")

add_docstr_all('log1p',
               r"""
log1p() -> Tensor

See :func:`torch.log1p`
""")

add_docstr_all('log1p_',
               r"""
log1p_() -> Tensor

In-place version of :meth:`~Tensor.log1p`
""")

add_docstr_all('log2',
               r"""
log2() -> Tensor

See :func:`torch.log2`
""")

add_docstr_all('log2_',
               r"""
log2_() -> Tensor

In-place version of :meth:`~Tensor.log2`
""")

add_docstr_all('log_normal_', u"""
log_normal_(mean=1, std=2, *, generator=None)

Fills :attr:`self` tensor with numbers samples from the log-normal distribution
parameterized by the given mean (\u00B5) and standard deviation (\u03C3).
Note that :attr:`mean` and :attr:`stdv` are the mean and standard deviation of
the underlying normal distribution, and not of the returned distribution:

.. math::

    f(x) = \\dfrac{1}{x \\sigma \\sqrt{2\\pi}}\ e^{-\\dfrac{(\\ln x - \\mu)^2}{2\\sigma^2}}
""")

add_docstr_all('lt',
               r"""
lt(other) -> Tensor

See :func:`torch.lt`
""")

add_docstr_all('lt_',
               r"""
lt_(other) -> Tensor

In-place version of :meth:`~Tensor.lt`
""")

add_docstr_all('map_',
               r"""
map_(tensor, callable)

Applies :attr:`callable` for each element in :attr:`self` tensor and the given
:attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.

The :attr:`callable` should have the signature::

    def callable(a, b) -> number
""")

add_docstr_all('masked_scatter_',
               r"""
masked_scatter_(mask, source)

Copies elements from :attr:`source` into :attr:`self` tensor at positions where
the :attr:`mask` is one.
The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
with the shape of the underlying tensor. The :attr:`source` should have at least
as many elements as the number of ones in :attr:`mask`

Args:
    mask (ByteTensor): the binary mask
    source (Tensor): the tensor to copy from

.. note::

    The :attr:`mask` operates on the :attr:`self` tensor, not on the given
    :attr:`source` tensor.
""")

add_docstr_all('masked_fill_',
               r"""
masked_fill_(mask, value)

Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
one. The shape of :attr:`mask` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
tensor.

Args:
    mask (ByteTensor): the binary mask
    value (float): the value to fill in with
""")

add_docstr_all('masked_select',
               r"""
masked_select(mask) -> Tensor

See :func:`torch.masked_select`
""")

add_docstr_all('max',
               r"""
max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

See :func:`torch.max`
""")

add_docstr_all('mean',
               r"""
mean(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

See :func:`torch.mean`
""")

add_docstr_all('median',
               r"""
median(dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.median`
""")

add_docstr_all('min',
               r"""
min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

See :func:`torch.min`
""")

add_docstr_all('mm',
               r"""
mm(mat2) -> Tensor

See :func:`torch.mm`
""")

add_docstr_all('mode',
               r"""
mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.mode`
""")

add_docstr_all('mul',
               r"""
mul(value) -> Tensor

See :func:`torch.mul`
""")

add_docstr_all('mul_',
               r"""
mul_(value)

In-place version of :meth:`~Tensor.mul`
""")

add_docstr_all('multinomial',
               r"""
multinomial(num_samples, replacement=False, *, generator=None) -> Tensor

See :func:`torch.multinomial`
""")

add_docstr_all('mv',
               r"""
mv(vec) -> Tensor

See :func:`torch.mv`
""")

add_docstr_all('narrow',
               r"""
narrow(dimension, start, length) -> Tensor

Returns a new tensor that is a narrowed version of :attr:`self` tensor. The
dimension :attr:`dim` is narrowed from :attr:`start` to :attr:`start + length`. The
returned tensor and :attr:`self` tensor share the same underlying storage.

Args:
    dimension (int): the dimension along which to narrow
    start (int): the starting dimension
    length (int): the distance to the ending dimension

Example::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> x.narrow(0, 0, 2)
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    >>> x.narrow(1, 1, 2)
    tensor([[ 2,  3],
            [ 5,  6],
            [ 8,  9]])
""")

add_docstr_all('ndimension',
               r"""
ndimension() -> int

Alias for :meth:`~Tensor.dim()`
""")

add_docstr_all('ne',
               r"""
ne(other) -> Tensor

See :func:`torch.ne`
""")

add_docstr_all('ne_',
               r"""
ne_(other) -> Tensor

In-place version of :meth:`~Tensor.ne`
""")

add_docstr_all('neg',
               r"""
neg() -> Tensor

See :func:`torch.neg`
""")

add_docstr_all('neg_',
               r"""
neg_() -> Tensor

In-place version of :meth:`~Tensor.neg`
""")

add_docstr_all('nelement',
               r"""
nelement() -> int

Alias for :meth:`~Tensor.numel`
""")

add_docstr_all('nonzero',
               r"""
nonzero() -> LongTensor

See :func:`torch.nonzero`
""")

add_docstr_all('norm',
               r"""
norm(p=2, dim=None, keepdim=False) -> Tensor

See :func:`torch.norm`
""")

add_docstr_all('normal_',
               r"""
normal_(mean=0, std=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements samples from the normal distribution
parameterized by :attr:`mean` and :attr:`std`.
""")

add_docstr_all('numel',
               r"""
numel() -> int

See :func:`torch.numel`
""")

add_docstr_all('numpy',
               r"""
numpy() -> numpy.ndarray

Returns :attr:`self` tensor as a NumPy :class:`ndarray`. This tensor and the
returned :class:`ndarray` share the same underlying storage. Changes to
:attr:`self` tensor will be reflected in the :class:`ndarray` and vice versa.
""")

add_docstr_all('orgqr',
               r"""
orgqr(input2) -> Tensor

See :func:`torch.orgqr`
""")

add_docstr_all('ormqr',
               r"""
ormqr(input2, input3, left=True, transpose=False) -> Tensor

See :func:`torch.ormqr`
""")

add_docstr_all('potrf',
               r"""
potrf(upper=True) -> Tensor

See :func:`torch.potrf`
""")

add_docstr_all('potri',
               r"""
potri(upper=True) -> Tensor

See :func:`torch.potri`
""")

add_docstr_all('potrs',
               r"""
potrs(input2, upper=True) -> Tensor

See :func:`torch.potrs`
""")

add_docstr_all('pow',
               r"""
pow(exponent) -> Tensor

See :func:`torch.pow`
""")

add_docstr_all('pow_',
               r"""
pow_(exponent) -> Tensor

In-place version of :meth:`~Tensor.pow`
""")

add_docstr_all('prod',
               r"""
prod(dim=None, keepdim=False) -> Tensor

See :func:`torch.prod`
""")

add_docstr_all('pstrf',
               r"""
pstrf(upper=True, tol=-1) -> (Tensor, IntTensor)

See :func:`torch.pstrf`
""")

add_docstr_all('put_',
               r"""
put_(indices, tensor, accumulate=False) -> Tensor

Copies the elements from :attr:`tensor` into the positions specified by
indices. For the purpose of indexing, the :attr:`self` tensor is treated as if
it were a 1-D tensor.

If :attr:`accumulate` is ``True``, the elements in :attr:`tensor` are added to
:attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
contain duplicate elements.

Args:
    indices (LongTensor): the indices into self
    tensor (Tensor): the tensor containing values to copy from
    accumulate (bool): whether to accumulate into self

Example::

    >>> src = torch.tensor([[4, 3, 5],
                            [6, 7, 8]])
    >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
    tensor([[  4,   9,   5],
            [ 10,   7,   8]])
""")

add_docstr_all('qr',
               r"""
qr() -> (Tensor, Tensor)

See :func:`torch.qr`
""")

add_docstr_all('random_',
               r"""
random_(from=0, to=None, *, generator=None) -> Tensor

Fills :attr:`self` tensor with numbers sampled from the discrete uniform
distribution over ``[from, to - 1]``. If not specified, the values are usually
only bounded by :attr:`self` tensor's data type. However, for floating point
types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
will be uniform in ``[0, 2^53]``.
""")

add_docstr_all('reciprocal',
               r"""
reciprocal() -> Tensor

See :func:`torch.reciprocal`
""")

add_docstr_all('reciprocal_',
               r"""
reciprocal_() -> Tensor

In-place version of :meth:`~Tensor.reciprocal`
""")

add_docstr_all('remainder',
               r"""
remainder(divisor) -> Tensor

See :func:`torch.remainder`
""")

add_docstr_all('remainder_',
               r"""
remainder_(divisor) -> Tensor

In-place version of :meth:`~Tensor.remainder`
""")

add_docstr_all('renorm',
               r"""
renorm(p, dim, maxnorm) -> Tensor

See :func:`torch.renorm`
""")

add_docstr_all('renorm_',
               r"""
renorm_(p, dim, maxnorm) -> Tensor

In-place version of :meth:`~Tensor.renorm`
""")

add_docstr_all('repeat',
               r"""
repeat(*sizes) -> Tensor

Repeats this tensor along the specified dimensions.

Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

Args:
    sizes (torch.Size or int...): The number of times to repeat this tensor along each
        dimension

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat(4, 2)
    tensor([[ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3]])
    >>> x.repeat(4, 2, 1).size()
    torch.Size([4, 2, 3])
""")

add_docstr_all('requires_grad_',
               r"""
requires_grad_(requires_grad=True) -> Tensor

Change if autograd should record operations on this tensor: sets this tensor's
:attr:`requires_grad` attribute in-place. Returns this tensor.

:func:`require_grad_`'s main use case is to tell autograd to begin recording
operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
(because it was obtained through a DataLoader, or required preprocessing or
initialization), ``tensor.requires_grad_()`` makes it so that autograd will
begin to record operations on ``tensor``.

Args:
    requires_grad (bool): If autograd should record operations on this tensor.
        Default: ``True``.

Example::

    >>> # Let's say we want to preprocess some saved weights and use
    >>> # the result as new weights.
    >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
    >>> loaded_weights = torch.tensor(saved_weights)
    >>> weights = preprocess(loaded_weights)  # some function
    >>> weights
    tensor([-0.5503,  0.4926, -2.1158, -0.8303])

    >>> # Now, start to record operations done to weights
    >>> weights.requires_grad_()
    >>> out = weights.pow(2).sum()
    >>> out.backward()
    >>> weights.grad
    tensor([-1.1007,  0.9853, -4.2316, -1.6606])

""")

add_docstr_all('reshape',
               r"""
reshape(*shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`self`,
but with the specified shape.

Args:
    shape (tuple of ints or int...): the desired shape

See :func:`torch.reshape`
""")

add_docstr_all('resize_',
               r"""
resize_(*sizes) -> Tensor

Resizes :attr:`self` tensor to the specified size. If the number of elements is
larger than the current storage size, then the underlying storage is resized
to fit the new number of elements. If the number of elements is smaller, the
underlying storage is not changed. Existing elements are preserved but any new
memory is uninitialized.

Args:
    sizes (torch.Size or int...): the desired size

Example::

    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    >>> x.resize_(2, 2)
    tensor([[ 1,  2],
            [ 3,  4]])
""")

add_docstr_all('resize_as_',
               r"""
resize_as_(tensor) -> Tensor

Resizes the :attr:`self` tensor to be the same size as the specified
:attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.
""")

add_docstr_all('round',
               r"""
round() -> Tensor

See :func:`torch.round`
""")

add_docstr_all('round_',
               r"""
round_() -> Tensor

In-place version of :meth:`~Tensor.round`
""")

add_docstr_all('rsqrt',
               r"""
rsqrt() -> Tensor

See :func:`torch.rsqrt`
""")

add_docstr_all('rsqrt_',
               r"""
rsqrt_() -> Tensor

In-place version of :meth:`~Tensor.rsqrt`
""")

add_docstr_all('scatter_',
               r"""
scatter_(dim, index, src) -> Tensor

Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
index is specified by its index in :attr:`src` for dimension != :attr:`dim` and
by the corresponding value in :attr:`index` for dimension = :attr:`dim`.

For a 3-D tensor, :attr:`self` is updated as::

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

:attr:`self`, :attr:`index` and :attr:`src` should have same number of
dimensions. It is also required that `index->size[d] <= src->size[d]` for all
dimension `d`, and that `index->size[d] <= real->size[d]` for all dimensions
`d != dim`.

Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
between `0` and `(self.size(dim) -1)` inclusive, and all values in a row along
the specified dimension :attr:`dim` must be unique.

Args:
    input (Tensor): the source tensor
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter
    src (Tensor or float): the source element(s) to scatter

Example::

    >>> x = torch.rand(2, 5)
    >>> x
    tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
            [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
    >>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
    tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
            [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
            [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])

    >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
    >>> z
    tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.2300]])
""")

add_docstr_all('select',
               r"""
select(dim, index) -> Tensor

Slices the :attr:`self` tensor along the selected dimension at the given index.
This function returns a tensor with the given dimension removed.

Args:
    dim (int): the dimension to slice
    index (int): the index to select with

.. note::

    :meth:`select` is equivalent to slicing. For example,
    ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and
    ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.
""")

add_docstr_all('set_',
               r"""
set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
:attr:`self` tensor will share the same storage and have the same size and
strides as :attr:`source`. Changes to elements in one tensor will be reflected
in the other.

If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
storage, offset, size, and stride.

Args:
    source (Tensor or Storage): the tensor or storage to use
    storage_offset (int, optional): the offset in the storage
    size (torch.Size, optional): the desired size. Defaults to the size of the source.
    stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
""")

add_docstr_all('sigmoid',
               r"""
sigmoid() -> Tensor

See :func:`torch.sigmoid`
""")

add_docstr_all('sigmoid_',
               r"""
sigmoid_() -> Tensor

In-place version of :meth:`~Tensor.sigmoid`
""")

add_docstr_all('sign',
               r"""
sign() -> Tensor

See :func:`torch.sign`
""")

add_docstr_all('sign_',
               r"""
sign_() -> Tensor

In-place version of :meth:`~Tensor.sign`
""")

add_docstr_all('sin',
               r"""
sin() -> Tensor

See :func:`torch.sin`
""")

add_docstr_all('sin_',
               r"""
sin_() -> Tensor

In-place version of :meth:`~Tensor.sin`
""")

add_docstr_all('sinh',
               r"""
sinh() -> Tensor

See :func:`torch.sinh`
""")

add_docstr_all('sinh_',
               r"""
sinh_() -> Tensor

In-place version of :meth:`~Tensor.sinh`
""")

add_docstr_all('size',
               r"""
size() -> torch.Size

Returns the size of the :attr:`self` tensor. The returned value is a subclass of
:class:`tuple`.

Example::

    >>> torch.empty(3, 4, 5).size()
    torch.Size([3, 4, 5])

""")

add_docstr_all('sort',
               r"""
sort(dim=None, descending=False) -> (Tensor, LongTensor)

See :func:`torch.sort`
""")

add_docstr_all('sqrt',
               r"""
sqrt() -> Tensor

See :func:`torch.sqrt`
""")

add_docstr_all('sqrt_',
               r"""
sqrt_() -> Tensor

In-place version of :meth:`~Tensor.sqrt`
""")

add_docstr_all('squeeze',
               r"""
squeeze(dim=None) -> Tensor

See :func:`torch.squeeze`
""")

add_docstr_all('squeeze_',
               r"""
squeeze_(dim=None) -> Tensor

In-place version of :meth:`~Tensor.squeeze`
""")

add_docstr_all('std',
               r"""
std(dim=None, unbiased=True, keepdim=False) -> Tensor

See :func:`torch.std`
""")

add_docstr_all('storage',
               r"""
storage() -> torch.Storage

Returns the underlying storage
""")

add_docstr_all('storage_offset',
               r"""
storage_offset() -> int

Returns :attr:`self` tensor's offset in the underlying storage in terms of
number of storage elements (not bytes).

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3

""")

add_docstr_all('stride',
               r"""
stride(dim) -> tuple or int

Returns the stride of :attr:`self` tensor.

Stride is the jump necessary to go from one element to the next one in the
specified dimension :attr:`dim`. A tuple of all strides is returned when no
argument is passed in. Otherwise, an integer value is returned as the stride in
the particular dimension :attr:`dim`.

Args:
    dim (int, optional): the desired dimension in which stride is required

Example::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>>x.stride(0)
    5
    >>> x.stride(-1)
    1

""")

add_docstr_all('sub',
               r"""
sub(value, other) -> Tensor

Subtracts a scalar or tensor from :attr:`self` tensor. If both :attr:`value` and
:attr:`other` are specified, each element of :attr:`other` is scaled by
:attr:`value` before being used.

When :attr:`other` is a tensor, the shape of :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
tensor.

""")

add_docstr_all('sub_',
               r"""
sub_(x) -> Tensor

In-place version of :meth:`~Tensor.sub`
""")

add_docstr_all('sum',
               r"""
sum(dim=None, keepdim=False) -> Tensor

See :func:`torch.sum`
""")

add_docstr_all('svd',
               r"""
svd(some=True) -> (Tensor, Tensor, Tensor)

See :func:`torch.svd`
""")

add_docstr_all('symeig',
               r"""
symeig(eigenvectors=False, upper=True) -> (Tensor, Tensor)

See :func:`torch.symeig`
""")

add_docstr_all('t',
               r"""
t() -> Tensor

See :func:`torch.t`
""")

add_docstr_all('t_',
               r"""
t_() -> Tensor

In-place version of :meth:`~Tensor.t`
""")

add_docstr_all('to',
               r"""
to(*args, **kwargs) -> Tensor

Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
inferred from the arguments of ``self.to(*args, **kwargs)``.

.. note::

    If the ``self`` Tensor already
    has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
    Otherwise, the returned tensor is a copy of ``self`` with the desired
    :class:`torch.dtype` and :class:`torch.device`.

Here are the ways to call ``to``:

.. function:: to(dtype) -> Tensor

    Returns a Tensor with the specified :attr:`dtype`

.. function:: to(device, dtype=None) -> Tensor

    Returns a Tensor with the specified :attr:`device` and (optional)
    :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.

.. function:: to(other) -> Tensor

    Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as the Tensor
    :attr:`other`.

Example::

    >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
    >>> tensor.to(torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64)

    >>> cuda0 = torch.device('cuda:0')
    >>> tensor.to(cuda0)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], device='cuda:0')

    >>> tensor.to(cuda0, dtype=torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
    >>> tensor.to(other)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

""")

add_docstr_all('byte',
               r"""
byte() -> Tensor

``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.
""")

add_docstr_all('char',
               r"""
char() -> Tensor

``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.
""")

add_docstr_all('double',
               r"""
double() -> Tensor

``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.
""")

add_docstr_all('float',
               r"""
float() -> Tensor

``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.
""")

add_docstr_all('half',
               r"""
half() -> Tensor

``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.
""")

add_docstr_all('int',
               r"""
int() -> Tensor

``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.
""")

add_docstr_all('long',
               r"""
long() -> Tensor

``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.
""")

add_docstr_all('short',
               r"""
short() -> Tensor

``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.
""")

add_docstr_all('take',
               r"""
take(indices) -> Tensor

See :func:`torch.take`
""")

add_docstr_all('tan_',
               r"""
tan_() -> Tensor

In-place version of :meth:`~Tensor.tan`
""")

add_docstr_all('tanh',
               r"""
tanh() -> Tensor

See :func:`torch.tanh`
""")

add_docstr_all('tanh_',
               r"""
tanh_() -> Tensor

In-place version of :meth:`~Tensor.tanh`
""")

add_docstr_all('topk',
               r"""
topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

See :func:`torch.topk`
""")

add_docstr_all('trace',
               r"""
trace() -> Tensor

See :func:`torch.trace`
""")

add_docstr_all('transpose',
               r"""
transpose(dim0, dim1) -> Tensor

See :func:`torch.transpose`
""")

add_docstr_all('transpose_',
               r"""
transpose_(dim0, dim1) -> Tensor

In-place version of :meth:`~Tensor.transpose`
""")

add_docstr_all('tril',
               r"""
tril(k=0) -> Tensor

See :func:`torch.tril`
""")

add_docstr_all('tril_',
               r"""
tril_(k=0) -> Tensor

In-place version of :meth:`~Tensor.tril`
""")

add_docstr_all('triu',
               r"""
triu(k=0) -> Tensor

See :func:`torch.triu`
""")

add_docstr_all('triu_',
               r"""
triu_(k=0) -> Tensor

In-place version of :meth:`~Tensor.triu`
""")

add_docstr_all('trtrs',
               r"""
trtrs(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

See :func:`torch.trtrs`
""")

add_docstr_all('trunc',
               r"""
trunc() -> Tensor

See :func:`torch.trunc`
""")

add_docstr_all('trunc_',
               r"""
trunc_() -> Tensor

In-place version of :meth:`~Tensor.trunc`
""")

add_docstr_all('type',
               r"""
type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
Returns the type if `dtype` is not provided, else casts this object to
the specified type.

If this is already of the correct type, no copy is performed and the
original object is returned.

Args:
    dtype (type or string): The desired type
    non_blocking (bool): If ``True``, and the source is in pinned memory
        and destination is on the GPU or vice versa, the copy is performed
        asynchronously with respect to the host. Otherwise, the argument
        has no effect.
    **kwargs: For compatibility, may contain the key ``async`` in place of
        the ``non_blocking`` argument. The ``async`` arg is deprecated.
""")

add_docstr_all('type_as',
               r"""
type_as(tensor) -> Tensor

Returns this tensor cast to the type of the given tensor.

This is a no-op if the tensor is already of the correct type. This is
equivalent to::

    self.type(tensor.type())

Params:
    tensor (Tensor): the tensor which has the desired type
""")

add_docstr_all('unfold',
               r"""
unfold(dim, size, step) -> Tensor

Returns a tensor which contains all slices of size :attr:`size` from
:attr:`self` tensor in the dimension :attr:`dim`.

Step between two slices is given by :attr:`step`.

If `sizedim` is the size of dimension dim for :attr:`self`, the size of
dimension :attr:`dim` in the returned tensor will be
`(sizedim - size) / step + 1`.

An additional dimension of size size is appended in the returned tensor.

Args:
    dim (int): dimension in which unfolding happens
    size (int): the size of each slice that is unfolded
    step (int): the step between each slice

Example::

    >>> x = torch.arange(1, 8)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> x.unfold(0, 2, 1)
    tensor([[ 1.,  2.],
            [ 2.,  3.],
            [ 3.,  4.],
            [ 4.,  5.],
            [ 5.,  6.],
            [ 6.,  7.]])
    >>> x.unfold(0, 2, 2)
    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 5.,  6.]])
""")

add_docstr_all('uniform_',
               r"""
uniform_(from=0, to=1) -> Tensor

Fills :attr:`self` tensor with numbers sampled from the continuous uniform
distribution:

.. math::
    P(x) = \dfrac{1}{\text{to} - \text{from}}
""")

add_docstr_all('unsqueeze',
               r"""
unsqueeze(dim) -> Tensor

See :func:`torch.unsqueeze`
""")

add_docstr_all('unsqueeze_',
               r"""
unsqueeze_(dim) -> Tensor

In-place version of :meth:`~Tensor.unsqueeze`
""")

add_docstr_all('var',
               r"""
var(dim=None, unbiased=True, keepdim=False) -> Tensor

See :func:`torch.var`
""")

add_docstr_all('view',
               r"""
view(*args) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different size.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed, the new
view size must be compatible with its original size and stride, i.e., each new
view dimension must either be a subspace of an original dimension, or only span
across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
contiguity-like condition that :math:`\forall i = 0, \dots, k-1`,

.. math::

  stride[i] = stride[i+1] \times size[i+1]

Otherwise, :func:`contiguous` needs to be called before the tensor can be
viewed.

Args:
    args (torch.Size or int...): the desired size

Example::

    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])

""")

add_docstr_all('expand',
               r"""
expand(*sizes) -> Tensor

Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
to a larger size.

Passing -1 as the size for a dimension means not changing the size of
that dimension.

Tensor can be also expanded to a larger number of dimensions, and the
new ones will be appended at the front. For the new dimensions, the
size cannot be set to -1.

Expanding a tensor does not allocate new memory, but only creates a
new view on the existing tensor where a dimension of size one is
expanded to a larger size by setting the ``stride`` to 0. Any dimension
of size 1 can be expanded to an arbitrary value without allocating new
memory.

Args:
    *sizes (torch.Size or int...): the desired expanded size

Example::

    >>> x = torch.tensor([[1], [2], [3]])
    >>> x.size()
    torch.Size([3, 1])
    >>> x.expand(3, 4)
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
    >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
""")

add_docstr_all('zero_',
               r"""
zero_() -> Tensor

Fills :attr:`self` tensor with zeros.
""")

add_docstr_all('matmul',
               r"""
matmul(tensor2) -> Tensor

See :func:`torch.matmul`
""")

add_docstr_all('chunk',
               r"""
chunk(chunks, dim=0) -> List of Tensors

See :func:`torch.chunk`
""")

add_docstr_all('stft',
               r"""
stft(frame_length, hop, fft_size=None, return_onesided=True, window=None, pad_end=0) -> Tensor

See :func:`torch.stft`
""")

add_docstr_all('fft',
               r"""
fft(signal_ndim, normalized=False) -> Tensor

See :func:`torch.fft`
""")

add_docstr_all('ifft',
               r"""
ifft(signal_ndim, normalized=False) -> Tensor

See :func:`torch.ifft`
""")

add_docstr_all('rfft',
               r"""
rfft(signal_ndim, normalized=False, onesided=True) -> Tensor

See :func:`torch.rfft`
""")

add_docstr_all('irfft',
               r"""
irfft(signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> Tensor

See :func:`torch.irfft`
""")

add_docstr_all('det',
               r"""
det() -> Tensor

See :func:`torch.det`
""")

add_docstr_all('where',
               r"""
where(condition, y) -> Tensor

``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
See :func:`torch.where`
""")

add_docstr_all('logdet',
               r"""
logdet() -> Tensor

See :func:`torch.logdet`
""")

add_docstr_all('slogdet',
               r"""
slogdet() -> (Tensor, Tensor)

See :func:`torch.slogdet`
""")
