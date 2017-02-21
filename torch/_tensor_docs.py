"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.FloatTensorBase.abs,
           """
abs() -> Tensor

See :func:`torch.abs`
""")

add_docstr(torch._C.FloatTensorBase.abs_,
           """
abs_() -> Tensor

In-place version of :meth:`~Tensor.abs`
""")

add_docstr(torch._C.FloatTensorBase.acos,
           """
acos() -> Tensor

See :func:`torch.acos`
""")

add_docstr(torch._C.FloatTensorBase.acos_,
           """
acos_() -> Tensor

In-place version of :meth:`~Tensor.acos`
""")

add_docstr(torch._C.FloatTensorBase.add,
           """
add(value)

See :func:`torch.add`
""")

add_docstr(torch._C.FloatTensorBase.add_,
           """
add_(value)

In-place version of :meth:`~Tensor.add`
""")

add_docstr(torch._C.FloatTensorBase.addbmm,
           """
addbmm(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

See :func:`torch.addbmm`
""")

add_docstr(torch._C.FloatTensorBase.addbmm_,
           """
addbmm_(beta=1, mat, alpha=1, batch1, batch2) -> Tensor

In-place version of :meth:`~Tensor.addbmm`
""")

add_docstr(torch._C.FloatTensorBase.addcdiv,
           """
addcdiv(value=1, tensor1, tensor2) -> Tensor

See :func:`torch.addcdiv`
""")

add_docstr(torch._C.FloatTensorBase.addcdiv_,
           """
addcdiv_(value=1, tensor1, tensor2) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""")

add_docstr(torch._C.FloatTensorBase.addcmul,
           """
addcmul(value=1, tensor1, tensor2) -> Tensor

See :func:`torch.addcmul`
""")

add_docstr(torch._C.FloatTensorBase.addcmul_,
           """
addcmul_(value=1, tensor1, tensor2) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""")

add_docstr(torch._C.FloatTensorBase.addmm,
           """
addmm(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

See :func:`torch.addmm`
""")

add_docstr(torch._C.FloatTensorBase.addmm_,
           """
addmm_(beta=1, mat, alpha=1, mat1, mat2) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""")

add_docstr(torch._C.FloatTensorBase.addmv,
           """
addmv(beta=1, tensor, alpha=1, mat, vec) -> Tensor

See :func:`torch.addmv`
""")

add_docstr(torch._C.FloatTensorBase.addmv_,
           """
addmv_(beta=1, tensor, alpha=1, mat, vec) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""")

add_docstr(torch._C.FloatTensorBase.addr,
           """
addr(beta=1, alpha=1, vec1, vec2) -> Tensor

See :func:`torch.addr`
""")

add_docstr(torch._C.FloatTensorBase.addr_,
           """
addr_(beta=1, alpha=1, vec1, vec2) -> Tensor

In-place version of :meth:`~Tensor.addr`
""")

add_docstr(torch._C.FloatTensorBase.apply_,
           """
apply_(callable) -> Tensor

Applies the function :attr:`callable` to each element in the tensor, replacing
each element with the value returned by :attr:`callable`.

.. note::

    This function only works with CPU tensors and should not be used in code
    sections that require high performance.
""")

add_docstr(torch._C.FloatTensorBase.asin,
           """
asin() -> Tensor

See :func:`torch.asin`
""")

add_docstr(torch._C.FloatTensorBase.asin_,
           """
asin_() -> Tensor

In-place version of :meth:`~Tensor.asin`
""")

add_docstr(torch._C.FloatTensorBase.atan,
           """
atan() -> Tensor

See :func:`torch.atan`
""")

add_docstr(torch._C.FloatTensorBase.atan2,
           """
atan2(other) -> Tensor

See :func:`torch.atan2`
""")

add_docstr(torch._C.FloatTensorBase.atan2_,
           """
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""")

add_docstr(torch._C.FloatTensorBase.atan_,
           """
atan_() -> Tensor

In-place version of :meth:`~Tensor.atan`
""")

add_docstr(torch._C.FloatTensorBase.baddbmm,
           """
baddbmm(beta=1, alpha=1, batch1, batch2) -> Tensor

See :func:`torch.baddbmm`
""")

add_docstr(torch._C.FloatTensorBase.baddbmm_,
           """
baddbmm_(beta=1, alpha=1, batch1, batch2) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""")

add_docstr(torch._C.FloatTensorBase.bernoulli,
           """
bernoulli() -> Tensor

See :func:`torch.bernoulli`
""")

add_docstr(torch._C.FloatTensorBase.bernoulli_,
           """
bernoulli_() -> Tensor

In-place version of :meth:`~Tensor.bernoulli`
""")

add_docstr(torch._C.FloatTensorBase.bmm,
           """
bmm(batch2) -> Tensor

See :func:`torch.bmm`
""")

add_docstr(torch._C.FloatTensorBase.cauchy_,
           """
cauchy_(generator=None, median=0, sigma=1) -> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

.. math::

    P(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - median)^2 + \sigma^2}
""")

add_docstr(torch._C.FloatTensorBase.ceil,
           """
ceil() -> Tensor

See :func:`torch.ceil`
""")

add_docstr(torch._C.FloatTensorBase.ceil_,
           """
ceil_() -> Tensor

In-place version of :meth:`~Tensor.ceil`
""")

add_docstr(torch._C.FloatTensorBase.clamp,
           """
clamp(min, max) -> Tensor

See :func:`torch.clamp`
""")

add_docstr(torch._C.FloatTensorBase.clamp_,
           """
clamp_(min, max) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""")

add_docstr(torch._C.FloatTensorBase.clone,
           """
clone() -> Tensor

Returns a copy of the tensor. The copy has the same size and data type as the
original tensor.
""")

add_docstr(torch._C.FloatTensorBase.contiguous,
           """
contiguous() -> Tensor

Returns a contiguous Tensor containing the same data as this tensor. If this
tensor is contiguous, this function returns the original tensor.
""")

add_docstr(torch._C.FloatTensorBase.copy_,
           """
copy_(src, async=False) -> Tensor

Copies the elements from :attr:`src` into this tensor and returns this tensor.

The source tensor should have the same number of elements as this tensor. It
may be of a different data type or reside on a different device.

Args:
    src (Tensor): Source tensor to copy
    async (bool): If True and this copy is between CPU and GPU, then the copy
                  may occur asynchronously with respect to the host. For other
                  copies, this argument has no effect.
""")

add_docstr(torch._C.FloatTensorBase.cos,
           """
cos() -> Tensor

See :func:`torch.cos`
""")

add_docstr(torch._C.FloatTensorBase.cos_,
           """
cos_() -> Tensor

In-place version of :meth:`~Tensor.cos`
""")

add_docstr(torch._C.FloatTensorBase.cosh,
           """
cosh() -> Tensor

See :func:`torch.cosh`
""")

add_docstr(torch._C.FloatTensorBase.cosh_,
           """
cosh_() -> Tensor

In-place version of :meth:`~Tensor.cosh`
""")

add_docstr(torch._C.FloatTensorBase.cross,
           """
cross(other, dim=-1) -> Tensor

See :func:`torch.cross`
""")

add_docstr(torch._C.FloatTensorBase.cumprod,
           """
cumprod(dim) -> Tensor

See :func:`torch.cumprod`
""")

add_docstr(torch._C.FloatTensorBase.cumsum,
           """
cumsum(dim) -> Tensor

See :func:`torch.cumsum`
""")

add_docstr(torch._C.FloatTensorBase.data_ptr,
           """
data_ptr() -> int

Returns the address of the first element of this tensor.
""")

add_docstr(torch._C.FloatTensorBase.diag,
           """
diag(diagonal=0) -> Tensor

See :func:`torch.diag`
""")

add_docstr(torch._C.FloatTensorBase.dim,
           """
dim() -> int

Returns the number of dimensions of this tensor.
""")

add_docstr(torch._C.FloatTensorBase.dist,
           """
dist(other, p=2) -> Tensor

See :func:`torch.dist`
""")

add_docstr(torch._C.FloatTensorBase.div,
           """
div(value)

See :func:`torch.div`
""")

add_docstr(torch._C.FloatTensorBase.div_,
           """
div_(value)

In-place version of :meth:`~Tensor.div`
""")

add_docstr(torch._C.FloatTensorBase.dot,
           """
dot(tensor2) -> float

See :func:`torch.dot`
""")

add_docstr(torch._C.FloatTensorBase.eig,
           """
eig(eigenvectors=False) -> (Tensor, Tensor)

See :func:`torch.eig`
""")

add_docstr(torch._C.FloatTensorBase.element_size,
           """
element_size() -> int

Returns the size in bytes of an individual element.

Example:
    >>> torch.FloatTensor().element_size()
    4
    >>> torch.ByteTensor().element_size()
    1
""")

add_docstr(torch._C.FloatTensorBase.eq,
           """
eq(other) -> Tensor

See :func:`torch.eq`
""")

add_docstr(torch._C.FloatTensorBase.eq_,
           """
eq_(other) -> Tensor

In-place version of :meth:`~Tensor.eq`
""")

add_docstr(torch._C.FloatTensorBase.equal,
           """
equal(other) -> bool

See :func:`torch.equal`
""")

add_docstr(torch._C.FloatTensorBase.exp,
           """
exp() -> Tensor

See :func:`torch.exp`
""")

add_docstr(torch._C.FloatTensorBase.exp_,
           """
exp_() -> Tensor

In-place version of :meth:`~Tensor.exp`
""")

add_docstr(torch._C.FloatTensorBase.exponential_,
           """
exponential_(generator=None, lambd=1) -> Tensor

Fills this tensor with elements drawn from the exponential distribution:

.. math::

    P(x) = \lambda e^{-\lambda x}
""")

add_docstr(torch._C.FloatTensorBase.fill_,
           """
fill_(value) -> Tensor

Fills this tensor with the specified value.
""")

add_docstr(torch._C.FloatTensorBase.floor,
           """
floor() -> Tensor

See :func:`torch.floor`
""")

add_docstr(torch._C.FloatTensorBase.floor_,
           """
floor_() -> Tensor

In-place version of :meth:`~Tensor.floor`
""")

add_docstr(torch._C.FloatTensorBase.fmod,
           """
fmod(divisor) -> Tensor

See :func:`torch.fmod`
""")

add_docstr(torch._C.FloatTensorBase.fmod_,
           """
fmod_(divisor) -> Tensor

In-place version of :meth:`~Tensor.fmod`
""")

add_docstr(torch._C.FloatTensorBase.frac,
           """
frac() -> Tensor

See :func:`torch.frac`
""")

add_docstr(torch._C.FloatTensorBase.frac_,
           """
frac_() -> Tensor

In-place version of :meth:`~Tensor.frac`
""")

add_docstr(torch._C.FloatTensorBase.gather,
           """
gather(dim, index) -> Tensor

See :func:`torch.gather`
""")

add_docstr(torch._C.FloatTensorBase.ge,
           """
ge(other) -> Tensor

See :func:`torch.ge`
""")

add_docstr(torch._C.FloatTensorBase.ge_,
           """
ge_(other) -> Tensor

In-place version of :meth:`~Tensor.ge`
""")

add_docstr(torch._C.FloatTensorBase.gels,
           """
gels(A) -> Tensor

See :func:`torch.gels`
""")

add_docstr(torch._C.FloatTensorBase.geometric_,
           """
geometric_(generator=None, p) -> Tensor

Fills this tensor with elements drawn from the geometric distribution:

.. math::

    P(X=k) = (1 - p)^{k - 1} p

""")

add_docstr(torch._C.FloatTensorBase.geqrf,
           """
geqrf() -> (Tensor, Tensor)

See :func:`torch.geqrf`
""")

add_docstr(torch._C.FloatTensorBase.ger,
           """
ger(vec2) -> Tensor

See :func:`torch.ger`
""")

add_docstr(torch._C.FloatTensorBase.gesv,
           """
gesv(A) -> Tensor, Tensor

See :func:`torch.gesv`
""")

add_docstr(torch._C.FloatTensorBase.gt,
           """
gt(other) -> Tensor

See :func:`torch.gt`
""")

add_docstr(torch._C.FloatTensorBase.gt_,
           """
gt_(other) -> Tensor

In-place version of :meth:`~Tensor.gt`
""")

add_docstr(torch._C.FloatTensorBase.histc,
           """
histc(bins=100, min=0, max=0) -> Tensor

See :func:`torch.histc`
""")

add_docstr(torch._C.FloatTensorBase.index,
           """
index(m) -> Tensor

Selects elements from this tensor using a binary mask or along a given
dimension. The expression ``tensor.index(m)`` is equivalent to ``tensor[m]``.

Args:
    m (int or ByteTensor or slice): The dimension or mask used to select elements
""")

add_docstr(torch._C.FloatTensorBase.index_add_,
           """
index_add_(dim, index, tensor) -> Tensor

Accumulate the elements of tensor into the original tensor by adding to the
indices in the order given in index. The shape of tensor must exactly match the
elements indexed or an error will be raised.

Args:
    dim (int): Dimension along which to index
    index (LongTensor): Indices to select from tensor
    tensor (Tensor): Tensor containing values to add

Example:
    >>> x = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    >>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2, 1])
    >>> x.index_add_(0, index, t)
    >>> x
      2   3   4
      8   9  10
      5   6   7
    [torch.FloatTensor of size 3x3]
""")

add_docstr(torch._C.FloatTensorBase.index_copy_,
           """
index_copy_(dim, index, tensor) -> Tensor

Copies the elements of tensor into the original tensor by selecting the
indices in the order given in index. The shape of tensor must exactly match the
elements indexed or an error will be raised.

Args:
    dim (int): Dimension along which to index
    index (LongTensor): Indices to select from tensor
    tensor (Tensor): Tensor containing values to copy

Example:
    >>> x = torch.Tensor(3, 3)
    >>> t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2, 1])
    >>> x.index_copy_(0, index, t)
    >>> x
     1  2  3
     7  8  9
     4  5  6
    [torch.FloatTensor of size 3x3]
""")

add_docstr(torch._C.FloatTensorBase.index_fill_,
           """
index_fill_(dim, index, tensor) -> Tensor

Fills the elements of the original tensor with value :attr:`val` by selecting
the indices in the order given in index.

Args:
    dim (int): Dimension along which to index
    index (LongTensor): Indices
    val (float): Value to fill

Example:
    >>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> index = torch.LongTensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    >>> x
    -1  2 -1
    -1  5 -1
    -1  8 -1
    [torch.FloatTensor of size 3x3]
""")

add_docstr(torch._C.FloatTensorBase.index_select,
           """
index_select(dim, index) -> Tensor

See :func:`torch.index_select`
""")

add_docstr(torch._C.FloatTensorBase.inverse,
           """
inverse() -> Tensor

See :func:`torch.inverse`
""")

add_docstr(torch._C.FloatTensorBase.is_contiguous,
           """
is_contiguous() -> bool

Returns True if this tensor is contiguous in memory in C order.
""")

add_docstr(torch._C.FloatTensorBase.is_set_to,
           """
is_set_to(tensor) -> bool

Returns True if this object refers to the same ``THTensor`` object from the
Torch C API as the given tensor.
""")

add_docstr(torch._C.FloatTensorBase.kthvalue,
           """
kthvalue(k, dim=None) -> (Tensor, LongTensor)

See :func:`torch.kthvalue`
""")

add_docstr(torch._C.FloatTensorBase.le,
           """
le(other) -> Tensor

See :func:`torch.le`
""")

add_docstr(torch._C.FloatTensorBase.le_,
           """
le_(other) -> Tensor

In-place version of :meth:`~Tensor.le`
""")

add_docstr(torch._C.FloatTensorBase.lerp,
           """
lerp(start, end, weight)

See :func:`torch.lerp`
""")

add_docstr(torch._C.FloatTensorBase.lerp_,
           """
lerp_(start, end, weight)

In-place version of :meth:`~Tensor.lerp`
""")

add_docstr(torch._C.FloatTensorBase.log,
           """
log() -> Tensor

See :func:`torch.log`
""")

add_docstr(torch._C.FloatTensorBase.log1p,
           """
log1p() -> Tensor

See :func:`torch.log1p`
""")

add_docstr(torch._C.FloatTensorBase.log1p_,
           """
log1p_() -> Tensor

In-place version of :meth:`~Tensor.log1p`
""")

add_docstr(torch._C.FloatTensorBase.log_, """
log_() -> Tensor

In-place version of :meth:`~Tensor.log`
""")

add_docstr(torch._C.FloatTensorBase.log_normal_, u"""
log_normal_(generator=None, mean=1, std=2)

Fills this tensor with numbers samples from the log-normal distribution
parameterized by the given mean (\u00B5) and standard deviation (\u03C3). Note that
:attr:`mean` and :attr:`stdv` are the mean and standard deviation of the
underlying normal distribution, and not of the returned distribution:

.. math::

    P(x) = \\dfrac{1}{x \\sigma \\sqrt{2\\pi}} e^{-\\dfrac{(\\ln x - \\mu)^2}{2\\sigma^2}}
""")

add_docstr(torch._C.FloatTensorBase.lt,
           """
lt(other) -> Tensor

See :func:`torch.lt`
""")

add_docstr(torch._C.FloatTensorBase.lt_,
           """
lt_(other) -> Tensor

In-place version of :meth:`~Tensor.lt`
""")

add_docstr(torch._C.FloatTensorBase.map_,
           """
map_(tensor, callable)

Applies :attr:`callable` for each element in this tensor and the given tensor
and stores the results in this tensor. The :attr:`callable` should have the
signature::

    def callable(a, b) -> number
""")

add_docstr(torch._C.FloatTensorBase.masked_copy_,
           """
masked_copy_(mask, source)

Copies elements from :attr:`source` into this tensor at positions where the
:attr:`mask` is one. The :attr:`mask` should have the same number of elements
as this tensor. The :attr:`source` should have at least as many elements as the
number of ones in :attr:`mask`

Args:
    mask (ByteTensor): The binary mask
    source (Tensor): The tensor to copy from

.. note::

    The :attr:`mask` operates on the :attr:`self` tensor, not on the given
    :attr:`source` tensor.
""")

add_docstr(torch._C.FloatTensorBase.masked_fill_,
           """
masked_fill_(mask, value)

Fills elements of this tensor with :attr:`value` where :attr:`mask` is one.
The :attr:`mask` should have the same number of elements as this tensor, but
the shape may differ.

Args:
    mask (ByteTensor): The binary mask
    value (Tensor): The value to fill
""")

add_docstr(torch._C.FloatTensorBase.masked_select,
           """
masked_select(mask) -> Tensor

See :func:`torch.masked_select`
""")

add_docstr(torch._C.FloatTensorBase.max,
           """
max(dim=None) -> float or (Tensor, Tensor)

See :func:`torch.max`
""")

add_docstr(torch._C.FloatTensorBase.mean,
           """
mean(dim=None) -> float or (Tensor, Tensor)

See :func:`torch.mean`
""")

add_docstr(torch._C.FloatTensorBase.median,
           """
median(dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

See :func:`torch.median`
""")

add_docstr(torch._C.FloatTensorBase.min,
           """
min(dim=None) -> float or (Tensor, Tensor)

See :func:`torch.min`
""")

add_docstr(torch._C.FloatTensorBase.mm,
           """
mm(mat2) -> Tensor

See :func:`torch.mm`
""")

add_docstr(torch._C.FloatTensorBase.mode,
           """
mode(dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

See :func:`torch.mode`
""")

add_docstr(torch._C.FloatTensorBase.mul,
           """
mul(value) -> Tensor

See :func:`torch.mul`
""")

add_docstr(torch._C.FloatTensorBase.mul_,
           """
mul_(value)

In-place version of :meth:`~Tensor.mul`
""")

add_docstr(torch._C.FloatTensorBase.multinomial,
           """
multinomial(generator=None, num_samples, replacement=False)

See :func:`torch.multinomial`
""")

add_docstr(torch._C.FloatTensorBase.mv,
           """
mv(vec) -> Tensor

See :func:`torch.mv`
""")

add_docstr(torch._C.FloatTensorBase.narrow,
           """
narrow(dimension, start, length) -> Tensor

Returns a new tensor that is a narrowed version of this tensor. The dimension
:attr:`dim` is narrowed from :attr:`start` to :attr:`start + length`. The
returned tensor and this tensor share the same underlying storage.

Args:
    dimension (int): The dimension along which to narrow
    start (int): The starting dimension
    length (int):

Example:
    >>> x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> x.narrow(0, 0, 2)
     1  2  3
     4  5  6
    [torch.FloatTensor of size 2x3]
    >>> x.narrow(1, 1, 2)
     2  3
     5  6
     8  9
    [torch.FloatTensor of size 3x2]
""")

add_docstr(torch._C.FloatTensorBase.ndimension,
           """
ndimension() -> int

Alias for :meth:`~Tensor.dim()`
""")

add_docstr(torch._C.FloatTensorBase.ne,
           """
ne(other) -> Tensor

See :func:`torch.ne`
""")

add_docstr(torch._C.FloatTensorBase.ne_,
           """
ne_(other) -> Tensor

In-place version of :meth:`~Tensor.ne`
""")

add_docstr(torch._C.FloatTensorBase.neg,
           """
neg() -> Tensor

See :func:`torch.neg`
""")

add_docstr(torch._C.FloatTensorBase.neg_,
           """
neg_() -> Tensor

In-place version of :meth:`~Tensor.neg`
""")

add_docstr(torch._C.FloatTensorBase.nelement,
           """
nelement() -> int

Alias for :meth:`~Tensor.numel`
""")

add_docstr(torch._C.FloatTensorBase.nonzero,
           """
nonzero() -> LongTensor

See :func:`torch.nonzero`
""")

add_docstr(torch._C.FloatTensorBase.norm,
           """
norm(p=2) -> float

See :func:`torch.norm`
""")

add_docstr(torch._C.FloatTensorBase.normal_,
           """
normal_(generator=None, mean=0, std=1)

Fills this tensor with elements samples from the normal distribution
parameterized by :attr:`mean` and :attr:`std`.
""")

add_docstr(torch._C.FloatTensorBase.numel,
           """
numel() -> int

See :func:`torch.numel`
""")

add_docstr(torch._C.FloatTensorBase.numpy,
           """
numpy() -> ndarray

Returns this tensor as a NumPy :class:`ndarray`. This tensor and the returned
:class:`ndarray` share the same underlying storage. Changes to this tensor will
be reflected in the :class:`ndarray` and vice versa.
""")

add_docstr(torch._C.FloatTensorBase.orgqr,
           """
orgqr(input2) -> Tensor

See :func:`torch.orgqr`
""")

add_docstr(torch._C.FloatTensorBase.ormqr,
           """
ormqr(input2, input3, left=True, transpose=False) -> Tensor

See :func:`torch.ormqr`
""")

add_docstr(torch._C.FloatTensorBase.potrf,
           """
potrf(upper=True) -> Tensor

See :func:`torch.potrf`
""")

add_docstr(torch._C.FloatTensorBase.potri,
           """
potri(upper=True) -> Tensor

See :func:`torch.potri`
""")

add_docstr(torch._C.FloatTensorBase.potrs,
           """
potrs(input2, upper=True) -> Tensor

See :func:`torch.potrs`
""")

add_docstr(torch._C.FloatTensorBase.pow,
           """
pow(exponent)

See :func:`torch.pow`
""")

add_docstr(torch._C.FloatTensorBase.pow_,
           """
pow_(exponent)

In-place version of :meth:`~Tensor.pow`
""")

add_docstr(torch._C.FloatTensorBase.prod,
           """
prod() -> float

See :func:`torch.prod`
""")

add_docstr(torch._C.FloatTensorBase.pstrf,
           """
pstrf(upper=True, tol=-1) -> (Tensor, IntTensor)

See :func:`torch.pstrf`
""")

add_docstr(torch._C.FloatTensorBase.qr,
           """
qr() -> (Tensor, Tensor)

See :func:`torch.qr`
""")

add_docstr(torch._C.FloatTensorBase.random_,
           """
random_(generator=None, from=0, to=None)

Fills this tensor with numbers sampled from the uniform distribution or
discrete uniform distribution over [from, to]. If not specified, :attr:`to`
defaults to the largest value representable by this tensor's data type.
""")

add_docstr(torch._C.FloatTensorBase.reciprocal,
           """
reciprocal() -> Tensor

See :func:`torch.reciprocal`
""")

add_docstr(torch._C.FloatTensorBase.reciprocal_,
           """
reciprocal_() -> Tensor

In-place version of :meth:`~Tensor.reciprocal`
""")

add_docstr(torch._C.FloatTensorBase.remainder,
           """
remainder(divisor) -> Tensor

See :func:`torch.remainder`
""")

add_docstr(torch._C.FloatTensorBase.remainder_,
           """
remainder_(divisor) -> Tensor

In-place version of :meth:`~Tensor.remainder`
""")

add_docstr(torch._C.FloatTensorBase.renorm,
           """
renorm(p, dim, maxnorm) -> Tensor

See :func:`torch.renorm`
""")

add_docstr(torch._C.FloatTensorBase.renorm_,
           """
renorm_(p, dim, maxnorm) -> Tensor

In-place version of :meth:`~Tensor.renorm`
""")

add_docstr(torch._C.FloatTensorBase.resize_,
           """
resize_(*sizes)

Resizes this tensor to the specified size. If the number of elements is
larger than the current storage size, then the underlying storage is resized
to fit the new number of elements. If the number of elements is smaller, the
underlying storage is not changed. Existing elements are preserved but any new
memory is uninitialized.

Args:
    sizes (torch.Size or int...): The desired size

Example:
    >>> x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    >>> x.resize_(2, 2)
    >>> x
     1  2
     3  4
    [torch.FloatTensor of size 2x2]
""")

add_docstr(torch._C.FloatTensorBase.resize_as_,
           """
resize_as_(tensor)

Resizes the current tensor to be the same size as the specified tensor. This is
equivalent to::

    self.resize_(tensor.size())
""")

add_docstr(torch._C.FloatTensorBase.round,
           """
round() -> Tensor

See :func:`torch.round`
""")

add_docstr(torch._C.FloatTensorBase.round_,
           """
round_() -> Tensor

In-place version of :meth:`~Tensor.round`
""")

add_docstr(torch._C.FloatTensorBase.rsqrt,
           """
rsqrt() -> Tensor

See :func:`torch.rsqrt`
""")

add_docstr(torch._C.FloatTensorBase.rsqrt_,
           """
rsqrt_() -> Tensor

In-place version of :meth:`~Tensor.rsqrt`
""")

add_docstr(torch._C.FloatTensorBase.scatter_,
           """
scatter_(input, dim, index, src) -> Tensor

Writes all values from the Tensor :attr:`src` into self at the indices specified
in the :attr:`index` Tensor. The indices are specified with respect to the
given dimension, dim, in the manner described in :meth:`~Tensor.gather`.

Note that, as for gather, the values of index must be between `0` and `(self.size(dim) -1)`
inclusive and all values in a row along the specified dimension must be unique.

Args:
    input (Tensor): The source tensor
    dim (int): The axis along which to index
    index (LongTensor): The indices of elements to scatter
    src (Tensor or float): The source element(s) to scatter

Example::

    >>> x = torch.rand(2, 5)
    >>> x

     0.4319  0.6500  0.4080  0.8760  0.2355
     0.2609  0.4711  0.8486  0.8573  0.1029
    [torch.FloatTensor of size 2x5]

    >>> torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

     0.4319  0.4711  0.8486  0.8760  0.2355
     0.0000  0.6500  0.0000  0.8573  0.0000
     0.2609  0.0000  0.4080  0.0000  0.1029
    [torch.FloatTensor of size 3x5]

    >>> z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
    >>> z

     0.0000  0.0000  1.2300  0.0000
     0.0000  0.0000  0.0000  1.2300
    [torch.FloatTensor of size 2x4]

""")

add_docstr(torch._C.FloatTensorBase.select,
           """
select(dim, index) -> Tensor or number

Slices the tensor along the selected dimension at the given index. If this
tensor is one dimensional, this function returns a number. Otherwise, it
returns a tensor with the given dimension removed.

Args:
    dim (int): Dimension to slice
    index (int): Index to select

.. note::

    :meth:`select` is equivalent to slicing. For example, ``tensor.select(0, index)``
    is equivalent to ``tensor[index]`` and ``tensor.select(2, index)`` is equivalent
    to ``tensor[:,:,index]``.
""")

add_docstr(torch._C.FloatTensorBase.set_,
           """
set_(source=None, storage_offset=0, size=None, stride=None)

Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
this tensor will share the same storage and have the same size and strides
as the given tensor. Changes to elements in one tensor will be reflected in the
other.

If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
storage, offset, size, and stride.

Args:
    source (Tensor or Storage): The tensor or storage to use
    storage_offset (int): The offset in the storage
    size (torch.Size): The desired size. Defaults to the size of the source.
    stride (tuple): The desired stride. Defaults to C-contiguous strides.
""")

add_docstr(torch._C.FloatTensorBase.sigmoid,
           """
sigmoid() -> Tensor

See :func:`torch.sigmoid`
""")

add_docstr(torch._C.FloatTensorBase.sigmoid_,
           """
sigmoid_() -> Tensor

In-place version of :meth:`~Tensor.sigmoid`
""")

add_docstr(torch._C.FloatTensorBase.sign,
           """
sign() -> Tensor

See :func:`torch.sign`
""")

add_docstr(torch._C.FloatTensorBase.sign_,
           """
sign_() -> Tensor

In-place version of :meth:`~Tensor.sign`
""")

add_docstr(torch._C.FloatTensorBase.sin,
           """
sin() -> Tensor

See :func:`torch.sin`
""")

add_docstr(torch._C.FloatTensorBase.sin_,
           """
sin_() -> Tensor

In-place version of :meth:`~Tensor.sin`
""")

add_docstr(torch._C.FloatTensorBase.sinh,
           """
sinh() -> Tensor

See :func:`torch.sinh`
""")

add_docstr(torch._C.FloatTensorBase.sinh_,
           """
sinh_() -> Tensor

In-place version of :meth:`~Tensor.sinh`
""")

add_docstr(torch._C.FloatTensorBase.size,
           """
size() -> torch.Size

Returns the size of the tensor. The returned value is a subclass of
:class:`tuple`.

Example:
    >>> torch.Tensor(3, 4, 5).size()
    torch.Size([3, 4, 5])
""")

add_docstr(torch._C.FloatTensorBase.sort,
           """
sort(dim=None, descending=False) -> (Tensor, LongTensor)

See :func:`torch.sort`
""")

add_docstr(torch._C.FloatTensorBase.sqrt,
           """
sqrt() -> Tensor

See :func:`torch.sqrt`
""")

add_docstr(torch._C.FloatTensorBase.sqrt_,
           """
sqrt_() -> Tensor

In-place version of :meth:`~Tensor.sqrt`
""")

add_docstr(torch._C.FloatTensorBase.squeeze,
           """
squeeze(dim=None)

See :func:`torch.squeeze`
""")

add_docstr(torch._C.FloatTensorBase.squeeze_,
           """
squeeze_(dim=None)

In-place version of :meth:`~Tensor.squeeze`
""")

add_docstr(torch._C.FloatTensorBase.std,
           """
std() -> float

See :func:`torch.std`
""")

add_docstr(torch._C.FloatTensorBase.storage,
           """
storage() -> torch.Storage

Returns the underlying storage
""")

add_docstr(torch._C.FloatTensorBase.storage_offset,
           """
storage_offset() -> int

Returns this tensor's offset in the underlying storage in terms of number of
storage elements (not bytes).

Example:
    >>> x = torch.Tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3
""")

add_docstr(torch._C.FloatTensorBase.stride,
           """
stride() -> tuple

Returns the stride of the tensor.
""")

add_docstr(torch._C.FloatTensorBase.sub,
           """
sub(value, other) -> Tensor

Subtracts a scalar or tensor from this tensor. If both :attr:`value` and
:attr:`other` are specified, each element of :attr:`other` is scaled by
:attr:`value` before being used.
""")

add_docstr(torch._C.FloatTensorBase.sub_,
           """
sub_(x) -> Tensor

In-place version of :meth:`~Tensor.sub`
""")

add_docstr(torch._C.FloatTensorBase.sum,
           """
sum(dim=None) -> float

See :func:`torch.sum`
""")

add_docstr(torch._C.FloatTensorBase.svd,
           """
svd(some=True) -> (Tensor, Tensor, Tensor)

See :func:`torch.svd`
""")

add_docstr(torch._C.FloatTensorBase.symeig,
           """
symeig(eigenvectors=False, upper=True) -> (Tensor, Tensor)

See :func:`torch.symeig`
""")

add_docstr(torch._C.FloatTensorBase.t,
           """
t() -> Tensor

See :func:`torch.t`
""")

add_docstr(torch._C.FloatTensorBase.t_,
           """
t_() -> Tensor

In-place version of :meth:`~Tensor.t`
""")

add_docstr(torch._C.FloatTensorBase.tan,
           """
tan() -> Tensor

See :func:`torch.tan`
""")

add_docstr(torch._C.FloatTensorBase.tan_,
           """
tan_() -> Tensor

In-place version of :meth:`~Tensor.tan`
""")

add_docstr(torch._C.FloatTensorBase.tanh,
           """
tanh() -> Tensor

See :func:`torch.tanh`
""")

add_docstr(torch._C.FloatTensorBase.tanh_,
           """
tanh_() -> Tensor

In-place version of :meth:`~Tensor.tanh`
""")

add_docstr(torch._C.FloatTensorBase.topk,
           """
topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

See :func:`torch.topk`
""")

add_docstr(torch._C.FloatTensorBase.trace,
           """
trace() -> float

See :func:`torch.trace`
""")

add_docstr(torch._C.FloatTensorBase.transpose,
           """
transpose(dim0, dim1) -> Tensor

See :func:`torch.transpose`
""")

add_docstr(torch._C.FloatTensorBase.transpose_,
           """
transpose_(dim0, dim1) -> Tensor

In-place version of :meth:`~Tensor.transpose`
""")

add_docstr(torch._C.FloatTensorBase.tril,
           """
tril(k=0) -> Tensor

See :func:`torch.tril`
""")

add_docstr(torch._C.FloatTensorBase.tril_,
           """
tril_(k=0) -> Tensor

In-place version of :meth:`~Tensor.tril`
""")

add_docstr(torch._C.FloatTensorBase.triu,
           """
triu(k=0) -> Tensor

See :func:`torch.triu`
""")

add_docstr(torch._C.FloatTensorBase.triu_,
           """
triu_(k=0) -> Tensor

In-place version of :meth:`~Tensor.triu`
""")

add_docstr(torch._C.FloatTensorBase.trtrs,
           """
trtrs(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

See :func:`torch.trtrs`
""")

add_docstr(torch._C.FloatTensorBase.trunc,
           """
trunc() -> Tensor

See :func:`torch.trunc`
""")

add_docstr(torch._C.FloatTensorBase.trunc_,
           """
trunc_() -> Tensor

In-place version of :meth:`~Tensor.trunc`
""")

add_docstr(torch._C.FloatTensorBase.unfold,
           """
unfold(dim, size, step) -> Tensor

Returns a tensor which contains all slices of size :attr:`size` in
the dimension :attr:`dim`.

Step between two slices is given by :attr:`step`.

If `sizedim` is the original size of dimension dim, the size of dimension `dim`
in the returned tensor will be `(sizedim - size) / step + 1`

An additional dimension of size size is appended in the returned tensor.

Args:
    dim (int): dimension in which unfolding happens
    size (int): size of each slice that is unfolded
    step (int): the step between each slice

Example::

    >>> x = torch.range(1, 7)
    >>> x

     1
     2
     3
     4
     5
     6
     7
    [torch.FloatTensor of size 7]

    >>> x.unfold(0, 2, 1)

     1  2
     2  3
     3  4
     4  5
     5  6
     6  7
    [torch.FloatTensor of size 6x2]

    >>> x.unfold(0, 2, 2)

     1  2
     3  4
     5  6
    [torch.FloatTensor of size 3x2]

""")

add_docstr(torch._C.FloatTensorBase.uniform_,
           """
uniform_(from=0, to=1) -> Tensor

Fills this tensor with numbers sampled from the uniform distribution:

.. math:

    P(x) = \dfrac{1}{to - from}
""")

add_docstr(torch._C.FloatTensorBase.unsqueeze,
           """
unsqueeze(dim)

See :func:`torch.unsqueeze`
""")

add_docstr(torch._C.FloatTensorBase.unsqueeze_,
           """
unsqueeze_(dim)

In-place version of :meth:`~Tensor.unsqueeze`
""")

add_docstr(torch._C.FloatTensorBase.var,
           """
var() -> float

See :func:`torch.var`
""")

add_docstr(torch._C.FloatTensorBase.view,
           """
view(*args) -> Tensor

Returns a new tensor with the same data but different size.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. A tensor must be
:func:`contiguous` to be viewed.

Args:
    args (torch.Size or int...): Desired size

Example:
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

add_docstr(torch._C.FloatTensorBase.zero_,
           """
zero_()

Fills this tensor with zeros.
""")
