"""Adds docstrings to functions defined in the torch._C"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.abs,
"""abs(tensor, out=None) -> tensor

Computes the element-wise absolute value of a tensor.

Example:
    >>> torch.abs(torch.FloatTensor([-1, -2, 3]))
    FloatTensor([1, 2, 3])
""")

add_docstr(torch._C.acos,
"""
acos(tensor, out=None) -> tensor

Computes the element-wise inverse cosine of a tensor.

Example:
    >>> torch.acos(torch.FloatTensor([1, -1]))
    FloatTensor([0.0000, 3.1416])
""")

add_docstr(torch._C.add,
"""
""")

add_docstr(torch._C.addbmm,
"""
""")

add_docstr(torch._C.addcdiv,
"""
""")

add_docstr(torch._C.addcmul,
"""
""")

add_docstr(torch._C.addmm,
"""
""")

add_docstr(torch._C.addmv,
"""
""")

add_docstr(torch._C.addr,
"""
""")

add_docstr(torch._C.all,
"""
""")

add_docstr(torch._C.any,
"""
""")

add_docstr(torch._C.asin,
"""
""")

add_docstr(torch._C.atan,
"""
""")

add_docstr(torch._C.atan2,
"""
""")

add_docstr(torch._C.baddbmm,
"""
""")

add_docstr(torch._C.bernoulli,
"""
""")

add_docstr(torch._C.bmm,
"""
""")

add_docstr(torch._C.cat,
"""
""")

add_docstr(torch._C.cauchy,
"""
""")

add_docstr(torch._C.cdiv,
"""
""")

add_docstr(torch._C.ceil,
"""
""")

add_docstr(torch._C.cfmod,
"""
""")

add_docstr(torch._C.cinv,
"""
""")

add_docstr(torch._C.clamp,
"""
""")

add_docstr(torch._C.cmax,
"""
""")

add_docstr(torch._C.cmin,
"""
""")

add_docstr(torch._C.cmod,
"""
""")

add_docstr(torch._C.cmul,
"""
""")

add_docstr(torch._C.cos,
"""
""")

add_docstr(torch._C.cosh,
"""
""")

add_docstr(torch._C.cpow,
"""
""")

add_docstr(torch._C.cremainder,
"""
""")

add_docstr(torch._C.cross,
"""
""")

add_docstr(torch._C.csub,
"""
""")

add_docstr(torch._C.cumprod,
"""
""")

add_docstr(torch._C.cumsum,
"""
""")

add_docstr(torch._C.diag,
"""
""")

add_docstr(torch._C.dist,
"""
""")

add_docstr(torch._C.div,
"""
""")

add_docstr(torch._C.dot,
"""
dot(tensor1, tensor2) -> float

Computes the dot product (inner product) of two tensors. Both tensors are
treated as 1-D vectors.

Example:
    >>> torch.dot(torch.Tensor([2, 3]), torch.Tensor([2, 1]))
    7.0
""")

add_docstr(torch._C.eig,
"""
eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)

Computes the eigenvalues and eigenvectors of a real square matrix.

Args:
    a (Tensor): A square matrix for which the eigenvalues and eigenvectors will
                be computed
    eigenvectors (bool): `True` to compute both eigenvalues and eigenvectors.
                         Otherwise, only eigenvalues will be computed.
    out (tuple, optional): Output tensors

Returns:
    (Tensor, Tensor): tuple containing

        - **e** (*Tensor*): the right eigenvalues of ``a``
        - **v** (*Tensor*): the eigenvectors of ``a`` if ``eigenvectors` is ``True``; otherwise an empty tensor
""")

add_docstr(torch._C.eq,
"""
eq(tensor, other, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where the tensors are equal and a 0 at every other location

Example:
    >>> torch.eq(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    1  0
    0  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.equal,
"""
equal(tensor1, tensor2) -> bool

True if two tensors have the same size and elements, False otherwise.

Example:
    >>> torch.equal(torch.Tensor([1, 2]), torch.Tensor([1, 2]))
    True
""")

add_docstr(torch._C.exp,
"""
exp(tensor, out=None) -> tensor

Computes the exponential of each element.

Example:
    >>> torch.exp(torch.Tensor([0, math.log(2)]))
    torch.FloatTensor([1, 2])
""")

add_docstr(torch._C.eye,
"""
eye(n, m=None, out=None)

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): Number of rows
    m (int, optional): Number of columns. If None, defaults to `n`
    out (Tensor, optional): Output tensor

Returns:
    Tensor: a 2-D tensor with ones on the diagonal and zeros elsewhere

Example:
    >>> torch.eye(3)
     1  0  0
     0  1  0
     0  0  1
    [torch.FloatTensor of size 3x3]
""")

add_docstr(torch._C.floor,
"""
floor(tensor, out=None) -> Tensor

Computes the floor of each element in `tensor`. That is, each element is
rounded down to the nearest integer.

Example:
    >>> torch.floor(torch.Tensor([1.5, -2.7, 3.0]))
    torch.FloatTensor([1, -3, 3])
""")

add_docstr(torch._C.fmod,
"""
fmod(tensor, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend `tensor`.

Args:
    tensor (Tensor): The dividend
    divisor (Tensor or float): The divisor. This may be either a number or a
                               tensor of the same shape as the dividend.
    out (Tensor, optional): Output tensor

Example:
    >>> torch.fmod(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    torch.FloatTensor([-1, -0, -1, 1, 0, 1])
    >>> torch.fmod(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    torch.FloatTensor([1.0, 0.5, 0.0, 1.0, 0.5])

    .. seealso::

        :func:`torch.remainder`, which computes the element-wise remainder of
        division equivalently to Python's `%` operator
""")

add_docstr(torch._C.frac,
"""
frac(tensor, out=None) -> Tensor

Computes the fractional portion of each element in `tensor`.

Example:
    >>> torch.frac(torch.Tensor([1, 2.5, -3.2])
    torch.FloatTensor([0, 0.5, -0.2])
""")

add_docstr(torch._C.from_numpy,
"""
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and `ndarray` share the same memory. Modifications to the
tensor will be reflected in the `ndarray` and vice versa. The returned tensor
is not resizable.

Example:
    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    torch.LongTensor([1, 2, 3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
""")

add_docstr(torch._C.gather,
"""
gather(tensor, dim, index, out=None) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = tensor[index[i][j][k]][j][k]  # dim=0
    out[i][j][k] = tensor[i][index[i][j][k]][k]  # dim=1
    out[i][j][k] = tensor[i][j][index[i][j][k]]  # dim=3

Args:
    tensor (Tensor): The source tensor
    dim (int): The axis along which to index
    index (LongTensor): The indices of elements to gather
    out (Tensor, optional): Destination tensor

Example:
    >>> t = torch.Tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
     1  1
     4  3
    [torch.FloatTensor of size 2x2]""")

add_docstr(torch._C.ge,
"""
ge(tensor, other, out=None) -> Tensor

Computes `tensor >= other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example:
    >>> torch.ge(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     1  1
     0  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.gels,
r"""
gels(B, A, out=None) -> Tensor

Computes the solution to the least squares and least norm problems for a full
rank :math:`m` by :math:`n` matrix :math:`A`.

If :math:`m >= n`, :func:`gels` solves the least-squares problem:

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \|AX-B\|_F.
   \end{array}

If :math:`m < n`, :func:`gels` solves the least-norm problem:

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \|X\|_F & \mbox{subject to} & AX = B.
   \end{array}

The first :math:`n` rows of the returned matrix :math:`X` contains the
solution. The remaining rows contain residual information: the euclidean norm
of each column starting at row :math:`n` is the residual for the corresponding
column.

Args:
    B (Tensor): The matrix :math:`B`
    A (Tensor): The :math:`m` by :math:`n` matrix :math:`A`
    out (tuple, optional): Optional destination tensor

Returns:
    (Tensor, Tensor): tuple containing:

        - **X** (*Tensor*): the least squares solution
        - **qr** (*Tensor*): the details of the QR factorization

.. note::

    The returned matrices will always be tranposed, irrespective of the strides
    of the input matrices. That is, they will have stride `(1, m)` instead of
    `(m, 1)`.

Example:

    >>> A = torch.Tensor([[1, 1, 1],
    ...                   [2, 3, 4],
    ...                   [3, 5, 2],
    ...                   [4, 2, 5],
    ...                   [5, 4, 3]])
    >>> B = torch.Tensor([[-10, -3],
                          [ 12, 14],
                          [ 14, 12],
                          [ 16, 16],
                          [ 18, 16]])
    >>> X, _ = torch.gels(B, A)
    >>> X
    2.0000  1.0000
    1.0000  1.0000
    1.0000  2.0000
    [torch.FloatTensor of size 3x2]
""")

add_docstr(torch._C.geqrf,
"""
""")

add_docstr(torch._C.ger,
"""
""")

add_docstr(torch._C.gesv,
"""
""")

add_docstr(torch._C.get_num_threads,
"""
get_num_threads() -> int

Gets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.gt,
"""
gt(tensor, other, out=None) -> Tensor

Computes `tensor > other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example:
    >>> torch.gt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  1
     0  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.histc,
"""
histc(tensor, bins=100, min=0, max=0, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between `min` and `max`. If `min`
and `max` are both zero, the minimum and maximum values of the data are used.

Args:
    tensor (Tensor): Input data
    bins (int): Number of histogram bins
    min (int): Lower end of the range (inclusive)
    max (int): Upper end of the range (inclusive)
    out (Tensor, optional): Output argument

Returns:
    Tensor: the histogram

Example:
    >>> torch.histc(torch.FloatTensor([1, 2, 1]), bins=4, min=0, max=3)
    FloatTensor([0, 2, 1, 0])

""")

add_docstr(torch._C.index_select,
"""
""")

add_docstr(torch._C.inverse,
"""
""")

add_docstr(torch._C.kthvalue,
"""
""")

add_docstr(torch._C.le,
"""
le(tensor, other, out=None) -> Tensor

Computes `tensor <= other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example:
    >>> torch.le(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     1  0
     1  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.lerp,
"""
""")

add_docstr(torch._C.linspace,
"""
""")

add_docstr(torch._C.log,
"""
""")

add_docstr(torch._C.log1p,
"""
""")

add_docstr(torch._C.log_normal,
"""
""")

add_docstr(torch._C.logspace,
"""
""")

add_docstr(torch._C.lt,
"""
lt(tensor, other, out=None) -> Tensor

Computes `tensor < other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example:
    >>> torch.lt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  0
     1  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.masked_select,
"""
""")

add_docstr(torch._C.max,
"""
""")

add_docstr(torch._C.mean,
"""
""")

add_docstr(torch._C.median,
"""
""")

add_docstr(torch._C.min,
"""
""")

add_docstr(torch._C.mm,
"""
""")

add_docstr(torch._C.mod,
"""
""")

add_docstr(torch._C.mode,
"""
""")

add_docstr(torch._C.mul,
"""
""")

add_docstr(torch._C.multinomial,
"""
""")

add_docstr(torch._C.mv,
"""
""")

add_docstr(torch._C.ne,
"""
ne(tensor, other, out=None) -> Tensor

Computes `tensor != other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    tensor (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example:
    >>> torch.ne(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  1
     1  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.neg,
"""
""")

add_docstr(torch._C.nonzero,
"""
""")

add_docstr(torch._C.norm,
"""
""")

add_docstr(torch._C.normal,
"""
""")

add_docstr(torch._C.numel,
"""
""")

add_docstr(torch._C.ones,
"""
""")

add_docstr(torch._C.orgqr,
"""
""")

add_docstr(torch._C.ormqr,
"""
""")

add_docstr(torch._C.potrf,
"""
""")

add_docstr(torch._C.potri,
"""
""")

add_docstr(torch._C.potrs,
"""
""")

add_docstr(torch._C.pow,
"""
""")

add_docstr(torch._C.prod,
"""
""")

add_docstr(torch._C.pstrf,
"""
""")

add_docstr(torch._C.qr,
"""
""")

add_docstr(torch._C.rand,
"""
""")

add_docstr(torch._C.randn,
"""
""")

add_docstr(torch._C.random,
"""
""")

add_docstr(torch._C.randperm,
"""
""")

add_docstr(torch._C.range,
"""
""")

add_docstr(torch._C.remainder,
"""
remainder(tensor, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The divisor and dividend may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor.

Args:
    tensor (Tensor): The dividend
    divisor (Tensor or float): The divisor. This may be either a number or a
                               tensor of the same shape as the dividend.
    out (Tensor, optional): Output tensor

Example:
    >>> torch.remainder(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    torch.FloatTensor([1, 0, 1, 1, 0, 1])
    >>> torch.remainder(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    torch.FloatTensor([1.0, 0.5, 0.0, 1.0, 0.5])

    .. seealso::

        :func:`torch.fmod`, which computes the element-wise remainder of
        division equivalently to the C library function ``fmod()``
""")

add_docstr(torch._C.renorm,
"""
""")

add_docstr(torch._C.reshape,
"""
""")

add_docstr(torch._C.round,
"""
""")

add_docstr(torch._C.rsqrt,
"""
""")

add_docstr(torch._C.scatter,
"""
""")

add_docstr(torch._C.set_num_threads,
"""
set_num_threads(int)

Sets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.sigmoid,
"""
""")

add_docstr(torch._C.sign,
"""
""")

add_docstr(torch._C.sin,
"""
""")

add_docstr(torch._C.sinh,
"""
""")

add_docstr(torch._C.sort,
"""
""")

add_docstr(torch._C.sqrt,
"""
""")

add_docstr(torch._C.squeeze,
"""
""")

add_docstr(torch._C.std,
"""
""")

add_docstr(torch._C.sum,
"""
""")

add_docstr(torch._C.svd,
"""
""")

add_docstr(torch._C.symeig,
"""
""")

add_docstr(torch._C.t,
"""
""")

add_docstr(torch._C.tan,
"""
""")

add_docstr(torch._C.tanh,
"""
""")

add_docstr(torch._C.topk,
"""
""")

add_docstr(torch._C.trace,
"""
""")

add_docstr(torch._C.transpose,
"""
""")

add_docstr(torch._C.tril,
"""
""")

add_docstr(torch._C.triu,
"""
""")

add_docstr(torch._C.trtrs,
"""
""")

add_docstr(torch._C.trunc,
"""
""")

add_docstr(torch._C.unfold,
"""
""")

add_docstr(torch._C.uniform,
"""
""")

add_docstr(torch._C.var,
"""
""")

add_docstr(torch._C.zero,
"""
""")

add_docstr(torch._C.zeros,
"""
""")
