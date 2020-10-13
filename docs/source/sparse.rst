.. currentmodule:: torch

.. _sparse-docs:

torch.sparse
============

Introduction
++++++++++++

PyTorch provides :class:`torch.Tensor` to represent a
multi-dimensional array containing elements of a single data type. By
default, array elements are stored contiguously in memory leading to
highly-efficient implementations of various array processing
algorithms because the memory locations of array elements can be
efficiently computed. However, there exists an important class of
multi-dimensional arrays, so-called sparse arrays, where the
contiguous storage of array elements in memory turns out to be
suboptimal. Namely, sparse arrays have a property of having a vast
portion of elements being equal to zero which means that a lot of
memory as well as processor resources can be saved if only the
non-zero elements of such arrays are stored or processed. However,
depending on a particular stucture of the locations of non-zero
elements in an array, different sparse storage formats have been
developed (see [Wikipedia:Sparse
Matrix](https://en.wikipedia.org/wiki/Sparse_matrix) for an overview)
that require storing the indicies of non-zero elements leading to
additional memory usage.

.. note::

   When talking about storing only non-zero elements of a sparse
   array, in practice, the usage of adjective "non-zero" is not
   strict: one is allowed to store zero values in the sparse array
   data structure. Hence, in the following, we use "specified
   elements" instead of "non-zero elements" when talking about storing
   elements of sparse arrays in some sparse storage format. In
   addition, since the unspecified elements are assumed to be the
   same, we use a single "fill value" to represent the values of
   unspecified array elements.

.. warning::

   Using a sparse storage format for storing sparse arrays can be
   advantageous only when the size and sparsity levels of arrays are
   high. Otherwise, for small-sized or low-sparsity arrays using the
   contiguous memory storage format is likely the most efficient
   approach.

.. warning::

  The PyTorch API of sparse tensors is in beta and may change in the near future.

.. _sparse-coo-docs:

Sparse COO tensors
++++++++++++++++++

Currently, PyTorch implements the so-called Coordinate format, or COO
format, as the default sparse storage format for storing sparse
tensors efficiently.  Conceptually, when using the COO format, the
specified elements of sparse tensors are stored as tuples of element
indices and the corresponding values. In PyTorch implementation of the
COO format:

  - the element indices are collected in ``indices`` tensor of size
    ``(ndim, nse)`` and with element type ``torch.int64``,

  - the corresponding values are collected in ``values`` tensor of
    size ``(nse,)`` and arbitrary integer or floating point number
    element type,

where ``ndim`` is the dimensionality of the tensor and ``nse`` is the
number of specified elements.

.. note::

   The memory consumption of a sparse COO tensor is at least ``(ndim *
   8 + <size of element type in bytes>) * nse`` bytes (plus a constant
   overhead from storing the shape and type information of the
   tensor).

   The memory consumption of a strided tensor is at least
   ``product(<tensor shape>) * <size of element type in bytes>``.

   For instance, the memory consumption of a 10 000 x 10 000 tensor with
   100 000 non-zero 32-bit floating point numbers is at least ``(2 *
   8 + 4) * 100 000 = 2 000 000`` bytes when using COO format layout and
   ``10 000 * 10 000 * 4 = 400 000 000`` bytes when using the default
   strided tensor layout.

Construction
------------

A sparse COO tensor can be constructed by providing the two tensors of
indices and values, as well as the size of the sparse tensor (when it
cannot be inferred from the indices and values tensors) to a function
:func:`torch.sparse_coo_tensor`. Suppose we want to define a sparse
tensor with the entry 3 at location (0, 2), entry 4 at location (1,
0), and entry 5 at location (1, 2).  Unspecified elements are assumed
to have the same value, fill value, which is zero by default. We would
then write:

    >>> i = [[0, 1, 1],
             [2, 0, 2]]
    >>> v =  [3, 4, 5]
    >>> s = torch.sparse_coo_tensor(i, v, (2, 3))
    >>> s
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3, 4, 5]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> s.to_dense()
    tensor([[0, 0, 3],
            [4, 0, 5]])

Note that the input ``i`` is NOT a list of index tuples.  If you want
to write your indices this way, you should transpose before passing them to
the sparse constructor:

    >>> i = [[0, 2], [1, 0], [1, 2]]
    >>> v =  [3,      4,      5    ]
    >>> s = torch.sparse_coo_tensor(list(zip(*i)), v, (2, 3))
    >>> # Or another equivalent formulation to get s
    >>> s = torch.sparse_coo_tensor(torch.tensor(i).t(), v, (2, 3))
    >>> torch.sparse_coo_tensor(i.t(), v, torch.Size([2,3])).to_dense()
    tensor([[0, 0, 3],
            [4, 0, 5]])

An empty sparse COO tensor can be constructed by specifying its size:

    >>> torch.sparse_coo_tensor(size=(2, 3))
    tensor(indices=tensor([], size=(2, 0)),
           values=tensor([], size=(0,)),
           size=(2, 3), nnz=0, layout=torch.sparse_coo)

Hybrid sparse COO tensors
-------------------------

Pytorch implements an extension of sparse tensors of scalar values to
sparse tensors of tensor values. Such tensors are called hybrid
tensors.

PyTorch hybrid sparse COO tensor extends the sparse COO tensor by
allowing the ``values`` tensor to be multi-dimensional so that we have:

- the element indices are collected in ``indices`` tensor of size
  ``(sparse_dims, nse)`` and with element type ``torch.int64``,
- the corresponding values are collected in ``values`` tensor of size
  ``(nse, dense_dims)`` and arbitrary integer or floating point number
  element type.

We use (M + K)-dimensional tensor to denote a N-dimensional hybrid
sparse tensor such that M + K == N holds.

Suppose we want to create a (2 + 1)-dimensional tensor with the entry
[3, 4] at location (0, 2), entry [5, 6] at location (1, 0), and entry
[7, 8] at location (1, 2). We would write

    >>> i = [[0, 1, 1],
             [2, 0, 2]]
    >>> v =  [[3, 4], [5, 6], [7, 8]]
    >>> s = torch.sparse_coo_tensor(i, v, (2, 3, 2))
    >>> s
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([[3, 4],
                          [5, 6],
                          [7, 8]]),
           size=(2, 3, 2), nnz=3, layout=torch.sparse_coo)

    >>> s.to_dense()
    tensor([[[0, 0],
             [0, 0],
             [3, 4]],

            [[5, 6],
             [0, 0],
             [7, 8]]])

In general, if ``s`` is a sparse COO tensor and set
``M=s.sparse_dim()``, ``K=s.dense_dim()`` then the following
invariants hold:

- ``M + K == len(s.shape) == s.ndim`` - dimensionality of a tensor is a sum of the number of sparse and dense dimensions,
- ``s.indices().shape == (M, nse)`` - sparse indices are stored explicitly,
- ``s.values().shape == (nse,) + s.shape[M : M + K]`` - values are K-dimensional tensors,
- ``s.values().layout == torch.strided`` - values are stored as strided tensors.

.. note::

   Dense dimensions always follow sparse dimensions, that is, mixing
   of dense and sparse dimensions is not supported.


Uncoalesced sparse COO tensors
------------------------------

PyTorch sparse COO tensor format permits *uncoalesced* sparse tensors,
where there may be duplicate coordinates in the indices; in this case,
the interpretation is that the value at that index is the sum of all
duplicate value entries. For example, one can specify multiple values,
``3`` and ``4``, for the same index ``1``, that leads to an
uncoalesced tensor:

    >>> i = [[1, 1]]
    >>> v =  [3, 4]
    >>> s=torch.sparse_coo_tensor(i, v, (3,))
    >>> s
    tensor(indices=tensor([[1, 1]]),
           values=tensor(  [3, 4]),
           size=(3,), nnz=2, layout=torch.sparse_coo)

while coalescing process will accumulate the multi-valued elements
into a single value using summation:

    >>> s.coalesce()
    tensor(indices=tensor([[1]]),
           values=tensor([7]),
           size=(3,), nnz=1, layout=torch.sparse_coo)

In general, the output of :meth:`torch.Tensor.coalesce` method is a
sparse tensor with the following properties:

- the indices of specified tensor elements are unique,
- the indices are sorted in lexicographical order,
- :meth:`torch.Tensor.is_coalesced()` returns ``True``.

.. note::

  For the most part, you shouldn't have to care whether or not a
  sparse tensor is coalesced or not, as most operations will work
  identically given a coalesced or uncoalesced sparse tensor.

  However, some operations can be implemented more efficiently on
  uncoalesced tensors, and some on coalesced tensors.

  For instance, addition of sparse COO tensors is implemented by
  simply concatenating the indices and values tensors:

    >>> a = torch.sparse_coo_tensor([[1, 1]], [5, 6], (2,))
    >>> b = torch.sparse_coo_tensor([[0, 0]], [7, 8], (2,))
    >>> a + b
    tensor(indices=tensor([[0, 0, 1, 1]]),
           values=tensor([7, 8, 5, 6]),
           size=(2,), nnz=4, layout=torch.sparse_coo)

  If you repeatedly perform an operation that can produce duplicate
  entries (e.g., :func:`torch.Tensor.add`), you should occasionally
  coalesce your sparse tensors to prevent them from growing too large.

  On the other hand, the lexicographical ordering of indices can be
  advantageous for implementing algorithms that involve many element
  selection operations, such as slicing or matrix products.


Working with sparse COO tensors
-------------------------------

Let's consider the following example:

    >>> i = [[0, 1, 1],
             [2, 0, 2]]
    >>> v =  [[3, 4], [5, 6], [7, 8]]
    >>> s = torch.sparse_coo_tensor(i, v, (2, 3, 2))

As mentioned above, a sparse COO tensor is a :class:`torch.Tensor`
instance and to distinguish it from the `Tensor` instances that use
some other layout, on can use :attr:`torch.Tensor.is_sparse` or
:attr:`torch.Tensor.layout` properties:

    >>> isinstance(s, torch.Tensor)
    True
    >>> s.is_sparse
    True
    >>> s.layout == torch.sparse_coo
    True

When working with hybrid sparse COO tensors then the number of sparse
and dense dimensions can be acquired using methods
:meth:`torch.Tensor.sparse_dim` and :meth:`torch.Tensor.dense_dim`,
respectively. For instance:

    >>> s.sparse_dim(), s.dense_dim()
    (2, 1)


If ``s`` is a sparse COO tensor then its COO format data can be
acquired using methods :meth:`torch.Tensor.indices()` and
:meth:`torch.Tensor.values()`.

.. note::

   Currently, one can acquire the COO format data only when the tensor
   instance is coalesced:

     >>> s.indices()
     RuntimeError: Cannot get indices on an uncoalesced tensor, please call .coalesce() first

   For acquiring the COO format data of uncoalesced tensor, the
   following methods are provided: :func:`torch.Tensor._values()` and
   :func:`torch.Tensor._indices()`.

     >>> s._indices()
     tensor([[0, 1, 1],
             [2, 0, 2]])

   .. See https://github.com/pytorch/pytorch/pull/45695 for a new API.

Constructing a new sparse COO tensor is not considered coalesced:

    >>> s.is_coalesced()
    False

but one can construct a coalesced copy by using the
:meth:`torch.Tensor.coalesce` method:

    >>> s2 = s.coalesce()
    >>> s2.indices()
    tensor([[0, 1, 1],
           [2, 0, 2]])

When working with uncoalesced sparse COO tensors, one must take into
an account the additive nature of uncoalesced data: the values of the
same indices are the terms of a sum that evaluation gives the value of
the corresponding tensor element. For example, the scalar
multiplication on an uncoalesced sparse tensor could be implemented by
multiplying all the uncoalesced values with the scalar because ``c *
(a + b) == c * a + c * b`` holds. However, any nonlinear operation,
say, a square root, cannot be implemented on applying the operation to
uncoalesced data because ``sqrt(a + b) == sqrt(a) + sqrt(b)`` does not
hold, for instance.

Slicing (with positive step) of a sparse COO tensor is supported only
for dense dimensions. Indexing is supported for both sparse and dense
dimensions:

    >>> s[1]
    tensor(indices=tensor([[0, 2]]),
           values=tensor([[5, 6],
                          [7, 8]]),
           size=(3, 2), nnz=2, layout=torch.sparse_coo)
    >>> s[1, 0, 1]
    tensor(6)
    >>> s[1, 0, 1:]
    tensor([6])

Fill value
----------

.. See https://github.com/Quansight-Labs/rfcs/tree/pearu/rfc-fill-value/RFC-0004-sparse-fill-value for a new API

In PyTorch, the fill value is not specified explicitly but is
implicitly interpreted by sparse algorithms. The implicit
interpretation may vary in-between algorithms of different fields and
applications. The most commonly assumed fill value is zero and it is
assumed for all tools that are available via :mod:`torch`
namespace. When some sparse tensor functionality assumes a non-zero or
indefinite fill value, the corresponding tools are provided via the
:mod:`torch.sparse` namespace. Please see the documentation strings of
the corresponding functions for a particular definition of the fill
value.

Supported Linear Algebra operations
+++++++++++++++++++++++++++++++++++

The following table summarizes supported Linear Algebra operations on
sparse matrices with different storage layouts. Here ``T[layout]``
denotes a tensor with a give layout. Similarly, ``M[layout]`` denotes
a matrix (2-D PyTorch tensor), and ``V[layout]`` denotes a vector (1-D
PyTorch tensor). In addition, ``f`` denotes a scalar (float or 0-D PyTorch
tensor), ``*`` is element-wise multiplication, ``@`` is matrix
multiplication.

.. csv-table::
   :header: "PyTorch operation", "Sparse grad?", "Layout signature"
   :widths: 20, 5, 60
   :delim: ;

   :func:`torch.mv`;no; ``M[sparse_coo] @ V[strided] -> V[strided]``
   :func:`torch.matmul`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.mm`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.sparse.mm`; yes; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.smm`; no; ``M[sparse_coo] @ M[strided] -> M[sparse_coo]``
   :func:`torch.hspmm`; no; ``M[sparse_coo] @ M[strided] -> M[hybrid sparse_coo]``
   :func:`torch.bmm`; no; ``T[sparse_coo] @ T[strided] -> T[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.sparse.addmm`; yes; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.sspaddmm`; no; ``f * M[sparse_coo] + f * (M[sparse_coo] @ M[strided]) -> M[sparse_coo]``
   :func:`torch.lobpcg`; no; ``GENEIG(M[sparse_coo]) -> M[strided], M[strided]``
   :func:`torch.pca_lowrank`; yes; ``PCA(M[sparse_coo]) -> M[strided], M[strided], M[strided]``
   :func:`torch.svd_lowrank`; yes; ``SVD(M[sparse_coo]) -> M[strided], M[strided], M[strided]``

where "Sparse grad?" column indicates if the PyTorch operation supports
backward with respect to sparse matrix argument. All PyTorch operations,
except :func:`torch.smm`, support backward with respect to strided
matrix arguments.

.. note::

   Currently, PyTorch does not support matrix multiplication with the
   layout signature ``M[strided] @ M[sparse_coo]``. However,
   applications can still compute this using the matrix relation ``D @
   S == (S.t() @ D.t()).t()``.

.. class:: Tensor()
   :noindex:

   The following methods are specific to :ref:`sparse tensors <sparse-docs>`:

    .. autoattribute:: is_sparse
    .. automethod:: dense_dim
    .. automethod:: sparse_dim
    .. automethod:: sparse_mask
    .. automethod:: sparse_resize_
    .. automethod:: sparse_resize_and_clear_
    .. automethod:: to_dense
    .. automethod:: to_sparse

   The following methods are specific to :ref:`sparse COO tensors
   <sparse-coo-docs>`:

    .. automethod:: coalesce
    .. automethod:: is_coalesced
    .. automethod:: indices
    .. automethod:: values

The following :class:`torch.Tensor` methods support :ref:`sparse COO
tensors <sparse-coo-docs>`:

:meth:`~torch.Tensor.add`
:meth:`~torch.Tensor.add_`
:meth:`~torch.Tensor.addmm`
:meth:`~torch.Tensor.addmm_`
:meth:`~torch.Tensor.any`
:meth:`~torch.Tensor.asin`
:meth:`~torch.Tensor.asin_`
:meth:`~torch.Tensor.arcsin`
:meth:`~torch.Tensor.arcsin_`
:meth:`~torch.Tensor.bmm`
:meth:`~torch.Tensor.clone`
:meth:`~torch.Tensor.deg2rad`
:meth:`~torch.Tensor.deg2rad_`
:meth:`~torch.Tensor.detach`
:meth:`~torch.Tensor.detach_`
:meth:`~torch.Tensor.dim`
:meth:`~torch.Tensor.div`
:meth:`~torch.Tensor.div_`
:meth:`~torch.Tensor.floor_divide`
:meth:`~torch.Tensor.floor_divide_`
:meth:`~torch.Tensor.get_device`
:meth:`~torch.Tensor.index_select`
:meth:`~torch.Tensor.isnan`
:meth:`~torch.Tensor.log1p`
:meth:`~torch.Tensor.log1p_`
:meth:`~torch.Tensor.mm`
:meth:`~torch.Tensor.mul`
:meth:`~torch.Tensor.mul_`
:meth:`~torch.Tensor.mv`
:meth:`~torch.Tensor.narrow_copy`
:meth:`~torch.Tensor.neg`
:meth:`~torch.Tensor.neg_`
:meth:`~torch.Tensor.negative`
:meth:`~torch.Tensor.negative_`
:meth:`~torch.Tensor.numel`
:meth:`~torch.Tensor.rad2deg`
:meth:`~torch.Tensor.rad2deg_`
:meth:`~torch.Tensor.resize_as_`
:meth:`~torch.Tensor.size`
:meth:`~torch.Tensor.pow`
:meth:`~torch.Tensor.square`
:meth:`~torch.Tensor.smm`
:meth:`~torch.Tensor.sspaddmm`
:meth:`~torch.Tensor.sub`
:meth:`~torch.Tensor.sub_`
:meth:`~torch.Tensor.t`
:meth:`~torch.Tensor.t_`
:meth:`~torch.Tensor.transpose`
:meth:`~torch.Tensor.transpose_`
:meth:`~torch.Tensor.zero_`


Sparse tensor functions
+++++++++++++++++++++++

.. autofunction:: torch.sparse_coo_tensor
.. autofunction:: torch.sparse.sum
.. autofunction:: torch.sparse.addmm
.. autofunction:: torch.sparse.mm
.. autofunction:: torch.sspaddmm
.. autofunction:: torch.hspmm
.. autofunction:: torch.smm
.. autofunction:: torch.sparse.softmax
.. autofunction:: torch.sparse.log_softmax

Other functions
***************

The following :mod:`torch` functions support :ref:`sparse COO tensors <sparse-coo-docs>`:

:func:`~torch.cat`
:func:`~torch.dstack`
:func:`~torch.empty`
:func:`~torch.empty_like`
:func:`~torch.hstack`
:func:`~torch.index_select`
:func:`~torch.is_complex`
:func:`~torch.is_floating_point`
:func:`~torch.is_nonzero`
:func:`~torch.is_same_size`
:func:`~torch.is_signed`
:func:`~torch.is_tensor`
:func:`~torch.lobpcg`
:func:`~torch.mm`
:func:`~torch.native_norm`
:func:`~torch.pca_lowrank`
:func:`~torch.select`
:func:`~torch.stack`
:func:`~torch.svd_lowrank`
:func:`~torch.unsqueeze`
:func:`~torch.vstack`
:func:`~torch.zeros`
:func:`~torch.zeros_like`
