.. currentmodule:: torch

.. _sparse-docs:

torch.sparse
============

Torch supports sparse tensors which provide memory-efficient storage
and processing algorithms of tensors for which the majority of
elements have the same value (fill value). This allows saving storage
and processing resources when the size and sparsity levels of tensors
are high.  Otherwise, for small-sized or low-sparsity tensors using
strided tensors is likely the most efficient approach.

.. _sparse-fill-value-docs:

Fill value
++++++++++

In Torch, the fill value is not specified explicitly but is implicitly
interpreted by sparse algorithms. The implicit interpretation may vary
in-between algorithms of different fields and applications. The most
commonly assumed fill value is zero and it is assumed for all tools
that are available via :mod:`torch` namespace. When some sparse tensor
functionality assumes a non-zero or indefinite fill value, the
corresponding tools are provided via :mod:`torch.sparse`
namespace. Please see the documentation strings of the corresponding
functions for a particular definition of the fill value.

.. _sparse-hybrid-docs:

Hybrid tensors
++++++++++++++

Torch supports hybrid sparse tensors with dimensions split into sparse
dimensions and dense dimensions. Dense dimensions always follow sparse
dimensions, that is, mixing of dense and sparse dimensions is not
supported. One can interpret hybrid tensors as sparse tensors with
elements being (contiguous) strided tensors. We use (M +
K)-dimensional tensor to denote a N-dimensional hybrid sparse tensor
where M and K are the number of sparse and dense dimensions,
respectively, such that M + K == N holds.

See :ref:`Hybrid COO tensors <sparse-hybrid-coo-docs>`.

Sparse tensor formats
+++++++++++++++++++++

Torch supported sparse storage formats are:

- COO(rdinate) format where tensor elements are stored using the lists
  of indices and the corresponding values. Unspecified elements have
  values equal to the fill value. To construct a sparse tensor in COO
  format, use :func:`torch.sparse_coo_tensor` function.

.. _sparse-coo-docs:

Sparse COO tensors
******************

The sparse COO tensor is represented as a pair of dense tensors: a 2D
tensor of indices and a tensor of values. A sparse COO tensor can be
constructed by providing these two tensors, as well as the size of the
sparse tensor (which cannot be inferred from the indices and values
tensors!). Suppose we want to define a sparse tensor with the entry 3
at location (0, 2), entry 4 at location (1, 0), and entry 5 at
location (1, 2).  We would then write:

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

Sparse COO tensors support indexing operations (except slicing in
sparse dimensions):

    >>> s[1]
    tensor(indices=tensor([[0, 2]]),
           values=tensor(  [4, 5]),
           size=(3,), nnz=2, layout=torch.sparse_coo)
    >>> s[1, 0]
    tensor(4)

An empty sparse tensor can be constructed by specifying its size:

    >>> torch.sparse_coo_tensor(size=(2, 3))
    tensor(indices=tensor([], size=(2, 0)),
           values=tensor([], size=(0,)),
           size=(2, 3), nnz=0, layout=torch.sparse_coo)

.. _sparse-hybrid-coo-docs:

Hybrid COO tensors
------------------

You can also construct hybrid tensors, where only the first dimensions
are sparse, and the rest of the dimensions are dense. For instance,

    >>> i = [[2, 4]]
    >>> v = [[1, 3], [5, 7]]
    >>> s = torch.sparse_coo_tensor(i, v, (5, 2))
    >>> s.to_dense()
    tensor([[0, 0],
            [0, 0],
            [1, 3],
            [0, 0],
            [5, 7]])

where ``s`` is a (1+1)-dimensional :ref:`hybrid tensor
<sparse-hybrid-docs>`:

    >>> s[0]
    tensor([0, 0])
    >>> s[2]
    tensor([1, 3])

Notice that slicing in dense dimensions is supported (except with
negative step):

    >>> s[2, 1:]
    tensor([3])

In general, a (M + K)-dimensional coalesced sparse COO tensor ``s``
with the number of specified elements ``nse`` has the following
invariants:

    >>> s = s.coalesce()
    >>> M, K = s.sparse_dim(), s.dense_dim()
    >>> assert M + K == len(s.shape) == s.ndim
    >>> assert s.indices().shape == (M, nse)
    >>> assert s.values().shape == (nse,) + s.shape[M:]

Since ``s.indices()`` is always a 2D tensor, the smallest sparse
dimension M = 1.  Therefore, representation of a sparse tensor of M =
0 is simply a dense tensor.

.. _sparse-uncoalesed-coo-docs:

Uncoalesced sparse COO tensors
------------------------------

Torch sparse COO tensor format permits *uncoalesced* sparse tensors,
where there may be duplicate coordinates in the indices; in this case,
the interpretation is that the value at that index is the sum of all
duplicate value entries:

    >>> i = [[1, 1]]
    >>> v =  [3, 4]
    >>> s=torch.sparse_coo_tensor(i, v, (3,))
    >>> s
    tensor(indices=tensor([[1, 1]]),
           values=tensor(  [3, 4]),
           size=(3,), nnz=2, layout=torch.sparse_coo)
    >>> s.to_dense()
    tensor([0, 7, 0])

Uncoalesced tensors permit us to implement certain operators more
efficiently.  For the most part, you shouldn't have to care whether or
not a sparse tensor is coalesced or not, as most operations will work
identically given a coalesced or uncoalesced sparse tensor.  However,
there are two cases in which you may need to care.

First, if you repeatedly perform an operation that can produce
duplicate entries (e.g., :func:`torch.Tensor.add`), you should
occasionally coalesce your sparse tensors to prevent them from growing
too large:

    >>> s.add_(torch.sparse_coo_tensor([[2, 2]], [5, 6]))
    tensor(indices=tensor([[1, 1, 2, 2]]),
           values=tensor(  [3, 4, 5, 6]),
           size=(3,), nnz=4, layout=torch.sparse_coo)
    >>> s.coalesce()
    tensor(indices=tensor([[1,  2]]),
           values=tensor(  [7, 11]),
           size=(3,), nnz=2, layout=torch.sparse_coo)

.. note::

   A coalesced tensor is a sparse COO tensor with unique indices in
   lexicographical ordering.

Second, some operators can be implemented on uncoalesced sparse
tensors. The corresponding implementations would use
:func:`torch.Tensor._values()` and :func:`torch.Tensor._indices()` to
retrieve the uncoalesced data of a tensor. One must take into an
account the additive nature of uncoalesced data: the values of the
same indices are the terms of a sum that evaluation gives the value of
the corresponding tensor element. For example, the scalar
multiplication on an uncoalesced sparse tensor could be implemented by
multiplying all the uncoalesced values with the scalar because ``c *
(a + b) == c * a + c * b`` holds. However, any nonlinear operation,
say, a square root, cannot be implemented on applying the operation to
uncoalesced data because ``sqrt(a + b) == sqrt(a) + sqrt(b)`` does not
hold, for instance.

Linear Algebra operations
+++++++++++++++++++++++++

The following table summarizes supported Linear Algebra operations on
sparse matrices with different storage layouts. Here ``T[layout]``
denotes a tensor with a give layout. Similarly, ``M[layout]`` denotes
a matrix (2-D Torch tensor), and ``V[layout]`` denotes a vector (1-D
Torch tensor). In addition, ``f`` denotes a scalar (float or 0-D Torch
tensor), ``*`` is element-wise multiplication, ``@`` is matrix
multiplication.

.. csv-table::
   :header: "Torch operation", "Sparse grad?", "Layout signature"
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

where "Sparse grad?" column indicates if the Torch operation supports
backward with respect to sparse matrix argument. All Torch operations,
except :func:`torch.smm`, support backward with respect to strided
matrix arguments.

.. note::

   :func:`torch.spmm` and :func:`torch.dsmm` are aliases to :func:`torch.mm`.

.. note::

   Currently, Torch does not support matrix multiplication with the
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
.. autofunction:: torch.dsmm
.. autofunction:: torch.spmm
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
