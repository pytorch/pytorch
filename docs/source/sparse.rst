.. currentmodule:: torch.sparse

.. _sparse-docs:

torch.sparse
============

.. warning::

    This API is currently experimental and may change in the near future.

Torch supports sparse tensors in COO(rdinate) format, which can
efficiently store and process tensors for which the majority of elements
are zeros.

A sparse tensor is represented as a pair of dense tensors: a tensor
of values and a 2D tensor of indices.  A sparse tensor can be constructed
by providing these two tensors, as well as the size of the sparse tensor
(which cannot be inferred from these tensors!)  Suppose we want to define
a sparse tensor with the entry 3 at location (0, 2), entry 4 at
location (1, 0), and entry 5 at location (1, 2).  We would then write:

    >>> i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
    >>> v = torch.tensor([3., 4., 5.])
    >>> torch.sparse_coo_tensor(i, v, size=[2, 3]).to_dense()
    tensor([[0., 0., 3.],
            [4., 0., 5.]])

Note that the input to LongTensor is NOT a list of index tuples.  If you want
to write your indices this way, you should transpose before passing them to
the sparse constructor:

    >>> i = torch.tensor([[0, 2], [1, 0], [1, 2]])
    >>> v = torch.tensor([ 3.,     4.,     5.   ])
    >>> torch.sparse_coo_tensor(i.t(), v, size=[2, 3]).to_dense()
    tensor([[0., 0., 3.],
            [4., 0., 5.]])

You can also construct hybrid sparse tensors, where only the first n
dimensions are sparse, and the rest of the dimensions are dense.

    >>> i = torch.tensor([[2, 4]])
    >>> v = torch.tensor([[1., 3.], [5., 7.]])
    >>> torch.sparse_coo_tensor(i, v).to_dense()
    tensor([[0., 0.],
            [0., 0.],
            [1., 3.],
            [0., 0.],
            [5., 7.]])

An empty sparse tensor can be constructed by specifying its size:

    >>> torch.sparse_coo_tensor(size=[2,3])
    tensor(indices=tensor([], size=(2, 0)),
           values=tensor([], size=(0,)),
           size=(2, 3), nnz=0, layout=torch.sparse_coo)

SparseTensor has the following invariants:

    1. sparse_dim + dense_dim = len(SparseTensor.shape)
    2. SparseTensor._indices().shape = (sparse_dim, nnz)
    3. SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])

Since SparseTensor._indices() is always a 2D tensor, the smallest sparse_dim = 1.
Therefore, representation of a SparseTensor of sparse_dim = 0 is simply a dense tensor.

.. note::

    Our sparse tensor format permits *uncoalesced* sparse tensors, where
    there may be duplicate coordinates in the indices; in this case,
    the interpretation is that the value at that index is the sum of all
    duplicate value entries. Uncoalesced tensors permit us to implement
    certain operators more efficiently.

    For the most part, you shouldn't have to care whether or not a
    sparse tensor is coalesced or not, as most operations will work
    identically given a coalesced or uncoalesced sparse tensor.
    One can call `is_coalesced()` to check whether or not a SparseTensor
    is coalesced. Specifically, the following invariants hold for
    a coalesced SparseTensor:

    1. indices are unique
    2. indices are sorted

    However, there are two cases in which you may need to care.

    First, if you repeatedly perform an operation that can produce
    duplicate entries (e.g., :func:`torch.sparse.FloatTensor.add`), you
    should occasionally coalesce your sparse tensors to prevent
    them from growing too large.

    Second, some operators will produce different values depending on
    whether or not they are coalesced or not (e.g.,
    :func:`torch.sparse.FloatTensor._values` and
    :func:`torch.sparse.FloatTensor._indices`, as well as
    :func:`torch.Tensor.sparse_mask`). These operators are
    prefixed by an underscore to indicate that they reveal internal
    implementation details and should be used with care, since code
    that works with coalesced sparse tensors may not work with
    uncoalesced sparse tensors; generally speaking, it is safest
    to explicitly coalesce before working with these operators.

    For example, suppose that we wanted to implement an operator
    by operating directly on :func:`torch.sparse.FloatTensor._values`.
    Multiplication by a scalar can be implemented in the obvious way,
    as multiplication distributes over addition; however, square root
    cannot be implemented directly, since ``sqrt(a + b) != sqrt(a) +
    sqrt(b)`` (which is what would be computed if you were given an
    uncoalesced tensor.)

.. class:: FloatTensor()

    .. method:: add
    .. method:: add_
    .. method:: clone
    .. method:: coalesce
    .. method:: dim
    .. method:: div
    .. method:: div_
    .. method:: get_device
    .. method:: hspmm
    .. method:: is_coalesced
    .. method:: mm
    .. method:: mul
    .. method:: mul_
    .. method:: narrow_copy
    .. method:: resizeAs_
    .. method:: size
    .. method:: spadd
    .. method:: spmm
    .. method:: squeeze

    See :func:`torch.sparse.squeeze` and :func:`torch.squeeze`

    .. method:: sspaddmm
    .. method:: sspmm
    .. method:: sub
    .. method:: sub_
    .. method:: t_
    .. method:: toDense
    .. method:: transpose
    .. method:: transpose_
    .. method:: unsqueeze

    See :func:`torch.sparse.unsqueeze` and :func:`torch.unsqueeze`

    .. method:: zero_
    .. method:: _indices
    .. method:: _values
    .. method:: _nnz

Functions
----------------------------------

.. autofunction:: torch.sparse.addmm
.. autofunction:: torch.sparse.mm
.. autofunction:: torch.sparse.squeeze
.. autofunction:: torch.sparse.sum
.. autofunction:: torch.sparse.unsqueeze
