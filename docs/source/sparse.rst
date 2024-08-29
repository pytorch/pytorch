.. automodule:: torch.sparse

.. currentmodule:: torch

.. _sparse-docs:

torch.sparse
============

.. warning::

  The PyTorch API of sparse tensors is in beta and may change in the near future.
  We highly welcome feature requests, bug reports and general suggestions as GitHub issues.

Why and when to use sparsity
++++++++++++++++++++++++++++

By default, PyTorch stores :class:`torch.Tensor` elements contiguously in
physical memory. This leads to efficient implementations of various array
processing algorithms that require fast access to elements.

Now, some users might decide to represent data such as graph adjacency
matrices, pruned weights or points clouds by Tensors whose *elements are
mostly zero valued*. We recognize these are important applications and aim
to provide performance optimizations for these use cases via sparse storage formats.

Various sparse storage formats such as COO, CSR/CSC, semi-structured, LIL, etc. have been
developed over the years. While they differ in exact layouts, they all
compress data through efficient representation of zero valued elements.
We call the uncompressed values *specified* in contrast to *unspecified*,
compressed elements.

By compressing repeat zeros sparse storage formats aim to save memory
and computational resources on various CPUs and GPUs. Especially for high
degrees of sparsity or highly structured sparsity this can have significant
performance implications. As such sparse storage formats can be seen as a
performance optimization.

Like many other performance optimization sparse storage formats are not
always advantageous. When trying sparse formats for your use case
you might find your execution time to increase rather than decrease.

Please feel encouraged to open a GitHub issue if you analytically
expected to see a stark increase in performance but measured a
degradation instead. This helps us prioritize the implementation
of efficient kernels and wider performance optimizations.

We make it easy to try different sparsity layouts, and convert between them,
without being opinionated on what's best for your particular application.

Functionality overview
++++++++++++++++++++++

We want it to be straightforward to construct a sparse Tensor from a
given dense Tensor by providing conversion routines for each layout.

In the next example we convert a 2D Tensor with default dense (strided)
layout to a 2D Tensor backed by the COO memory layout. Only values and
indices of non-zero elements are stored in this case.

    >>> a = torch.tensor([[0, 2.], [3, 0]])
    >>> a.to_sparse()
    tensor(indices=tensor([[0, 1],
                           [1, 0]]),
           values=tensor([2., 3.]),
           size=(2, 2), nnz=2, layout=torch.sparse_coo)

PyTorch currently supports :ref:`COO<sparse-coo-docs>`, :ref:`CSR<sparse-csr-docs>`,
:ref:`CSC<sparse-csc-docs>`, :ref:`BSR<sparse-bsr-docs>`, and :ref:`BSC<sparse-bsc-docs>`.

We also have a prototype implementation to support :ref: `semi-structured sparsity<sparse-semi-structured-docs>`.
Please see the references for more details.

Note that we provide slight generalizations of these formats.

Batching: Devices such as GPUs require batching for optimal performance and
thus we support batch dimensions.

We currently offer a very simple version of batching where each component of a sparse format
itself is batched. This also requires the same number of specified elements per batch entry.
In this example we construct a 3D (batched) CSR Tensor from a 3D dense Tensor.

    >>> t = torch.tensor([[[1., 0], [2., 3.]], [[4., 0], [5., 6.]]])
    >>> t.dim()
    3
    >>> t.to_sparse_csr()
    tensor(crow_indices=tensor([[0, 1, 3],
                                [0, 1, 3]]),
           col_indices=tensor([[0, 0, 1],
                               [0, 0, 1]]),
           values=tensor([[1., 2., 3.],
                          [4., 5., 6.]]), size=(2, 2, 2), nnz=3,
           layout=torch.sparse_csr)


Dense dimensions: On the other hand, some data such as Graph embeddings might be
better viewed as sparse collections of vectors instead of scalars.

In this example we create a 3D Hybrid COO Tensor with 2 sparse and 1 dense dimension
from a 3D strided Tensor. If an entire row in the 3D strided Tensor is zero, it is
not stored. If however any of the values in the row are non-zero, they are stored
entirely. This reduces the number of indices since we need one index one per row instead
of one per element. But it also increases the amount of storage for the values. Since
only rows that are *entirely* zero can be emitted and the presence of any non-zero
valued elements cause the entire row to be stored.

    >>> t = torch.tensor([[[0., 0], [1., 2.]], [[0., 0], [3., 4.]]])
    >>> t.to_sparse(sparse_dim=2)
    tensor(indices=tensor([[0, 1],
                           [1, 1]]),
           values=tensor([[1., 2.],
                          [3., 4.]]),
           size=(2, 2, 2), nnz=2, layout=torch.sparse_coo)


Operator overview
+++++++++++++++++

Fundamentally, operations on Tensor with sparse storage formats behave the same as
operations on Tensor with strided (or other) storage formats. The particularities of
storage, that is the physical layout of the data, influences the performance of
an operation but should not influence the semantics.


We are actively increasing operator coverage for sparse tensors. Users should not
expect support same level of support as for dense Tensors yet.
See our :ref:`operator<sparse-ops-docs>` documentation for a list.

    >>> b = torch.tensor([[0, 0, 1, 2, 3, 0], [4, 5, 0, 6, 0, 0]])
    >>> b_s = b.to_sparse_csr()
    >>> b_s.cos()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: unsupported tensor layout: SparseCsr
    >>> b_s.sin()
    tensor(crow_indices=tensor([0, 3, 6]),
           col_indices=tensor([2, 3, 4, 0, 1, 3]),
           values=tensor([ 0.8415,  0.9093,  0.1411, -0.7568, -0.9589, -0.2794]),
           size=(2, 6), nnz=6, layout=torch.sparse_csr)

As shown in the example above, we don't support non-zero preserving unary
operators such as cos. The output of a non-zero preserving unary operation
will not be able to take advantage of sparse storage formats to the same
extent as the input and potentially result in a catastrophic increase in memory.
We instead rely on the user to explicitly convert to a dense Tensor first and
then run the operation.

    >>> b_s.to_dense().cos()
    tensor([[ 1.0000, -0.4161],
            [-0.9900,  1.0000]])

We are aware that some users want to ignore compressed zeros for operations such
as `cos` instead of preserving the exact semantics of the operation. For this we
can point to torch.masked and its MaskedTensor, which is in turn also backed and
powered by sparse storage formats and kernels.

Also note that, for now, the user doesn't have a choice of the output layout. For example,
adding a sparse Tensor to a regular strided Tensor results in a strided Tensor. Some
users might prefer for this to stay a sparse layout, because they know the result will
still be sufficiently sparse.

    >>> a + b.to_sparse()
    tensor([[0., 3.],
            [3., 0.]])

We acknowledge that access to kernels that can efficiently produce different output
layouts can be very useful. A subsequent operation might significantly benefit from
receiving a particular layout. We are working on an API to control the result layout
and recognize it is an important feature to plan a more optimal path of execution for
any given model.

.. _sparse-semi-structured-docs:

Sparse Semi-Structured Tensors
++++++++++++++++++++++++++++++

.. warning::

   Sparse semi-structured tensors are currently a prototype feature and subject to change. Please feel free to open an issue to report a bug or if you have feedback to share.

Semi-Structured sparsity is a sparse data layout that was first introduced in NVIDIA's Ampere architecture. It is also referred to as **fine-grained structured sparsity** or **2:4 structured sparsity**.

This sparse layout stores `n` elements out of every `2n` elements, with `n` being determined by the width of the Tensor's data type (dtype). The most frequently used dtype is float16, where `n=2`, thus the term "2:4 structured sparsity."

Semi-structured sparsity is explained in greater detail in `this NVIDIA blog post <https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt>`_.

In PyTorch, semi-structured sparsity is implemented via a Tensor subclass.
By subclassing, we can override ``__torch_dispatch__`` , allowing us to use faster sparse kernels when performing matrix multiplication.
We can also store the tensor in it's compressed form inside the subclass to reduce memory overhead.

In this compressed form, the sparse tensor is stored by retaining only the *specified* elements and some metadata, which encodes the mask.

.. note::
    The specified elements and metadata mask of a semi-structured sparse tensor are stored together in a single
    flat compressed tensor. They are appended to each other to form a contiguous chunk of memory.

    compressed tensor = [ specified elements of original tensor |   metadata_mask ]

    For an original tensor of size `(r, c)` we expect the first `m * k // 2` elements to be the kept elements
    and the rest of the tensor is metadata.

    In order to make it easier for the user to view the specified elements
    and mask, one can use ``.indices()`` and ``.values()`` to access the mask and specified elements respectively.


    - ``.values()`` returns the specified elements in a tensor of size `(r, c//2)` and with the same dtype as the dense matrix.

    - ``.indices()`` returns the metadata_mask in a tensor of size `(r, c//2 )` and with element type ``torch.int16`` if dtype is torch.float16 or torch.bfloat16, and element type ``torch.int32`` if dtype is torch.int8.


For 2:4 sparse tensors, the metadata overhead is minor - just 2 bits per specified element.

.. note::
  It's important to note that ``torch.float32`` is only supported for 1:2 sparsity. Therefore, it does not follow the same formula as above.

Here, we break down how to calculate the compression ratio ( size dense / size sparse) of a 2:4 sparse tensor.

Let `(r, c) = tensor.shape` and `e = bitwidth(tensor.dtype)`, so `e = 16` for ``torch.float16`` and ``torch.bfloat16`` and `e = 8` for ``torch.int8``.

.. math::
  M_{dense} = r \times c \times e \\
  M_{sparse} = M_{specified} + M_{metadata} = r \times \frac{c}{2} \times e + r \times \frac{c}{2} \times 2 = \frac{rce}{2} + rc =rce(\frac{1}{2} +\frac{1}{e})

Using these calculations, we can determine the total memory footprint for both the original dense and the new sparse representation.

This gives us a simple formula for the compression ratio, which is dependent only on the bitwidth of the tensor datatype.

.. math::
  C = \frac{M_{sparse}}{M_{dense}} =  \frac{1}{2} + \frac{1}{e}

By using this formula, we find that the compression ratio is 56.25% for ``torch.float16`` or ``torch.bfloat16``, and 62.5% for ``torch.int8``.

Constructing Sparse Semi-Structured Tensors
-------------------------------------------

You can transform a dense tensor into a sparse semi-structured tensor by simply using the ``torch.to_sparse_semi_structured`` function.

Please also note that we only support CUDA tensors since hardware compatibility for semi-structured sparsity is limited to NVIDIA GPUs.


The following datatypes are supported for semi-structured sparsity. Note that each datatype has its own shape constraints and compression factor.

.. csv-table::
   :header: "PyTorch dtype", "Shape Constraints", "Compression Factor", "Sparsity Pattern"
   :widths: 15, 45, 10, 10
   :delim: ;

   ``torch.float16``; Tensor must be 2D and (r, c) must both be a positive multiple of 64;9/16;2:4
   ``torch.bfloat16``; Tensor must be 2D and (r, c) must both be a positive multiple of 64;9/16;2:4
   ``torch.int8``; Tensor must be 2D and (r, c) must both be a positive multiple of 128;10/16;2:4


To construct a semi-structured sparse tensor, start by creating a regular dense tensor that adheres to a 2:4 (or semi-structured) sparse format.
To do this we  tile a small 1x4 strip to create a 16x16 dense float16 tensor.
Afterwards, we can call ``to_sparse_semi_structured`` function to compress it for accelerated inference.

    >>> from torch.sparse import to_sparse_semi_structured
    >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
    tensor([[0., 0., 1.,  ..., 0., 1., 1.],
            [0., 0., 1.,  ..., 0., 1., 1.],
            [0., 0., 1.,  ..., 0., 1., 1.],
            ...,
            [0., 0., 1.,  ..., 0., 1., 1.],
            [0., 0., 1.,  ..., 0., 1., 1.],
            [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
    >>> A_sparse = to_sparse_semi_structured(A)
    SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
            [1., 1., 1.,  ..., 1., 1., 1.],
            [1., 1., 1.,  ..., 1., 1., 1.],
            ...,
            [1., 1., 1.,  ..., 1., 1., 1.],
            [1., 1., 1.,  ..., 1., 1., 1.],
            [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16), metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
            [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
            [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
            ...,
            [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
            [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
            [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
    dtype=torch.int16))

Sparse Semi-Structured Tensor Operations
----------------------------------------

Currently, the following operations are supported for semi-structured sparse tensors:

- torch.addmm(bias, dense, sparse.t())
- torch.mm(dense, sparse)
- torch.mm(sparse, dense)
- aten.linear.default(dense, sparse, bias)
- aten.t.default(sparse)
- aten.t.detach(sparse)

To use these ops, simply pass the output of ``to_sparse_semi_structured(tensor)``  instead of using ``tensor`` once your tensor has 0s in a semi-structured sparse format, like this:

    >>> a = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
    >>> b = torch.rand(64, 64).half().cuda()
    >>> c = torch.mm(a, b)
    >>> a_sparse = to_sparse_semi_structured(a)
    >>> torch.allclose(c, torch.mm(a_sparse, b))
    True

Accelerating nn.Linear with semi-structured sparsity
----------------------------------------------------
You can accelerate the linear layers in your model if the weights are already semi-structured sparse with just a few lines of code:

    >>> input = torch.rand(64, 64).half().cuda()
    >>> mask = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).cuda().bool()
    >>> linear = nn.Linear(64, 64).half().cuda()
    >>> linear.weight = nn.Parameter(to_sparse_semi_structured(linear.weight.masked_fill(~mask, 0)))


.. _sparse-coo-docs:

Sparse COO tensors
++++++++++++++++++

PyTorch implements the so-called Coordinate format, or COO
format, as one of the storage formats for implementing sparse
tensors.  In COO format, the specified elements are stored as tuples
of element indices and the corresponding values. In particular,

  - the indices of specified elements are collected in ``indices``
    tensor of size ``(ndim, nse)`` and with element type
    ``torch.int64``,

  - the corresponding values are collected in ``values`` tensor of
    size ``(nse,)`` and with an arbitrary integer or floating point
    number element type,

where ``ndim`` is the dimensionality of the tensor and ``nse`` is the
number of specified elements.

.. note::

   The memory consumption of a sparse COO tensor is at least ``(ndim *
   8 + <size of element type in bytes>) * nse`` bytes (plus a constant
   overhead from storing other tensor data).

   The memory consumption of a strided tensor is at least
   ``product(<tensor shape>) * <size of element type in bytes>``.

   For example, the memory consumption of a 10 000 x 10 000 tensor
   with 100 000 non-zero 32-bit floating point numbers is at least
   ``(2 * 8 + 4) * 100 000 = 2 000 000`` bytes when using COO tensor
   layout and ``10 000 * 10 000 * 4 = 400 000 000`` bytes when using
   the default strided tensor layout. Notice the 200 fold memory
   saving from using the COO storage format.

Construction
------------

A sparse COO tensor can be constructed by providing the two tensors of
indices and values, as well as the size of the sparse tensor (when it
cannot be inferred from the indices and values tensors) to a function
:func:`torch.sparse_coo_tensor`.

Suppose we want to define a sparse tensor with the entry 3 at location
(0, 2), entry 4 at location (1, 0), and entry 5 at location (1, 2).
Unspecified elements are assumed to have the same value, fill value,
which is zero by default. We would then write:

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

An empty sparse COO tensor can be constructed by specifying its size
only:

    >>> torch.sparse_coo_tensor(size=(2, 3))
    tensor(indices=tensor([], size=(2, 0)),
           values=tensor([], size=(0,)),
           size=(2, 3), nnz=0, layout=torch.sparse_coo)

.. _sparse-hybrid-coo-docs:

Sparse hybrid COO tensors
-------------------------

PyTorch implements an extension of sparse tensors with scalar values
to sparse tensors with (contiguous) tensor values. Such tensors are
called hybrid tensors.

PyTorch hybrid COO tensor extends the sparse COO tensor by allowing
the ``values`` tensor to be a multi-dimensional tensor so that we
have:

  - the indices of specified elements are collected in ``indices``
    tensor of size ``(sparse_dims, nse)`` and with element type
    ``torch.int64``,

  - the corresponding (tensor) values are collected in ``values``
    tensor of size ``(nse, dense_dims)`` and with an arbitrary integer
    or floating point number element type.

.. note::

   We use (M + K)-dimensional tensor to denote a N-dimensional sparse
   hybrid tensor, where M and K are the numbers of sparse and dense
   dimensions, respectively, such that M + K == N holds.

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

In general, if ``s`` is a sparse COO tensor and ``M =
s.sparse_dim()``, ``K = s.dense_dim()``, then we have the following
invariants:

  - ``M + K == len(s.shape) == s.ndim`` - dimensionality of a tensor
    is the sum of the number of sparse and dense dimensions,
  - ``s.indices().shape == (M, nse)`` - sparse indices are stored
    explicitly,
  - ``s.values().shape == (nse,) + s.shape[M : M + K]`` - the values
    of a hybrid tensor are K-dimensional tensors,
  - ``s.values().layout == torch.strided`` - values are stored as
    strided tensors.

.. note::

   Dense dimensions always follow sparse dimensions, that is, mixing
   of dense and sparse dimensions is not supported.

.. note::

   To be sure that a constructed sparse tensor has consistent indices,
   values, and size, the invariant checks can be enabled per tensor
   creation via ``check_invariants=True`` keyword argument, or
   globally using :class:`torch.sparse.check_sparse_tensor_invariants`
   context manager instance. By default, the sparse tensor invariants
   checks are disabled.

.. _sparse-uncoalesced-coo-docs:

Uncoalesced sparse COO tensors
------------------------------

PyTorch sparse COO tensor format permits sparse *uncoalesced* tensors,
where there may be duplicate coordinates in the indices; in this case,
the interpretation is that the value at that index is the sum of all
duplicate value entries. For example, one can specify multiple values,
``3`` and ``4``, for the same index ``1``, that leads to an 1-D
uncoalesced tensor:

    >>> i = [[1, 1]]
    >>> v =  [3, 4]
    >>> s=torch.sparse_coo_tensor(i, v, (3,))
    >>> s
    tensor(indices=tensor([[1, 1]]),
           values=tensor(  [3, 4]),
           size=(3,), nnz=2, layout=torch.sparse_coo)

while the coalescing process will accumulate the multi-valued elements
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
  identically given a sparse coalesced or uncoalesced tensor.

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
some other layout, one can use :attr:`torch.Tensor.is_sparse` or
:attr:`torch.Tensor.layout` properties:

    >>> isinstance(s, torch.Tensor)
    True
    >>> s.is_sparse
    True
    >>> s.layout == torch.sparse_coo
    True

The number of sparse and dense dimensions can be acquired using
methods :meth:`torch.Tensor.sparse_dim` and
:meth:`torch.Tensor.dense_dim`, respectively. For instance:

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

  For acquiring the COO format data of an uncoalesced tensor, use
  :func:`torch.Tensor._values()` and :func:`torch.Tensor._indices()`:

    >>> s._indices()
    tensor([[0, 1, 1],
            [2, 0, 2]])

  .. See https://github.com/pytorch/pytorch/pull/45695 for a new API.

  .. warning::
    Calling :meth:`torch.Tensor._values()` will return a *detached* tensor.
    To track gradients, :meth:`torch.Tensor.coalesce().values()` must be
    used instead.

Constructing a new sparse COO tensor results a tensor that is not
coalesced:

    >>> s.is_coalesced()
    False

but one can construct a coalesced copy of a sparse COO tensor using
the :meth:`torch.Tensor.coalesce` method:

    >>> s2 = s.coalesce()
    >>> s2.indices()
    tensor([[0, 1, 1],
           [2, 0, 2]])

When working with uncoalesced sparse COO tensors, one must take into
an account the additive nature of uncoalesced data: the values of the
same indices are the terms of a sum that evaluation gives the value of
the corresponding tensor element. For example, the scalar
multiplication on a sparse uncoalesced tensor could be implemented by
multiplying all the uncoalesced values with the scalar because ``c *
(a + b) == c * a + c * b`` holds. However, any nonlinear operation,
say, a square root, cannot be implemented by applying the operation to
uncoalesced data because ``sqrt(a + b) == sqrt(a) + sqrt(b)`` does not
hold in general.

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


In PyTorch, the fill value of a sparse tensor cannot be specified
explicitly and is assumed to be zero in general. However, there exists
operations that may interpret the fill value differently. For
instance, :func:`torch.sparse.softmax` computes the softmax with the
assumption that the fill value is negative infinity.

.. See https://github.com/Quansight-Labs/rfcs/tree/pearu/rfc-fill-value/RFC-0004-sparse-fill-value for a new API

.. _sparse-compressed-docs:

Sparse Compressed Tensors
+++++++++++++++++++++++++

Sparse Compressed Tensors represents a class of sparse tensors that
have a common feature of compressing the indices of a certain dimension
using an encoding that enables certain optimizations on linear algebra
kernels of sparse compressed tensors. This encoding is based on the
`Compressed Sparse Row (CSR)`__ format that PyTorch sparse compressed
tensors extend with the support of sparse tensor batches, allowing
multi-dimensional tensor values, and storing sparse tensor values in
dense blocks.

__ https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

.. note::

   We use (B + M + K)-dimensional tensor to denote a N-dimensional
   sparse compressed hybrid tensor, where B, M, and K are the numbers
   of batch, sparse, and dense dimensions, respectively, such that
   ``B + M + K == N`` holds. The number of sparse dimensions for
   sparse compressed tensors is always two, ``M == 2``.

.. note::

   We say that an indices tensor ``compressed_indices`` uses CSR
   encoding if the following invariants are satisfied:

   - ``compressed_indices`` is a contiguous strided 32 or 64 bit
     integer tensor
   - ``compressed_indices`` shape is ``(*batchsize,
     compressed_dim_size + 1)`` where ``compressed_dim_size`` is the
     number of compressed dimensions (e.g. rows or columns)
   - ``compressed_indices[..., 0] == 0`` where ``...`` denotes batch
     indices
   - ``compressed_indices[..., compressed_dim_size] == nse`` where
     ``nse`` is the number of specified elements
   - ``0 <= compressed_indices[..., i] - compressed_indices[..., i -
     1] <= plain_dim_size`` for ``i=1, ..., compressed_dim_size``,
     where ``plain_dim_size`` is the number of plain dimensions
     (orthogonal to compressed dimensions, e.g. columns or rows).

   To be sure that a constructed sparse tensor has consistent indices,
   values, and size, the invariant checks can be enabled per tensor
   creation via ``check_invariants=True`` keyword argument, or
   globally using :class:`torch.sparse.check_sparse_tensor_invariants`
   context manager instance. By default, the sparse tensor invariants
   checks are disabled.

.. note::

   The generalization of sparse compressed layouts to N-dimensional
   tensors can lead to some confusion regarding the count of specified
   elements. When a sparse compressed tensor contains batch dimensions
   the number of specified elements will correspond to the number of such
   elements per-batch. When a sparse compressed tensor has dense dimensions
   the element considered is now the K-dimensional array. Also for block
   sparse compressed layouts the 2-D block is considered as the element
   being specified.  Take as an example a 3-dimensional block sparse
   tensor, with one batch dimension of length ``b``, and a block
   shape of ``p, q``. If this tensor has ``n`` specified elements, then
   in fact we have ``n`` blocks specified per batch. This tensor would
   have ``values`` with shape ``(b, n, p, q)``. This interpretation of the
   number of specified elements comes from all sparse compressed layouts
   being derived from the compression of a 2-dimensional matrix. Batch
   dimensions are treated as stacking of sparse matrices, dense dimensions
   change the meaning of the element from a simple scalar value to an
   array with its own dimensions.

.. _sparse-csr-docs:

Sparse CSR Tensor
-----------------

The primary advantage of the CSR format over the COO format is better
use of storage and much faster computation operations such as sparse
matrix-vector multiplication using MKL and MAGMA backends.

In the simplest case, a (0 + 2 + 0)-dimensional sparse CSR tensor
consists of three 1-D tensors: ``crow_indices``, ``col_indices`` and
``values``:

  - The ``crow_indices`` tensor consists of compressed row
    indices. This is a 1-D tensor of size ``nrows + 1`` (the number of
    rows plus 1). The last element of ``crow_indices`` is the number
    of specified elements, ``nse``. This tensor encodes the index in
    ``values`` and ``col_indices`` depending on where the given row
    starts. Each successive number in the tensor subtracted by the
    number before it denotes the number of elements in a given row.

  - The ``col_indices`` tensor contains the column indices of each
    element. This is a 1-D tensor of size ``nse``.

  - The ``values`` tensor contains the values of the CSR tensor
    elements. This is a 1-D tensor of size ``nse``.

.. note::

   The index tensors ``crow_indices`` and ``col_indices`` should have
   element type either ``torch.int64`` (default) or
   ``torch.int32``. If you want to use MKL-enabled matrix operations,
   use ``torch.int32``. This is as a result of the default linking of
   pytorch being with MKL LP64, which uses 32 bit integer indexing.

In the general case, the (B + 2 + K)-dimensional sparse CSR tensor
consists of two (B + 1)-dimensional index tensors ``crow_indices`` and
``col_indices``, and of (1 + K)-dimensional ``values`` tensor such
that

  - ``crow_indices.shape == (*batchsize, nrows + 1)``

  - ``col_indices.shape == (*batchsize, nse)``

  - ``values.shape == (nse, *densesize)``

while the shape of the sparse CSR tensor is ``(*batchsize, nrows,
ncols, *densesize)`` where ``len(batchsize) == B`` and
``len(densesize) == K``.

.. note::

   The batches of sparse CSR tensors are dependent: the number of
   specified elements in all batches must be the same. This somewhat
   artificial constraint allows efficient storage of the indices of
   different CSR batches.

.. note::

   The number of sparse and dense dimensions can be acquired using
   :meth:`torch.Tensor.sparse_dim` and :meth:`torch.Tensor.dense_dim`
   methods. The batch dimensions can be computed from the tensor
   shape: ``batchsize = tensor.shape[:-tensor.sparse_dim() -
   tensor.dense_dim()]``.

.. note::

   The memory consumption of a sparse CSR tensor is at least
   ``(nrows * 8 + (8 + <size of element type in bytes> *
   prod(densesize)) * nse) * prod(batchsize)`` bytes (plus a constant
   overhead from storing other tensor data).

   With the same example data of :ref:`the note in sparse COO format
   introduction<sparse-coo-docs>`, the memory consumption of a 10 000
   x 10 000 tensor with 100 000 non-zero 32-bit floating point numbers
   is at least ``(10000 * 8 + (8 + 4 * 1) * 100 000) * 1 = 1 280 000``
   bytes when using CSR tensor layout. Notice the 1.6 and 310 fold
   savings from using CSR storage format compared to using the COO and
   strided formats, respectively.

Construction of CSR tensors
'''''''''''''''''''''''''''

Sparse CSR tensors can be directly constructed by using the
:func:`torch.sparse_csr_tensor` function. The user must supply the row
and column indices and values tensors separately where the row indices
must be specified using the CSR compression encoding.  The ``size``
argument is optional and will be deduced from the ``crow_indices`` and
``col_indices`` if it is not present.

    >>> crow_indices = torch.tensor([0, 2, 4])
    >>> col_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([1, 2, 3, 4])
    >>> csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
    >>> csr
    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([0, 1, 0, 1]),
           values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
           dtype=torch.float64)
    >>> csr.to_dense()
    tensor([[1., 2.],
            [3., 4.]], dtype=torch.float64)

.. note::

   The values of sparse dimensions in deduced ``size`` is computed
   from the size of ``crow_indices`` and the maximal index value in
   ``col_indices``. If the number of columns needs to be larger than
   in the deduced ``size`` then the ``size`` argument must be
   specified explicitly.

The simplest way of constructing a 2-D sparse CSR tensor from a
strided or sparse COO tensor is to use
:meth:`torch.Tensor.to_sparse_csr` method. Any zeros in the (strided)
tensor will be interpreted as missing values in the sparse tensor:

    >>> a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)
    >>> sp = a.to_sparse_csr()
    >>> sp
    tensor(crow_indices=tensor([0, 1, 3, 3]),
          col_indices=tensor([2, 0, 1]),
          values=tensor([1., 1., 2.]), size=(3, 4), nnz=3, dtype=torch.float64)

CSR Tensor Operations
'''''''''''''''''''''

The sparse matrix-vector multiplication can be performed with the
:meth:`tensor.matmul` method. This is currently the only math operation
supported on CSR tensors.

    >>> vec = torch.randn(4, 1, dtype=torch.float64)
    >>> sp.matmul(vec)
    tensor([[0.9078],
            [1.3180],
            [0.0000]], dtype=torch.float64)

.. _sparse-csc-docs:

Sparse CSC Tensor
-----------------

The sparse CSC (Compressed Sparse Column) tensor format implements the
CSC format for storage of 2 dimensional tensors with an extension to
supporting batches of sparse CSC tensors and values being
multi-dimensional tensors.

.. note::

   Sparse CSC tensor is essentially a transpose of the sparse CSR
   tensor when the transposition is about swapping the sparse
   dimensions.

Similarly to :ref:`sparse CSR tensors <sparse-csr-docs>`, a sparse CSC
tensor consists of three tensors: ``ccol_indices``, ``row_indices``
and ``values``:

  - The ``ccol_indices`` tensor consists of compressed column
    indices. This is a (B + 1)-D tensor of shape ``(*batchsize, ncols + 1)``.
    The last element is the number of specified
    elements, ``nse``. This tensor encodes the index in ``values`` and
    ``row_indices`` depending on where the given column starts. Each
    successive number in the tensor subtracted by the number before it
    denotes the number of elements in a given column.

  - The ``row_indices`` tensor contains the row indices of each
    element. This is a (B + 1)-D tensor of shape ``(*batchsize, nse)``.

  - The ``values`` tensor contains the values of the CSC tensor
    elements. This is a (1 + K)-D tensor of shape ``(nse, *densesize)``.

Construction of CSC tensors
'''''''''''''''''''''''''''

Sparse CSC tensors can be directly constructed by using the
:func:`torch.sparse_csc_tensor` function. The user must supply the row
and column indices and values tensors separately where the column indices
must be specified using the CSR compression encoding.  The ``size``
argument is optional and will be deduced from the ``row_indices`` and
``ccol_indices`` tensors if it is not present.

    >>> ccol_indices = torch.tensor([0, 2, 4])
    >>> row_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([1, 2, 3, 4])
    >>> csc = torch.sparse_csc_tensor(ccol_indices, row_indices, values, dtype=torch.float64)
    >>> csc
    tensor(ccol_indices=tensor([0, 2, 4]),
           row_indices=tensor([0, 1, 0, 1]),
           values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
           dtype=torch.float64, layout=torch.sparse_csc)
    >>> csc.to_dense()
    tensor([[1., 3.],
            [2., 4.]], dtype=torch.float64)

.. note::

   The sparse CSC tensor constructor function has the compressed
   column indices argument before the row indices argument.

The (0 + 2 + 0)-dimensional sparse CSC tensors can be constructed from
any two-dimensional tensor using :meth:`torch.Tensor.to_sparse_csc`
method. Any zeros in the (strided) tensor will be interpreted as
missing values in the sparse tensor:

    >>> a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.float64)
    >>> sp = a.to_sparse_csc()
    >>> sp
    tensor(ccol_indices=tensor([0, 1, 2, 3, 3]),
           row_indices=tensor([1, 1, 0]),
           values=tensor([1., 2., 1.]), size=(3, 4), nnz=3, dtype=torch.float64,
           layout=torch.sparse_csc)

.. _sparse-bsr-docs:

Sparse BSR Tensor
-----------------

The sparse BSR (Block compressed Sparse Row) tensor format implements the
BSR format for storage of two-dimensional tensors with an extension to
supporting batches of sparse BSR tensors and values being blocks of
multi-dimensional tensors.

A sparse BSR tensor consists of three tensors: ``crow_indices``,
``col_indices`` and ``values``:

  - The ``crow_indices`` tensor consists of compressed row
    indices. This is a (B + 1)-D tensor of shape ``(*batchsize,
    nrowblocks + 1)``.  The last element is the number of specified blocks,
    ``nse``. This tensor encodes the index in ``values`` and
    ``col_indices`` depending on where the given column block
    starts. Each successive number in the tensor subtracted by the
    number before it denotes the number of blocks in a given row.

  - The ``col_indices`` tensor contains the column block indices of each
    element. This is a (B + 1)-D tensor of shape ``(*batchsize,
    nse)``.

  - The ``values`` tensor contains the values of the sparse BSR tensor
    elements collected into two-dimensional blocks. This is a (1 + 2 +
    K)-D tensor of shape ``(nse, nrowblocks, ncolblocks,
    *densesize)``.

Construction of BSR tensors
'''''''''''''''''''''''''''

Sparse BSR tensors can be directly constructed by using the
:func:`torch.sparse_bsr_tensor` function. The user must supply the row
and column block indices and values tensors separately where the row block indices
must be specified using the CSR compression encoding.
The ``size`` argument is optional and will be deduced from the ``crow_indices`` and
``col_indices`` tensors if it is not present.

    >>> crow_indices = torch.tensor([0, 2, 4])
    >>> col_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
    ...                        [[3, 4, 5], [9, 10, 11]],
    ...                        [[12, 13, 14], [18, 19, 20]],
    ...                        [[15, 16, 17], [21, 22, 23]]])
    >>> bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
    >>> bsr
    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([0, 1, 0, 1]),
           values=tensor([[[ 0.,  1.,  2.],
                           [ 6.,  7.,  8.]],
                          [[ 3.,  4.,  5.],
                           [ 9., 10., 11.]],
                          [[12., 13., 14.],
                           [18., 19., 20.]],
                          [[15., 16., 17.],
                           [21., 22., 23.]]]),
           size=(4, 6), nnz=4, dtype=torch.float64, layout=torch.sparse_bsr)
    >>> bsr.to_dense()
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
            [ 6.,  7.,  8.,  9., 10., 11.],
            [12., 13., 14., 15., 16., 17.],
            [18., 19., 20., 21., 22., 23.]], dtype=torch.float64)

The (0 + 2 + 0)-dimensional sparse BSR tensors can be constructed from
any two-dimensional tensor using :meth:`torch.Tensor.to_sparse_bsr`
method that also requires the specification of the values block size:

    >>> dense = torch.tensor([[0, 1, 2, 3, 4, 5],
    ...                       [6, 7, 8, 9, 10, 11],
    ...                       [12, 13, 14, 15, 16, 17],
    ...                       [18, 19, 20, 21, 22, 23]])
    >>> bsr = dense.to_sparse_bsr(blocksize=(2, 3))
    >>> bsr
    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([0, 1, 0, 1]),
           values=tensor([[[ 0,  1,  2],
                           [ 6,  7,  8]],
                          [[ 3,  4,  5],
                           [ 9, 10, 11]],
                          [[12, 13, 14],
                           [18, 19, 20]],
                          [[15, 16, 17],
                           [21, 22, 23]]]), size=(4, 6), nnz=4,
           layout=torch.sparse_bsr)

.. _sparse-bsc-docs:

Sparse BSC Tensor
-----------------

The sparse BSC (Block compressed Sparse Column) tensor format implements the
BSC format for storage of two-dimensional tensors with an extension to
supporting batches of sparse BSC tensors and values being blocks of
multi-dimensional tensors.

A sparse BSC tensor consists of three tensors: ``ccol_indices``,
``row_indices`` and ``values``:

  - The ``ccol_indices`` tensor consists of compressed column
    indices. This is a (B + 1)-D tensor of shape ``(*batchsize,
    ncolblocks + 1)``.  The last element is the number of specified blocks,
    ``nse``. This tensor encodes the index in ``values`` and
    ``row_indices`` depending on where the given row block
    starts. Each successive number in the tensor subtracted by the
    number before it denotes the number of blocks in a given column.

  - The ``row_indices`` tensor contains the row block indices of each
    element. This is a (B + 1)-D tensor of shape ``(*batchsize,
    nse)``.

  - The ``values`` tensor contains the values of the sparse BSC tensor
    elements collected into two-dimensional blocks. This is a (1 + 2 +
    K)-D tensor of shape ``(nse, nrowblocks, ncolblocks,
    *densesize)``.

Construction of BSC tensors
'''''''''''''''''''''''''''

Sparse BSC tensors can be directly constructed by using the
:func:`torch.sparse_bsc_tensor` function. The user must supply the row
and column block indices and values tensors separately where the column block indices
must be specified using the CSR compression encoding.
The ``size`` argument is optional and will be deduced from the ``ccol_indices`` and
``row_indices`` tensors if it is not present.

    >>> ccol_indices = torch.tensor([0, 2, 4])
    >>> row_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
    ...                        [[3, 4, 5], [9, 10, 11]],
    ...                        [[12, 13, 14], [18, 19, 20]],
    ...                        [[15, 16, 17], [21, 22, 23]]])
    >>> bsc = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, dtype=torch.float64)
    >>> bsc
    tensor(ccol_indices=tensor([0, 2, 4]),
           row_indices=tensor([0, 1, 0, 1]),
           values=tensor([[[ 0.,  1.,  2.],
                           [ 6.,  7.,  8.]],
                          [[ 3.,  4.,  5.],
                           [ 9., 10., 11.]],
                          [[12., 13., 14.],
                           [18., 19., 20.]],
                          [[15., 16., 17.],
                           [21., 22., 23.]]]), size=(4, 6), nnz=4,
           dtype=torch.float64, layout=torch.sparse_bsc)

Tools for working with sparse compressed tensors
------------------------------------------------

All sparse compressed tensors --- CSR, CSC, BSR, and BSC tensors ---
are conceptionally very similar in that their indices data is split
into two parts: so-called compressed indices that use the CSR
encoding, and so-called plain indices that are orthogonal to the
compressed indices. This allows various tools on these tensors to
share the same implementations that are parameterized by tensor
layout.

Construction of sparse compressed tensors
'''''''''''''''''''''''''''''''''''''''''

Sparse CSR, CSC, BSR, and CSC tensors can be constructed by using
:func:`torch.sparse_compressed_tensor` function that have the same
interface as the above discussed constructor functions
:func:`torch.sparse_csr_tensor`, :func:`torch.sparse_csc_tensor`,
:func:`torch.sparse_bsr_tensor`, and :func:`torch.sparse_bsc_tensor`,
respectively, but with an extra required ``layout`` argument. The
following example illustrates a method of constructing CSR and CSC
tensors using the same input data by specifying the corresponding
layout parameter to the :func:`torch.sparse_compressed_tensor`
function:

    >>> compressed_indices = torch.tensor([0, 2, 4])
    >>> plain_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([1, 2, 3, 4])
    >>> csr = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=torch.sparse_csr)
    >>> csr
    tensor(crow_indices=tensor([0, 2, 4]),
           col_indices=tensor([0, 1, 0, 1]),
           values=tensor([1, 2, 3, 4]), size=(2, 2), nnz=4,
           layout=torch.sparse_csr)
    >>> csc = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=torch.sparse_csc)
    >>> csc
    tensor(ccol_indices=tensor([0, 2, 4]),
           row_indices=tensor([0, 1, 0, 1]),
           values=tensor([1, 2, 3, 4]), size=(2, 2), nnz=4,
           layout=torch.sparse_csc)
    >>> (csr.transpose(0, 1).to_dense() == csc.to_dense()).all()
    tensor(True)

.. _sparse-ops-docs:

Supported operations
+++++++++++++++++++++++++++++++++++

Linear Algebra operations
-------------------------

The following table summarizes supported Linear Algebra operations on
sparse matrices where the operands layouts may vary. Here
``T[layout]`` denotes a tensor with a given layout. Similarly,
``M[layout]`` denotes a matrix (2-D PyTorch tensor), and ``V[layout]``
denotes a vector (1-D PyTorch tensor). In addition, ``f`` denotes a
scalar (float or 0-D PyTorch tensor), ``*`` is element-wise
multiplication, and ``@`` is matrix multiplication.

.. csv-table::
   :header: "PyTorch operation", "Sparse grad?", "Layout signature"
   :widths: 20, 5, 60
   :delim: ;

   :func:`torch.mv`;no; ``M[sparse_coo] @ V[strided] -> V[strided]``
   :func:`torch.mv`;no; ``M[sparse_csr] @ V[strided] -> V[strided]``
   :func:`torch.matmul`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[sparse_csr] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
   :func:`torch.mm`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.mm`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
   :func:`torch.mm`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
   :func:`torch.sparse.mm`; yes; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.smm`; no; ``M[sparse_coo] @ M[strided] -> M[sparse_coo]``
   :func:`torch.hspmm`; no; ``M[sparse_coo] @ M[strided] -> M[hybrid sparse_coo]``
   :func:`torch.bmm`; no; ``T[sparse_coo] @ T[strided] -> T[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[SparseSemiStructured] @ M[strided]) -> M[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[strided] @ M[SparseSemiStructured]) -> M[strided]``
   :func:`torch.sparse.addmm`; yes; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.sparse.spsolve`; no; ``SOLVE(M[sparse_csr], V[strided]) -> V[strided]``
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

Tensor methods and sparse
-------------------------

The following Tensor methods are related to sparse tensors:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.is_sparse
    Tensor.is_sparse_csr
    Tensor.dense_dim
    Tensor.sparse_dim
    Tensor.sparse_mask
    Tensor.to_sparse
    Tensor.to_sparse_coo
    Tensor.to_sparse_csr
    Tensor.to_sparse_csc
    Tensor.to_sparse_bsr
    Tensor.to_sparse_bsc
    Tensor.to_dense
    Tensor.values

The following Tensor methods are specific to sparse COO tensors:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.coalesce
    Tensor.sparse_resize_
    Tensor.sparse_resize_and_clear_
    Tensor.is_coalesced
    Tensor.indices

The following methods are specific to :ref:`sparse CSR tensors <sparse-csr-docs>` and :ref:`sparse BSR tensors <sparse-bsr-docs>`:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.crow_indices
    Tensor.col_indices

The following methods are specific to :ref:`sparse CSC tensors <sparse-csc-docs>` and :ref:`sparse BSC tensors <sparse-bsc-docs>`:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.row_indices
    Tensor.ccol_indices

The following Tensor methods support sparse COO tensors:

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
:meth:`~torch.Tensor.sqrt`
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

Torch functions specific to sparse Tensors
------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse_coo_tensor
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    sparse_compressed_tensor
    sparse.sum
    sparse.addmm
    sparse.sampled_addmm
    sparse.mm
    sspaddmm
    hspmm
    smm
    sparse.softmax
    sparse.spsolve
    sparse.log_softmax
    sparse.spdiags

Other functions
---------------

The following :mod:`torch` functions support sparse tensors:

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

To manage checking sparse tensor invariants, see:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse.check_sparse_tensor_invariants

To use sparse tensors with :func:`~torch.autograd.gradcheck` function,
see:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse.as_sparse_gradcheck

Zero-preserving unary functions
-------------------------------

We aim to support all 'zero-preserving unary functions': functions of one argument that map zero to zero.

If you find that we are missing a zero-preserving unary function
that you need, please feel encouraged to open an issue for a feature request.
As always please kindly try the search function first before opening an issue.

The following operators currently support sparse COO/CSR/CSC/BSR/CSR tensor inputs.

:func:`~torch.abs`
:func:`~torch.asin`
:func:`~torch.asinh`
:func:`~torch.atan`
:func:`~torch.atanh`
:func:`~torch.ceil`
:func:`~torch.conj_physical`
:func:`~torch.floor`
:func:`~torch.log1p`
:func:`~torch.neg`
:func:`~torch.round`
:func:`~torch.sin`
:func:`~torch.sinh`
:func:`~torch.sign`
:func:`~torch.sgn`
:func:`~torch.signbit`
:func:`~torch.tan`
:func:`~torch.tanh`
:func:`~torch.trunc`
:func:`~torch.expm1`
:func:`~torch.sqrt`
:func:`~torch.angle`
:func:`~torch.isinf`
:func:`~torch.isposinf`
:func:`~torch.isneginf`
:func:`~torch.isnan`
:func:`~torch.erf`
:func:`~torch.erfinv`
