.. automodule:: torch.masked
.. automodule:: torch.masked.maskedtensor

.. currentmodule:: torch

.. _masked-docs:

torch.masked
============

Introduction
++++++++++++

Motivation
----------

.. warning::

  The PyTorch API of masked tensors is in prototype stage and may or may not change in the future.

MaskedTensor serves as an extension to :class:`torch.Tensor` that provides the user with the ability to:
- use any masked semantics (e.g. variable length tensors, nan* operators, etc.)
- differentiation between 0 and NaN gradients
- various sparse applications (see tutorial below)

"Specified" and "unspecified" have a long history in PyTorch without formal semantics and certainly without
consistency; indeed, MaskedTensor was born out of a build up of :ref:`issues` that the vanilla :class:`torch.Tensor`
class could not properly address. Thus, a primary goal of MaskedTensor is to become the source of truth for
said "specified" and "unspecified" values in PyTorch where they are a first class citizen instead of an afterthought.
In turn, this should further unlock `sparsity's <https://pytorch.org/docs/stable/sparse.html>`__ potential,
enable safer and more consistent operators, and provide a smoother and more intuitive experience
for users and developers alike.

What is a MaskedTensor?
-----------------------

A MaskedTensor is a tensor subclass that consists of 1) an input (data), and 2) a mask. The mask tells us
which entries from the input should be included or ignored.

By way of example, suppose that we wanted to mask out all values that are equal to 0 (represented by the gray)
and take the max:

.. image:: _static/img/masked/tensor_comparison.png
      :scale: 50 %

On top is the vanilla tensor example while the bottom is MaskedTensor where all the 0's are masked out.
This clearly yields a different result depending on whether we have the mask, but this flexible structure
allows the user to systematically ignore any elements they'd like during computation.

Using MaskedTensor
++++++++++++++++++

Construction
------------

There are a few different ways to construct a MaskedTensor:

* The first way is to directly invoke the MaskedTensor class
* The second (and our recommended way) is to use :func:`masked.masked_tensor` and :func:`masked.as_masked_tensor` factory functions,
  which are analogous to :func:`torch.tensor` and :func:`torch.as_tensor`

  .. autosummary::
    :toctree: generated
    :nosignatures:

    masked.masked_tensor
    masked.as_masked_tensor

Accessing the data and mask
---------------------------

The underlying fields in a MaskedTensor can be accessed through:

* the :meth:`MaskedTensor.get_data` function
* the :meth:`MaskedTensor.get_mask` function. Recall that ``True`` indicates "specified" or "valid" while ``False`` indicates
  "unspecified" or "invalid".

In general, the underlying data that is returned may not be valid in the unspecified entries, so we recommend that
when users require a Tensor without any masked entries, that they use :meth:`MaskedTensor.to_tensor` (as shown above) to
return a Tensor with filled values.

Indexing and slicing
--------------------

:class:`MaskedTensor` is a Tensor subclass, which means that it inherits the same semantics for indexing and slicing
as :class:`torch.Tensor`. Below are some examples of common indexing and slicing patterns:

    >>> data = torch.arange(60).reshape(3, 4, 5)
    >>> mask = data % 2 == 0
    >>> mt = masked_tensor(data.float(), mask)
    >>> mt[0]
    MaskedTensor(
      [
        [  0.0000,       --,   2.0000,       --,   4.0000],
        [      --,   6.0000,       --,   8.0000,       --],
        [ 10.0000,       --,  12.0000,       --,  14.0000],
        [      --,  16.0000,       --,  18.0000,       --]
      ]
    )
    >>> mt[[0,2]]
    MaskedTensor(
      [
        [
          [  0.0000,       --,   2.0000,       --,   4.0000],
          [      --,   6.0000,       --,   8.0000,       --],
          [ 10.0000,       --,  12.0000,       --,  14.0000],
          [      --,  16.0000,       --,  18.0000,       --]
        ],
        [
          [ 40.0000,       --,  42.0000,       --,  44.0000],
          [      --,  46.0000,       --,  48.0000,       --],
          [ 50.0000,       --,  52.0000,       --,  54.0000],
          [      --,  56.0000,       --,  58.0000,       --]
        ]
      ]
    )
    >>> mt[:, :2]
    MaskedTensor(
      [
        [
          [  0.0000,       --,   2.0000,       --,   4.0000],
          [      --,   6.0000,       --,   8.0000,       --]
        ],
        [
          [ 20.0000,       --,  22.0000,       --,  24.0000],
          [      --,  26.0000,       --,  28.0000,       --]
        ],
        [
          [ 40.0000,       --,  42.0000,       --,  44.0000],
          [      --,  46.0000,       --,  48.0000,       --]
        ]
      ]
    )

Sparsity
++++++++

Sparsity has been an area of rapid growth and importance within PyTorch; if any sparsity terms are confusing below,
please refer to the `sparsity tutorial <https://pytorch.org/docs/stable/sparse.html>`__ for additional details.

Sparse storage formats have been proven to be powerful in a variety of ways. As a primer, the first use case
most practitioners think about is when the majority of elements are equal to zero (a high degree of sparsity),
but even in cases of lower sparsity, certain formats (e.g. BSR) can take advantage of substructures within a matrix.

.. note::

    At the moment, MaskedTensor supports COO and CSR tensors with plans to support additional formats
    (e.g. BSR and CSC) in the future. If you have any requests for additional formats, please file a feature request!

Principles
----------

When creating a :class:`MaskedTensor` with sparse tensors, there are a few principles that must be observed:

1. ``data`` and ``mask`` must have the same storage format, whether that's :attr:`torch.strided`, :attr:`torch.sparse_coo`, or :attr:`torch.sparse_csr`
2. ``data`` and ``mask`` must have the same size, indicated by :func:`size()`

Sparse COO tensors
------------------

In accordance with Principle #1, a sparse COO MaskedTensor is created by passing in two sparse COO tensors,
which can be initialized by any of its constructors, e.g. :func:`torch.sparse_coo_tensor`.

As a recap of `sparse COO tensors <https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors>`__, the COO format
stands for "coordinate format", where the specified elements are stored as tuples of their indices and the
corresponding values. That is, the following are provided:

* ``indices``: array of size ``(ndim, nse)`` and dtype ``torch.int64``
* ``values``: array of size `(nse,)` with any integer or floating point dtype

where ``ndim`` is the dimensionality of the tensor and ``nse`` is the number of specified elements

For both sparse COO and CSR tensors, you can construct a :class:`MaskedTensor` by doing either:

1. ``masked_tensor(sparse_tensor_data, sparse_tensor_mask)``
2. ``dense_masked_tensor.to_sparse_coo()`` or ``dense_masked_tensor.to_sparse_csr()``

The second method is easier to illustrate so we've shown that below, but for more on the first and the nuances behind
the approach, please read the :ref:`sparse-coo-appendix`.

    >>> values = torch.tensor([[0, 0, 3], [4, 0, 5]])
    >>> mask = torch.tensor([[False, False, True], [False, False, True]])
    >>> mt = masked_tensor(values, mask)
    >>> sparse_coo_mt = mt.to_sparse_coo()
    >>> mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> sparse_coo_mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> sparse_coo_mt.get_data()
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([3, 5]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)

Sparse CSR tensors
------------------

Similarly, :class:`MaskedTensor` also supports the
`CSR (Compressed Sparse Row) <https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor>`__
sparse tensor format. Instead of storing the tuples of the indices like sparse COO tensors, sparse CSR tensors
aim to decrease the memory requirements by storing compressed row indices.
In particular, a CSR sparse tensor consists of three 1-D tensors:

* ``crow_indices``: array of compressed row indices with size ``(size[0] + 1,)``. This array indicates which row
  a given entry in values lives in. The last element is the number of specified elements,
  while crow_indices[i+1] - crow_indices[i] indicates the number of specified elements in row i.
* ``col_indices``: array of size ``(nnz,)``. Indicates the column indices for each value.
* ``values``: array of size ``(nnz,)``. Contains the values of the CSR tensor.

Of note, both sparse COO and CSR tensors are in a `beta <https://pytorch.org/docs/stable/index.html>`__ state.

By way of example:

    >>> mt_sparse_csr = mt.to_sparse_csr()
    >>> mt_sparse_csr
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )
    >>> mt_sparse_csr.get_data()
    tensor(crow_indices=tensor([0, 1, 2]),
          col_indices=tensor([2, 2]),
          values=tensor([3, 5]), size=(2, 3), nnz=2, layout=torch.sparse_csr)

Semantics
+++++++++

MaskedTensor vs NumPy's MaskedArray
-----------------------------------

NumPy's ``MaskedArray`` has a few fundamental semantics differences from MaskedTensor.

1. Their factory function and basic definition inverts the mask (similar to ``torch.nn.MHA``); that is, MaskedTensor
uses ``True`` to denote "specified" and ``False`` to denote "unspecified", or "valid"/"invalid", whereas NumPy does the
opposite.
2. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be
masked out as well -- in practice, they
`apply the logical_or operator <https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024>`__.

    >>> data = torch.arange(5.)
    >>> mask = torch.tensor([True, True, False, True, False])
    >>> npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
    >>> npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())
    >>> npm0
    masked_array(data=[0.0, 1.0, --, 3.0, --],
                mask=[False, False,  True, False,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm1
    masked_array(data=[--, --, 2.0, --, 4.0],
                mask=[ True,  True, False,  True, False],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0 + npm1
    masked_array(data=[--, --, --, --, --],
                mask=[ True,  True,  True,  True,  True],
          fill_value=1e+20,
                dtype=float32)

Meanwhile, MaskedTensor does not support addition or binary operators with masks that don't match -- to understand why,
please find the section on reductions.

    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> m0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt0 = masked_tensor(data, mask)
    >>> mt1 = masked_tensor(data, ~mask)
    >>> mt0
    MaskedTensor(
      [  0.0000,   1.0000,       --,   3.0000,       --]
    )
    >>> mt1
    MaskedTensor(
      [      --,       --,   2.0000,       --,   4.0000]
    )
    >>> mt0 + mt1
    ValueError: Input masks must match. If you need support for this, please open an issue on Github.

However, if this behavior is desired, MaskedTensor does support these semantics by giving access to the data and masks
and conveniently converting a MaskedTensor to a Tensor with masked values filled in using :func:`to_tensor`.

    >>> t0 = mt0.to_tensor(0)
    >>> t1 = mt1.to_tensor(0)
    >>> mt2 = masked_tensor(t0 + t1, mt0.get_mask() & mt1.get_mask())
    >>> t0
    tensor([0., 1., 0., 3., 0.])
    >>> t1
    tensor([0., 0., 2., 0., 4.])
    >>> mt2
    MaskedTensor(
      [      --,       --,       --,       --,       --]

.. _reduction-semantics:

Reduction semantics
-------------------

The basis for reduction semantics `has been documented and discussed at length <https://github.com/pytorch/rfcs/pull/27>`__,
but again, by way of example:

    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> mask = torch.randint(2, (3, 4), dtype=torch.bool)
    >>> mt = masked_tensor(data, mask)
    >>> mt
    MaskedTensor(
      [
        [      --,   1.0000,       --,       --],
        [      --,   5.0000,   6.0000,   7.0000],
        [  8.0000,   9.0000,       --,  11.0000]
      ]
    )

    >>> torch.sum(mt, 1)
    MaskedTensor(
      [  1.0000,  18.0000,  28.0000]
    )
    >>> torch.mean(mt, 1)
    MaskedTensor(
      [  1.0000,   6.0000,   9.3333]
    )
    >>> torch.prod(mt, 1)
    MaskedTensor(
      [  1.0000, 210.0000, 792.0000]
    )
    >>> torch.amin(mt, 1)
    MaskedTensor(
      [  1.0000,   5.0000,   8.0000]
    )
    >>> torch.amax(mt, 1)
    MaskedTensor(
      [  1.0000,   7.0000,  11.0000]
    )

Now we can revisit the question: why do we enforce the invariant that masks must match for binary operators?
In other words, why don't we use the same semantics as ``np.ma.masked_array``? Consider the following example:

    >>> data0 = torch.arange(10.).reshape(2, 5)
    >>> data1 = torch.arange(10.).reshape(2, 5) + 10
    >>> mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
    >>> mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])

    >>> npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
    >>> npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())
    >>> npm0
    masked_array(
      data=[[--, --, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, --, --]],
      mask=[[ True,  True, False, False, False],
            [False, False, False,  True,  True]],
      fill_value=1e+20,
      dtype=float32)
    >>> npm1
    masked_array(
      data=[[10.0, 11.0, 12.0, --, --],
            [--, --, 17.0, 18.0, 19.0]],
      mask=[[False, False, False,  True,  True],
            [ True,  True, False, False, False]],
      fill_value=1e+20,
      dtype=float32)
    >>> (npm0 + npm1).sum(0)
    masked_array(data=[--, --, 38.0, --, --],
                mask=[ True,  True, False,  True,  True],
          fill_value=1e+20,
                dtype=float32)
    >>> npm0.sum(0) + npm1.sum(0)
    masked_array(data=[15.0, 17.0, 38.0, 21.0, 23.0],
                mask=[False, False, False, False, False],
          fill_value=1e+20,
                dtype=float32)

Sum and addition should clearly be associative, but with NumPy's semantics, they are allowed to not be,
which can certainly be confusing for the user. That being said, if the user wishes, there are ways around this
(e.g. filling in the MaskedTensor's undefined elements with 0 values using :func:`to_tensor` as shown in a previous
example), but the user must now be more explicit with their intentions.

.. _issues:

Issues
++++++

:class:`MaskedTensor` has fixed a number of different issues that have persisted across PyTorch for a number of years
by making masking a first class citizen.

Distinguishing between 0 and NaN gradient
-----------------------------------------

One issue that :class:`torch.Tensor` runs into is the inability to distinguish between gradients that are not
defined (NaN) vs. gradients that are actually 0. By way of example, below are several different issues where
:class:`MaskedTensor` can resolve and/or work around the NaN gradient problem.

`Issue 10729 <https://github.com/pytorch/pytorch/issues/10729>`__ -- torch.where
--------------------------------------------------------------------------------

Current result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True, dtype=torch.float)
    >>> y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
    >>> y.sum().backward()
    >>> x.grad
    tensor([4.5400e-05, 6.7379e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00,        nan,        nan])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100])
    >>> mask = x < 0
    >>> mx = masked_tensor(x, mask, requires_grad=True)
    >>> my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
    >>> y = torch.where(mask, torch.exp(mx), my)
    >>> y.sum().backward()
    >>> mx.grad
    MaskedTensor(
      [  0.0000,   0.0067,       --,       --,       --,       --,       --,       --,       --,       --,       --]
    )

`Issue 52248 <https://github.com/pytorch/pytorch/issues/52248>`__ -- another torch.where
----------------------------------------------------------------------------------------

Current result:

    >>> a = torch.randn((), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    tensor(1., grad_fn=<WhereBackward0>)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (tensor(nan),)

:class:`MaskedTensor` result:

    >>> a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
    >>> b = torch.tensor(False)
    >>> c = torch.ones(())
    >>> torch.where(b, a/0, c)
    MaskedTensor(  1.0000, True)
    >>> torch.autograd.grad(torch.where(b, a/0, c), a)
    (MaskedTensor(--, False),)

`Issue 67180 <https://github.com/pytorch/pytorch/issues/67180>`__ -- :func:`torch.nansum` and :func:`torch.nanmean`
-------------------------------------------------------------------------------------------------------------------

Current result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> c = a * b
    >>> c1 = torch.nansum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    tensor(nan)

:class:`MaskedTensor` result:

    >>> a = torch.tensor([1., 2., float('nan')])
    >>> b = torch.tensor(1.0, requires_grad=True)
    >>> mt = masked_tensor(a, ~torch.isnan(a))
    >>> c = mt * b
    >>> c1 = torch.sum(c)
    >>> bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
    >>> bgrad1
    MaskedTensor(  3.0000, True)

`Issue 4132 <https://github.com/pytorch/pytorch/issues/4132>`__ -- when using mask, x/0 yields NaN grad
-------------------------------------------------------------------------------------------------------

Current result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>> mask = (div != 0) # => mask is [0, 1]
    >>> y[mask].backward()
    >>> x.grad # grad is [nan, 1], but expected [0, 1]
    tensor([nan, 1.])

:class:`MaskedTensor` result:

    >>> x = torch.tensor([1., 1.], requires_grad=True)
    >>> div = torch.tensor([0., 1.])
    >>> y = x/div # => y is [inf, 1]
    >>>
    >>> mask = (div != 0) # => mask is [0, 1]
    >>> loss = as_masked_tensor(y, mask)
    >>> loss.sum().backward()
    >>> x.grad
    MaskedTensor(
      [      --,   1.0000]
    )

`Issue 55056 <https://github.com/pytorch/pytorch/issues/55056>`__ -- Safe Softmax
---------------------------------------------------------------------------------

One of the issues that frequently comes up is the necessity for a safe softmax -- that is, if there is an entire
batch that is "masked out" or consists entirely of padding (which, in the softmax case, translates to being set `-inf`),
then this will result in NaNs, which can leading to training divergence. For more detail on why this functionality
is necessary, please find refer to the issue.

Luckily, :class:`MaskedTensor` has solved this issue:

    >>> data = torch.randn(3, 3)
    >>> mask = torch.tensor([[True, False, False], [True, False, True], [False, False, False]])
    >>> x = data.masked_fill(~mask, float('-inf'))
    >>> x.softmax(0)
    tensor([[0.3548,    nan, 0.0000],
            [0.6452,    nan, 1.0000],
            [0.0000,    nan, 0.0000]])

    >>> mt = masked_tensor(data, mask)
    >>> mt.softmax(0)
    MaskedTensor(
      [
        [  0.3548,       --,       --],
        [  0.6452,       --,   1.0000],
        [      --,       --,       --]
      ]
    )


`Issue 61474 <https://github.com/pytorch/pytorch/issues/61474>`__ -- Implementing missing torch.nan* operators
--------------------------------------------------------------------------------------------------------------

In the above issue, there is a request to add additional operators to cover the various `torch.nan*` applications,
such as ``torch.nanmax``, ``torch.nanmin``, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using MaskedTensors instead. Since
`nanmean has already landed <https://github.com/pytorch/pytorch/issues/21987>`__, we can use it as a comparison point:

    >>> x = torch.arange(16).float()
    >>> y = x * x.fmod(4)
    >>> y = y.masked_fill(y ==0, float('nan'))
    >>> y
    tensor([nan,  1.,  4.,  9., nan,  5., 12., 21., nan,  9., 20., 33., nan, 13.,
            28., 45.])
    >>> y.nanmean()
    tensor(16.6667)
    >>> torch.mean(masked_tensor(y, ~torch.isnan(y)))
    MaskedTensor( 16.6667, True)

Furthermore, :class:`MaskedTensor` supports reductions when the data is fully masked out, which is equivalent
to the case above when the data Tensor is completely ``nan``. ``nanmean`` would return ``nan``
(an ambiguous return value), while MaskedTensor would more accurately indicate a masked out result.

    >>> x = torch.empty(16).fill_(float('nan'))
    >>> x
    tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    >>> torch.nanmean(x)
    tensor(nan)
    >>> torch.mean(masked_tensor(x, ~torch.isnan(x)))
    MaskedTensor(--, False)


Supported Operators
+++++++++++++++++++

Unary Operators
---------------

Unary operators are operators that only contain only a single input.
Applying them to MaskedTensors is relatively straightforward: if the data is masked out at a given index,
we apply the operator, otherwise we'll continue to mask out the data.

The available unary operators are:

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    bitwise_not
    ceil
    clamp
    clip
    conj_physical
    cos
    cosh
    deg2rad
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fix
    floor
    frac
    lgamma
    log
    log10
    log1p
    log2
    logit
    i0
    isnan
    nan_to_num
    neg
    negative
    positive
    pow
    rad2deg
    reciprocal
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    sqrt
    square
    tan
    tanh
    trunc

The available inplace unary operators are all of the above **except**:

.. autosummary::
    :toctree: generated
    :nosignatures:

    angle
    positive
    signbit
    isnan

Binary Operators
----------------

As you may have seen in the tutorial, :class:`MaskedTensor` also has binary operations implemented with the caveat
that the masks in the two MaskedTensors must match or else an error will be raised. As noted in the error, if you
need support for a particular operator or have proposed semantics for how they should be behave instead, please open
an issue on Github. For now, we have decided to go with the most conservative implementation to ensure that users
know exactly what is going on and are being intentional about their decisions with masked semantics.

The available binary operators are:

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    atan2
    arctan2
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    div
    divide
    floor_divide
    fmod
    logaddexp
    logaddexp2
    mul
    multiply
    nextafter
    remainder
    sub
    subtract
    true_divide
    eq
    ne
    le
    ge
    greater
    greater_equal
    gt
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    not_equal

The available inplace binary operators are all of the above **except**:

.. autosummary::
    :toctree: generated
    :nosignatures:

    logaddexp
    logaddexp2
    equal
    fmin
    minimum
    fmax

Reductions
----------

The following reductions are available (with autograd support):

.. autosummary::
    :toctree: generated
    :nosignatures:

    sum
    mean
    amin
    amax
    argmin
    argmax
    prod
    all
    norm
    var
    std

View and select functions
-------------------------

We've included a number of view and select functions as well; intuitively, these operators will apply to
both the data and the mask and then wrap the result in a :class:`MaskedTensor`. For a quick example,
consider :func:`select`:

    >>> data = torch.arange(12, dtype=torch.float).reshape(3, 4)
    >>> data
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]])
    >>> mask = torch.tensor([[True, False, False, True], [False, True, False, False], [True, True, True, True]])
    >>> mt = masked_tensor(data, mask)
    >>> data.select(0, 1)
    tensor([4., 5., 6., 7.])
    >>> mask.select(0, 1)
    tensor([False,  True, False, False])
    >>> mt.select(0, 1)
    MaskedTensor(
      [      --,   5.0000,       --,       --]
    )

The following ops are currently supported:

.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    broadcast_tensors
    broadcast_to
    cat
    chunk
    column_stack
    dsplit
    flatten
    hsplit
    hstack
    kron
    meshgrid
    narrow
    ravel
    select
    split
    t
    transpose
    vsplit
    vstack
    Tensor.expand
    Tensor.expand_as
    Tensor.reshape
    Tensor.reshape_as
    Tensor.view

.. _Appendix:

Appendix
++++++++

.. _sparse-coo-appendix:

Sparse COO construction
-----------------------

Recall in our original example, we created a :class:`MaskedTensor` and then converted it to a sparse COO MaskedTensor
with :meth:`MaskedTensor.to_sparse_coo`.

Alternatively, we can also construct a sparse COO MaskedTensor directly by passing in two sparse COO tensors:

    >>> values = torch.tensor([[0, 0, 3], [4, 0, 5]]).to_sparse()
    >>> mask = torch.tensor([[False, False, True], [False, False, True]]).to_sparse()
    >>> mt = masked_tensor(values, mask)
    >>> values
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mask
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([True, True]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)
    >>> mt
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )

Instead of using :meth:`torch.Tensor.to_sparse`, we can also create the sparse COO tensors directly, which brings us to a warning:

.. warning::

  When using a function like :meth:`MaskedTensor.to_sparse_coo`, if the user does not specify the indices like in the above
  example, then the 0 values will be "unspecified" by default.

Below, we explicitly specify the 0's:

    >>> values = torch.sparse_coo_tensor(i, v, (2, 3))
    >>> mask = torch.sparse_coo_tensor(i, m, (2, 3))
    >>> mt2 = masked_tensor(values, mask)
    >>> values
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mask
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([ True, False,  True]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)
    >>> mt2
    MaskedTensor(
      [
        [      --,       --, 3],
        [      --,       --, 5]
      ]
    )

Note that ``mt`` and ``mt2`` look identical on the surface, and in the vast majority of operations, will yield the same
result. But this brings us to a detail on the implementation:

``data`` and ``mask`` -- only for sparse MaskedTensors -- can have a different number of elements (:func:`nnz`)
**at creation**, but the indices of ``mask`` must then be a subset of the indices of ``data``. In this case,
``data`` will assume the shape of ``mask`` by ``data = data.sparse_mask(mask)``; in other words, any of the elements
in ``data`` that are not ``True`` in ``mask`` (i.e. not specified) will be thrown away.

Therefore, under the hood, the data looks slightly different; ``mt2`` has the "4" value masked out and ``mt`` is completely
without it. Their underlying data has different shapes, which would make operations like ``mt + mt2`` invalid.

    >>> mt.get_data()
    tensor(indices=tensor([[0, 1],
                          [2, 2]]),
          values=tensor([3, 5]),
          size=(2, 3), nnz=2, layout=torch.sparse_coo)
    >>> mt2.get_data()
    tensor(indices=tensor([[0, 1, 1],
                          [2, 0, 2]]),
          values=tensor([3, 4, 5]),
          size=(2, 3), nnz=3, layout=torch.sparse_coo)

.. _sparse-csr-appendix:

Sparse CSR construction
-----------------------

We can also construct a sparse CSR MaskedTensor using sparse CSR tensors,
and like the example above, this results in a similar treatment under the hood.

    >>> crow_indices = torch.tensor([0, 2, 4])
    >>> col_indices = torch.tensor([0, 1, 0, 1])
    >>> values = torch.tensor([1, 2, 3, 4])
    >>> mask_values = torch.tensor([True, False, False, True])
    >>>
    >>> csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
    >>> mask = torch.sparse_csr_tensor(crow_indices, col_indices, mask_values, dtype=torch.bool)
    >>>
    >>> mt = masked_tensor(csr, mask)
    >>> mt
    MaskedTensor(
      [
        [  1.0000,       --],
        [      --,   4.0000]
      ]
    )
    >>> mt.get_data()
    tensor(crow_indices=tensor([0, 2, 4]),
          col_indices=tensor([0, 1, 0, 1]),
          values=tensor([1., 2., 3., 4.]), size=(2, 2), nnz=4,
          dtype=torch.float64, layout=torch.sparse_csr)
