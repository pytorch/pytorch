torch.masked
============

.. automodule:: torch.masked
.. automodule:: torch.masked.maskedtensor

Introduction
++++++++++++

.. warning::

  The PyTorch API of masked tensors is in prototype stage and may or may not change in the future.

MaskedTensor serves as an extension to `torch.Tensor` that provides the user with the ability to:
- use any masked semantics (e.g. variable length tensors, nan* operators, etc.)
- differentiation between 0 and NaN gradients
- various sparse applications (see tutorial below)

The goal of MaskedTensor is to become the source of truth for "specified" and "unspecified" values in PyTorch;
in turn, this should further unlock [sparsity's](https://pytorch.org/docs/stable/sparse.html) potential,
enable safer and more consistent operators, and provide a smoother and more intuitive experience
for users and developers alike.

Overview
++++++++

A MaskedTensor is a tensor subclass that consists of 1) an input (data), and 2) a mask. The mask tells us
which entries from the input should be included or ignored.

By way of example, suppose that we wanted to mask out all values that are equal to 0 (represented by the gray)
and take the max:

.. figure:: _static/img/masked/tensor_comparison.png

  On top is the vanilla tensor example while the bottom is MaskedTensor where all the 0's are masked out.
  This clearly yields a different result depending on whether we have the mask, but this flexible structure
  allows the user to systematically ignore any elements they'd like during computation.

Using MaskedTensor
------------------

Construction
^^^^^^^^^^^^

There are a few different ways to construct a MaskedTensor:

* The first way is to directly invoke the MaskedTensor class

* The second (and our recommended way) is to use :func:`masked_tensor` and :func:`as_masked_tensor` factory functions, which are
  analogous to `torch.tensor` and `torch.as_tensor`

  .. autosummary::
    :toctree: generated
    :nosignatures:

    torch.masked.masked_tensor
    torch.masked.as_masked_tensor

Accessing the data and mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^


The underlying fields in a MaskedTensor can be accessed through:

* the :func:`get_data` function
* the :func:`get_mask` function. Recall that `True` indicates "specified" or "valid" while `False` indicates
"unspecified" or "invalid".

In general, the underlying data that is returned may not be valid in the unspecified entries, so we recommend that
when users require a Tensor without any masked entries, that they use :func:`to_tensor` (as shown above) to
return a Tensor with filled values.

Indexing and slicing
^^^^^^^^^^^^^^^^^^^^

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


Semantics
---------

MaskedTensor vs NumPy's MaskedArray semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy's `MaskedArray` has a few fundamental semantics differences from MaskedTensor.

1. Their factory function and basic definition inverts the mask (similar to `torch.nn.MHA`); that is, we (MaskedTensor)
uses `True` to denote "specified" and `False` to denote "unspecified", or "valid"/"invalid", whereas NumPy does the
opposite. 

2. Intersection semantics. In NumPy, if one of two elements are masked out, the resulting element will be
masked out as well -- in practice, they [apply the `logical_or` operator](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024).

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
and conveniently converting a MaskedTensor to a Tensor with masked values filled in.

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



Issues fixed
------------

:class:`MaskedTensor` fixes a number of different issues that have persisted across PyTorch for a number of years
by making masking a first class citizen.

Distinguishing between 0 and NaN gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One issue that :class:`torch.Tensor` runs into is the inability to distinguish between gradients that are not
defined (NaN) vs. gradients that are actually 0. By way of example, below are several different issues where
:class:`MaskedTensor` can resolve and/or work around the NaN gradient problem.

[Issue 10729](https://github.com/pytorch/pytorch/issues/10729) -- torch.where
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

[Issue 52248](https://github.com/pytorch/pytorch/issues/52248) -- another torch.where
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

[Issue 67180](https://github.com/pytorch/pytorch/issues/67180) -- torch.nansum and torch.nanmean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

[Issue 4132](https://github.com/pytorch/pytorch/issues/4132) -- when using mask, x/0 yields NaN grad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Safe Softmax
^^^^^^^^^^^^

One of the issues that frequently comes up is the necessity for a safe softmax -- that is, if there is an entire
batch that is "masked out" or consists entirely of padding (which, in the softmax case, translates to being set `-inf`),
then this will result in NaNs, which can leading to training divergence. For more detail on why this functionality
is necessary, please find
[Issue 55056 -- Feature Request for Safe Softmax](https://github.com/pytorch/pytorch/issues/55056).

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


[Issue 61474](https://github.com/pytorch/pytorch/issues/61474) Implementing missing torch.nan* operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the above issue, there is a request to add additional operators to cover the various `torch.nan*` applications,
such as `torch.nanmax`, `torch.nanmin`, etc.

In general, these problems lend themselves more naturally to masked semantics, so instead of introducing additional
operators, we propose using MaskedTensors instead. Since [nanmean](https://github.com/pytorch/pytorch/issues/21987)
has already landed, we can use it as a comparison point:

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

Furthermore, :class:`MaskedTensor` supports reductions when the input is fully masked out, which is equivalent
to the case above when the input Tensor is completely `nan`. `nanmean` would return `nan` (an ambiguous return value)
while MaskedTensor would more accurately indicate a masked out result.

    >>> x = torch.empty(16).fill_(float('nan'))
    >>> x
    tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    >>> torch.nanmean(x)
    tensor(nan)
    >>> torch.mean(masked_tensor(x, ~torch.isnan(x)))
    MaskedTensor(--, False)


Available Operators
-------------------

Unary Operators
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^





