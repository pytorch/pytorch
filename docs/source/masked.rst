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
==================

Construction
++++++++++++

There are a few different ways to construct a MaskedTensor:

- The first way is to directly invoke the MaskedTensor class
- The second (and our recommended way) is to use `masked_tensor` and `as_masked_tensor` factory functions, which are
analogous to `torch.tensor` and `torch.as_tensor`

Semantics
+++++++++

MaskedTensor vs NumPy's MaskedArray semantics
+++++++++++++++++++++++++++++++++++++++++++++

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

WIP. For more information, you can go to github.com/pytorch/maskedtensor for the source code
or http://pytorch.org/maskedtensor for a number of tutorials
