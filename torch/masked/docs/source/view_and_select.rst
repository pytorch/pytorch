.. currentmodule:: torch

View and select functions
=========================

In 0.11, we've included a number of different view and select functions -- under the hood, these are implemented
as pass through functions -- i.e. functions that apply the operator to both the mask and the data.

By way of example, consider :mod:`select`; this operation can be applied to both the data
and the mask of a :mod:`MaskedTensor`, and the result will then be wrapped into a new :mod:`MaskedTensor`.

A quick example of this:

::

    >>> data = torch.arange(12, dtype=torch.float).reshape((3,4))
    >>> mask = torch.tensor([
            [True, False, False, True],
            [False, True, False, False],
            [True, True, True, True]])
    >>> mt = masked_tensor(data, mask)
    >>> data.select(0, 1)
    tensor([4., 5., 6., 7.])
    >>> mask.select(0, 1)
    tensor([False,  True, False, False])
    >>> mt.select(0, 1)
    masked_tensor(
    [      --,   5.0000,       --,       --]
    )

Below is a list of the ops that are currently supported:

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
