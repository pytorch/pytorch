.. currentmodule:: torch

Reductions
============

As an intro to masked reductions, please find the document on reduction semantics `here <https://github.com/pytorch/rfcs/pull/27>`_.
In general, an operator is a reduction operator if it reduces one or more dimensions of the input tensor to a single value
(e.g. think :mod:`nanmean` or :mod:`nansum`)

MaskedTensor currently supports the following reductions:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sum
    mean
    amin
    amax
    prod

The next ops to be implemented will be in the ones in the `MaskedTensor Reduction RFC <https://github.com/pytorch/rfcs/blob/8cb9ce7fe84724099138dc281080c74ad1bc2cca/RFC-0016-Masked-reductions-and-normalizations.md>`_.
If you would like any others implemented, please create a feature request with proposed input/output semantics!
