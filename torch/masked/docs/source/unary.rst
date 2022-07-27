.. currentmodule:: torch

Unary Operations
================

Unary operators are operators that only contain only a single input. Applying them to MaskedTensors
is relatively straightforward at the moment: if the data is masked out at a given index, we apply the operator,
and if not, then we'll continue to mask out the data.

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
