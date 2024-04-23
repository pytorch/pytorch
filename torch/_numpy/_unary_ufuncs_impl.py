# mypy: ignore-errors

"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""

import torch

from torch import (  # noqa: F401
    absolute as fabs,  # noqa: F401
    arccos,  # noqa: F401
    arccosh,  # noqa: F401
    arcsin,  # noqa: F401
    arcsinh,  # noqa: F401
    arctan,  # noqa: F401
    arctanh,  # noqa: F401
    bitwise_not,  # noqa: F401
    bitwise_not as invert,  # noqa: F401
    ceil,  # noqa: F401
    conj_physical as conjugate,  # noqa: F401
    cos,  # noqa: F401
    cosh,  # noqa: F401
    deg2rad,  # noqa: F401
    deg2rad as radians,  # noqa: F401
    exp,  # noqa: F401
    exp2,  # noqa: F401
    expm1,  # noqa: F401
    floor,  # noqa: F401
    isfinite,  # noqa: F401
    isinf,  # noqa: F401
    isnan,  # noqa: F401
    log,  # noqa: F401
    log10,  # noqa: F401
    log1p,  # noqa: F401
    log2,  # noqa: F401
    logical_not,  # noqa: F401
    negative,  # noqa: F401
    rad2deg,  # noqa: F401
    rad2deg as degrees,  # noqa: F401
    reciprocal,  # noqa: F401
    round as fix,  # noqa: F401
    round as rint,  # noqa: F401
    sign,  # noqa: F401
    signbit,  # noqa: F401
    sin,  # noqa: F401
    sinh,  # noqa: F401
    sqrt,  # noqa: F401
    square,  # noqa: F401
    tan,  # noqa: F401
    tanh,  # noqa: F401
    trunc,  # noqa: F401
)


# special cases: torch does not export these names
def cbrt(x):
    return torch.pow(x, 1 / 3)


def positive(x):
    return +x


def absolute(x):
    # work around torch.absolute not impl for bools
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)


# TODO set __name__ and __qualname__
abs = absolute
conj = conjugate
