"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""

import torch

# renames
from torch import absolute as fabs
from torch import arccos, arccosh, arcsin, arcsinh, arctan, arctanh
from torch import bitwise_not
from torch import bitwise_not as invert
from torch import ceil
from torch import conj_physical as conjugate
from torch import cos, cosh
from torch import deg2rad
from torch import deg2rad as radians
from torch import (
    exp,
    exp2,
    expm1,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    log1p,
    log2,
    log10,
    logical_not,
    negative,
)
from torch import rad2deg
from torch import rad2deg as degrees
from torch import reciprocal
from torch import round as fix
from torch import round as rint
from torch import sign, signbit, sin, sinh, sqrt, square, tan, tanh, trunc


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


abs = absolute
conj = conjugate
