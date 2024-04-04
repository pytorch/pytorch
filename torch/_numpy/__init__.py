# mypy: ignore-errors

from . import fft, linalg, random
from ._dtypes import *  # noqa: F403
from ._funcs import *  # noqa: F403
from ._getlimits import finfo, iinfo
from ._ndarray import (
    array,
    asarray,
    ascontiguousarray,
    can_cast,
    from_dlpack,
    ndarray,
    newaxis,
    result_type,
)
from ._ufuncs import *  # noqa: F403
from ._util import AxisError, UFuncTypeError

# from . import testing

alltrue = all
sometrue = any

inf = float("inf")
nan = float("nan")
from math import pi, e  # isort: skip

False_ = False
True_ = True
