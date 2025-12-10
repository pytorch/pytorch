"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

import numpy as np
import numpy.typing as npt

f8: np.float64
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]
AR_b: npt.NDArray[np.bool]

ctypes_obj = AR_f8.ctypes

f8.argpartition(0)  # type: ignore[attr-defined]
f8.diagonal()  # type: ignore[attr-defined]
f8.dot(1)  # type: ignore[attr-defined]
f8.nonzero()  # type: ignore[attr-defined]
f8.partition(0)  # type: ignore[attr-defined]
f8.put(0, 2)  # type: ignore[attr-defined]
f8.setfield(2, np.float64)  # type: ignore[attr-defined]
f8.sort()  # type: ignore[attr-defined]
f8.trace()  # type: ignore[attr-defined]

AR_M.__complex__()  # type: ignore[misc]
AR_b.__index__()  # type: ignore[misc]

AR_f8[1.5]  # type: ignore[call-overload]
AR_f8["field_a"]  # type: ignore[call-overload]
AR_f8[["field_a", "field_b"]]  # type: ignore[index]

AR_f8.__array_finalize__(object())  # type: ignore[arg-type]
