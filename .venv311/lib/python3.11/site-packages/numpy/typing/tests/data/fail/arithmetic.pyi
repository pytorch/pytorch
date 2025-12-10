from typing import Any

import numpy as np
import numpy.typing as npt

b_ = np.bool()
dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

AR_b: npt.NDArray[np.bool]
AR_u: npt.NDArray[np.uint32]
AR_i: npt.NDArray[np.int64]
AR_f: npt.NDArray[np.longdouble]
AR_c: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]

ANY: Any

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_m: list[np.timedelta64]
AR_LIKE_M: list[np.datetime64]

# Array subtraction

# NOTE: mypys `NoReturn` errors are, unfortunately, not that great
_1 = AR_b - AR_LIKE_b  # type: ignore[var-annotated]
_2 = AR_LIKE_b - AR_b  # type: ignore[var-annotated]
AR_i - bytes()  # type: ignore[operator]

AR_f - AR_LIKE_m  # type: ignore[operator]
AR_f - AR_LIKE_M  # type: ignore[operator]
AR_c - AR_LIKE_m  # type: ignore[operator]
AR_c - AR_LIKE_M  # type: ignore[operator]

AR_m - AR_LIKE_f  # type: ignore[operator]
AR_M - AR_LIKE_f  # type: ignore[operator]
AR_m - AR_LIKE_c  # type: ignore[operator]
AR_M - AR_LIKE_c  # type: ignore[operator]

AR_m - AR_LIKE_M  # type: ignore[operator]
AR_LIKE_m - AR_M  # type: ignore[operator]

# array floor division

AR_M // AR_LIKE_b  # type: ignore[operator]
AR_M // AR_LIKE_u  # type: ignore[operator]
AR_M // AR_LIKE_i  # type: ignore[operator]
AR_M // AR_LIKE_f  # type: ignore[operator]
AR_M // AR_LIKE_c  # type: ignore[operator]
AR_M // AR_LIKE_m  # type: ignore[operator]
AR_M // AR_LIKE_M  # type: ignore[operator]

AR_b // AR_LIKE_M  # type: ignore[operator]
AR_u // AR_LIKE_M  # type: ignore[operator]
AR_i // AR_LIKE_M  # type: ignore[operator]
AR_f // AR_LIKE_M  # type: ignore[operator]
AR_c // AR_LIKE_M  # type: ignore[operator]
AR_m // AR_LIKE_M  # type: ignore[operator]
AR_M // AR_LIKE_M  # type: ignore[operator]

_3 = AR_m // AR_LIKE_b  # type: ignore[var-annotated]
AR_m // AR_LIKE_c  # type: ignore[operator]

AR_b // AR_LIKE_m  # type: ignore[operator]
AR_u // AR_LIKE_m  # type: ignore[operator]
AR_i // AR_LIKE_m  # type: ignore[operator]
AR_f // AR_LIKE_m  # type: ignore[operator]
AR_c // AR_LIKE_m  # type: ignore[operator]

# regression tests for https://github.com/numpy/numpy/issues/28957
AR_c // 2  # type: ignore[operator]
AR_c // AR_i  # type: ignore[operator]
AR_c // AR_c  # type: ignore[operator]

# Array multiplication

AR_b *= AR_LIKE_u  # type: ignore[arg-type]
AR_b *= AR_LIKE_i  # type: ignore[arg-type]
AR_b *= AR_LIKE_f  # type: ignore[arg-type]
AR_b *= AR_LIKE_c  # type: ignore[arg-type]
AR_b *= AR_LIKE_m  # type: ignore[arg-type]

AR_u *= AR_LIKE_f  # type: ignore[arg-type]
AR_u *= AR_LIKE_c  # type: ignore[arg-type]
AR_u *= AR_LIKE_m  # type: ignore[arg-type]

AR_i *= AR_LIKE_f  # type: ignore[arg-type]
AR_i *= AR_LIKE_c  # type: ignore[arg-type]
AR_i *= AR_LIKE_m  # type: ignore[arg-type]

AR_f *= AR_LIKE_c  # type: ignore[arg-type]
AR_f *= AR_LIKE_m  # type: ignore[arg-type]

# Array power

AR_b **= AR_LIKE_b  # type: ignore[misc]
AR_b **= AR_LIKE_u  # type: ignore[misc]
AR_b **= AR_LIKE_i  # type: ignore[misc]
AR_b **= AR_LIKE_f  # type: ignore[misc]
AR_b **= AR_LIKE_c  # type: ignore[misc]

AR_u **= AR_LIKE_f  # type: ignore[arg-type]
AR_u **= AR_LIKE_c  # type: ignore[arg-type]

AR_i **= AR_LIKE_f  # type: ignore[arg-type]
AR_i **= AR_LIKE_c  # type: ignore[arg-type]

AR_f **= AR_LIKE_c  # type: ignore[arg-type]

# Scalars

b_ - b_  # type: ignore[operator]

dt + dt  # type: ignore[operator]
td - dt  # type: ignore[operator]
td % 1  # type: ignore[operator]
td / dt  # type: ignore[operator]
td % dt  # type: ignore[operator]

-b_  # type: ignore[operator]
+b_  # type: ignore[operator]
