from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

def func1(ar: npt.NDArray[Any], a: int) -> npt.NDArray[np.str_]: ...

def func2(ar: npt.NDArray[Any], a: float) -> float: ...

AR_b: npt.NDArray[np.bool]
AR_m: npt.NDArray[np.timedelta64]

AR_LIKE_b: list[bool]

np.eye(10, M=20.0)  # type: ignore[call-overload]
np.eye(10, k=2.5, dtype=int)  # type: ignore[call-overload]

np.diag(AR_b, k=0.5)  # type: ignore[call-overload]
np.diagflat(AR_b, k=0.5)  # type: ignore[call-overload]

np.tri(10, M=20.0)  # type: ignore[call-overload]
np.tri(10, k=2.5, dtype=int)  # type: ignore[call-overload]

np.tril(AR_b, k=0.5)  # type: ignore[call-overload]
np.triu(AR_b, k=0.5)  # type: ignore[call-overload]

np.vander(AR_m)  # type: ignore[arg-type]

np.histogram2d(AR_m)  # type: ignore[call-overload]

np.mask_indices(10, func1)  # type: ignore[arg-type]
np.mask_indices(10, func2, 10.5)  # type: ignore[arg-type]
