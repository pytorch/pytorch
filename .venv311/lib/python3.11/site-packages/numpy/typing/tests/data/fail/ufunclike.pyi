import numpy as np
import numpy.typing as npt

AR_c: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]

np.fix(AR_c)  # type: ignore[arg-type]
np.fix(AR_m)  # type: ignore[arg-type]
np.fix(AR_M)  # type: ignore[arg-type]

np.isposinf(AR_c)  # type: ignore[arg-type]
np.isposinf(AR_m)  # type: ignore[arg-type]
np.isposinf(AR_M)  # type: ignore[arg-type]
np.isposinf(AR_O)  # type: ignore[arg-type]

np.isneginf(AR_c)  # type: ignore[arg-type]
np.isneginf(AR_m)  # type: ignore[arg-type]
np.isneginf(AR_M)  # type: ignore[arg-type]
np.isneginf(AR_O)  # type: ignore[arg-type]
