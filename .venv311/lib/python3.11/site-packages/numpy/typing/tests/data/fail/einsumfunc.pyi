import numpy as np
import numpy.typing as npt

AR_i: npt.NDArray[np.int64]
AR_f: npt.NDArray[np.float64]
AR_m: npt.NDArray[np.timedelta64]
AR_U: npt.NDArray[np.str_]

np.einsum("i,i->i", AR_i, AR_m)  # type: ignore[arg-type]
np.einsum("i,i->i", AR_f, AR_f, dtype=np.int32)  # type: ignore[arg-type]
np.einsum("i,i->i", AR_i, AR_i, out=AR_U)  # type: ignore[type-var]
np.einsum("i,i->i", AR_i, AR_i, out=AR_U, casting="unsafe")  # type: ignore[call-overload]
