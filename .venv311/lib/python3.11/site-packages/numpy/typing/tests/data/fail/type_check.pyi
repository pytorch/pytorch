import numpy as np
import numpy.typing as npt

DTYPE_i8: np.dtype[np.int64]

np.mintypecode(DTYPE_i8)  # type: ignore[arg-type]
np.iscomplexobj(DTYPE_i8)  # type: ignore[arg-type]
np.isrealobj(DTYPE_i8)  # type: ignore[arg-type]

np.typename(DTYPE_i8)  # type: ignore[call-overload]
np.typename("invalid")  # type: ignore[call-overload]

np.common_type(np.timedelta64())  # type: ignore[arg-type]
