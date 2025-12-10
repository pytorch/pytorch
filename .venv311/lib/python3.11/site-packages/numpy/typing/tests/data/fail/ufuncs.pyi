import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]

np.sin.nin + "foo"  # type: ignore[operator]
np.sin(1, foo="bar")  # type: ignore[call-overload]

np.abs(None)  # type: ignore[call-overload]

np.add(1, 1, 1)  # type: ignore[call-overload]
np.add(1, 1, axis=0)  # type: ignore[call-overload]

np.matmul(AR_f8, AR_f8, where=True)  # type: ignore[call-overload]

np.frexp(AR_f8, out=None)  # type: ignore[call-overload]
np.frexp(AR_f8, out=AR_f8)  # type: ignore[call-overload]
