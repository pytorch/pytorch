import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]

np.rec.fromarrays(1)  # type: ignore[call-overload]
np.rec.fromarrays([1, 2, 3], dtype=[("f8", "f8")], formats=["f8", "f8"])  # type: ignore[call-overload]

np.rec.fromrecords(AR_i8)  # type: ignore[arg-type]
np.rec.fromrecords([(1.5,)], dtype=[("f8", "f8")], formats=["f8", "f8"])  # type: ignore[call-overload]

np.rec.fromstring("string", dtype=[("f8", "f8")])  # type: ignore[call-overload]
np.rec.fromstring(b"bytes")  # type: ignore[call-overload]
np.rec.fromstring(b"(1.5,)", dtype=[("f8", "f8")], formats=["f8", "f8"])  # type: ignore[call-overload]

with open("test", "r") as f:
    np.rec.fromfile(f, dtype=[("f8", "f8")])  # type: ignore[call-overload]
