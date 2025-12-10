import numpy as np
import numpy.typing as npt

i8: np.int64

AR_b: npt.NDArray[np.bool]
AR_u1: npt.NDArray[np.uint8]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]

M: np.datetime64

AR_LIKE_f: list[float]

def func(a: int) -> None: ...

np.where(AR_b, 1)  # type: ignore[call-overload]

np.can_cast(AR_f8, 1)  # type: ignore[arg-type]

np.vdot(AR_M, AR_M)  # type: ignore[arg-type]

np.copyto(AR_LIKE_f, AR_f8)  # type: ignore[arg-type]

np.putmask(AR_LIKE_f, [True, True, False], 1.5)  # type: ignore[arg-type]

np.packbits(AR_f8)  # type: ignore[arg-type]
np.packbits(AR_u1, bitorder=">")  # type: ignore[arg-type]

np.unpackbits(AR_i8)  # type: ignore[arg-type]
np.unpackbits(AR_u1, bitorder=">")  # type: ignore[arg-type]

np.shares_memory(1, 1, max_work=i8)  # type: ignore[arg-type]
np.may_share_memory(1, 1, max_work=i8)  # type: ignore[arg-type]

np.arange(stop=10)  # type: ignore[call-overload]

np.datetime_data(int)  # type: ignore[arg-type]

np.busday_offset("2012", 10)  # type: ignore[call-overload]

np.datetime_as_string("2012")  # type: ignore[call-overload]

np.char.compare_chararrays("a", b"a", "==", False)  # type: ignore[call-overload]

np.nested_iters([AR_i8, AR_i8])  # type: ignore[call-arg]
np.nested_iters([AR_i8, AR_i8], 0)  # type: ignore[arg-type]
np.nested_iters([AR_i8, AR_i8], [0])  # type: ignore[list-item]
np.nested_iters([AR_i8, AR_i8], [[0], [1]], flags=["test"])  # type: ignore[list-item]
np.nested_iters([AR_i8, AR_i8], [[0], [1]], op_flags=[["test"]])  # type: ignore[list-item]
np.nested_iters([AR_i8, AR_i8], [[0], [1]], buffersize=1.0)  # type: ignore[arg-type]
