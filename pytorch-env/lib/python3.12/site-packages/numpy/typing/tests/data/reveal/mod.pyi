import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy._typing import _32Bit, _64Bit

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

td = np.timedelta64(0, "D")
b_ = np.bool()

b = bool()
f = float()
i = int()

AR_b: npt.NDArray[np.bool]
AR_m: npt.NDArray[np.timedelta64]

# Time structures

assert_type(td % td, np.timedelta64)
assert_type(AR_m % td, npt.NDArray[np.timedelta64])
assert_type(td % AR_m, npt.NDArray[np.timedelta64])

assert_type(divmod(td, td), tuple[np.int64, np.timedelta64])
assert_type(divmod(AR_m, td), tuple[npt.NDArray[np.int64], npt.NDArray[np.timedelta64]])
assert_type(divmod(td, AR_m), tuple[npt.NDArray[np.int64], npt.NDArray[np.timedelta64]])

# Bool

assert_type(b_ % b, np.int8)
assert_type(b_ % i, np.int_)
assert_type(b_ % f, np.float64)
assert_type(b_ % b_, np.int8)
assert_type(b_ % i8, np.int64)
assert_type(b_ % u8, np.uint64)
assert_type(b_ % f8, np.float64)
assert_type(b_ % AR_b, npt.NDArray[np.int8])

assert_type(divmod(b_, b), tuple[np.int8, np.int8])
assert_type(divmod(b_, i), tuple[np.int_, np.int_])
assert_type(divmod(b_, f), tuple[np.float64, np.float64])
assert_type(divmod(b_, b_), tuple[np.int8, np.int8])
assert_type(divmod(b_, i8), tuple[np.int64, np.int64])
assert_type(divmod(b_, u8), tuple[np.uint64, np.uint64])
assert_type(divmod(b_, f8), tuple[np.float64, np.float64])
assert_type(divmod(b_, AR_b), tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]])

assert_type(b % b_, np.int8)
assert_type(i % b_, np.int_)
assert_type(f % b_, np.float64)
assert_type(b_ % b_, np.int8)
assert_type(i8 % b_, np.int64)
assert_type(u8 % b_, np.uint64)
assert_type(f8 % b_, np.float64)
assert_type(AR_b % b_, npt.NDArray[np.int8])

assert_type(divmod(b, b_), tuple[np.int8, np.int8])
assert_type(divmod(i, b_), tuple[np.int_, np.int_])
assert_type(divmod(f, b_), tuple[np.float64, np.float64])
assert_type(divmod(b_, b_), tuple[np.int8, np.int8])
assert_type(divmod(i8, b_), tuple[np.int64, np.int64])
assert_type(divmod(u8, b_), tuple[np.uint64, np.uint64])
assert_type(divmod(f8, b_), tuple[np.float64, np.float64])
assert_type(divmod(AR_b, b_), tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]])

# int

assert_type(i8 % b, np.int64)
assert_type(i8 % f, np.float64)
assert_type(i8 % i8, np.int64)
assert_type(i8 % f8, np.float64)
assert_type(i4 % i8, np.signedinteger[_32Bit | _64Bit])
assert_type(i4 % f8, np.floating[_32Bit | _64Bit])
assert_type(i4 % i4, np.int32)
assert_type(i4 % f4, np.float32)
assert_type(i8 % AR_b, npt.NDArray[np.signedinteger[Any]])

assert_type(divmod(i8, b), tuple[np.int64, np.int64])
assert_type(divmod(i8, f), tuple[np.float64, np.float64])
assert_type(divmod(i8, i8), tuple[np.int64, np.int64])
assert_type(divmod(i8, f8), tuple[np.float64, np.float64])
assert_type(divmod(i8, i4), tuple[np.signedinteger[_32Bit | _64Bit], np.signedinteger[_32Bit | _64Bit]])
assert_type(divmod(i8, f4), tuple[np.floating[_32Bit | _64Bit], np.floating[_32Bit | _64Bit]])
assert_type(divmod(i4, i4), tuple[np.int32, np.int32])
assert_type(divmod(i4, f4), tuple[np.float32, np.float32])
assert_type(divmod(i8, AR_b), tuple[npt.NDArray[np.signedinteger[Any]], npt.NDArray[np.signedinteger[Any]]])

assert_type(b % i8, np.int64)
assert_type(f % i8, np.float64)
assert_type(i8 % i8, np.int64)
assert_type(f8 % i8, np.float64)
assert_type(i8 % i4, np.signedinteger[_32Bit | _64Bit])
assert_type(f8 % i4, np.floating[_32Bit | _64Bit])
assert_type(i4 % i4, np.int32)
assert_type(f4 % i4, np.float32)
assert_type(AR_b % i8, npt.NDArray[np.signedinteger[Any]])

assert_type(divmod(b, i8), tuple[np.int64, np.int64])
assert_type(divmod(f, i8), tuple[np.float64, np.float64])
assert_type(divmod(i8, i8), tuple[np.int64, np.int64])
assert_type(divmod(f8, i8), tuple[np.float64, np.float64])
assert_type(divmod(i4, i8), tuple[np.signedinteger[_32Bit | _64Bit], np.signedinteger[_32Bit | _64Bit]])
assert_type(divmod(f4, i8), tuple[np.floating[_32Bit | _64Bit], np.floating[_32Bit | _64Bit]])
assert_type(divmod(i4, i4), tuple[np.int32, np.int32])
assert_type(divmod(f4, i4), tuple[np.float32, np.float32])
assert_type(divmod(AR_b, i8), tuple[npt.NDArray[np.signedinteger[Any]], npt.NDArray[np.signedinteger[Any]]])

# float

assert_type(f8 % b, np.float64)
assert_type(f8 % f, np.float64)
assert_type(i8 % f4, np.floating[_32Bit | _64Bit])
assert_type(f4 % f4, np.float32)
assert_type(f8 % AR_b, npt.NDArray[np.floating[Any]])

assert_type(divmod(f8, b), tuple[np.float64, np.float64])
assert_type(divmod(f8, f), tuple[np.float64, np.float64])
assert_type(divmod(f8, f8), tuple[np.float64, np.float64])
assert_type(divmod(f8, f4), tuple[np.floating[_32Bit | _64Bit], np.floating[_32Bit | _64Bit]])
assert_type(divmod(f4, f4), tuple[np.float32, np.float32])
assert_type(divmod(f8, AR_b), tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]])

assert_type(b % f8, np.float64)
assert_type(f % f8, np.float64)
assert_type(f8 % f8, np.float64)
assert_type(f8 % f8, np.float64)
assert_type(f4 % f4, np.float32)
assert_type(AR_b % f8, npt.NDArray[np.floating[Any]])

assert_type(divmod(b, f8), tuple[np.float64, np.float64])
assert_type(divmod(f, f8), tuple[np.float64, np.float64])
assert_type(divmod(f8, f8), tuple[np.float64, np.float64])
assert_type(divmod(f4, f8), tuple[np.floating[_32Bit | _64Bit], np.floating[_32Bit | _64Bit]])
assert_type(divmod(f4, f4), tuple[np.float32, np.float32])
assert_type(divmod(AR_b, f8), tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]])
