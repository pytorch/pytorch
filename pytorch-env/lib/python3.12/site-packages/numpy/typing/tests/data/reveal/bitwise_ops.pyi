import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy._typing import _64Bit, _32Bit

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

i8 = np.int64(1)
u8 = np.uint64(1)

i4 = np.int32(1)
u4 = np.uint32(1)

b_ = np.bool(1)

b = bool(1)
i = int(1)

AR = np.array([0, 1, 2], dtype=np.int32)
AR.setflags(write=False)


assert_type(i8 << i8, np.int64)
assert_type(i8 >> i8, np.int64)
assert_type(i8 | i8, np.int64)
assert_type(i8 ^ i8, np.int64)
assert_type(i8 & i8, np.int64)

assert_type(i8 << AR, npt.NDArray[np.signedinteger[Any]])
assert_type(i8 >> AR, npt.NDArray[np.signedinteger[Any]])
assert_type(i8 | AR, npt.NDArray[np.signedinteger[Any]])
assert_type(i8 ^ AR, npt.NDArray[np.signedinteger[Any]])
assert_type(i8 & AR, npt.NDArray[np.signedinteger[Any]])

assert_type(i4 << i4, np.int32)
assert_type(i4 >> i4, np.int32)
assert_type(i4 | i4, np.int32)
assert_type(i4 ^ i4, np.int32)
assert_type(i4 & i4, np.int32)

assert_type(i8 << i4, np.signedinteger[_32Bit | _64Bit])
assert_type(i8 >> i4, np.signedinteger[_32Bit | _64Bit])
assert_type(i8 | i4, np.signedinteger[_32Bit | _64Bit])
assert_type(i8 ^ i4, np.signedinteger[_32Bit | _64Bit])
assert_type(i8 & i4, np.signedinteger[_32Bit | _64Bit])

assert_type(i8 << b_, np.int64)
assert_type(i8 >> b_, np.int64)
assert_type(i8 | b_, np.int64)
assert_type(i8 ^ b_, np.int64)
assert_type(i8 & b_, np.int64)

assert_type(i8 << b, np.int64)
assert_type(i8 >> b, np.int64)
assert_type(i8 | b, np.int64)
assert_type(i8 ^ b, np.int64)
assert_type(i8 & b, np.int64)

assert_type(u8 << u8, np.uint64)
assert_type(u8 >> u8, np.uint64)
assert_type(u8 | u8, np.uint64)
assert_type(u8 ^ u8, np.uint64)
assert_type(u8 & u8, np.uint64)

assert_type(u8 << AR, npt.NDArray[np.signedinteger[Any]])
assert_type(u8 >> AR, npt.NDArray[np.signedinteger[Any]])
assert_type(u8 | AR, npt.NDArray[np.signedinteger[Any]])
assert_type(u8 ^ AR, npt.NDArray[np.signedinteger[Any]])
assert_type(u8 & AR, npt.NDArray[np.signedinteger[Any]])

assert_type(u4 << u4, np.uint32)
assert_type(u4 >> u4, np.uint32)
assert_type(u4 | u4, np.uint32)
assert_type(u4 ^ u4, np.uint32)
assert_type(u4 & u4, np.uint32)

assert_type(u4 << i4, np.signedinteger[Any])
assert_type(u4 >> i4, np.signedinteger[Any])
assert_type(u4 | i4, np.signedinteger[Any])
assert_type(u4 ^ i4, np.signedinteger[Any])
assert_type(u4 & i4, np.signedinteger[Any])

assert_type(u4 << i, np.signedinteger[Any])
assert_type(u4 >> i, np.signedinteger[Any])
assert_type(u4 | i, np.signedinteger[Any])
assert_type(u4 ^ i, np.signedinteger[Any])
assert_type(u4 & i, np.signedinteger[Any])

assert_type(u8 << b_, np.uint64)
assert_type(u8 >> b_, np.uint64)
assert_type(u8 | b_, np.uint64)
assert_type(u8 ^ b_, np.uint64)
assert_type(u8 & b_, np.uint64)

assert_type(u8 << b, np.uint64)
assert_type(u8 >> b, np.uint64)
assert_type(u8 | b, np.uint64)
assert_type(u8 ^ b, np.uint64)
assert_type(u8 & b, np.uint64)

assert_type(b_ << b_, np.int8)
assert_type(b_ >> b_, np.int8)
assert_type(b_ | b_, np.bool)
assert_type(b_ ^ b_, np.bool)
assert_type(b_ & b_, np.bool)

assert_type(b_ << AR, npt.NDArray[np.signedinteger[Any]])
assert_type(b_ >> AR, npt.NDArray[np.signedinteger[Any]])
assert_type(b_ | AR, npt.NDArray[np.signedinteger[Any]])
assert_type(b_ ^ AR, npt.NDArray[np.signedinteger[Any]])
assert_type(b_ & AR, npt.NDArray[np.signedinteger[Any]])

assert_type(b_ << b, np.int8)
assert_type(b_ >> b, np.int8)
assert_type(b_ | b, np.bool)
assert_type(b_ ^ b, np.bool)
assert_type(b_ & b, np.bool)

assert_type(b_ << i, np.int_)
assert_type(b_ >> i, np.int_)
assert_type(b_ | i, np.int_)
assert_type(b_ ^ i, np.int_)
assert_type(b_ & i, np.int_)

assert_type(~i8, np.int64)
assert_type(~i4, np.int32)
assert_type(~u8, np.uint64)
assert_type(~u4, np.uint32)
assert_type(~b_, np.bool)
assert_type(~AR, npt.NDArray[np.int32])
