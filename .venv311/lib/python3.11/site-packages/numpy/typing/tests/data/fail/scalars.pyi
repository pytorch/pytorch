import sys
import numpy as np

f2: np.float16
f8: np.float64
c8: np.complex64

# Construction

np.float32(3j)  # type: ignore[arg-type]

# Technically the following examples are valid NumPy code. But they
# are not considered a best practice, and people who wish to use the
# stubs should instead do
#
# np.array([1.0, 0.0, 0.0], dtype=np.float32)
# np.array([], dtype=np.complex64)
#
# See e.g. the discussion on the mailing list
#
# https://mail.python.org/pipermail/numpy-discussion/2020-April/080566.html
#
# and the issue
#
# https://github.com/numpy/numpy-stubs/issues/41
#
# for more context.
np.float32([1.0, 0.0, 0.0])  # type: ignore[arg-type]
np.complex64([])  # type: ignore[call-overload]

# TODO: protocols (can't check for non-existent protocols w/ __getattr__)

np.datetime64(0)  # type: ignore[call-overload]

class A:
    def __float__(self) -> float: ...

np.int8(A())  # type: ignore[arg-type]
np.int16(A())  # type: ignore[arg-type]
np.int32(A())  # type: ignore[arg-type]
np.int64(A())  # type: ignore[arg-type]
np.uint8(A())  # type: ignore[arg-type]
np.uint16(A())  # type: ignore[arg-type]
np.uint32(A())  # type: ignore[arg-type]
np.uint64(A())  # type: ignore[arg-type]

np.void("test")  # type: ignore[call-overload]
np.void("test", dtype=None)  # type: ignore[call-overload]

np.generic(1)  # type: ignore[abstract]
np.number(1)  # type: ignore[abstract]
np.integer(1)  # type: ignore[abstract]
np.inexact(1)  # type: ignore[abstract]
np.character("test")  # type: ignore[abstract]
np.flexible(b"test")  # type: ignore[abstract]

np.float64(value=0.0)  # type: ignore[call-arg]
np.int64(value=0)  # type: ignore[call-arg]
np.uint64(value=0)  # type: ignore[call-arg]
np.complex128(value=0.0j)  # type: ignore[call-overload]
np.str_(value='bob')  # type: ignore[call-overload]
np.bytes_(value=b'test')  # type: ignore[call-overload]
np.void(value=b'test')  # type: ignore[call-overload]
np.bool(value=True)  # type: ignore[call-overload]
np.datetime64(value="2019")  # type: ignore[call-overload]
np.timedelta64(value=0)  # type: ignore[call-overload]

np.bytes_(b"hello", encoding='utf-8')  # type: ignore[call-overload]
np.str_("hello", encoding='utf-8')  # type: ignore[call-overload]

f8.item(1)  # type: ignore[call-overload]
f8.item((0, 1))  # type: ignore[arg-type]
f8.squeeze(axis=1)  # type: ignore[arg-type]
f8.squeeze(axis=(0, 1))  # type: ignore[arg-type]
f8.transpose(1)  # type: ignore[arg-type]

def func(a: np.float32) -> None: ...

func(f2)  # type: ignore[arg-type]
func(f8)  # type: ignore[arg-type]

c8.__getnewargs__()  # type: ignore[attr-defined]
f2.__getnewargs__()  # type: ignore[attr-defined]
f2.hex()  # type: ignore[attr-defined]
np.float16.fromhex("0x0.0p+0")  # type: ignore[attr-defined]
f2.__trunc__()  # type: ignore[attr-defined]
f2.__getformat__("float")  # type: ignore[attr-defined]
