import sys
from typing import Any

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

assert_type(np.char.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.not_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.less_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.greater(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.char.less(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.char.multiply(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.multiply(AR_S, [5, 4, 3]), npt.NDArray[np.bytes_])

assert_type(np.char.mod(AR_U, "test"), npt.NDArray[np.str_])
assert_type(np.char.mod(AR_S, "test"), npt.NDArray[np.bytes_])

assert_type(np.char.capitalize(AR_U), npt.NDArray[np.str_])
assert_type(np.char.capitalize(AR_S), npt.NDArray[np.bytes_])

assert_type(np.char.center(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.center(AR_S, [2, 3, 4], b"a"), npt.NDArray[np.bytes_])

assert_type(np.char.encode(AR_U), npt.NDArray[np.bytes_])
assert_type(np.char.decode(AR_S), npt.NDArray[np.str_])

assert_type(np.char.expandtabs(AR_U), npt.NDArray[np.str_])
assert_type(np.char.expandtabs(AR_S, tabsize=4), npt.NDArray[np.bytes_])

assert_type(np.char.join(AR_U, "_"), npt.NDArray[np.str_])
assert_type(np.char.join(AR_S, [b"_", b""]), npt.NDArray[np.bytes_])

assert_type(np.char.ljust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.ljust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.char.rjust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.rjust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

assert_type(np.char.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.lstrip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])
assert_type(np.char.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.rstrip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])
assert_type(np.char.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.char.strip(AR_S, chars=b"_"), npt.NDArray[np.bytes_])

assert_type(np.char.partition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.char.partition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.char.rpartition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.char.rpartition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

assert_type(np.char.replace(AR_U, "_", "-"), npt.NDArray[np.str_])
assert_type(np.char.replace(AR_S, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])

assert_type(np.char.split(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.char.split(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])
assert_type(np.char.rsplit(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.char.rsplit(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

assert_type(np.char.splitlines(AR_U), npt.NDArray[np.object_])
assert_type(np.char.splitlines(AR_S, keepends=[True, True, False]), npt.NDArray[np.object_])

assert_type(np.char.swapcase(AR_U), npt.NDArray[np.str_])
assert_type(np.char.swapcase(AR_S), npt.NDArray[np.bytes_])

assert_type(np.char.title(AR_U), npt.NDArray[np.str_])
assert_type(np.char.title(AR_S), npt.NDArray[np.bytes_])

assert_type(np.char.upper(AR_U), npt.NDArray[np.str_])
assert_type(np.char.upper(AR_S), npt.NDArray[np.bytes_])

assert_type(np.char.zfill(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.char.zfill(AR_S, [2, 3, 4]), npt.NDArray[np.bytes_])

assert_type(np.char.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.char.count(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.char.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.char.endswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.char.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.char.startswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

assert_type(np.char.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.char.find(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.char.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.char.rfind(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.char.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.char.index(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.char.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.char.rindex(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.char.isalpha(AR_U), npt.NDArray[np.bool])
assert_type(np.char.isalpha(AR_S), npt.NDArray[np.bool])

assert_type(np.char.isalnum(AR_U), npt.NDArray[np.bool])
assert_type(np.char.isalnum(AR_S), npt.NDArray[np.bool])

assert_type(np.char.isdecimal(AR_U), npt.NDArray[np.bool])

assert_type(np.char.isdigit(AR_U), npt.NDArray[np.bool])
assert_type(np.char.isdigit(AR_S), npt.NDArray[np.bool])

assert_type(np.char.islower(AR_U), npt.NDArray[np.bool])
assert_type(np.char.islower(AR_S), npt.NDArray[np.bool])

assert_type(np.char.isnumeric(AR_U), npt.NDArray[np.bool])

assert_type(np.char.isspace(AR_U), npt.NDArray[np.bool])
assert_type(np.char.isspace(AR_S), npt.NDArray[np.bool])

assert_type(np.char.istitle(AR_U), npt.NDArray[np.bool])
assert_type(np.char.istitle(AR_S), npt.NDArray[np.bool])

assert_type(np.char.isupper(AR_U), npt.NDArray[np.bool])
assert_type(np.char.isupper(AR_S), npt.NDArray[np.bool])

assert_type(np.char.str_len(AR_U), npt.NDArray[np.int_])
assert_type(np.char.str_len(AR_S), npt.NDArray[np.int_])

assert_type(np.char.array(AR_U), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(np.char.array(AR_S, order="K"), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.array("bob", copy=True), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(np.char.array(b"bob", itemsize=5), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.array(1, unicode=False), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.array(1, unicode=True), np.char.chararray[Any, np.dtype[np.str_]])

assert_type(np.char.asarray(AR_U), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(np.char.asarray(AR_S, order="K"), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.asarray("bob"), np.char.chararray[Any, np.dtype[np.str_]])
assert_type(np.char.asarray(b"bob", itemsize=5), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.asarray(1, unicode=False), np.char.chararray[Any, np.dtype[np.bytes_]])
assert_type(np.char.asarray(1, unicode=True), np.char.chararray[Any, np.dtype[np.str_]])
