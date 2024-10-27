import sys

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

AR_U: npt.NDArray[np.str_]
AR_S: npt.NDArray[np.bytes_]

assert_type(np.strings.equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.not_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.not_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less_equal(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less_equal(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.greater(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.greater(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.less(AR_U, AR_U), npt.NDArray[np.bool])
assert_type(np.strings.less(AR_S, AR_S), npt.NDArray[np.bool])

assert_type(np.strings.multiply(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.multiply(AR_S, [5, 4, 3]), npt.NDArray[np.bytes_])

assert_type(np.strings.mod(AR_U, "test"), npt.NDArray[np.str_])
assert_type(np.strings.mod(AR_S, "test"), npt.NDArray[np.bytes_])

assert_type(np.strings.capitalize(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.capitalize(AR_S), npt.NDArray[np.bytes_])

assert_type(np.strings.center(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.center(AR_S, [2, 3, 4], b"a"), npt.NDArray[np.bytes_])

assert_type(np.strings.encode(AR_U), npt.NDArray[np.bytes_])
assert_type(np.strings.decode(AR_S), npt.NDArray[np.str_])

assert_type(np.strings.expandtabs(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.expandtabs(AR_S, tabsize=4), npt.NDArray[np.bytes_])

assert_type(np.strings.join(AR_U, "_"), npt.NDArray[np.str_])
assert_type(np.strings.join(AR_S, [b"_", b""]), npt.NDArray[np.bytes_])

assert_type(np.strings.ljust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.ljust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.rjust(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.rjust(AR_S, [4, 3, 1], fillchar=[b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

assert_type(np.strings.lstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.lstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.rstrip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.rstrip(AR_S, b"_"), npt.NDArray[np.bytes_])
assert_type(np.strings.strip(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.strip(AR_S, b"_"), npt.NDArray[np.bytes_])

assert_type(np.strings.count(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.count(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.strings.partition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.strings.partition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])
assert_type(np.strings.rpartition(AR_U, "\n"), npt.NDArray[np.str_])
assert_type(np.strings.rpartition(AR_S, [b"a", b"b", b"c"]), npt.NDArray[np.bytes_])

assert_type(np.strings.replace(AR_U, "_", "-"), npt.NDArray[np.str_])
assert_type(np.strings.replace(AR_S, [b"_", b""], [b"a", b"b"]), npt.NDArray[np.bytes_])

assert_type(np.strings.split(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.strings.split(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])
assert_type(np.strings.rsplit(AR_U, "_"), npt.NDArray[np.object_])
assert_type(np.strings.rsplit(AR_S, maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

assert_type(np.strings.splitlines(AR_U), npt.NDArray[np.object_])
assert_type(np.strings.splitlines(AR_S, keepends=[True, True, False]), npt.NDArray[np.object_])

assert_type(np.strings.swapcase(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.swapcase(AR_S), npt.NDArray[np.bytes_])

assert_type(np.strings.title(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.title(AR_S), npt.NDArray[np.bytes_])

assert_type(np.strings.upper(AR_U), npt.NDArray[np.str_])
assert_type(np.strings.upper(AR_S), npt.NDArray[np.bytes_])

assert_type(np.strings.zfill(AR_U, 5), npt.NDArray[np.str_])
assert_type(np.strings.zfill(AR_S, [2, 3, 4]), npt.NDArray[np.bytes_])

assert_type(np.strings.endswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.endswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(np.strings.startswith(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

assert_type(np.strings.find(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.find(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rfind(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.strings.index(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.index(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_U, "a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.strings.rindex(AR_S, [b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(np.strings.isalpha(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isalpha(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isalnum(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isalnum(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isdecimal(AR_U), npt.NDArray[np.bool])

assert_type(np.strings.isdigit(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isdigit(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.islower(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.islower(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isnumeric(AR_U), npt.NDArray[np.bool])

assert_type(np.strings.isspace(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isspace(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.istitle(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.istitle(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.isupper(AR_U), npt.NDArray[np.bool])
assert_type(np.strings.isupper(AR_S), npt.NDArray[np.bool])

assert_type(np.strings.str_len(AR_U), npt.NDArray[np.int_])
assert_type(np.strings.str_len(AR_S), npt.NDArray[np.int_])
