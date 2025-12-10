from typing import Any, TypeAlias, assert_type

import numpy as np
import numpy.typing as npt

_BytesCharArray: TypeAlias = np.char.chararray[tuple[Any, ...], np.dtype[np.bytes_]]
_StrCharArray: TypeAlias = np.char.chararray[tuple[Any, ...], np.dtype[np.str_]]

AR_U: _StrCharArray
AR_S: _BytesCharArray

assert_type(AR_U == AR_U, npt.NDArray[np.bool])
assert_type(AR_S == AR_S, npt.NDArray[np.bool])

assert_type(AR_U != AR_U, npt.NDArray[np.bool])
assert_type(AR_S != AR_S, npt.NDArray[np.bool])

assert_type(AR_U >= AR_U, npt.NDArray[np.bool])
assert_type(AR_S >= AR_S, npt.NDArray[np.bool])

assert_type(AR_U <= AR_U, npt.NDArray[np.bool])
assert_type(AR_S <= AR_S, npt.NDArray[np.bool])

assert_type(AR_U > AR_U, npt.NDArray[np.bool])
assert_type(AR_S > AR_S, npt.NDArray[np.bool])

assert_type(AR_U < AR_U, npt.NDArray[np.bool])
assert_type(AR_S < AR_S, npt.NDArray[np.bool])

assert_type(AR_U * 5, _StrCharArray)
assert_type(AR_S * [5], _BytesCharArray)

assert_type(AR_U % "test", _StrCharArray)
assert_type(AR_S % b"test", _BytesCharArray)

assert_type(AR_U.capitalize(), _StrCharArray)
assert_type(AR_S.capitalize(), _BytesCharArray)

assert_type(AR_U.center(5), _StrCharArray)
assert_type(AR_S.center([2, 3, 4], b"a"), _BytesCharArray)

assert_type(AR_U.encode(), _BytesCharArray)
assert_type(AR_S.decode(), _StrCharArray)

assert_type(AR_U.expandtabs(), _StrCharArray)
assert_type(AR_S.expandtabs(tabsize=4), _BytesCharArray)

assert_type(AR_U.join("_"), _StrCharArray)
assert_type(AR_S.join([b"_", b""]), _BytesCharArray)

assert_type(AR_U.ljust(5), _StrCharArray)
assert_type(AR_S.ljust([4, 3, 1], fillchar=[b"a", b"b", b"c"]), _BytesCharArray)
assert_type(AR_U.rjust(5), _StrCharArray)
assert_type(AR_S.rjust([4, 3, 1], fillchar=[b"a", b"b", b"c"]), _BytesCharArray)

assert_type(AR_U.lstrip(), _StrCharArray)
assert_type(AR_S.lstrip(chars=b"_"), _BytesCharArray)
assert_type(AR_U.rstrip(), _StrCharArray)
assert_type(AR_S.rstrip(chars=b"_"), _BytesCharArray)
assert_type(AR_U.strip(), _StrCharArray)
assert_type(AR_S.strip(chars=b"_"), _BytesCharArray)

assert_type(AR_U.partition("\n"), _StrCharArray)
assert_type(AR_S.partition([b"a", b"b", b"c"]), _BytesCharArray)
assert_type(AR_U.rpartition("\n"), _StrCharArray)
assert_type(AR_S.rpartition([b"a", b"b", b"c"]), _BytesCharArray)

assert_type(AR_U.replace("_", "-"), _StrCharArray)
assert_type(AR_S.replace([b"_", b""], [b"a", b"b"]), _BytesCharArray)

assert_type(AR_U.split("_"), npt.NDArray[np.object_])
assert_type(AR_S.split(maxsplit=[1, 2, 3]), npt.NDArray[np.object_])
assert_type(AR_U.rsplit("_"), npt.NDArray[np.object_])
assert_type(AR_S.rsplit(maxsplit=[1, 2, 3]), npt.NDArray[np.object_])

assert_type(AR_U.splitlines(), npt.NDArray[np.object_])
assert_type(AR_S.splitlines(keepends=[True, True, False]), npt.NDArray[np.object_])

assert_type(AR_U.swapcase(), _StrCharArray)
assert_type(AR_S.swapcase(), _BytesCharArray)

assert_type(AR_U.title(), _StrCharArray)
assert_type(AR_S.title(), _BytesCharArray)

assert_type(AR_U.upper(), _StrCharArray)
assert_type(AR_S.upper(), _BytesCharArray)

assert_type(AR_U.zfill(5), _StrCharArray)
assert_type(AR_S.zfill([2, 3, 4]), _BytesCharArray)

assert_type(AR_U.count("a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(AR_S.count([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(AR_U.endswith("a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(AR_S.endswith([b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])
assert_type(AR_U.startswith("a", start=[1, 2, 3]), npt.NDArray[np.bool])
assert_type(AR_S.startswith([b"a", b"b", b"c"], end=9), npt.NDArray[np.bool])

assert_type(AR_U.find("a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(AR_S.find([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(AR_U.rfind("a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(AR_S.rfind([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(AR_U.index("a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(AR_S.index([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])
assert_type(AR_U.rindex("a", start=[1, 2, 3]), npt.NDArray[np.int_])
assert_type(AR_S.rindex([b"a", b"b", b"c"], end=9), npt.NDArray[np.int_])

assert_type(AR_U.isalpha(), npt.NDArray[np.bool])
assert_type(AR_S.isalpha(), npt.NDArray[np.bool])

assert_type(AR_U.isalnum(), npt.NDArray[np.bool])
assert_type(AR_S.isalnum(), npt.NDArray[np.bool])

assert_type(AR_U.isdecimal(), npt.NDArray[np.bool])
assert_type(AR_S.isdecimal(), npt.NDArray[np.bool])

assert_type(AR_U.isdigit(), npt.NDArray[np.bool])
assert_type(AR_S.isdigit(), npt.NDArray[np.bool])

assert_type(AR_U.islower(), npt.NDArray[np.bool])
assert_type(AR_S.islower(), npt.NDArray[np.bool])

assert_type(AR_U.isnumeric(), npt.NDArray[np.bool])
assert_type(AR_S.isnumeric(), npt.NDArray[np.bool])

assert_type(AR_U.isspace(), npt.NDArray[np.bool])
assert_type(AR_S.isspace(), npt.NDArray[np.bool])

assert_type(AR_U.istitle(), npt.NDArray[np.bool])
assert_type(AR_S.istitle(), npt.NDArray[np.bool])

assert_type(AR_U.isupper(), npt.NDArray[np.bool])
assert_type(AR_S.isupper(), npt.NDArray[np.bool])

assert_type(AR_U.__array_finalize__(object()), None)
assert_type(AR_S.__array_finalize__(object()), None)
