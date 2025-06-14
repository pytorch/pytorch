# mypy: enable-error-code=unused-ignore

from typing_extensions import assert_type, Never

from torch import Size


class ZeroIndex:
    def __index__(self) -> int:
        return 0


tup0: tuple[()] = ()
tup1: tuple[int] = (1,)
tup2: tuple[int, int] = (1, 2)
tupN: tuple[int, int, int] = (1, 2, 3)
tupX: tuple[Never, ...] = tuple()
s = Size([1, 2, 3])

# assignability to tuple
t: tuple[int, ...] = s

# __getitem__
assert_type(s[0], int)
assert_type(s[ZeroIndex()], int)
assert_type(s[:2], Size)
# __add__
assert_type(s + s, Size)
assert_type(s + tup0, Size)
assert_type(s + tup1, Size)
assert_type(s + tup2, Size)
assert_type(s + tupN, Size)
assert_type(s + tupX, Size)
# __radd__
# NOTE: currently incorrect inference, see: https://github.com/python/mypy/issues/19006
assert_type(tup0 + s, Size)  # type: ignore[assert-type]
assert_type(tup1 + s, Size)  # type: ignore[assert-type]
assert_type(tup2 + s, Size)  # type: ignore[assert-type]
assert_type(tupN + s, Size)  # type: ignore[assert-type]
assert_type(tupX + s, Size)  # type: ignore[assert-type]
# __mul__
assert_type(s * 3, Size)
assert_type(s * ZeroIndex(), Size)
assert_type(3 * s, Size)
assert_type(ZeroIndex() * s, Size)
