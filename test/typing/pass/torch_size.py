from typing_extensions import assert_type

from torch import Size


s1 = Size([1, 2, 3])
s2 = Size([1, 2, 3])


class ZeroIndex:
    def __index__(self) -> int:
        return 0

# assignability to tuple
tup: tuple[int, ...] = s1
# __getitem__
assert_type(s1[0], int)
assert_type(s1[ZeroIndex()], int)
assert_type(s1[:2], Size)
# __add__
assert_type(s1 + s2, Size)
assert_type(s1 + tup, Size)
assert_type(s1 + (), Size)
assert_type(s1 + (1,2), Size)
# __radd__
assert_type(tup + s1, Size)
assert_type(() + s1, Size)  # type: ignore[assert-type]
assert_type((1, 2) + s1, Size)  # type: ignore[assert-type]
# __mul__
assert_type(s1 * 3, Size)
assert_type(s1 * ZeroIndex(), Size)
assert_type(3 * s1, Size)
assert_type(ZeroIndex() * s1, Size)
