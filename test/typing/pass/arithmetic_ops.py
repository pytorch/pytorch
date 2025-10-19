from typing import Union
from typing_extensions import assert_type, TypeAlias

from torch import randn, Tensor


# Test deduced types of arithmetic operations between tensors, ints, floats and bools
# The expected type should always be `Tensor`, but isn't.
# See https://github.com/pytorch/pytorch/issues/145838

TENSOR, INT, FLOAT, BOOL = randn(3), 2, 1.5, True

#
# Unary ops
#

assert_type(+TENSOR, Tensor)
assert_type(-TENSOR, Tensor)
assert_type(~TENSOR, Tensor)

#
# Binary ops that return a boolean
#

# Operator ==
assert_type(TENSOR == TENSOR, Tensor)
assert_type(TENSOR == BOOL, Tensor)
assert_type(BOOL == TENSOR, bool)  # Should be Tensor
assert_type(TENSOR == INT, Tensor)
assert_type(INT == TENSOR, bool)  # Should be Tensor
assert_type(TENSOR == FLOAT, Tensor)
assert_type(FLOAT == TENSOR, bool)  # Should be Tensor

# Operator !=
assert_type(TENSOR != TENSOR, Tensor)
assert_type(TENSOR != BOOL, Tensor)
assert_type(BOOL != TENSOR, bool)  # Should be Tensor
assert_type(TENSOR != INT, Tensor)
assert_type(INT != TENSOR, bool)  # Should be Tensor
assert_type(TENSOR != FLOAT, Tensor)
assert_type(FLOAT != TENSOR, bool)  # Should be Tensor

# Operator <
assert_type(TENSOR < TENSOR, Tensor)
assert_type(TENSOR < BOOL, Tensor)
assert_type(BOOL < TENSOR, Tensor)
assert_type(TENSOR < INT, Tensor)
assert_type(INT < TENSOR, Tensor)
assert_type(TENSOR < FLOAT, Tensor)
assert_type(FLOAT < TENSOR, Tensor)

# Operator >
assert_type(TENSOR > TENSOR, Tensor)
assert_type(TENSOR > BOOL, Tensor)
assert_type(BOOL > TENSOR, Tensor)
assert_type(TENSOR > INT, Tensor)
assert_type(INT > TENSOR, Tensor)
assert_type(TENSOR > FLOAT, Tensor)
assert_type(FLOAT > TENSOR, Tensor)

# Operator <=
assert_type(TENSOR <= TENSOR, Tensor)
assert_type(TENSOR <= BOOL, Tensor)
assert_type(BOOL <= TENSOR, Tensor)
assert_type(TENSOR <= INT, Tensor)
assert_type(INT <= TENSOR, Tensor)
assert_type(TENSOR <= FLOAT, Tensor)
assert_type(FLOAT <= TENSOR, Tensor)

# Operator >=
assert_type(TENSOR >= TENSOR, Tensor)
assert_type(TENSOR >= BOOL, Tensor)
assert_type(BOOL >= TENSOR, Tensor)
assert_type(TENSOR >= INT, Tensor)
assert_type(INT >= TENSOR, Tensor)
assert_type(TENSOR >= FLOAT, Tensor)
assert_type(FLOAT >= TENSOR, Tensor)

#
# Binary ops that take and return ints or floats
#

# Operator +
assert_type(TENSOR + TENSOR, Tensor)
assert_type(TENSOR + BOOL, Tensor)
assert_type(BOOL + TENSOR, Tensor)
assert_type(TENSOR + INT, Tensor)
assert_type(INT + TENSOR, Tensor)
assert_type(TENSOR + FLOAT, Tensor)
assert_type(FLOAT + TENSOR, Tensor)

# Operator -
assert_type(TENSOR - TENSOR, Tensor)
assert_type(TENSOR - BOOL, Tensor)
assert_type(BOOL - TENSOR, Tensor)
assert_type(TENSOR - INT, Tensor)
assert_type(INT - TENSOR, Tensor)
assert_type(TENSOR - FLOAT, Tensor)
assert_type(FLOAT - TENSOR, Tensor)

# Operator *
assert_type(TENSOR * TENSOR, Tensor)
assert_type(TENSOR * BOOL, Tensor)
assert_type(BOOL * TENSOR, Tensor)
assert_type(TENSOR * INT, Tensor)
assert_type(INT * TENSOR, Tensor)
assert_type(TENSOR * FLOAT, Tensor)
assert_type(FLOAT * TENSOR, Tensor)

# Operator //
assert_type(TENSOR // TENSOR, Tensor)
assert_type(TENSOR // BOOL, Tensor)
assert_type(BOOL // TENSOR, Tensor)
assert_type(TENSOR // INT, Tensor)
assert_type(INT // TENSOR, Tensor)
assert_type(TENSOR // FLOAT, Tensor)
assert_type(FLOAT // TENSOR, Tensor)

# Operator /
assert_type(TENSOR / TENSOR, Tensor)
assert_type(TENSOR / BOOL, Tensor)
assert_type(BOOL / TENSOR, Tensor)
assert_type(TENSOR / INT, Tensor)
assert_type(INT / TENSOR, Tensor)
assert_type(TENSOR / FLOAT, Tensor)
assert_type(FLOAT / TENSOR, Tensor)

# Operator %
assert_type(TENSOR % TENSOR, Tensor)
assert_type(TENSOR % BOOL, Tensor)
assert_type(BOOL % TENSOR, Tensor)
assert_type(TENSOR % INT, Tensor)
assert_type(INT % TENSOR, Tensor)
assert_type(TENSOR % FLOAT, Tensor)
assert_type(FLOAT % TENSOR, Tensor)

# Operator **
assert_type(TENSOR**TENSOR, Tensor)
assert_type(TENSOR**BOOL, Tensor)
assert_type(BOOL**TENSOR, Tensor)
assert_type(TENSOR**INT, Tensor)
assert_type(INT**TENSOR, Tensor)
assert_type(TENSOR**FLOAT, Tensor)
assert_type(FLOAT**TENSOR, Tensor)

#
# Matrix multiplication
#

# Operator @
assert_type(TENSOR @ TENSOR, Tensor)
assert_type(TENSOR @ BOOL, Tensor)  # Should fail type checking
assert_type(BOOL @ TENSOR, Tensor)  # type: ignore[operator]
assert_type(TENSOR @ INT, Tensor)  # Should fail type checking
assert_type(INT @ TENSOR, Tensor)  # type: ignore[operator]
assert_type(TENSOR @ FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT @ TENSOR, Tensor)  # type: ignore[operator]

#
# Binary ops that take and return ints only
#

# Operator <<
assert_type(TENSOR << TENSOR, Tensor)
assert_type(TENSOR << BOOL, Tensor)
assert_type(BOOL << TENSOR, Tensor)
assert_type(TENSOR << INT, Tensor)
assert_type(INT << TENSOR, Tensor)
assert_type(TENSOR << FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT << TENSOR, Tensor)  # Should fail type checking

# Operator >>
assert_type(TENSOR >> TENSOR, Tensor)
assert_type(TENSOR >> BOOL, Tensor)
assert_type(BOOL >> TENSOR, Tensor)
assert_type(TENSOR >> INT, Tensor)
assert_type(INT >> TENSOR, Tensor)
assert_type(TENSOR >> FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT >> TENSOR, Tensor)  # Should fail type checking

# Operator &
assert_type(TENSOR & TENSOR, Tensor)
assert_type(TENSOR & BOOL, Tensor)
assert_type(BOOL & TENSOR, Tensor)
assert_type(TENSOR & INT, Tensor)
assert_type(INT & TENSOR, Tensor)
assert_type(TENSOR & FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT & TENSOR, Tensor)  # type: ignore[operator]

# Operator |
assert_type(TENSOR | TENSOR, Tensor)
assert_type(TENSOR | BOOL, Tensor)
assert_type(BOOL | TENSOR, Tensor)
assert_type(TENSOR | INT, Tensor)
assert_type(INT | TENSOR, Tensor)
assert_type(TENSOR | FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT | TENSOR, Tensor)  # type: ignore[operator]

# Operator ^
assert_type(TENSOR ^ TENSOR, Tensor)
assert_type(TENSOR ^ BOOL, Tensor)
assert_type(BOOL ^ TENSOR, Tensor)
assert_type(TENSOR ^ INT, Tensor)
assert_type(INT ^ TENSOR, Tensor)
assert_type(TENSOR ^ FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT ^ TENSOR, Tensor)  # type: ignore[operator]


NUMBER: TypeAlias = Union[int, float, bool]


class Binary:
    """
    This class demonstrates what is possible by overriding every magic method
    relating to binary operations.
    """

    def __add__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __and__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __div__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __eq__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __floordiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __ge__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __gt__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __le__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __lshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __lt__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __mod__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __mul__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __ne__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __or__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __pow__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __radd__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rand__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rdiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rfloordiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rlshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rmod__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rmul__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __ror__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rpow__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rrshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rsub__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rtruediv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __rxor__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __sub__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __truediv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self

    def __xor__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        return self


BINARY = Binary()

assert_type(BINARY + INT, Binary)
assert_type(BINARY & INT, Binary)
assert_type(BINARY / INT, Binary)
assert_type(BINARY == INT, Binary)
assert_type(BINARY // INT, Binary)
assert_type(BINARY >= INT, Binary)
assert_type(BINARY > INT, Binary)
assert_type(BINARY <= INT, Binary)
assert_type(BINARY << INT, Binary)
assert_type(BINARY < INT, Binary)
assert_type(BINARY % INT, Binary)
assert_type(BINARY * INT, Binary)
assert_type(BINARY != INT, Binary)
assert_type(BINARY | INT, Binary)
assert_type(BINARY**INT, Binary)
assert_type(BINARY >> INT, Binary)
assert_type(BINARY - INT, Binary)
assert_type(BINARY ^ INT, Binary)

assert_type(INT + BINARY, Binary)
assert_type(INT & BINARY, Binary)
assert_type(INT / BINARY, Binary)
assert_type(INT == BINARY, bool)
assert_type(INT // BINARY, Binary)
assert_type(INT >= BINARY, Binary)
assert_type(INT > BINARY, Binary)
assert_type(INT <= BINARY, Binary)
assert_type(INT << BINARY, Binary)
assert_type(INT < BINARY, Binary)
assert_type(INT % BINARY, Binary)
assert_type(INT * BINARY, Binary)
assert_type(INT != BINARY, bool)
assert_type(INT | BINARY, Binary)
assert_type(INT**BINARY, Binary)
assert_type(INT >> BINARY, Binary)
assert_type(INT - BINARY, Binary)
assert_type(INT ^ BINARY, Binary)

assert_type(BINARY + FLOAT, Binary)
assert_type(BINARY & FLOAT, Binary)
assert_type(BINARY / FLOAT, Binary)
assert_type(BINARY == FLOAT, Binary)
assert_type(BINARY // FLOAT, Binary)
assert_type(BINARY >= FLOAT, Binary)
assert_type(BINARY > FLOAT, Binary)
assert_type(BINARY <= FLOAT, Binary)
assert_type(BINARY << FLOAT, Binary)
assert_type(BINARY < FLOAT, Binary)
assert_type(BINARY % FLOAT, Binary)
assert_type(BINARY * FLOAT, Binary)
assert_type(BINARY != FLOAT, Binary)
assert_type(BINARY | FLOAT, Binary)
assert_type(BINARY**FLOAT, Binary)
assert_type(BINARY >> FLOAT, Binary)
assert_type(BINARY - FLOAT, Binary)
assert_type(BINARY ^ FLOAT, Binary)

assert_type(FLOAT + BINARY, Binary)
assert_type(FLOAT & BINARY, Binary)
assert_type(FLOAT / BINARY, Binary)
assert_type(FLOAT == BINARY, bool)
assert_type(FLOAT // BINARY, Binary)
assert_type(FLOAT >= BINARY, Binary)
assert_type(FLOAT > BINARY, Binary)
assert_type(FLOAT <= BINARY, Binary)
assert_type(FLOAT << BINARY, Binary)
assert_type(FLOAT < BINARY, Binary)
assert_type(FLOAT % BINARY, Binary)
assert_type(FLOAT * BINARY, Binary)
assert_type(FLOAT != BINARY, bool)
assert_type(FLOAT | BINARY, Binary)
assert_type(FLOAT**BINARY, Binary)
assert_type(FLOAT >> BINARY, Binary)
assert_type(FLOAT - BINARY, Binary)
assert_type(FLOAT ^ BINARY, Binary)

assert_type(BINARY + BOOL, Binary)
assert_type(BINARY & BOOL, Binary)
assert_type(BINARY / BOOL, Binary)
assert_type(BINARY == BOOL, Binary)
assert_type(BINARY // BOOL, Binary)
assert_type(BINARY >= BOOL, Binary)
assert_type(BINARY > BOOL, Binary)
assert_type(BINARY <= BOOL, Binary)
assert_type(BINARY << BOOL, Binary)
assert_type(BINARY < BOOL, Binary)
assert_type(BINARY % BOOL, Binary)
assert_type(BINARY * BOOL, Binary)
assert_type(BINARY != BOOL, Binary)
assert_type(BINARY | BOOL, Binary)
assert_type(BINARY**BOOL, Binary)
assert_type(BINARY >> BOOL, Binary)
assert_type(BINARY - BOOL, Binary)
assert_type(BINARY ^ BOOL, Binary)

assert_type(BOOL + BINARY, Binary)
assert_type(BOOL & BINARY, Binary)
assert_type(BOOL / BINARY, Binary)
assert_type(BOOL == BINARY, bool)
assert_type(BOOL // BINARY, Binary)
assert_type(BOOL >= BINARY, Binary)
assert_type(BOOL > BINARY, Binary)
assert_type(BOOL <= BINARY, Binary)
assert_type(BOOL << BINARY, Binary)
assert_type(BOOL < BINARY, Binary)
assert_type(BOOL % BINARY, Binary)
assert_type(BOOL * BINARY, Binary)
assert_type(BOOL != BINARY, bool)
assert_type(BOOL | BINARY, Binary)
assert_type(BOOL**BINARY, Binary)
assert_type(BOOL >> BINARY, Binary)
assert_type(BOOL - BINARY, Binary)
assert_type(BOOL ^ BINARY, Binary)
