# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from array import array

from hypothesis.internal.floats import float_to_int, int_to_float

"""
This module implements support for arbitrary floating point numbers in
Conjecture. It doesn't make any attempt to get a good distribution, only to
get a format that will shrink well.

It works by defining an encoding of non-negative floating point numbers
(including NaN values with a zero sign bit) that has good lexical shrinking
properties.

This encoding is a tagged union of two separate encodings for floating point
numbers, with the tag being the first bit of 64 and the remaining 63-bits being
the payload.

If the tag bit is 0, the next 7 bits are ignored, and the remaining 7 bytes are
interpreted as a 7 byte integer in big-endian order and then converted to a
float (there is some redundancy here, as 7 * 8 = 56, which is larger than the
largest integer that floating point numbers can represent exactly, so multiple
encodings may map to the same float).

If the tag bit is 1, we instead use something that is closer to the normal
representation of floats (and can represent every non-negative float exactly)
but has a better ordering:

1. NaNs are ordered after everything else.
2. Infinity is ordered after every finite number.
3. The sign is ignored unless two floating point numbers are identical in
   absolute magnitude. In that case, the positive is ordered before the
   negative.
4. Positive floating point numbers are ordered first by int(x) where
   encoding(x) < encoding(y) if int(x) < int(y).
5. If int(x) == int(y) then x and y are sorted towards lower denominators of
   their fractional parts.

The format of this encoding of floating point goes as follows:

    [exponent] [mantissa]

Each of these is the same size their equivalent in IEEE floating point, but are
in a different format.

We translate exponents as follows:

    1. The maximum exponent (2 ** 11 - 1) is left unchanged.
    2. We reorder the remaining exponents so that all of the positive exponents
       are first, in increasing order, followed by all of the negative
       exponents in decreasing order (where positive/negative is done by the
       unbiased exponent e - 1023).

We translate the mantissa as follows:

    1. If the unbiased exponent is <= 0 we reverse it bitwise.
    2. If the unbiased exponent is >= 52 we leave it alone.
    3. If the unbiased exponent is in the range [1, 51] then we reverse the
       low k bits, where k is 52 - unbiased exponent.

The low bits correspond to the fractional part of the floating point number.
Reversing it bitwise means that we try to minimize the low bits, which kills
off the higher powers of 2 in the fraction first.
"""


MAX_EXPONENT = 0x7FF

BIAS = 1023
MAX_POSITIVE_EXPONENT = MAX_EXPONENT - 1 - BIAS


def exponent_key(e: int) -> float:
    if e == MAX_EXPONENT:
        return float("inf")
    unbiased = e - BIAS
    if unbiased < 0:
        return 10000 - unbiased
    else:
        return unbiased


ENCODING_TABLE = array("H", sorted(range(MAX_EXPONENT + 1), key=exponent_key))
DECODING_TABLE = array("H", [0]) * len(ENCODING_TABLE)

for i, b in enumerate(ENCODING_TABLE):
    DECODING_TABLE[b] = i

del i, b


def decode_exponent(e: int) -> int:
    """Take an integer and turn it into a suitable floating point exponent
    such that lexicographically simpler leads to simpler floats."""
    assert 0 <= e <= MAX_EXPONENT
    return ENCODING_TABLE[e]


def encode_exponent(e: int) -> int:
    """Take a floating point exponent and turn it back into the equivalent
    result from conjecture."""
    assert 0 <= e <= MAX_EXPONENT
    return DECODING_TABLE[e]


def reverse_byte(b: int) -> int:
    result = 0
    for _ in range(8):
        result <<= 1
        result |= b & 1
        b >>= 1
    return result


# Table mapping individual bytes to the equivalent byte with the bits of the
# byte reversed. e.g. 1=0b1 is mapped to 0xb10000000=0x80=128. We use this
# precalculated table to simplify calculating the bitwise reversal of a longer
# integer.
REVERSE_BITS_TABLE = bytearray(map(reverse_byte, range(256)))


def reverse64(v: int) -> int:
    """Reverse a 64-bit integer bitwise.

    We do this by breaking it up into 8 bytes. The 64-bit integer is then the
    concatenation of each of these bytes. We reverse it by reversing each byte
    on its own using the REVERSE_BITS_TABLE above, and then concatenating the
    reversed bytes.

    In this case concatenating consists of shifting them into the right
    position for the word and then oring the bits together.
    """
    assert v.bit_length() <= 64
    return (
        (REVERSE_BITS_TABLE[(v >> 0) & 0xFF] << 56)
        | (REVERSE_BITS_TABLE[(v >> 8) & 0xFF] << 48)
        | (REVERSE_BITS_TABLE[(v >> 16) & 0xFF] << 40)
        | (REVERSE_BITS_TABLE[(v >> 24) & 0xFF] << 32)
        | (REVERSE_BITS_TABLE[(v >> 32) & 0xFF] << 24)
        | (REVERSE_BITS_TABLE[(v >> 40) & 0xFF] << 16)
        | (REVERSE_BITS_TABLE[(v >> 48) & 0xFF] << 8)
        | (REVERSE_BITS_TABLE[(v >> 56) & 0xFF] << 0)
    )


MANTISSA_MASK = (1 << 52) - 1


def reverse_bits(x: int, n: int) -> int:
    assert x.bit_length() <= n <= 64
    x = reverse64(x)
    x >>= 64 - n
    return x


def update_mantissa(unbiased_exponent: int, mantissa: int) -> int:
    if unbiased_exponent <= 0:
        mantissa = reverse_bits(mantissa, 52)
    elif unbiased_exponent <= 51:
        n_fractional_bits = 52 - unbiased_exponent
        fractional_part = mantissa & ((1 << n_fractional_bits) - 1)
        mantissa ^= fractional_part
        mantissa |= reverse_bits(fractional_part, n_fractional_bits)
    return mantissa


def lex_to_float(i: int) -> float:
    assert i.bit_length() <= 64
    has_fractional_part = i >> 63
    if has_fractional_part:
        exponent = (i >> 52) & ((1 << 11) - 1)
        exponent = decode_exponent(exponent)
        mantissa = i & MANTISSA_MASK
        mantissa = update_mantissa(exponent - BIAS, mantissa)

        assert mantissa.bit_length() <= 52

        return int_to_float((exponent << 52) | mantissa)
    else:
        integral_part = i & ((1 << 56) - 1)
        return float(integral_part)


def float_to_lex(f: float) -> int:
    if is_simple(f):
        assert f >= 0
        return int(f)
    return base_float_to_lex(f)


def base_float_to_lex(f: float) -> int:
    i = float_to_int(f)
    i &= (1 << 63) - 1
    exponent = i >> 52
    mantissa = i & MANTISSA_MASK
    mantissa = update_mantissa(exponent - BIAS, mantissa)
    exponent = encode_exponent(exponent)

    assert mantissa.bit_length() <= 52
    return (1 << 63) | (exponent << 52) | mantissa


def is_simple(f: float) -> int:
    try:
        i = int(f)
    except (ValueError, OverflowError):
        return False
    if i != f:
        return False
    return i.bit_length() <= 56
