"""
Python polyfills for common builtins.
"""
import math
import torch


def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True


def index(iterator, item, start=0, end=-1):
    for i, elem in enumerate(list(iterator))[start:end]:
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item, count):
    for i in range(count):
        yield item


def radians(x):
    return math.pi / 180.0 * x


def round(number, ndigits=None):
    if isinstance(number, (bool, torch.SymBool, int, torch.SymInt)):
        return number
    elif not isinstance(number, (float, torch.SymFloat)):
        raise TypeError(f"round is not support for inputs of type {type(number)}.")

    if ndigits is None:
        result = _round_half_to_even(number)
        return torch.sym_int(result) if isinstance(number, torch.SymFloat) else int(result)
    else:
        return _round_half_to_even(number * 10 ** ndigits) * 10 ** -ndigits


def _round_half_to_even(x):
    # Pythons builtin 'round' implements the 'round half to even' strategy
    # See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
    # This is implemented here in terms of the 'round half up' strategy (how non-programmers think about rounding)
    # and correcting the relevant numbers.
    rounded_half_up = math.floor(x + 0.5)

    # Correction is needed for numbers with an even integer part and exactly 0.5 as decimal. They are rounded up by
    # 'round half to even', but need to be rounded down for 'round half to even'. Thus, we need to subtract 1.
    # To avoid branching, the correction is implemented by treating 0.0 and 1.0 as boolean proxys.
    # Thus, 'not x' is replaced by '1 - x' and 'x and y' is replaced by 'x * y'.

    integer = math.floor(x)
    is_even_integer = 1 - (integer % 2)

    decimal = x - integer
    is_le_half = math.floor(2 * (1 - decimal))
    is_ge_half = math.floor(2 * decimal)
    is_half = is_le_half * is_ge_half

    round_half_to_even_correction = is_even_integer * is_half

    return rounded_half_up - round_half_to_even_correction
