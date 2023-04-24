import triton
import triton.language as tl


@triton.jit
def is_floating(x):
    if isinstance(x, tl.constexpr):
        ret = isinstance(x.value, float)
    else:
        ret = x.dtype.is_floating()
    return ret


@triton.jit
def _prod_accumulate(a, b):
    return a * b


@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def minimum(a, b):
    mask = a < b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def maximum(a, b):
    mask = a > b
    if is_floating(a):
        mask |= a != a
    return tl.where(mask, a, b)


@triton.jit
def min(a, dim):
    return tl.reduce(a, dim, minimum)


@triton.jit
def max(a, dim):
    return tl.reduce(a, dim, maximum)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    if is_floating(a_value):
        # Consider NaN as equal
        equal = not (a_value < b_value) and not (a_value > b_value)
    else:
        equal = a_value == b_value

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    if is_floating(a_value):
        # Consider NaN as equal
        equal = not (a_value < b_value) and not (a_value > b_value)
    else:
        equal = a_value == b_value

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)
