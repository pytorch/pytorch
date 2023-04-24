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
