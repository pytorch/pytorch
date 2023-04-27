import triton
import triton.language as tl


@triton.jit
def is_floating(x):
    # Addition to promote scalars to tensor
    x += tl.zeros((1,), tl.int1)
    return x.dtype.is_floating()


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
