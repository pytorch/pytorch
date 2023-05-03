import triton
import triton.language as tl


@triton.jit
def _prod_accumulate(a, b):
    return a * b


@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r
