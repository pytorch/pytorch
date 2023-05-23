import triton
import triton.language as tl

TRITON_HAS_REDUCE = hasattr(tl, "reduce")


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


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


if TRITON_HAS_REDUCE:

    @triton.jit
    def min2(a, dim):
        return tl.reduce(a, dim, minimum)

    @triton.jit
    def max2(a, dim):
        return tl.reduce(a, dim, maximum)

else:

    @triton.jit
    def min2(a, dim):
        min_values = tl.min(a, dim)
        if is_floating(a):
            has_nan = tl.sum(a != a, dim)
            nan = tl.full([1], float("nan"), tl.float32).to(a.dtype)
            min_values = tl.where(has_nan, nan, min_values)
        return min_values

    @triton.jit
    def max2(a, dim):
        max_values = tl.max(a, dim)
        if is_floating(a):
            has_nan = tl.sum(a != a, dim)
            nan = tl.full([1], float("nan"), tl.float32).to(a.dtype)
            max_values = tl.where(has_nan, nan, max_values)
        return max_values


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


if TRITON_HAS_REDUCE:

    @triton.jit
    def min_with_index(value, index, dim):
        return tl.reduce((value, index), dim, minimum_with_index)

    @triton.jit
    def max_with_index(value, index, dim):
        return tl.reduce((value, index), dim, maximum_with_index)

else:

    @triton.jit
    def _argreduce_index(reduction_result, value, index, dim):
        reduction_result_keepdim = reduction_result[None, :]
        if dim == 0:
            pass
        elif dim == 1:
            reduction_result_keepdim = reduction_result[:, None]
        else:
            tl.device_assert(False)

        equal = value == reduction_result_keepdim
        if is_floating(value):
            # Treat nan as equal
            result_is_nan = reduction_result_keepdim != reduction_result_keepdim
            equal |= (value != value) and result_is_nan

        invalid_index = 2**62
        indices = tl.where(equal, index, invalid_index)
        index = tl.min(indices, dim)
        return index

    @triton.jit
    def min_with_index(value, index, dim):
        min_values = min2(value, dim)
        min_index = _argreduce_index(min_values, value, index, dim)
        return min_values, min_index

    @triton.jit
    def max_with_index(value, index, dim):
        max_values = max2(value, dim)
        max_index = _argreduce_index(max_values, value, index, dim)
        return max_values, max_index


@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r
