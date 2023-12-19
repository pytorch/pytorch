import math
import random

import torch
from torch.testing._internal.inputgen.variable.type import ScalarDtype
from torch.testing._internal.inputgen.variable.utils import nextdown, nextup


def safe_ix(array, ix, default=0):
    if len(array) == 0:
        return default
    return array[ix % len(array)]


def safe_size(t, d):
    if t.dim() == 0:
        return 1
    else:
        return t.size(d % t.dim())


def normalize(d, r):
    if r == 0:
        return 0
    return d % r


def promote_type_with_scalar(t, scalar):
    if isinstance(scalar, bool):
        return t
    elif isinstance(scalar, int):
        if t != torch.bool:
            return t
        else:
            return torch.int64
    if isinstance(scalar, float):
        if t in [torch.float32, torch.float64]:
            return t
        else:
            return torch.float32


def dt_to_st(dt):
    if dt is None:
        return None
    if dt == torch.bool:
        return ScalarDtype.bool
    if dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return ScalarDtype.int
    if dt in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return ScalarDtype.float
    raise NotImplementedError("Unsupported type")


def st_ge(st):
    if st is None:
        return None
    if st == ScalarDtype.bool:
        return [ScalarDtype.bool, ScalarDtype.int, ScalarDtype.float]
    if st == ScalarDtype.int:
        return [ScalarDtype.int, ScalarDtype.float]
    if st == ScalarDtype.float:
        return [ScalarDtype.float]
    raise NotImplementedError("Unsupported type")


def st_le(st):
    if st is None:
        return None
    if st == ScalarDtype.bool:
        return [ScalarDtype.bool]
    if st == ScalarDtype.int:
        return [ScalarDtype.bool, ScalarDtype.int]
    if st == ScalarDtype.float:
        return [ScalarDtype.bool, ScalarDtype.int, ScalarDtype.float]
    raise NotImplementedError("Unsupported type")


def add_alpha_st(st):
    if st is None:
        return None
    if st == ScalarDtype.bool:
        return [ScalarDtype.bool, ScalarDtype.int]
    if st == ScalarDtype.int:
        return [ScalarDtype.int]
    if st == ScalarDtype.float:
        return [ScalarDtype.int, ScalarDtype.float]
    raise NotImplementedError("Unsupported type")


def dtype_lower_bound(dtype):
    if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.iinfo(dtype).min
    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return torch.finfo(dtype).min
    return None


def dtype_strict_lower_bound(dtype):
    if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.iinfo(dtype).min - 1
    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return nextdown(torch.finfo(dtype).min)
    return None


def dtype_upper_bound(dtype):
    if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.iinfo(dtype).max
    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return torch.finfo(dtype).max
    return None


def dtype_strict_upper_bound(dtype):
    if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.iinfo(dtype).max + 1
    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return nextup(torch.finfo(dtype).max)
    return None


def factorize_into_primes(n):
    factors = []
    d = 2
    while d <= n:
        if n % d == 0:
            n = n // d
            factors.append(d)
        else:
            d += 1
    return factors


def factorize(n, length):
    if length == 0 and n != 1:
        return set()

    if n == 0:
        factor_list = []
        prod = 1
        for _ in range(length):
            x = random.choice(range(10))
            factor_list.append(x)
            prod *= x
        if prod != 0:
            i = random.choice(range(length))
            factor_list[i] = 0
        return {tuple(factor_list)}

    if n == 1:
        return {(1,) * length}

    factors = factorize_into_primes(n)
    factor_list = [1] * length
    for factor in factors:
        x = random.choice(range(length))
        factor_list[x] *= factor
    return {tuple(factor_list)}


def valid_view_copy_size(tensor, length):
    n = tensor.numel()
    valids = factorize(n, length)
    factors = random.choice(list(factorize(n, length)))
    if length >= 1:
        if n > 0:
            x = random.choice(range(length))
            factor_list = list(factors)
            factor_list[x] = -1
            valids |= {tuple(factor_list)}
        else:
            zeros = [i for i in range(length) if factors[i] == 0]
            z = random.choice(zeros)
            factor_list = list(factors)
            factor_list[z] = -1
            for i in range(length):
                if i != z and factors[i] == 0:
                    factor_list[i] = random.choice(range(1, 10))
            valids |= {tuple(factor_list)}
    return valids


def invalid_view_copy_size(tensor, length):
    n = tensor.numel()
    invalids = set()
    if n != 0:
        invalids |= factorize(0, length)
    if n != 1:
        invalids |= factorize(1, length)
    if n > 0 and n % 2 == 0:
        invalids |= factorize(n // 2, length)
    if n > 0 and n % 3 == 0:
        invalids |= factorize(n // 3, length)
    if n > 2:
        invalids |= factorize(n - 1, length)
    if n > 3:
        x = random.choice(range(2, n - 1))
        invalids |= factorize(x, length)
    invalids |= factorize(n + 1, length)
    if n > 0:
        invalids |= factorize(2 * n, length)
        invalids |= factorize(3 * n, length)
    factors = random.choice(list(factorize(n, length)))
    potential_negative = []
    for ix, factor in enumerate(factors):
        if factor > 1:
            potential_negative.append(ix)
    if len(potential_negative) >= 1:
        x = random.choice(potential_negative)
        factor_list = list(factors)
        factor_list[x] = -factors[x]
        invalids |= {tuple(factor_list)}
    if len(potential_negative) >= 2:
        x, y = random.sample(potential_negative, 2)
        factor_list = list(factors)
        factor_list[x] = -factors[x]
        factor_list[y] = -factors[y]
        invalids |= {tuple(factor_list)}
    if length >= 2:
        x, y = random.sample(range(length), 2)
        factor_list = list(factors)
        factor_list[x], factor_list[y] = -1, -1
        invalids |= {tuple(factor_list)}
    if length >= 1 and n == 0:
        zeros = [i for i in range(length) if factors[i] == 0]
        z = random.choice(zeros)
        non_z = [i for i in range(length) if i != z]
        if len(non_z) >= 1:
            x = random.choice([i for i in range(length) if i != z])
            factor_list = list(factors)
            factor_list[x] = -1
            invalids |= {tuple(factor_list)}
    return invalids


def as_strided_min_numel(sizes, strides, storage_offset):
    if storage_offset is None:
        storage_offset = 0
    m = storage_offset
    for i in range(len(sizes)):
        if sizes[i] == 0:
            return 0
        m += (sizes[i] - 1) * strides[i]
    return m + 1


def valid_as_strided_sizes(sizes, strides, storage_offset, rank):
    m = as_strided_min_numel(sizes, strides, storage_offset)
    return factorize(m, rank) | factorize(m + 1, rank) | factorize(2 * m, rank)


def invalid_as_strided_sizes(sizes, strides, storage_offset, rank):
    m = as_strided_min_numel(sizes, strides, storage_offset)
    if m == 0:
        return set()
    return factorize(m - 1, rank)


def valid_dim_list_helper(tensor, pool, length):
    if length > len(pool):
        return {}

    n = max(tensor.dim(), 1)

    sample = tuple(random.sample(pool, length))
    neg_sample = tuple(s - n for s in sample)
    mix_sample = tuple(random.choice([s, s - n]) for s in sample)

    return {sample, neg_sample, mix_sample}


def valid_dim_list(tensor, length):
    n = max(tensor.dim(), 1)
    pool = list(range(n))
    return valid_dim_list_helper(tensor, pool, length)


def valid_dim_list_non_zero_size(tensor, length):
    n = tensor.dim()
    if n == 0:
        pool = [0]
    else:
        pool = [d for d in range(n) if tensor.size(d) != 0]
    return valid_dim_list_helper(tensor, pool, length)


def invalid_dim_list(tensor, length):
    return {
        (
            0,
            0,
        )
    }


def invalid_dim_list_non_zero_size(tensor, length):
    return {
        (
            0,
            0,
        )
    }


def scatter_add_index_size_max(x, dim, src, d):
    max_d = safe_size(src, d)
    if d != dim:
        max_d = min(max_d, safe_size(x, d))
    return max_d


def cat_rank_in(ctx):
    for i in range(ctx.index.rank):
        r = ctx.rank(i)
        if r != 1:
            return [1, r]
    return None


def cat_size_in(ctx):
    return None


def cat_common_rank(tensors):
    for t in tensors:
        if t.dim() == 1 and t.size(0) == 0:
            continue
        return t.dim()
    return -1


def cat_dim_value_in(tensors):
    common_rank = cat_common_rank(tensors)
    if common_rank < 0:
        return None
    if common_rank in (0, 1):
        return (-1, 0)
    ix = 0
    for t in tensors:
        if t.dim() == 1 and t.size(0) == 0:
            ix += 1
            continue
        break
    t0 = tensors[ix]
    valid_dim = None
    for t in tensors[ix + 1 :]:
        for d in range(t.dim()):
            if safe_size(t, d) != safe_size(t0, d):
                valid_dim = d
    if valid_dim is not None:
        return (valid_dim - common_rank, valid_dim)
    return None


def clamp_max_is_optional(tensor, min_s):
    if min_s is None or (isinstance(min_s, bool) and tensor.dtype == torch.bool):
        return False
    return None


def clamp_max_ne_dtype(tensor, min_s):
    if (min_s is None or isinstance(min_s, bool)) and tensor.dtype == torch.bool:
        return torch.bool
    return None


def expand_copy_size_in(shape, length, ix):
    n = len(shape)
    if ix < length - n:
        return None
    if shape[ix - (length - n)] == 1:
        return None
    else:
        return [-1, shape[ix - (length - n)]]


def nlm_input_size(shape, rank, d):
    n = len(shape)
    if d < rank - n:
        return None
    else:
        return shape[d - (rank - n)]


def conv_input_size_eq(weight, transposed, groups, dim):
    if transposed:
        return safe_size(weight, 0) if dim == 1 else None
    else:
        return groups * safe_size(weight, 1) if dim == 1 else None


def conv_input_size_min(
    weight, stride, padding, dilation, transposed, output_padding, dim
):
    if dim < 2:
        return 0
    s = stride[0] if len(stride) == 1 else safe_ix(stride, dim - 2)
    p = padding[0] if len(padding) == 1 else safe_ix(padding, dim - 2)
    d = dilation[0] if len(dilation) == 1 else safe_ix(dilation, dim - 2)
    op = (
        output_padding[0]
        if len(output_padding) == 1
        else safe_ix(output_padding, dim - 2)
    )
    if transposed:
        return max(1, (2 * p - d * (safe_size(weight, dim) - 1) - op + s - 1) // s + 1)
    else:
        return max(1, d * (safe_size(weight, dim) - 1) + 1 - 2 * p)


def conv_bias_size_eq(weight, transposed, groups):
    return safe_size(weight, 1) * groups if transposed else safe_size(weight, 0)


def conv_output_padding_max(stride, dilation, transposed, length, ix):
    if not transposed:
        return None
    if length != 1:
        s = safe_ix(stride, ix)
        d = safe_ix(dilation, ix)
        return max(s, d) - 1
    max_val = None
    for i in range(max(len(stride), len(dilation))):
        s = safe_ix(stride, i)
        d = safe_ix(dilation, i)
        v = max(s, d) - 1
        if max_val is None or v < max_val:
            max_val = v
    return max_val


def pool_input_size_min(
    kernel_ndim, kernel_size, stride, padding, dilation, ceil_mode, rank, dim
):
    if dim == 0:
        return 0 if rank == 4 else 1
    if dim == 1 and rank == 4:
        return 1

    kdim = dim - (rank - kernel_ndim)

    k = 1
    if len(kernel_size) > 0:
        k = kernel_size[0] if len(kernel_size) == 1 else safe_ix(kernel_size, kdim)
    s = k
    if len(stride) > 0:
        s = stride[0] if len(stride) == 1 else safe_ix(stride, kdim)
    p = 0
    if len(padding) > 0:
        p = padding[0] if len(padding) == 1 else safe_ix(padding, kdim)
    d = 1
    if len(dilation) > 0:
        d = dilation[0] if len(dilation) == 1 else safe_ix(dilation, kdim)
    return max(1, d * (k - 1) + 1 - 2 * p)


def pool_padding_max(kernel_size, length, ix):
    if length == 1:
        return min(kernel_size) // 2
    k = 1
    if len(kernel_size) > 0:
        k = kernel_size[0] if len(kernel_size) == 1 else safe_ix(kernel_size, ix)
    return k // 2


def bmm_mat2_size_eq(input, d):
    if d == 0:
        return input.size(0)
    if d == 1:
        return input.size(2)
    if d == 2:
        return None


def dim_non_zero_size(tensor):
    n = tensor.dim()
    if n == 0:
        return [-1, 0]
    return [d for d in range(-n, n) if tensor.size(d) != 0]


def broadcast_to(shape, rank, d):
    if len(shape) < rank:
        return set()
    return list({1, shape[len(shape) - rank + d]})


def broadcast_with(shape, rank, d):
    n = len(shape)
    if d < rank - n:
        return None
    for ix, s in enumerate(shape):
        if n - ix == rank - d:
            if s == 1:
                return None
            return list({1, s})


def broadcasted_shape(shape_a, shape_b):
    a_ndim = len(shape_a)
    b_ndim = len(shape_b)
    res = [1 for _ in range(max(a_ndim, b_ndim))]
    n = len(res)
    for ix in range(1, n + 1):
        a_size = shape_a[-ix] if a_ndim >= ix else 1
        b_size = shape_b[-ix] if b_ndim >= ix else 1
        if a_size != 1 and b_size != 1:
            assert a_size == b_size
        res[-ix] = a_size if a_size != 1 else b_size
    return res
