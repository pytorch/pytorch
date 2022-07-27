# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

UNARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

INPLACE_UNARY_NAMES = [
    n + "_"
    for n in (list(set(UNARY_NAMES) - {"angle", "positive", "signbit", "isnan"}))
]

# Explicitly tracking functions we know are currently not supported
# This might be due to missing code gen or because of complex semantics
UNARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]


def torch_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs, _wrap_result

    def unary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 1:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        if args[0].layout() == torch.sparse_coo:
            data_args[0] = data_args[0].coalesce()
            s = data_args[0].size()
            i = data_args[0].indices()
            data_args[0] = data_args[0].coalesce().values()
            v = fn(*data_args)
            result_data = torch.sparse_coo_tensor(i, v, size=s)
        elif args[0].layout() == torch.sparse_csr:
            crow = data_args[0].crow_indices()
            col = data_args[0].col_indices()
            data_args[0] = data_args[0].values()
            v = fn(*data_args)
            result_data = torch.sparse_csr_tensor(crow, col, v)
        else:
            result_data = fn(*data_args)
        return _wrap_result(result_data, mask_args[0])

    return unary_fn


def torch_inplace_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs

    def unary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 1:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        if args[0].layout() == torch.sparse_coo:
            s = data_args[0].size()
            i = data_args[0].indices()
            data_args[0] = data_args[0].values()
            v = fn(*data_args)
            result_data = torch.sparse_coo_tensor(i, v, size=s)
        elif args[0].layout() == torch.sparse_csr:
            crow = data_args[0].crow_indices()
            col = data_args[0].col_indices()
            data_args[0] = data_args[0].values()
            v = fn(*data_args)
            result_data = torch.sparse_csr_tensor(crow, col, v)
        else:
            result_data = fn(*data_args)
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]

    return unary_fn


NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): torch_unary(name) for name in UNARY_NAMES
}
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


def is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


def apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented
