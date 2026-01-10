# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


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


def _unary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError(
            "MaskedTensor unary ops require that len(kwargs) == 0. "
            "If you need support for this, please open an issue on Github."
        )
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor unary ops do not support additional Tensor arguments"
            )

    mask_args, _mask_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_mask
    )
    data_args, _data_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_data
    )

    if args[0].layout == torch.sparse_coo:
        data_args[0] = data_args[0].coalesce()
        s = data_args[0].size()
        i = data_args[0].indices()
        data_args[0] = data_args[0].coalesce().values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size=s)

    elif args[0].layout == torch.sparse_csr:
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        return _wrap_result(result_data, mask_args[0])


def _torch_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=False)

    return unary_fn


def _torch_inplace_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=True)

    return unary_fn


NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_unary(name) for name in UNARY_NAMES
}
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


def _is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


def _apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented
