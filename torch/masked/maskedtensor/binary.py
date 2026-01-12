# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from .core import (
    _map_mt_args_kwargs,
    _masks_match,
    _tensors_match,
    _wrap_result,
    is_masked_tensor,
)


__all__ = []  # type: ignore[var-annotated]

BINARY_NAMES = [
    "add",
    "atan2",
    "arctan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "div",
    "divide",
    "floor_divide",
    "fmod",
    "logaddexp",
    "logaddexp2",
    "mul",
    "multiply",
    "nextafter",
    "remainder",
    "sub",
    "subtract",
    "true_divide",
    "eq",
    "ne",
    "le",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "less_equal",
    "lt",
    "less",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "not_equal",
]

INPLACE_BINARY_NAMES = [
    n + "_"
    for n in (
        list(
            set(BINARY_NAMES)
            - {
                "logaddexp",
                "logaddexp2",
                "equal",
                "fmin",
                "minimum",
                "maximum",
                "fmax",
            }
        )
    )
]


def _get_at_least_one_mask(a, b):
    if not is_masked_tensor(a) and not is_masked_tensor(b):
        raise TypeError("At least one of `a` and `b` must be a MaskedTensor")
    if not _masks_match(a, b):
        raise ValueError("a and b must have matching masks")
    if is_masked_tensor(a):
        return a.get_mask()
    return b.get_mask()


def _binary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError("len(kwargs) must equal 0")
    for a in args[2:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor binary ops do not support Tensor arguments aside from the lhs and rhs"
            )

    if not _masks_match(*args[:2]):
        raise ValueError(
            "Input masks must match. If you need support for this, please open an issue on Github."
        )

    data_args, _data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    mask_args, _mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())

    args0_layout = data_args[0].layout
    same_layout = (
        torch.is_tensor(data_args[1]) or is_masked_tensor(data_args[1])
    ) and (args0_layout == data_args[1].layout)

    if args0_layout == torch.sparse_coo:
        if same_layout:
            if not _tensors_match(data_args[0].indices(), data_args[1].indices()):
                raise ValueError(
                    "sparse_coo indices must match. If you need support for this, please open an issue on Github."
                )
            if data_args[0].size() != data_args[1].size():
                raise ValueError(
                    "input1 and input2 must have the same size for binary functions."
                )

            data_args[1] = data_args[1].values()

        i = data_args[0].indices()
        size = data_args[0].size()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size)

    elif args0_layout == torch.sparse_csr:
        if same_layout:
            if not (
                _tensors_match(data_args[0].crow_indices(), data_args[1].crow_indices())
                and _tensors_match(
                    data_args[0].col_indices(), data_args[1].col_indices()
                )
            ):
                raise ValueError(
                    "sparse_csr indices must match. If you need support for this, please open an issue on Github."
                )

            data_args[1] = data_args[1].values()

        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        size = data_args[0].size()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v, size)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        result_mask = _get_at_least_one_mask(*args[:2])
        # sparse tensors don't have strides so we can only expand if the layout is strided
        if args0_layout == torch.strided:
            result_mask = result_mask.expand_as(result_data)
        return _wrap_result(result_data, result_mask)


def _torch_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=False)

    return binary_fn


def _torch_inplace_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=True)

    return binary_fn


NATIVE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_binary(name) for name in BINARY_NAMES
}
NATIVE_INPLACE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_binary(name)
    for name in INPLACE_BINARY_NAMES
}

NATIVE_BINARY_FNS = list(NATIVE_BINARY_MAP.keys())
NATIVE_INPLACE_BINARY_FNS = list(NATIVE_INPLACE_BINARY_MAP.keys())


def _is_native_binary(fn):
    return fn in NATIVE_BINARY_FNS or fn in NATIVE_INPLACE_BINARY_FNS


def _apply_native_binary(fn, *args, **kwargs):
    if fn in NATIVE_BINARY_FNS:
        return NATIVE_BINARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_BINARY_FNS:
        return NATIVE_INPLACE_BINARY_MAP[fn](*args, **kwargs)
    return NotImplemented
