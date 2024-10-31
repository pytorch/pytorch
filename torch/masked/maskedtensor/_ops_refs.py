# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from functools import partial
from typing import Any, Callable, Dict, TYPE_CHECKING

import torch

from .binary import _apply_native_binary, NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS
from .core import (
    _get_data,
    _masks_match,
    _maybe_get_mask,
    is_masked_tensor,
    MaskedTensor,
)
from .passthrough import _apply_pass_through_fn, PASSTHROUGH_FNS
from .reductions import (
    _apply_reduction,
    NATIVE_REDUCE_FNS,
    TENSOR_REDUCE_FNS,
    TORCH_REDUCE_FNS,
)
from .unary import _apply_native_unary, NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS


if TYPE_CHECKING:
    from torch._ops import OpOverload


__all__ = []  # type: ignore[var-annotated]


def _check_args_kwargs_length(
    args, kwargs, error_prefix, len_args=None, len_kwargs=None
):
    if len_args is not None and len_args != len(args):
        raise ValueError(
            f"{error_prefix}: len(args) must be {len_args} but got {len(args)}"
        )
    if len_kwargs is not None and len_kwargs != len(kwargs):
        raise ValueError(
            f"{error_prefix}: len(kwargs) must be {len_kwargs} but got {len(kwargs)}"
        )


class _MaskedContiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedContiguous forward: input must be a MaskedTensor.")

        if input.is_contiguous():
            return input

        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.contiguous(), mask.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _MaskedToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToDense forward: input must be a MaskedTensor.")

        if input.layout == torch.strided:
            return input

        ctx.layout = input.layout
        data = input.get_data()
        mask = input.get_mask()

        return MaskedTensor(data.to_dense(), mask.to_dense())

    @staticmethod
    def backward(ctx, grad_output):
        layout = ctx.layout

        if layout == torch.sparse_coo:
            return grad_output.to_sparse_coo()
        elif layout == torch.sparse_csr:
            return grad_output.to_sparse_csr()
        elif layout == torch.strided:
            return grad_output.to_dense()
        raise ValueError("to_dense: Unsupported input layout: ", layout)


class _MaskedToSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparse forward: input must be a MaskedTensor.")

        # Following the convention from sparse tensors that to_sparse always means that we convert to sparse_coo
        if input.layout == torch.sparse_coo:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_coo().coalesce()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedToSparseCsr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparseCsr forward: input must be a MaskedTensor.")

        if input._masked_data.ndim != 2:
            raise ValueError(
                f"Only 2D tensors can be converted to the SparseCsr layout but got shape: {input._masked_data.size()}"
            )

        if input.layout == torch.sparse_csr:
            return input

        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_csr()
        sparse_data = data.sparse_mask(sparse_mask)

        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()


class _MaskedWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond, self, other):
        ctx.mark_non_differentiable(cond)
        ctx.save_for_backward(cond)
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    def backward(ctx, grad_output):
        (cond,) = ctx.saved_tensors

        def masked_out_like(mt):
            return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())

        return (
            None,
            torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)),
            torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output),
        )


_MASKEDTENSOR_FUNCTION_TABLE = {}

_function_fn_apply_map = {
    (
        tuple(NATIVE_REDUCE_FNS),
        tuple(TORCH_REDUCE_FNS),
        tuple(TENSOR_REDUCE_FNS),
    ): _apply_reduction,
}

for fn_map_list, apply_fn in _function_fn_apply_map.items():
    for fn_map in fn_map_list:
        for fn in fn_map:
            _MASKEDTENSOR_FUNCTION_TABLE[fn] = partial(apply_fn, fn)


def register_function_func(ops):
    """
    Used for registering a new __torch_function__ function to MaskedTensor
    Called via _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_function_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for op in ops:
            _MASKEDTENSOR_FUNCTION_TABLE[op] = partial(func, op)

    return wrapper


@register_function_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_function_reductions(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)


@register_function_func([torch.Tensor.where, torch.where])
def _function_where(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, "__torch_function__, torch.where", len_args=3, len_kwargs=0
    )
    return _MaskedWhere.apply(*args)


@register_function_func([torch.Tensor.contiguous])
def _function_contiguous(func, *args, **kwargs):
    return _MaskedContiguous.apply(args[0])


@register_function_func([torch.Tensor.to_dense])
def _function_to_dense(func, *args, **kwargs):
    return _MaskedToDense.apply(args[0])


@register_function_func([torch.Tensor.to_sparse])
def _function_to_sparse(func, *args, **kwargs):
    return _MaskedToSparse.apply(args[0])


@register_function_func([torch.Tensor.to_sparse_csr])
def _function_to_sparse_csr(func, *args, **kwargs):
    return _MaskedToSparseCsr.apply(args[0])


_MASKEDTENSOR_DISPATCH_TABLE: Dict["OpOverload", Callable[..., Any]] = {}


def register_dispatch_func(aten_ops):
    """
    Used for registering a new __torch_dispatch__ function to MaskedTensor
    Called via _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_dispatch_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for aten_op in aten_ops:
            _MASKEDTENSOR_DISPATCH_TABLE[aten_op] = partial(func, aten_op)

    return wrapper


@register_dispatch_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_reduction(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)


@register_dispatch_func(PASSTHROUGH_FNS)
def _general_passthrough(func, *args, **kwargs):
    return _apply_pass_through_fn(func, *args, **kwargs)


@register_dispatch_func(NATIVE_UNARY_FNS + NATIVE_INPLACE_UNARY_FNS)
def _general_unary(func, *args, **kwargs):
    return _apply_native_unary(func, *args, **kwargs)


@register_dispatch_func(NATIVE_BINARY_FNS + NATIVE_INPLACE_BINARY_FNS)
def _general_binary(func, *args, **kwargs):
    return _apply_native_binary(func, *args, **kwargs)


@register_dispatch_func([torch.ops.aten.stride])
def stride(func, *args, **kwargs):
    return None


@register_dispatch_func([torch.ops.aten.sym_stride])
def sym_stride(func, *args, **kwargs):
    return None


@register_dispatch_func([torch.ops.prim.layout])
def layout(func, *args, **kwargs):
    return _get_data(args[0]).layout


@register_dispatch_func([torch.ops.aten.is_contiguous])
def is_contiguous(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have is_contiguous")
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.is_strides_like_format])
def is_strides_like_format(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_strides_like_format"
        )
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.is_non_overlapping_and_dense])
def is_non_overlapping_and_dense(func, *args, **kwargs):
    data = _get_data(args[0])
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_non_overlapping_and_dense"
        )
    return func(data, *args[1:], **kwargs)


@register_dispatch_func([torch.ops.aten.contiguous])
def contiguous(func, *args, **kwargs):
    if _get_data(args[0]).is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have contiguous")
    return _MaskedContiguous.apply(args[0])


@register_dispatch_func([torch.ops.aten.new_empty_strided])
def new_empty_strided(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3)
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if tuple(args[1]) != tuple(data.size()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[1] expected to be the same as data.size()"
        )
    if tuple(args[2]) != tuple(data.stride()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[2] expected to be the same as data.stride()"
        )
    return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)


@register_dispatch_func([torch.ops.aten._local_scalar_dense])
def _local_scalar_dense(func, *args, **kwargs):
    if not _maybe_get_mask(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected a mask tensor")
    return torch.ops.aten._local_scalar_dense(_get_data(args[0]))


@register_dispatch_func([torch.ops.aten.detach, torch.ops.aten.clone])
def _apply_fn_on_data(func, *args, **kwargs):
    return MaskedTensor(func(_get_data(args[0])), _maybe_get_mask(args[0]))


@register_dispatch_func([torch.ops.aten._to_copy])
def _to_copy(func, *args, **kwargs):
    new_data = func(_get_data(args[0]), *args[1:], **kwargs)
    return MaskedTensor(new_data, _maybe_get_mask(args[0]))


@register_dispatch_func([torch.ops.aten._softmax])
def _softmax(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    result_data = torch.ops.aten._masked_softmax(data, ~mask, args[1], 2)
    return MaskedTensor(result_data, mask)


@register_dispatch_func([torch.ops.aten.ones_like])
def ones_like(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
    result_data = func(_get_data(args[0]), **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))


@register_dispatch_func([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=4)
    grad, output, dim, _input_dtype = args
    if is_masked_tensor(grad) and is_masked_tensor(output):
        if not _masks_match(grad, output):
            raise ValueError(
                "__torch_dispatch__, {func}: expected the masks of grad and output to match"
            )
        grad_data = _get_data(grad)
        new_grad_data = torch.ops.aten._masked_softmax_backward(
            grad_data,
            _get_data(output),
            ~_maybe_get_mask(grad),
            dim % grad_data.ndim,
        )
        res = MaskedTensor(new_grad_data, _maybe_get_mask(grad))
        return res
    else:
        raise ValueError(
            f"__torch_dispatch__, {func}: grad and output must both be MaskedTensors"
        )


@register_dispatch_func([torch.ops.aten.copy_])
def copy_(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    if not _masks_match(_maybe_get_mask(args[0]), _maybe_get_mask(args[1])):
        raise ValueError("args[0] mask and args[1] mask must match but do not")
    func(_get_data(args[0]), _get_data(args[1]))
    return args[0]


@register_dispatch_func([torch.ops.aten.where])
def where(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mx = args[1]
    my = args[2]
    if not is_masked_tensor(mx):
        mx = MaskedTensor(mx, torch.ones_like(mx, dtype=torch.bool))
    if not is_masked_tensor(my):
        my = MaskedTensor(my, torch.ones_like(my, dtype=torch.bool))
    new_data = func(args[0], mx.get_data(), my.get_data())
    new_mask = func(args[0], mx.get_mask(), my.get_mask())
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse])
def _to_sparse(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise TypeError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt, dtype=torch.bool))
    if mt.is_sparse_coo():
        return mt
    new_mask = func(_maybe_get_mask(args[0])).coalesce()
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse_csr])
def _to_sparse_csr(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    if mt.is_sparse_csr():
        return mt
    new_mask = func(_maybe_get_mask(args[0]))
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_dense])
def _to_dense(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    mt = args[0]
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    new_data = func(_get_data(args[0]))
    new_mask = func(_maybe_get_mask(args[0]))
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._indices])
def _indices(func, *args, **kwargs):
    # Assumes data is sparse
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0]).indices()
    return MaskedTensor(data, torch.ones_like(data).bool())


@register_dispatch_func([torch.ops.aten._values])
def _values(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0]).values()
    return MaskedTensor(data, torch.ones_like(data).bool())


@register_dispatch_func([torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors])
def _sparse_coo_tensor_with_dims_and_tensors(func, *args, **kwargs):
    new_args = list(args)
    if is_masked_tensor(args[-1]):
        new_args[-1] = args[-1].get_data()
    if is_masked_tensor(args[-2]):
        new_args[-2] = args[-2].get_data()

    new_data = func(*new_args, **kwargs)
    new_args[-1] = torch.ones_like(new_args[-1])
    new_mask = func(*new_args, **kwargs).bool()

    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten.is_same_size])
def is_same_size(func, *args, **kwargs):
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    return _get_data(args[0]).is_same_size(_get_data(args[1]))


@register_dispatch_func([torch.ops.aten._is_any_true])
def _is_any_true(func, *args, **kwargs):
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    if mask is None:
        raise ValueError(
            f"__torch_dispatch__, {func}: expected args[0] to be a MaskedTensor"
        )
    if data.dtype != torch.bool:
        raise ValueError(f"__torch_dispatch__, {func}: expected a boolean tensor")
    if data.is_sparse:
        raise ValueError(f"MaskedTensors with sparse data do not have {func}")

    return MaskedTensor(func(data & mask), torch.tensor(True))
