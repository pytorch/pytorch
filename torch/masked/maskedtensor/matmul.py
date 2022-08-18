# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import warnings

from .core import is_masked_tensor
from .creation import masked_tensor

__all__ = [
    'MaskedBmm',
    'masked_bmm'
]


class MaskedBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, attn_mask):
        if is_masked_tensor(q):
            raise TypeError("q cannot be a MaskedTensor")
        if not is_masked_tensor(k):
            raise TypeError("k must be a MaskedTensor")
        k_mask = k.get_mask()
        ctx.mark_non_differentiable(attn_mask, k_mask)
        ctx.save_for_backward(attn_mask, k_mask, q, k)
        attn = torch.bmm(q, k)
        return_mask = attn_mask.expand_as(attn.get_data())  # type: ignore[attr-defined]
        return masked_tensor(attn.get_data() + return_mask, return_mask == 0)  # type: ignore[attr-defined]

    @staticmethod
    def backward(ctx, grad):
        _, k_mask, q, k = ctx.saved_tensors
        grad_data = grad.get_data()

        k_trans = k.transpose(1, 2)
        q_grad = torch.bmm(grad_data, k_trans)

        q_trans = q.transpose(1, 2)
        k_grad = torch.bmm(q_trans, grad)
        k_grad = masked_tensor(k_grad.get_data(), k_mask)  # type: ignore[attr-defined]

        return q_grad, k_grad, None


def masked_bmm(q, k, attn_mask):
    return MaskedBmm.apply(q, k, attn_mask)


def _torch_matmul(func_name):
    func = getattr(torch.ops.aten, func_name)

    def matmul(input0, input1):
        if not is_masked_tensor(input0) and not is_masked_tensor(input1):
            warnings.warn("At least one of input0 or input1 must be a MaskedTensor")
            return NotImplemented

        data0 = input0.get_data() if is_masked_tensor(input0) else input0
        data1 = input1.get_data() if is_masked_tensor(input1) else input1
        mask0 = input0.get_mask() if is_masked_tensor(input0) else torch.ones_like(input0)
        mask1 = input1.get_mask() if is_masked_tensor(input1) else torch.ones_like(input1)

        input_data0 = data0.masked_fill(~input0.get_mask(), 0) if is_masked_tensor(input0) else data0
        input_data1 = data1.masked_fill(~input1.get_mask(), 0) if is_masked_tensor(input1) else data1

        result_data = func(input_data0, input_data1)
        result_mask = func(mask0.float(), mask1.float())
        result_mask = result_mask > 0

        if is_masked_tensor(input0) and is_masked_tensor(input1):
            if func is torch.ops.aten.mm:
                if not torch.equal(mask0, mask1.transpose(0, 1)):
                    raise ValueError("for torch.mm, input0_mask must equal input1_mask.transpose(0, 1)")
            if func is torch.ops.aten.bmm:
                if not torch.equal(mask0, mask1.transpose(1, 2)):
                    raise ValueError("for torch.bmm, input0_mask must equal input1_mask.transpose(1, 2)")

        return masked_tensor(result_data, result_mask)

    return matmul


MATMUL_NAMES = ["mm", "bmm"]

NATIVE_MATMUL_MAP = {
    getattr(torch.ops.aten, name): _torch_matmul(name) for name in MATMUL_NAMES
}

NATIVE_MATMUL_FNS = list(NATIVE_MATMUL_MAP.keys())


def _is_native_matmul(fn):
    return fn in NATIVE_MATMUL_FNS


def _apply_native_matmul(fn, *args, **kwargs):
    if fn in NATIVE_MATMUL_FNS:
        return NATIVE_MATMUL_MAP[fn](*args, **kwargs)
    return NotImplemented
