# Copyright (c) Meta Platforms, Inc. and affiliates

import logging

import torch

from .creation import masked_tensor


class MaskedBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, attn_mask):
        from .core import is_masked_tensor

        assert not is_masked_tensor(q)
        assert is_masked_tensor(k)
        k_mask = k.get_mask()
        ctx.mark_non_differentiable(attn_mask, k_mask)
        ctx.save_for_backward(attn_mask, k_mask, q, k)
        attn = torch.bmm(q, k)
        return_mask = attn_mask.expand_as(attn.get_data())  # type: ignore[attr-defined]
        return masked_tensor(attn.get_data() + return_mask, return_mask == 0)  # type: ignore[attr-defined]

    @staticmethod
    def backward(ctx, grad):
        attn_mask, k_mask, q, k = ctx.saved_tensors
        grad_data = grad.get_data()

        k_trans = k.transpose(1, 2)
        q_grad = torch.bmm(grad_data, k_trans)

        q_trans = q.transpose(1, 2)
        k_grad = torch.bmm(q_trans, grad)
        k_grad = masked_tensor(k_grad.get_data(), k_mask)  # type: ignore[attr-defined]

        return q_grad, k_grad, None


def masked_bmm(q, k, attn_mask):
    return MaskedBmm.apply(q, k, attn_mask)


def torch_matmul(func_name):
    func = getattr(torch.ops.aten, func_name)

    def matmul(input0, input1):
        from .core import is_masked_tensor, MaskedTensor

        logging.debug("Calling matmul with type({type(input0)}, {type(input1)})")
        if is_masked_tensor(input0) and is_masked_tensor(input1):
            data0 = input0.get_data()
            data1 = input1.get_data()
            input_mask0 = input0.get_mask()
            input_mask1 = input1.get_mask()
            input_data0 = data0.masked_fill(~input_mask0, 0)
            input_data1 = data1.masked_fill(~input_mask1, 0)
            result_data = func(input_data0, input_data1)
            result_mask = func(input_mask0.float(), input_mask1.float())
            result_mask = result_mask > 0
            if func is torch.ops.aten.mm:
                assert torch.equal(input_mask0, input_mask1.transpose(0, 1))
            if func is torch.ops.aten.bmm:
                assert torch.equal(input_mask0, input_mask1.transpose(1, 2))
            return MaskedTensor(result_data, result_mask)
        if is_masked_tensor(input0) and not is_masked_tensor(input1):
            data0 = input0.get_data()
            input_mask0 = input0.get_mask()
            input_data0 = data0.masked_fill(~input_mask0, 0)
            result_data = func(input_data0, input1)
            result_mask = func(input_mask0.float(), torch.ones_like(input1).float())
            result_mask = result_mask > 0
            return MaskedTensor(result_data, result_mask)
        if not is_masked_tensor(input0) and is_masked_tensor(input1):
            data1 = input1.get_data()
            input_mask1 = input1.get_mask()
            input_data1 = data1.masked_fill(~input_mask1, 0)
            result_data = func(input0, input_data1)
            result_mask = func(torch.ones_like(input0).float(), input_mask1.float())
            result_mask = result_mask > 0
            return MaskedTensor(result_data, result_mask)

        return NotImplemented

    return matmul


MATMUL_NAMES = ["mm", "bmm"]

NATIVE_MATMUL_MAP = {
    getattr(torch.ops.aten, name): torch_matmul(name) for name in MATMUL_NAMES
}

NATIVE_MATMUL_FNS = list(NATIVE_MATMUL_MAP.keys())


def is_native_matmul(fn):
    return fn in NATIVE_MATMUL_FNS


def apply_native_matmul(fn, *args, **kwargs):
    if fn in NATIVE_MATMUL_FNS:
        return NATIVE_MATMUL_MAP[fn](*args, **kwargs)
    return NotImplemented
