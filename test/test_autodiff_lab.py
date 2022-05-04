# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import torch
from torch.autograd import gradcheck, gradgradcheck

class AttentionLab(torch.autograd.Function):
    """Define the forward and backward pass"""
    @staticmethod
    def forward(ctx, q, k, v):
        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        ctx.save_for_backward(q, k, v, a)
        return o, a


    @staticmethod
    def backward(ctx, grad_o, grad_a):
        q, k, v, a = ctx.saved_tensors

        grad_q = grad_k = grad_v = None

        tanh_x_2 = (1 - torch.square(a))
        grad_q = grad_o @ v.T * tanh_x_2 @ k + grad_a * tanh_x_2 @ k
        grad_k = (grad_o @ v.T * tanh_x_2).T @ q + (grad_a * tanh_x_2).T @ q
        grad_v = a.T @ grad_o

        return grad_q, grad_k, grad_v

def test_attention_grad():
    attention_lab = AttentionLab.apply
    q = torch.rand((2, 3), dtype=torch.double, requires_grad=True)
    k = torch.rand((2, 3), dtype=torch.double, requires_grad=True)
    v = torch.rand((2, 4), dtype=torch.double, requires_grad=True)
    input = (q, k, v)
    assert gradcheck(attention_lab, input)

def test_attention_grad_grad():
    attention_lab = AttentionLab.apply
    q = torch.rand((2, 3), dtype=torch.double, requires_grad=True)
    k = torch.rand((2, 3), dtype=torch.double, requires_grad=True)
    v = torch.rand((2, 4), dtype=torch.double, requires_grad=True)
    input = (q, k, v)
    assert gradgradcheck(attention_lab, input)
