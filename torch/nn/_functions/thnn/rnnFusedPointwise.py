import torch
from torch.autograd.function import Function, InplaceFunction, once_differentiable
from torch._thnn import type2backend


class GRUFused(Function):
    @staticmethod
    def forward(ctx, input_gate, hidden_gate, hx, ibias=None, hbias=None):
        ctx.backend = type2backend[input_gate.type()]

        hy = input_gate.new()
        workspace = input_gate.new(hx.numel() * 5)

        ctx.has_bias = False
        if ibias is not None:
            ctx.has_bias = True
            if ibias.dim() == 1:
                ibias = ibias.unsqueeze(0)
            if hbias.dim() == 1:
                hbias = hbias.unsqueeze(0)

        ctx.backend.GRUFused_updateOutput(
            ctx.backend.library_state,
            input_gate, hidden_gate, ibias, hbias, hx, hy, workspace)

        ctx.workspace = workspace
        ctx.igate_size = input_gate.size()
        ctx.hgate_size = hidden_gate.size()

        return hy

    @staticmethod
    @once_differentiable
    def backward(ctx, gradOutput):
        ctx.backend = type2backend[gradOutput.type()]

        gradInputHx = gradOutput.new()
        gradInInput = gradOutput.new(*ctx.igate_size)
        gradInHidden = gradOutput.new(*ctx.hgate_size)

        ctx.backend.GRUFused_updateGradInput(
            ctx.backend.library_state,
            gradInInput, gradInHidden, gradOutput, gradInputHx, ctx.workspace)

        gb1 = gb2 = None
        if ctx.has_bias:
            gb1 = gradInInput.sum(0, keepdim=False)
            gb2 = gradInHidden.sum(0, keepdim=False)
        return gradInInput, gradInHidden, gradInputHx, gb1, gb2


class LSTMFused(Function):
    @staticmethod
    def forward(ctx, input_gate, hidden_gate, cx, ibias=None, hbias=None):
        ctx.backend = type2backend[input_gate.type()]
        hy = input_gate.new()
        cy = input_gate.new()

        ctx.has_bias = False
        if ibias is not None:
            ctx.has_bias = True
            if ibias.dim() == 1:
                ibias = ibias.unsqueeze(0)
            if hbias.dim() == 1:
                hbias = hbias.unsqueeze(0)

        # input_gate gets overwritten with some intermediate values to use in backwards
        ctx.backend.LSTMFused_updateOutput(
            ctx.backend.library_state,
            input_gate, hidden_gate,
            ibias, hbias,
            cx, hy, cy)

        ctx.hgate_size = hidden_gate.size()
        ctx.save_for_backward(input_gate, cx, cy)

        return hy, cy

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradOutput):
        ctx.backend = type2backend[gradOutput[0].type()]
        gradInputCx = gradOutput[0].new()
        gradInGates = gradOutput[0].new(*ctx.hgate_size)

        saved_tens, cx, cy = ctx.saved_tensors
        ctx.backend.LSTMFused_updateGradInput(
            ctx.backend.library_state,
            saved_tens, gradInGates, cx, cy,
            gradOutput[0], gradOutput[1], gradInputCx)

        gb1 = gb2 = None
        if ctx.has_bias:
            gb1 = gradInGates.sum(0, keepdim=False)
            gb2 = gradInGates.sum(0, keepdim=False)

        return gradInGates, gradInGates, gradInputCx, gb1, gb2
