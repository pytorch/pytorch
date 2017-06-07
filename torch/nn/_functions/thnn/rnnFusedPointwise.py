import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend


class GRUFused(Function):
    def __init__(self):
        self.backend = None

    def forward(self, input_gate, hidden_gate, hx, ibias=None, hbias=None):
        if self.backend is None:
            self.backend = type2backend[type(input_gate)]

        hy = input_gate.new()
        workspace = input_gate.new(hx.numel() * 5)

        self.has_bias = False
        if ibias is not None:
            self.has_bias = True
            if ibias.dim() == 1:
                ibias = ibias.unsqueeze(0)
            if hbias.dim() == 1:
                hbias = hbias.unsqueeze(0)

        self.backend.GRUFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate, ibias, hbias, hx, hy, workspace)

        self.workspace = workspace
        self.igate_size = input_gate.size()
        self.hgate_size = hidden_gate.size()

        return hy

    def backward(self, gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(grad_output)]

        gradInputHx = gradOutput.new()
        gradInInput = gradOutput.new(*self.igate_size)
        gradInHidden = gradOutput.new(*self.hgate_size)

        self.backend.GRUFused_updateGradInput(
            self.backend.library_state,
            gradInInput, gradInHidden, gradOutput, gradInputHx, self.workspace)

        if self.has_bias:
            gb1 = gradInInput.sum(0, keepdim=False)
            gb2 = gradInHidden.sum(0, keepdim=False)
            return gradInInput, gradInHidden, gradInputHx, gb1, gb2
        else:
            return gradInInput, gradInHidden, gradInputHx


class LSTMFused(Function):
    def __init__(self):
        self.backend = None

    def forward(self, input_gate, hidden_gate, cx, ibias=None, hbias=None):
        if self.backend is None:
            self.backend = type2backend[type(input_gate)]
        hy = input_gate.new()
        cy = input_gate.new()

        self.has_bias = False
        if ibias is not None:
            self.has_bias = True
            if ibias.dim() == 1:
                ibias = ibias.unsqueeze(0)
            if hbias.dim() == 1:
                hbias = hbias.unsqueeze(0)

        # input_gate gets overwritten with some intermediate values to use in backwards
        self.backend.LSTMFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate,
            ibias, hbias,
            cx, hy, cy)

        self.hgate_size = hidden_gate.size()
        self.save_for_backward(input_gate, cx, cy)

        return hy, cy

    def backward(self, *gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(gradOutput[0])]

        gradInputCx = gradOutput[0].new()
        gradInGates = gradOutput[0].new(*self.hgate_size)

        saved_tens, cx, cy = self.saved_tensors
        self.backend.LSTMFused_updateGradInput(
            self.backend.library_state,
            saved_tens, gradInGates, cx, cy,
            gradOutput[0], gradOutput[1], gradInputCx)

        if self.has_bias:
            gb1 = gradInGates.sum(0, keepdim=False)
            gb2 = gradInGates.sum(0, keepdim=False)

            return gradInGates, gradInGates, gradInputCx, gb1, gb2
        else:
            return gradInGates, gradInGates, gradInputCx
