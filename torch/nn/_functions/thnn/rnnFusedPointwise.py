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
        if ibias is not None:
            if ibias.dim() == 1:
                ibias.unsqueeze_(0)
            if hbias.dim() == 1:
                hbias.unsqueeze_(0)

        self.backend.GRUFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate, ibias, hbias, hx, hy)
        self.save_for_backward(input_gate, hidden_gate, ibias)
        return hy

    def backward(self, gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(grad_output)]
        gradInput = gradOutput.new()
        input_gate, hidden_gate, bias = self.saved_tensors

        igc = input_gate.clone()
        hgc = hidden_gate.clone()
        self.backend.GRUFused_updateGradInput(
            self.backend.library_state,
            igc, hgc, gradOutput, gradInput)
        if bias is not None:
            gb1 = igc.sum(0).squeeze()
            gb2 = hgc.sum(0).squeeze()

            return igc, hgc, gradInput, gb1, gb2
        else:
            return igc, hgc, gradInput


class LSTMFused(Function):
    def __init__(self):
        self.backend = None

    def forward(self, input_gate, hidden_gate, cx, ibias=None, hbias=None):
        if self.backend is None:
            self.backend = type2backend[type(input_gate)]
        hy = input_gate.new()
        cy = input_gate.new()
        if ibias is not None:
            if ibias.dim() == 1:
                ibias.unsqueeze_(0)
            if hbias.dim() == 1:
                hbias.unsqueeze_(0)
        self.backend.LSTMFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate,
            ibias, hbias,
            cx, hy, cy)
        self.save_for_backward(input_gate, hidden_gate, cx, cy, ibias)
        return hy, cy

    def backward(self, *gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(gradOutput[0])]

        gradInput = gradOutput[0].new()
        gradInputCell = gradOutput[0].new()
        saved_tens, local_go, cx, cy, bias = self.saved_tensors
        lgo_clone = local_go.clone()
        self.backend.LSTMFused_updateGradInput(
            self.backend.library_state,
            saved_tens, lgo_clone, cx, cy,
            gradOutput[0], gradOutput[1], gradInput)

        if bias is not None:
            gb1 = lgo_clone.sum(0).squeeze()
            gb2 = lgo_clone.sum(0).squeeze()

            return lgo_clone, lgo_clone, gradInput, gb1, gb2
        else:
            return lgo_clone, lgo_clone, gradInput
