import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend


class GRUFused(Function):
    def forward(self, input_gate, hidden_gate, hx, ibias=None, hbias=None):
        hy = input_gate.new()
        type2backend[type(input_gate)].GRUFused_updateOutput(
            type2backend[type(input_gate)].library_state,
            input_gate, hidden_gate, ibias, hbias, hx, hy)
        self.save_for_backward(input_gate, hidden_gate, ibias)
        return hy

    def backward(self, gradOutput):
        gradInput = gradOutput.new()
        input_gate, hidden_gate, bias = self.saved_tensors

        type2backend[type(input_gate)].GRUFused_updateGradInput(
            type2backend[type(input_gate)].library_state,
            input_gate, hidden_gate, gradOutput, gradInput)
        if bias is not None:
            gb1 = input_gate.sum(0).squeeze()
            gb2 = hidden_gate.sum(0).squeeze()

            return input_gate, hidden_gate, gradInput, gb1, gb2
        else:
            return input_gate, hidden_gate, gradInput


class LSTMFused(Function):
    def forward(self, input_gate, hidden_gate, cx, ibias=None, hbias=None):
        hy = input_gate.new()
        cy = input_gate.new()
        type2backend[type(input_gate)].LSTMFused_updateOutput(
            type2backend[type(input_gate)].library_state,
            input_gate, hidden_gate,
            ibias, hbias,
            cx, hy, cy)
        self.save_for_backward(input_gate, hidden_gate, cx, cy, ibias)
        return hy, cy

    def backward(self, *gradOutput):
        gradInput = gradOutput[0].new()
        gradInputCell = gradOutput[0].new()
        saved_tens, local_go, cx, cy, bias = self.saved_tensors

        type2backend[type(saved_tens)].LSTMFused_updateGradInput(
            type2backend[type(saved_tens)].library_state,
            saved_tens, local_go, cx, cy,
            gradOutput[0], gradOutput[1], gradInput)

        if bias is not None:
            gb1 = local_go.sum(0).squeeze()
            gb2 = local_go.sum(0).squeeze()

            return local_go, local_go, gradInput, gb1, gb2
        else:
            return local_go, local_go, gradInput
