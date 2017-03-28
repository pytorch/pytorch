import torch
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend

from . import _all_functions


class GRUFused(Function):
    def forward(self, input_gate, hidden_gate, ibias, hbias, hidden):
        output = input_gate.new()
        type2backend[type(input_gate)].GRUFused_updateOutput(
            type2backend[type(input_gate)].library_state,
            input_gate, hidden_gate, ibias, hbias, hidden, output)
        self.save_for_backward(input_gate, hidden_gate)
        return output

    def backward(self, gradOutput):
        gradInput = gradOutput.new()
        input_gate, hidden_gate = self.saved_tensors

        type2backend[type(input_gate)].GRUFused_updateGradInput(
            type2backend[type(input_gate)].library_state,
            input_gate, hidden_gate, gradOutput, gradInput)

        gb1 = input_gate.sum(0)
        gb2 = hidden_gate.sum(0)

        return input_gate, hidden_gate, gb1, gb2, gradInput


_all_functions.append(GRUFused)
