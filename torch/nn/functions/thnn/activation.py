from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class PReLU(Function):

    def forward(self, input, weight):
        self._backend = type2backend[type(input)]
        output = input.new()
        self.num_parameters = weight.numel()
        if self.num_parameters == 1:
            self.num_parameters = 0
        self._backend.PReLU_updateOutput(
            self._backend.library_state,
            input,
            output,
            weight,
            self.num_parameters
        )
        self.save_for_backward(input, weight)
        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors
        # TODO: check if requires grad
        grad_input = input.new()
        self._backend.PReLU_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            self.num_parameters
        )

        buf = weight.new()
        buf2 = weight.new()
        # TODO: this won't have to be zeroed in the future
        grad_weight = weight.new().resize_as_(weight).zero_()
        self._backend.PReLU_accGradParameters(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            weight,
            grad_weight,
            buf,
            buf2,
            self.num_parameters,
            1
        )
        return grad_input, grad_weight


class Softmin(Function):

    def forward(self, input):
        self._backend = type2backend[type(input)]
        self.mininput = input.clone().mul(-1)
        output = input.new()
        self._backend.SoftMax_updateOutput(
            self._backend.library_state,
            self.mininput,
            output
        )
        self.save_for_backward(output)
        return output

    def backward(self, grad_output):
        output, = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SoftMax_updateGradInput(
            self._backend.library_state,
            self.mininput,
            grad_output,
            grad_input,
            output
        )
        return grad_input.mul(-1)


_all_functions.append(PReLU)
_all_functions.append(Softmin)

