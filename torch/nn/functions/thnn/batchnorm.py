from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class BatchNorm(Function):
    def __init__(self, *args):
        super(BatchNorm, self).__init__()
        self.additional_args = args

    def forward(self, input, *params):
        self.backend = type2backend[type(input)]
        self.save_for_backward(input, *params)
        self.num_features = input.size(1)
        # Add save_input and save_std
        self.additional_args = self.additional_args[:2] + \
            (input.new(self.num_features), input.new(self.num_features)) + \
            self.additional_args[2:]
        num_params = len(params)
        if num_params < 2:
            params = params + tuple(None for i in range(2 - num_params))
        additional_args = params + self.additional_args
        output = input.new().resizeAs_(input)
        self.backend.BatchNormalization_updateOutput(self.backend.library_state,
                input, output, *additional_args)
        return output

    def backward(self, grad_output):
        tensors = self.saved_tensors
        input, params = tensors[0], tensors[1:]
        grad_input = (input.new().resizeAs_(input).zero_()
                if self.needs_input_grad[0] else None,)
        grad_param = tuple(p.new().resizeAs_(p).zero_() if self.needs_input_grad[i+1]
                else None for i, p in enumerate(params))
        result_grad = grad_input + grad_param

        num_params = len(params)
        if num_params < 2:
            grad_param = grad_param + tuple(None for i in range(2 - num_params))

        weight_tuple = (params[0],) if len(params) > 0 else (None,)
        # backward takes scale instead of momentum
        additional_args = self.additional_args[:-2] + (1,) + self.additional_args[-1:]
        args = grad_input + grad_param + weight_tuple + additional_args
        self.backend.BatchNormalization_backward(self.backend.library_state,
                input, grad_output, *args)
        return result_grad


_all_functions.append(BatchNorm)

