import torch
from torch.autograd.function import Function
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn


class BatchNorm(Function):
    def __init__(self, running_mean, running_var, training, momentum, eps):
        super(BatchNorm, self).__init__()
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, weight=None, bias=None):
        self.save_for_backward(input, weight, bias)

        self.use_cudnn = (cudnn.is_acceptable(input)
                          and weight is not None and bias is not None)
        self.use_cudnn = False

        # temporary buffers used in forward and backward
        num_features = input.size(1)
        self._save_mean = input.new(num_features)
        self._save_std = input.new(num_features)

        output = input.new(input.size())

        if self.use_cudnn:
            torch._C._cudnn_batch_norm_forward(
                input, output, weight, bias,
                self.running_mean, self.running_var, self._save_mean,
                self._save_std, self.training, self.momentum, self.eps)
        else:
            backend = type2backend[type(input)]
            backend.BatchNormalization_updateOutput(
                backend.library_state, input, output, weight, bias,
                self.running_mean, self.running_var, self._save_mean,
                self._save_std, self.training, self.momentum, self.eps)

        return output

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None

        if self.needs_input_grad[0] or self.use_cudnn:
            grad_input = input.new(input.size())
        if (len(self.needs_input_grad) > 1 and self.needs_input_grad[1]) or self.use_cudnn:
            grad_weight = weight.new(weight.size()).zero_()
        if (len(self.needs_input_grad) > 1 and self.needs_input_grad[2]) or self.use_cudnn:
            grad_bias = bias.new(bias.size()).zero_()

        if self.use_cudnn:
            torch._C._cudnn_batch_norm_backward(
                input, grad_output, grad_input,
                grad_weight, grad_bias, weight,
                self.running_mean, self.running_var,
                self._save_mean, self._save_std, self.training, self.eps)
        else:
            backend = type2backend[type(input)]
            backend.BatchNormalization_backward(
                backend.library_state, input, grad_output, grad_input,
                grad_weight, grad_bias, weight,
                self.running_mean, self.running_var,
                self._save_mean, self._save_std, self.training, 1.0, self.eps)

        return grad_input, grad_weight, grad_bias
