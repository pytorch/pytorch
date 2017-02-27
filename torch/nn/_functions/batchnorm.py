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

        # don't use cuDNN for half inputs because cuDNN requires the weight and
        # bias tensors to be floats, unlike THCUNN which requires half tensors.
        self.use_cudnn = (cudnn.is_acceptable(input) and
                          weight is not None and bias is not None and
                          not isinstance(input, torch.cuda.HalfTensor))

        # temporary buffers used in forward and backward
        num_features = input.size(1)
        _save_mean = input.new(num_features)
        _save_std = input.new(num_features)

        output = input.new(input.size())

        if self.use_cudnn:
            torch._C._cudnn_batch_norm_forward(
                input, output, weight, bias,
                self.running_mean, self.running_var, _save_mean,
                _save_std, self.training, self.momentum, self.eps)
        else:
            backend = type2backend[type(input)]
            backend.BatchNormalization_updateOutput(
                backend.library_state, input, output, weight, bias,
                self.running_mean, self.running_var, _save_mean,
                _save_std, self.training, self.momentum, self.eps)

        if self.requires_grad:
            self._save_mean = _save_mean
            self._save_std = _save_std

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

        if self.use_cudnn and self.training:
            # cudnn does not support backward in evaluate mode
            torch._C._cudnn_batch_norm_backward(
                input, grad_output, grad_input,
                grad_weight, grad_bias, weight,
                self.running_mean, self.running_var,
                self._save_mean, self._save_std, self.training, self.eps)
        else:
            grad_output = grad_output.contiguous()
            backend = type2backend[type(input)]
            backend.BatchNormalization_backward(
                backend.library_state, input, grad_output, grad_input,
                grad_weight, grad_bias, weight,
                self.running_mean, self.running_var,
                self._save_mean, self._save_std, self.training, 1.0, self.eps)

        return grad_input, grad_weight, grad_bias
