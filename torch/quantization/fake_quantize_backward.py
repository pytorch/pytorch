from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from .fake_quantize import FakeQuantize
from .observer import MovingAverageMinMaxObserver

class _QuantizeBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X

    @staticmethod
    def backward(ctx, grad_X):
        scale, zero_point = torch._choose_qparams_per_tensor(grad_X, reduce_range=False)
        grad_X = torch.fake_quantize_per_tensor_affine(grad_X, scale, zero_point, 0, 255)

        return grad_X


class _FakeQuantizeWithBackward(FakeQuantize):
    r""" Simulate the quantize and dequantize operations in training time.
    See documentation for parent module torch.quantization.FakeQuantize.

    * :attr:`quantize_backward` controls the application of fake quantization on tensor gradients in
      the backward pass. This quantization is always done dynamically, and uses affine per-tensor
      quantization on unsigned 8-bit ints.

    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        quantize_forward (bool): If true, quantize on the forward pass. (default: True)
        quantize_backward (bool): If true, quantize on the backward pass. (default: False)
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                 quantize_forward=True, quantize_backward=False, **observer_kwargs):
        super(_FakeQuantizeWithBackward, self).__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.enable_fake_quant(quantize_forward)
        self.quantize_backward = quantize_backward

    def forward(self, X):
        X = super(_FakeQuantizeWithBackward, self).forward(X)
        if self.quantize_backward:
            X = _QuantizeBackwardFunction.apply(X)
        return X
