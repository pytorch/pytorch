from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from .fake_quantize import FakeQuantize
from .observer import MovingAverageMinMaxObserver  # , HistogramObserver, MovingAveragePerChannelMinMaxObserver

class FakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, quantize_fn, q_backward):
        ctx.others = q_backward

        X = quantize_fn(X)

        return X

    @staticmethod
    def backward(ctx, grad_X):
        quantize_backward = ctx.others
        if quantize_backward:
            scale, zero_point = torch._choose_qparams_per_tensor(grad_X, reduce_range=False)
            grad_X = torch.fake_quantize_per_tensor_affine(X, scale, zero_point, 0, 255)

        return grad_X, None, None


class FakeQuantizeBackward(FakeQuantize):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        quant_forward (bool): If true, quantize on the forward pass.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                 quant_forward=True, quant_backward=False, **observer_kwargs):
        super(FakeQuantizeBackward, self).__init__(observer, quant_min, quant_max, observer_kwargs)
        self.enable_fake_quant(quant_forward)
        self.quant_backward = quant_backward

    def forward(self, X):
        quantize_fn = super().forward
        return FakeQuantizeFunction.apply(X, quantize_fn, self.quant_backward)


default_fake_quant_backward = FakeQuantizeBackward.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    quant_forward=True,
    quant_backward=True,
    reduce_range=True,
)

default_weight_fake_quant_backward = FakeQuantizeBackward.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    quant_forward=True,
    quant_backward=True,
    reduce_range=False,
)
