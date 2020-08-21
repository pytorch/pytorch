from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional
import torch.nn.intrinsic as nni

class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    """Base class for the batch normalizations.

    You HAVE TO specify the following:
    - self._name => custom name of the current class
    - self._quantized_fn => Quantized function that takes the following args:
        1. x: Tensor
        2. weight: Tensor
        3. bias: Tensor
        4. running_mean: Tensor
        5. running_var: Tensor
        6. eps: float
        7. scale: float
        8. zero_point: int

    In addition to the members, you can also specify the class-level
    _INTRINSIC_BN_RELU, which is used to identify the equivalent fused version.
    """

    _INTRINSIC_BN_RELU = None

    def __init__(self, *args, **kwargs):
        super(_BatchNorm, self).__init__(*args, **kwargs)
        self._quantized_fn = None

    def forward(self, x):
        return self._quantized_fn(x, self.weight, self.bias, self.running_mean,
                                  self.running_var, self.eps, self.scale,
                                  self.zero_point)

    def _get_name(self):
        return self._name

    @classmethod
    def from_float(cls, mod):
        if cls._INTRINSIC_BN_RELU is not None and \
                type(mod) == cls._INTRINSIC_BN_RELU:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod

class BatchNorm1d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm1d`.
    """
    _INTRINSIC_BN_RELU = nni.BNReLU1d  # There is no intrinsic for 1d (yet!)

    def __init__(self, *args, **kwargs):
        super(BatchNorm1d, self).__init__(*args, **kwargs)
        self._quantized_fn = torch.ops.quantized.batch_norm1d
        self.scale = 1.0
        self.zero_point = 0
        self._name = 'QuantizedBatchNorm1d'


class BatchNorm2d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`.
    """
    _INTRINSIC_BN_RELU = nni.BNReLU2d

    def __init__(self, *args, **kwargs):
        super(BatchNorm2d, self).__init__(*args, **kwargs)
        self.scale = 1.0
        self.zero_point = 0
        self._name = 'QuantizedBatchNorm2d'
        self._quantized_fn = torch.ops.quantized.batch_norm2d


class BatchNorm3d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """
    _INTRINSIC_BN_RELU = nni.BNReLU3d

    def __init__(self, *args, **kwargs):
        super(BatchNorm3d, self).__init__(*args, **kwargs)
        self.scale = 1.0
        self.zero_point = 0
        self._name = 'QuantizedBatchNorm3d'
        self._quantized_fn = torch.ops.quantized.batch_norm3d
