from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional

class BatchNorm2d(torch.nn.BatchNorm2d):
    r"""Applies Quantized Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:

        >>> m = nn.quantized.BatchNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> quantized_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__(num_features)
        self.eps = eps
        self.scale = 1.0
        self.zero_point = 0

    def forward(self, input):
        return torch.ops.quantized.batch_norm(input, self.weight, self.bias, self.running_mean,
                                              self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedBatchNorm2d'

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == torch.nn.BatchNorm2d,\
            "QuantizedBatchNorm2d expects an instance of BatchNorm2d"
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = BatchNorm2d(mod.num_features, mod.eps)
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod

class BatchNorm3d(torch.nn.BatchNorm3d):
    r"""Applies Quantized Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:

        >>> m = nn.quantized.BatchNorm3d(100)
        >>> input = torch.randn(20, 100, 25, 35, 45)
        >>> quantized_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm3d, self).__init__(num_features)
        self.eps = eps
        self.scale = 1.0
        self.zero_point = 0

    def forward(self, input):
        return torch.ops.quantized.batch_norm3d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedBatchNorm3d'

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == torch.nn.BatchNorm3d,\
            "QuantizedBatchNorm3d expects an instance of BatchNorm3d"
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = BatchNorm3d(mod.num_features, mod.eps)
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
