from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional

class LayerNorm(torch.nn.LayerNorm):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)
    """

    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps=1e-5,
                 elementwise_affine=True):
        super(LayerNorm, self).__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.scale, output_zero_point=self.zero_point)

    def _get_name(self):
        return 'QuantizedLayerNorm'

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)
        return new_mod

class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of `torch.nn.GroupNorm`.
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedGroupNorm'

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_groups, mod.num_channels, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of `torch.nn.InstanceNorm1d`.
    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm1d'

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of `torch.nn.InstanceNorm2d`.
    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm2d'

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of `torch.nn.InstanceNorm2d`.
    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm3d'

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod
