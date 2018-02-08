from .batchnorm import _BatchNorm
from .. import functional as F


class _InstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine)
        self._use_running_stats = False

    def forward(self, input):
        b = input.size(0)

        weight, bias = None, None
        if self.affine:
            weight = self.weight.repeat(b)
            bias = self.bias.repeat(b)

        training = not self._use_running_stats
        return F.instance_norm(input, weight=weight, bias=bias,
                               saved_running_mean=self.running_mean,
                               saved_running_var=self.running_var,
                               training=training, momentum=self.momentum,
                               eps=self.eps, affine=self.affine)

    def use_running_stats(self, mode=True):
        r"""Set using running statistics or instance statistics.

        Instance normalization usually use instance statistics in both training
        and evaluation modes. But users can set this method to use running
        statistics in the fashion similar to batch normalization in eval mode.
        """
        self._use_running_stats = mode


class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization over a 3d input that is seen as a mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    At evaluation time (`.eval()`), the default behaviour of the InstanceNorm module stays the same
    i.e. running mean/variance is NOT used for normalization. One can force using stored
    mean and variance with `.use_running_stats(mode=True)` method, and switch back to normal
    behavior with `.use_running_stats(mode=False)` method.

    Args:
        num_features: num_features from an expected input of size `batch_size x num_features x width`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 40))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm1d, self)._check_input_dim(input)


class InstanceNorm2d(_InstanceNorm):
    r"""Applies Instance Normalization over a 4d input that is seen as a mini-batch of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    At evaluation time (`.eval()`), the default behaviour of the InstanceNorm module stays the same
    i.e. running mean/variance is NOT used for normalization. One can force using stored
    mean and variance with `.use_running_stats(mode=True)` method, and switch back to normal
    behavior with `.use_running_stats(mode=False)` method.

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm2d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm2d, self)._check_input_dim(input)


class InstanceNorm3d(_InstanceNorm):
    r"""Applies Instance Normalization over a 5d input that is seen as a mini-batch of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    At evaluation time (`.eval()`), the default behaviour of the InstanceNorm module stays the same
    i.e. running mean/variance is NOT used for normalization. One can force using stored
    mean and variance with `.use_running_stats(mode=True)` method, and switch back to normal
    behavior with `.use_running_stats(mode=False)` method.


    Args:
        num_features: num_features from an expected input of size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm3d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm3d, self)._check_input_dim(input)
