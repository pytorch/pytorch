from .batchnorm import _BatchNorm
from .. import functional as F


class _InstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine)

    def forward(self, input):
        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = None, None
        if self.affine:
            weight = self.weight.repeat(b)
            bias = self.bias.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def eval(self):
        return self


class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization over a 2d or 3d input that is seen as a mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    At evaluation time (`.eval()`), the default behaviour of the InstanceNorm module stays the same
    i.e. running mean/variance is NOT used for normalization. One can force using stored
    mean and variance with `.train(False)` method.

    Args:
        num_features: num_features from an expected input of size `batch_size x num_features x width`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters. Default: False

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
    mean and variance with `.train(False)` method.

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters. Default: False

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
    mean and variance with `.train(False)` method.


    Args:
        num_features: num_features from an expected input of size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters. Default: False

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
