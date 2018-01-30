import torch
from .module import Module
from .batchnorm import _BatchNorm
from .. import functional as F
from torch.nn.parameter import Parameter


class _InstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine)
        self._use_running_stats = False

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
            not self._use_running_stats, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def use_running_stats(self, mode=True):
        r"""Set using running statistics or instance statistics.

        Instance normalization usually use instance statistics in both training
        and evaluation modes. But users can set this method to use running
        statistics in the fashion similar to batch normalization in eval mode.
        """
        self._use_running_stats = mode


class LayerNorm(Module):
    r"""Applies Layer Normalization over a 2D input that is seen
    as a mini-batch of 1D inputs.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

    The mean and standard deviation are calculated for each object in a
    mini-batch (over `num_features`). Gamma and beta are
    optional learnable parameter vectors of size C (where C is the input size).

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features`. Specified only if learnable parameters
            are desired. Default: None
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5

    Shape:
        - Input: :math:`(N, C)`
        - Output: :math:`(N, C)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm()
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(100)
        >>> input = autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def __init__(self, num_features=None, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = num_features is not None
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return F.layer_norm(input, weight=self.weight, bias=self.bias,
                            eps=self.eps)

    def __repr__(self):
        if self.affine:
            return ('{name}({num_features}, eps={eps})'
                    .format(name=self.__class__.__name__, **self.__dict__))
        else:
            return ('{name}(eps={eps})'
                    .format(name=self.__class__.__name__, **self.__dict__))


class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization over a 3D input that is seen as a mini-batch.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

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
    r"""Applies Instance Normalization over a 4D input that is seen as a mini-batch of 3D inputs.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

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
    r"""Applies Instance Normalization over a 5D input that is seen as a mini-batch of 4D inputs.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

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
