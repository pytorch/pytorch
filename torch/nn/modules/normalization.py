import torch
import numbers
from torch.nn.parameter import Parameter
from .module import Module
from .batchnorm import _BatchNorm
from .. import functional as F


class LocalResponseNorm(Module):
    r"""Applies local response normalization over an input signal composed
    of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, ...)`
        - Output: :math:`(N, C, ...)` (same shape as input)

    Examples:
        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)
    """

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(input, self.size, self.alpha, self.beta,
                                     self.k)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.size) \
            + ', alpha=' + str(self.alpha) \
            + ', beta=' + str(self.beta) \
            + ', k=' + str(self.k) + ')'


class CrossMapLRN2d(Module):

    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(CrossMapLRN2d, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return self._backend.CrossMapLRN2d(self.size, self.alpha, self.beta,
                                           self.k)(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.size) \
            + ', alpha=' + str(self.alpha) \
            + ', beta=' + str(self.beta) \
            + ', k=' + str(self.k) + ')'


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x]} + \epsilon} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions with shape specified by :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable parameters of :attr:`normalized_shape` if
    :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    By default, this layer uses statistics computed from input data in both
    training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \hat{x}_\text{new} + \text{momemtum} x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
                    \times \ldots \times \text{normalized_shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension with that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples:
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

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1,
                 elementwise_affine=True, track_running_stats=False):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.momentum = momentum
        self.elementwise_affine = elementwise_affine
        self.track_running_stats = track_running_stats
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.running_mean, self.running_var,
            self.weight, self.bias, self.training or not self.track_running_stats,
            self.momentum, self.eps)

    def __repr__(self):
        return ('{name}({normalized_shape}, eps={eps}, momentum={momentum},'
                ' elementwise_affine={elementwise_affine},'
                ' track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))


# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
