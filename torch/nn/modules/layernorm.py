import torch
from .module import Module
from torch.nn.parameter import Parameter


class _LayerNorm(Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(_LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.fill_(1)
            self.bias.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.weight.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)

        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        return self.gamma * (input - mean) / (std + self.eps) + self.beta

        """
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
        """

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class LayerNorm1d(_LayerNorm):
    r"""Applies Layer Normalization over a 2d or 3d input that is seen as a mini-batch.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    Args:
        num_features: num_features from an expected input of size `batch_size x num_features x width`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.

    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm1d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 40))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        super(LayerNorm1d, self)._check_input_dim(input)


class LayerNorm2d(_LayerNorm):
    r"""Applies Layer Normalization over a 4d input that is seen as a mini-batch of 3d inputs.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm2d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(LayerNorm2d, self)._check_input_dim(input)


class LayerNorm3d(_LayerNorm):
    r"""Applies Layer Normalization over a 5d input that is seen as a mini-batch of 4d inputs.

    .. math::

        y = \gamma * \frac{x - \mu_x}{\sigma_x + \epsilon} + \beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    Gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm3d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(LayerNorm3d, self)._check_input_dim(input)
