import torch
from torch.autograd import Variable

from .module import Module

# TODO: check contiguous in THNN
# TODO: use separate backend functions?
class _BatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):

        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        if self.affine:
            super(_BatchNorm, self).__init__(
                weight=Variable(torch.Tensor(num_features)),
                bias=Variable(torch.Tensor(num_features)),
            )
        else:
            super(_BatchNorm, self).__init__()
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != self.expected_dim:
            raise RuntimeError('only mini-batch supported ({}D tensor), got {}D tensor instead'.format(self.expected_dim, input.dim()))
        if input.size(1) != self.running_mean.nelement():
            raise RuntimeError('got {}-feature tensor, expected {}'.format(input.size(1), self.running_mean.nelement()))

    def forward(self, input):
        self._check_input_dim(input)
        args = (input,)
        if self.weight is not None:
            args = args + (self.weight, self.bias)
        return self._backend.BatchNorm(self.running_mean,
                self.running_var, self.train, self.momentum, self.eps)(*args)

    def type(self, type, *forwarded_args):
        self.running_var = self.running_var.type(type, *forwarded_args)
        self.running_mean = self.running_mean.type(type, *forwarded_args)
        return super(_BatchNorm, self).type(type, *forwarded_args)


class BatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over a 2d input that is seen as a mini-batch of 1d inputs

    ```
                  x - mean(x)
    y =  ----------------------------- * gamma + beta
          standard_deviation(x) + eps
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size N (where N is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1
    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: the size of each 1D input in the mini-batch
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.
    Input Shape: [ * , num_features ] : 2D Tensor of nBatches x num_features
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a normalized tensor in the batch dimension
    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100))
        >>> output = m.forward(input)
    """
    expected_dim = 2


class BatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch of 3d inputs

    ```
                  x - mean(x)
    y =  ----------------------------- * gamma + beta
          standard_deviation(x) + eps
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size N (where N is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1
    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.
    Input Shape: [ * , num_features , *, * ] : 4D Tensor of batch_size x num_features x height x width
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a normalized tensor in the batch dimension
    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m.forward(input)
    """
    expected_dim = 4


class BatchNorm3d(_BatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch of 4d inputs

    ```
                  x - mean(x)
    y =  ----------------------------- * gamma + beta
          standard_deviation(x) + eps
    ```

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size N (where N is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1
    During evaluation, this running mean/variance is used for normalization.

    Args:
        num_features: num_features from an expected input of size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to true, gives the layer learnable affine parameters.
    Input Shape: [ * , num_features , * , * , * ] : 5D Tensor of batch_size x num_features x depth x height x width
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a normalized tensor in the batch dimension
    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m.forward(input)
    """
    expected_dim = 5

