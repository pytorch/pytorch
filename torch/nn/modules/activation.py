import torch
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F


class Threshold(Module):
    """Thresholds each element of the input Tensor

    Threshold is defined as::

         y =  x        if x >= threshold
              value    if x <  threshold

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.threshold) \
            + ', ' + str(self.value) \
            + inplace_str + ')'


class ReLU(Threshold):
    """Applies the rectified linear unit function element-wise :math:`{ReLU}(x)= max(0, x)`

    Args:
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class RReLU(Module):

    def __init__(self, lower=1. / 8, upper=1. / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.lower) \
            + ', ' + str(self.upper) \
            + inplace_str + ')'


class Hardtanh(Module):
    """Applies the HardTanh function element-wise

    HardTanh is defined as::

       f(x) = +1, if x  >  1
       f(x) = -1, if x  < -1
       f(x) =  x,  otherwise

    The range of the linear region :math:`[-1, 1]` can be adjusted

    Args:
        min_value: minimum value of the linear region range
        max_value: maximum value of the linear region range
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.HardTanh(-2, 2)
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, min_value=-1, max_value=1, inplace=False):
        super(Hardtanh, self).__init__()
        self.min_val = min_value
        self.max_val = max_value
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'min_val=' + str(self.min_val) \
            + ', max_val=' + str(self.max_val) \
            + inplace_str + ')'


class ReLU6(Hardtanh):
    """Applies the element-wise function :math:`{ReLU6}(x) = min(max(0,x), 6)`

    Args:
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU6()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class Sigmoid(Module):
    """Applies the element-wise function :math:`f(x) = 1 / ( 1 + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Tanh(Module):
    """Applies element-wise, :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Tanh()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return torch.tanh(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class ELU(Module):
    """Applies element-wise, :math:`f(x) = max(0,x) + min(0, alpha * (exp(x) - 1))`

    Args:
        alpha: the alpha value for the ELU formulation
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ELU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, alpha=1., inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'alpha=' + str(self.alpha) \
            + inplace_str + ')'


class Hardshrink(Module):
    """Applies the hard shrinkage function element-wise
    Hardshrink is defined as::
        f(x) = x, if x >  lambda
        f(x) = x, if x < -lambda
        f(x) = 0, otherwise

    Args:
        lambd: the lambda value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.hardshrink(input, self.lambd)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.lambd) + ')'


class LeakyReLU(Module):
    """Applies element-wise, :math:`f(x) = max(0, x) + {negative\_slope} * min(0, x)`

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.negative_slope) \
            + inplace_str + ')'


class LogSigmoid(Module):
    """Applies element-wise :math:`LogSigmoid(x) = log( 1 / (1 + exp(-x_i)))`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.logsigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softplus(Module):
    """Applies element-wise :math:`f(x) = 1/beta * log(1 + exp(beta * x_i))`

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    for inputs above a certain value.

    Args:
        beta: the beta value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Softplus()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, beta=1, threshold=20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'beta=' + str(self.beta) \
            + ', threshold=' + str(self.threshold) + ')'


class Softshrink(Module):
    """Applies the soft shrinkage function elementwise

    SoftShrinkage operator is defined as::

        f(x) = x-lambda, if x > lambda >  f(x) = x+lambda, if x < -lambda
        f(x) = 0, otherwise

    Args:
        lambd: the lambda value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Softshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.softshrink(input, self.lambd)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.lambd) + ')'


class PReLU(Module):
    """Applies element-wise the function :math:`PReLU(x) = max(0,x) + a * min(0,x)`
    Here "a" is a learnable parameter.
    When called without arguments, nn.PReLU() uses a single parameter "a"
    across all input channels. If called with nn.PReLU(nChannels), a separate
    "a" is used for each input channel.


    .. note::
        weight decay should not be used when learning "a" for good performance.

    Args:
        num_parameters: number of "a" to learn. Default: 1
        init: the initial value of "a". Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.PReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return F.prelu(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'


class Softsign(Module):
    """Applies element-wise, the function :math:`f(x) = x / (1 + |x|)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Softsign()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.softsign(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Tanhshrink(Module):
    """Applies element-wise, :math:`Tanhshrink(x) = x - Tanh(x)`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.tanhshrink(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softmin(Module):
    """Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `(0, 1)` and sum to 1

    :math:`f(x) = exp(-x_i - {shift}) / sum_j exp(-x_j - {shift})`

    where :math:`{shift} = max_i - x_i`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.softmin(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as :math:`f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)`
    where `shift = max_i x_i`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use Logsoftmax instead (it's faster).

    Examples::

        >>> m = nn.Softmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        assert input.dim() == 2, 'Softmax requires a 2D tensor as input'
        return F.softmax(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softmax2d(Module):
    """Applies SoftMax over features to each spatial location

    When given an image of Channels x Height x Width, it will

    apply Softmax to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = autograd.Variable(torch.randn(2, 3, 12, 13))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return F.softmax(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class LogSoftmax(Module):
    """Applies the Log(Softmax(x)) function to an n-dimensional input Tensor.
    The LogSoftmax formulation can be simplified as

    :math:`f_i(x) = log(1 / a * exp(x_i))` where :math:`a = sum_j exp(x_j)`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.log_softmax(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
