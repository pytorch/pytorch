import torch
from torch.autograd import Variable

from .module import Module


class Threshold(Module):
    """Thresholds each element of the input Tensor
    Threshold is defined as:
         y =  x        if x >= threshold
              value    if x <  threshold
    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        Tensor of same dimension and shape as the input
    Examples:
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
        return self._backend.Threshold(self.threshold, self.value, self.inplace)(input)


class ReLU(Threshold):
    """Applies the rectified linear unit function element-wise ReLU(x)= max(0,x)
    Args:
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: relu.png
    Examples:
        >>> m = nn.ReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)


class RReLU(Module):
    def __init__(self, lower=1./8, upper=1./3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return self._backend.RReLU(self.lower, self.upper, self.train,
                self.inplace)(input)


class Hardtanh(Module):
    """Applies the HardTanh function element-wise
    HardTanh is defined as:
       f(x) = +1, if x  >  1
       f(x) = -1, if x  < -1
       f(x) =  x,  otherwise
    The range of the linear region [-1, 1] can be adjusted
    Args:
        min_value: minimum value of the linear region range
        max_value: maximum value of the linear region range
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: htanh.png
    Examples:
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
        return self._backend.Hardtanh(self.min_val, self.max_val, self.inplace)(input)


class ReLU6(Hardtanh):
    """Applies the element-wise function ReLU6(x) = min( max(0,x), 6)
    Args:
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: relu6.png
    Examples:
        >>> m = nn.ReLU6()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)


class Sigmoid(Module):
    """Applies the element-wise function sigmoid(x) = 1 / ( 1 + exp(-x))
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: sigmoid.png
    Examples:
        >>> m = nn.Sigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.Sigmoid()(input)


class Tanh(Module):
    """Applies element-wise, Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: tanh.png
    Examples:
        >>> m = nn.Tanh()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.Tanh()(input)


class ELU(Module):
    """Applies element-wise, ELU(x) = max(0,x) + min(0, alpha * (exp(x) - 1))
    Args:
        alpha: the alpha value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: elu.png
    Examples:
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
        return self._backend.ELU(self.alpha, self.inplace)(input)


class Hardshrink(Module):
    """Applies the hard shrinkage function element-wise
    Hardshrink is defined as f(x) = x, if x >  lambda
                             f(x) = x, if x < -lambda
                             f(x) = 0, otherwise
    Args:
        lambd: the lambda value for the Hardshrink formulation. Default: 0.5
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: hshrink.png
    Examples:
        >>> m = nn.Hardshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return self._backend.Hardshrink(self.lambd)(input)


class LeakyReLU(Module):
    """Applies element-wise, f(x) = max(0, x) + negative_slope * min(0, x)
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Examples:
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
        return self._backend.LeakyReLU(self.negative_slope, self.inplace)(input)


class LogSigmoid(Module):
    """Applies element-wise LogSigmoid(x) = log( 1 / (1 + exp(-x_i)))
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: logsigmoid.png
    Examples:
        >>> m = nn.LogSigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.LogSigmoid()(input)


class Softplus(Module):
    """Applies element-wise SoftPlus(x) = 1/beta * log(1 + exp(beta * x_i))
    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    For numerical stability the implementation reverts to the linear function
    for inputs above a certain value.
    Args:
        beta: the beta value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: softplus.png
    Examples:
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
        return self._backend.Softplus(self.beta, self.threshold)(input)


class Softshrink(Module):
    """Applies the soft shrinkage function elementwise
    SoftShrinkage operator is defined as:
        f(x) = x-lambda, if x > lambda >  f(x) = x+lambda, if x < -lambda
        f(x) = 0, otherwise
    Args:
        lambd: the lambda value for the Softshrink formulation. Default: 0.5
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: sshrink.png
    Examples:
        >>> m = nn.Softshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return self._backend.Softshrink(self.lambd)(input)


class PReLU(Module):
    """Applies element-wise the function PReLU(x) = max(0,x) + a * min(0,x)
    Here "a" is a learnable parameter.
    When called without arguments, nn.PReLU() uses a single parameter "a"
    across all input channels. If called with nn.PReLU(nChannels), a separate
    "a" is used for each input channel.
    Note that weight decay should not be used when learning "a" for good
    performance.
    Args:
        num_parameters: number of "a" to learn. Default: 1
        init: the initial value of "a". Default: 0.25
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: prelu.png
    Examples:
        >>> m = nn.PReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__(
            weight=Variable(torch.Tensor(num_parameters).fill_(init))
        )

    def forward(self, input):
        return self._backend.PReLU()(input, self.weight)


class Softsign(Module):
    """Applies element-wise, the function Softsign(x) = x / (1 + |x|)
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Image: softsign.png
    Examples:
        >>> m = nn.Softsign()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.Softsign()(input)


class Tanhshrink(Module):
    """Applies element-wise, Tanhshrink(x) = x - Tanh(x)
    Input Shape:   Any : Tensor of any size and dimension
    Output Shape: Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input
    Examples:
        >>> m = nn.Tanhshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        tanh = self._backend.Tanh()(input)
        return input - tanh


class Softmin(Module):
    """Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1
    Softmin(x) = exp(-x_i - shift) / sum_j exp(-x_j - shift)
                 where shift = max_i - x_i
    Input Shape: [ * , * ] : 2D Tensor of any size
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]
    Image: softmin.png
    Examples:
        >>> m = nn.Softmin()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.Softmin()(input)


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
                          where shift = max_i x_i

    Input Shape: [ * , * ] : 2D Tensor of any size
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]
    Image: softmax.png
    Notes:
        Note that this module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use Logsoftmax instead (it's faster).
    Examples:
        >>> m = nn.Softmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        assert input.dim() == 2, 'Softmax requires a 2D tensor as input'
        return self._backend.Softmax()(input)


class Softmax2d(Module):
    """Applies SoftMax over features to each spatial location
    When given an image of Channels x Height x Width, it will
    apply Softmax to each location [Channels, h_i, w_j]

    Input Shape: [ * , * , * , * ] : 4D Tensor of any size
    Output Shape:             Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]
    Examples:
        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = autograd.Variable(torch.randn(2, 3, 12, 13))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return self._backend.Softmax()(input)

class LogSoftmax(Module):
    """Applies the Log(Softmax(x)) function to an n-dimensional input Tensor.
    The LogSoftmax formulation can be simplified as
         f_i(x) = log(1 / a * exp(x_i)) where a = sum_j exp(x_j) .
    Input Shape: [ * , * ] : 2D Tensor of any size
    Output Shape:     Same : Output has the same shape as input
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)
    Image: logsoftmax.png
    Examples:
        >>> m = nn.LogSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """
    def forward(self, input):
        return self._backend.LogSoftmax()(input)



# TODO: RReLU
