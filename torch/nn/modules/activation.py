import warnings
import torch
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F
from ..._jit_internal import weak_module, weak_script_method


@weak_module
class Threshold(Module):
    r"""Thresholds each element of the input Tensor

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['threshold', 'value', 'inplace']

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    @weak_script_method
    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )


@weak_module
class ReLU(Threshold):
    r"""Applies the rectified linear unit function element-wise
    :math:`\text{ReLU}(x)= \max(0, x)`

    .. image:: scripts/activation_images/ReLU.png

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0., 0., inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


@weak_module
class RReLU(Module):
    r"""Applies the randomized leaky rectified liner unit function, element-wise,
    as described in the paper:

    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.

     See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    __constants__ = ['lower', 'upper', 'inplace']

    def __init__(self, lower=1. / 8, upper=1. / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    @weak_script_method
    def forward(self, input):
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)


@weak_module
class Hardtanh(Module):
    r"""Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    .. image:: scripts/activation_images/Hardtanh.png

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['min_val', 'max_val', 'inplace']

    def __init__(self, min_val=-1., max_val=1., inplace=False, min_value=None, max_value=None):
        super(Hardtanh, self).__init__()
        if min_value is not None:
            warnings.warn("keyword argument min_value is deprecated and renamed to min_val")
            min_val = min_value
        if max_value is not None:
            warnings.warn("keyword argument max_value is deprecated and renamed to max_val")
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    @weak_script_method
    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(
            self.min_val, self.max_val, inplace_str
        )


@weak_module
class ReLU6(Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0., 6., inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


@weak_module
class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        return torch.sigmoid(input)


@weak_module
class Tanh(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        return torch.tanh(input)


@weak_module
class ELU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ELU.png

    Examples::

        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['alpha', 'inplace']

    def __init__(self, alpha=1., inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    @weak_script_method
    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


@weak_module
class CELU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/CELU.png

    Examples::

        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """
    __constants__ = ['alpha', 'inplace']

    def __init__(self, alpha=1., inplace=False):
        super(CELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    @weak_script_method
    def forward(self, input):
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


@weak_module
class SELU(Module):
    r"""Applied element-wise, as:

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    .. image:: scripts/activation_images/SELU.png

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.inplace = inplace

    @weak_script_method
    def forward(self, input):
        return F.selu(input, self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


@weak_module
class GLU(Module):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input vector and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.GLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    __constants__ = ['dim']

    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    @weak_script_method
    def forward(self, input):
        return F.glu(input, self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


@weak_module
class Hardshrink(Module):
    r"""Applies the hard shrinkage function element-wise:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']

    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    @weak_script_method
    def forward(self, input):
        return F.hardshrink(input, self.lambd)

    def extra_repr(self):
        return '{}'.format(self.lambd)


@weak_module
class LeakyReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


    or

    .. math::
        \text{LeakyRELU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/LeakyReLU.png

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace', 'negative_slope']

    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    @weak_script_method
    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


@weak_module
class LogSigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/LogSigmoid.png

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        return F.logsigmoid(input)


@weak_module
class Softplus(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    for inputs above a certain value.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Softplus.png

    Examples::

        >>> m = nn.Softplus()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['beta', 'threshold']

    def __init__(self, beta=1, threshold=20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    @weak_script_method
    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


@weak_module
class Softshrink(Module):
    r"""Applies the soft shrinkage function elementwise:

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']

    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    @weak_script_method
    def forward(self, input):
        return F.softshrink(input, self.lambd)

    def extra_repr(self):
        return str(self.lambd)


@weak_module
class PReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Attributes:
        weight (Tensor): the learnable weights of shape (attr:`num_parameters`).
            The attr:`dtype` is default to

    .. image:: scripts/activation_images/PReLU.png

    Examples::

        >>> m = nn.PReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    @weak_script_method
    def forward(self, input):
        return F.prelu(input, self.weight)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)


@weak_module
class Softsign(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Softsign.png

    Examples::

        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        return F.softsign(input)


@weak_module
class Tanhshrink(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanhshrink}(x) = x - \text{Tanh}(x)

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        return F.tanhshrink(input)


@weak_module
class Softmin(Module):
    r"""Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `(0, 1)` and sum to 1

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Arguments:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(Softmin, self).__init__()
        self.dim = dim

    @weak_script_method
    def forward(self, input):
        return F.softmin(input, self.dim, _stacklevel=5)


@weak_module
class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @weak_script_method
    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)


@weak_module
class Softmax2d(Module):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """

    @weak_script_method
    def forward(self, input):
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return F.softmax(input, 1, _stacklevel=5)


@weak_module
class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @weak_script_method
    def forward(self, input):
        return F.log_softmax(input, self.dim, _stacklevel=5)
