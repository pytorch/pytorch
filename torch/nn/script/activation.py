import warnings
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..modules.module import Module
from .. import functional as F
from torch.jit import ScriptModule


# NB: We don't support inplace operation since script don't support it.


class Threshold(ScriptModule):
    __doc__ = nn.Threshold.__doc__
    __constants__ = ['threshold', 'value', 'inplace']

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    @torch.jit.script_method
    def forward(self, input):
        return F.threshold(input, self.threshold, self.value)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )


class ReLU(Threshold):
    __doc__ = nn.ReLU.__doc__

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class RReLU(ScriptModule):
    __doc__ = nn.RReLU.__doc__
    __constants__ = ['lower', 'upper', 'inplace', 'training']

    def __init__(self, lower=1. / 8, upper=1. / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    @torch.jit.script_method
    def forward(self, input):
        # NB: aten::rrelu(Tensor self, Scalar lower=<default>, Scalar upper=<default>, int training=<default>
        # , Tensor generator=<default>) function signature has additional argument `generator`

        return F.rrelu(input, self.lower, self.upper, self.training, None)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)


class Hardtanh(ScriptModule):
    __doc__ = nn.Hardtanh.__doc__

    def __init__(self, min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None):
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

    @torch.jit.script_method
    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(
            self.min_val, self.max_val, inplace_str
        )


class ReLU6(Hardtanh):
    __doc__ = nn.ReLU6.__doc__

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class Sigmoid(ScriptModule):
    __doc__ = nn.Sigmoid.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return torch.sigmoid(input)


class Tanh(ScriptModule):
    __doc__ = nn.Tanh.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return torch.tanh(input)


class ELU(ScriptModule):
    __doc__ = nn.ELU.__doc__
    __constants__ = ['alpha', 'inplace']

    def __init__(self, alpha=1., inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    @torch.jit.script_method
    def forward(self, input):
        return F.elu(input, self.alpha)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


class SELU(ScriptModule):
    __doc__ = nn.SELU.__doc__
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.inplace = inplace

    @torch.jit.script_method
    def forward(self, input):
        return F.selu(input)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class GLU(ScriptModule):
    __doc__ = nn.GLU.__doc__
    __constants__ = ['dim']

    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    @torch.jit.script_method
    def forward(self, input):
        return F.glu(input, self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class Hardshrink(ScriptModule):
    __doc__ = nn.Hardshrink.__doc__
    __constants__ = ['lambd']

    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    @torch.jit.script_method
    def forward(self, input):
        return F.hardshrink(input, self.lambd)

    def extra_repr(self):
        return '{}'.format(self.lambd)


class LeakyReLU(ScriptModule):
    __doc__ = nn.LeakyReLU.__doc__
    __constants__ = ['negative_slope', 'inplace']

    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    @torch.jit.script_method
    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


class LogSigmoid(ScriptModule):
    __doc__ = nn.LogSigmoid.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.logsigmoid(input)


class Softplus(ScriptModule):
    __doc__ = nn.Softplus.__doc__
    __constants__ = ['beta', 'threshold']

    def __init__(self, beta=1, threshold=20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    @torch.jit.script_method
    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class Softshrink(ScriptModule):
    __doc__ = nn.Softshrink.__doc__
    __constants__ = ['lambd']

    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    @torch.jit.script_method
    def forward(self, input):
        return F.softshrink(input, self.lambd)

    def extra_repr(self):
        return str(self.lambd)


class PReLU(ScriptModule):
    __doc__ = nn.PReLU.__doc__
    # __constants__ = ['weight']

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    @torch.jit.script_method
    def forward(self, input):
        return F.prelu(input, self.weight)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)


class Softsign(ScriptModule):
    __doc__ = nn.Softsign.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.softsign(input)


class Tanhshrink(ScriptModule):
    __doc__ = nn.Tanhshrink.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.tanhshrink(input)


class Softmin(ScriptModule):
    __doc__ = nn.Softmin.__doc__
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(Softmin, self).__init__()
        self.dim = dim

    @torch.jit.script_method
    def forward(self, input):
        return F.softmin(input, self.dim, _stacklevel=5)


class Softmax(ScriptModule):
    __doc__ = nn.Softmax.__doc__
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @torch.jit.script_method
    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)


class Softmax2d(ScriptModule):
    __doc__ = nn.Softmax2d.__doc__

    @torch.jit.script_method
    def forward(self, input):
        # assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        if (input.dim() != 4):
            print("Error: Softmax2d requires a 4D tensor as input")
        return F.softmax(input, 1, _stacklevel=5)


class LogSoftmax(ScriptModule):
    __doc__ = nn.LogSoftmax.__doc__
    __constants__ = ['dim']

    def __init__(self, dim=None):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @torch.jit.script_method
    def forward(self, input):
        return F.log_softmax(input, self.dim, _stacklevel=5)
