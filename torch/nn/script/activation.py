import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..modules.module import Module
from .. import functional as F
from torch.jit import ScriptModule


# NB: We don't support inplace operation since script don't support it.


class Threshold(nn.Threshold, ScriptModule):
    __doc__ = nn.Threshold.__doc__
    __constants__ = ['threshold', 'value', 'inplace']

    @torch.jit.script_method
    def forward(self, input):
        return F.threshold(input, self.threshold, self.value)


class ReLU(nn.ReLU, Threshold):
    __doc__ = nn.ReLU.__doc__


class RReLU(nn.RReLU, ScriptModule):
    __doc__ = nn.RReLU.__doc__
    __constants__ = ['lower', 'upper', 'inplace', 'training']

    @torch.jit.script_method
    def forward(self, input):
        # NB: aten::rrelu(Tensor self, Scalar lower=<default>, Scalar upper=<default>, int training=<default>
        # , Tensor generator=<default>) function signature has additional argument `generator`

        return F.rrelu(input, self.lower, self.upper, self.training, None)


class Hardtanh(nn.Hardtanh, ScriptModule):
    __doc__ = nn.Hardtanh.__doc__
    __constants__ = ['min_val', 'max_val']

    @torch.jit.script_method
    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val)


class ReLU6(nn.ReLU6, Hardtanh):
    __doc__ = nn.ReLU6.__doc__


class Sigmoid(nn.Sigmoid, ScriptModule):
    __doc__ = nn.Sigmoid.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return torch.sigmoid(input)


class Tanh(nn.Tanh, ScriptModule):
    __doc__ = nn.Tanh.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return torch.tanh(input)


class ELU(nn.ELU, ScriptModule):
    __doc__ = nn.ELU.__doc__
    __constants__ = ['alpha', 'inplace']

    @torch.jit.script_method
    def forward(self, input):
        return F.elu(input, self.alpha)


class SELU(nn.SELU, ScriptModule):
    __doc__ = nn.SELU.__doc__
    __constants__ = ['inplace']

    @torch.jit.script_method
    def forward(self, input):
        return F.selu(input)


class GLU(nn.GLU, ScriptModule):
    __doc__ = nn.GLU.__doc__
    __constants__ = ['dim']

    @torch.jit.script_method
    def forward(self, input):
        return F.glu(input, self.dim)


class Hardshrink(nn.Hardshrink, ScriptModule):
    __doc__ = nn.Hardshrink.__doc__
    __constants__ = ['lambd']

    @torch.jit.script_method
    def forward(self, input):
        return F.hardshrink(input, self.lambd)


class LeakyReLU(nn.LeakyReLU, ScriptModule):
    __doc__ = nn.LeakyReLU.__doc__
    __constants__ = ['negative_slope', 'inplace']

    @torch.jit.script_method
    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope)


class LogSigmoid(nn.LogSigmoid, ScriptModule):
    __doc__ = nn.LogSigmoid.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.logsigmoid(input)


class Softplus(nn.Softplus, ScriptModule):
    __doc__ = nn.Softplus.__doc__
    __constants__ = ['beta', 'threshold']

    @torch.jit.script_method
    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)


class Softshrink(nn.Softshrink, ScriptModule):
    __doc__ = nn.Softshrink.__doc__
    __constants__ = ['lambd']

    @torch.jit.script_method
    def forward(self, input):
        return F.softshrink(input, self.lambd)


class PReLU(nn.PReLU, ScriptModule):
    __doc__ = nn.PReLU.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.prelu(input, self.weight)


class Softsign(nn.Softsign, ScriptModule):
    __doc__ = nn.Softsign.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.softsign(input)


class Tanhshrink(nn.Tanhshrink, ScriptModule):
    __doc__ = nn.Tanhshrink.__doc__

    @torch.jit.script_method
    def forward(self, input):
        return F.tanhshrink(input)


class Softmin(nn.Softmin, ScriptModule):
    __doc__ = nn.Softmin.__doc__
    __constants__ = ['dim']

    @torch.jit.script_method
    def forward(self, input):
        return F.softmin(input, self.dim, _stacklevel=5)


class Softmax(nn.Softmax, ScriptModule):
    __doc__ = nn.Softmax.__doc__
    __constants__ = ['dim']

    @torch.jit.script_method
    def forward(self, input):
        # return F.softmax(input, self.dim, _stacklevel=5)
        return F.softmax(input, self.dim)


class Softmax2d(nn.Softmax2d, ScriptModule):
    __doc__ = nn.Softmax2d.__doc__

    @torch.jit.script_method
    def forward(self, input):
        # assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        if (input.dim() != 4):
            print("Error: Softmax2d requires a 4D tensor as input")
        # return F.softmax(input, 1, _stacklevel=5)
        return F.softmax(input, 1)


class LogSoftmax(nn.LogSoftmax, ScriptModule):
    __doc__ = nn.LogSoftmax.__doc__
    __constants__ = ['dim']

    @torch.jit.script_method
    def forward(self, input):
        # return F.log_softmax(input, self.dim, _stacklevel=5)
        return F.log_softmax(input, self.dim)
