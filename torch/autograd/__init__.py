import torch

from .variable import Variable
from .function import Function, NestedIOFunction
from .stochastic_function import StochasticFunction

assert torch._C._autograd_init()
