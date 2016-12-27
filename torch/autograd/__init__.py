import torch

from .variable import Variable
from .function import Function, NestedIOFunction
from .stochastic_function import StochasticFunction

def backward(variables, grad_variables, retain_variables=False):
    Variable._execution_engine.run_backward(
            tuple(variables), tuple(grad_variables), retain_variables)

assert torch._C._autograd_init()
