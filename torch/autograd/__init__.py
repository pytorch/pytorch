import torch

from .variable import Variable
from .function import Function, NestedIOFunction

assert torch._C._autograd_init()
