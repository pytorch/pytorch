import torch

from .variable import Variable
from .function import Function, NestedInputFunction

assert torch._C._autograd_init()
