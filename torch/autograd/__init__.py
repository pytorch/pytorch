import torch

from .variable import Variable
from .function import Function

assert torch._C._autograd_init()
