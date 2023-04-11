import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from .module import Module
from .. import functional as F

__all__ = ['Residual']

class Residual(Module):
    r"""Applies the inner block, then adds the original input as a skip-connection

    TODO Docs

    TODO Open question - should "inner" be named something else. Potentially anonymous, at least for named_parameters
    """
    __constants__ = []

    def __init__(self, inner: Module):
        super().__init__()
        self._modules["inner"] = inner

    def forward(self, input: Tensor) -> Tensor:
        return self._modules["inner"](input) + input

    # def extra_repr(self) -> str:
    #     pass

