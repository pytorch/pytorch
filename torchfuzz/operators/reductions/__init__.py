"""Reductions operators module."""

from .sum import SumOperator
from .mean import MeanOperator
from .max import MaxOperator
from .min import MinOperator
from .std import StdOperator
from .var import VarOperator
from .norm import NormOperator
from .argmax import ArgmaxOperator
from .argmin import ArgminOperator

__all__ = [
    'SumOperator',
    'MeanOperator',
    'MaxOperator',
    'MinOperator',
    'StdOperator',
    'VarOperator',
    'NormOperator',
    'ArgmaxOperator',
    'ArgminOperator',
]
