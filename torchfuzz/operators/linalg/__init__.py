"""Linear algebra operators module."""

from .norm import NormOperator
from .matrix_norm import MatrixNormOperator
from .vector_norm import VectorNormOperator
from .inv import InvOperator
from .pinv import PinvOperator
from .svd import SvdOperator
from .qr import QrOperator
from .eig import EigOperator
from .eigh import EighOperator
from .cholesky import CholeskyOperator
from .solve import SolveOperator
from .lstsq import LstsqOperator
from .det import DetOperator
from .slogdet import SlogdetOperator

__all__ = [
    'NormOperator',
    'MatrixNormOperator',
    'VectorNormOperator',
    'InvOperator',
    'PinvOperator',
    'SvdOperator',
    'QrOperator',
    'EigOperator',
    'EighOperator',
    'CholeskyOperator',
    'SolveOperator',
    'LstsqOperator',
    'DetOperator',
    'SlogdetOperator',
]
