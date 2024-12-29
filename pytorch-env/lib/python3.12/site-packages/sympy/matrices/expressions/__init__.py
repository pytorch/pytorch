""" A module which handles Matrix Expressions """

from .slice import MatrixSlice
from .blockmatrix import BlockMatrix, BlockDiagMatrix, block_collapse, blockcut
from .companion import CompanionMatrix
from .funcmatrix import FunctionMatrix
from .inverse import Inverse
from .matadd import MatAdd
from .matexpr import MatrixExpr, MatrixSymbol, matrix_symbols
from .matmul import MatMul
from .matpow import MatPow
from .trace import Trace, trace
from .determinant import Determinant, det, Permanent, per
from .transpose import Transpose
from .adjoint import Adjoint
from .hadamard import hadamard_product, HadamardProduct, hadamard_power, HadamardPower
from .diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from .dotproduct import DotProduct
from .kronecker import kronecker_product, KroneckerProduct, combine_kronecker
from .permutation import PermutationMatrix, MatrixPermute
from .sets import MatrixSet
from .special import ZeroMatrix, Identity, OneMatrix

__all__ = [
    'MatrixSlice',

    'BlockMatrix', 'BlockDiagMatrix', 'block_collapse', 'blockcut',
    'FunctionMatrix',

    'CompanionMatrix',

    'Inverse',

    'MatAdd',

    'Identity', 'MatrixExpr', 'MatrixSymbol', 'ZeroMatrix', 'OneMatrix',
    'matrix_symbols', 'MatrixSet',

    'MatMul',

    'MatPow',

    'Trace', 'trace',

    'Determinant', 'det',

    'Transpose',

    'Adjoint',

    'hadamard_product', 'HadamardProduct', 'hadamard_power', 'HadamardPower',

    'DiagonalMatrix', 'DiagonalOf', 'DiagMatrix', 'diagonalize_vector',

    'DotProduct',

    'kronecker_product', 'KroneckerProduct', 'combine_kronecker',

    'PermutationMatrix', 'MatrixPermute',

    'Permanent', 'per'
]
