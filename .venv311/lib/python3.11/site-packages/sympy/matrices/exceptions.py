"""
Exceptions raised by the matrix module.
"""


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass


class NonInvertibleMatrixError(ValueError, MatrixError):
    """The matrix in not invertible (division by multidimensional zero error)."""
    pass


class NonPositiveDefiniteMatrixError(ValueError, MatrixError):
    """The matrix is not a positive-definite matrix."""
    pass
