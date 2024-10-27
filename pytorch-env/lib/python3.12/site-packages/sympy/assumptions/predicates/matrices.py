from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

class SquarePredicate(Predicate):
    """
    Square matrix predicate.

    Explanation
    ===========

    ``Q.square(x)`` is true iff ``x`` is a square matrix. A square matrix
    is a matrix with the same number of rows and columns.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('X', 2, 3)
    >>> ask(Q.square(X))
    True
    >>> ask(Q.square(Y))
    False
    >>> ask(Q.square(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.square(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_matrix

    """
    name = 'square'
    handler = Dispatcher("SquareHandler", doc="Handler for Q.square.")


class SymmetricPredicate(Predicate):
    """
    Symmetric matrix predicate.

    Explanation
    ===========

    ``Q.symmetric(x)`` is true iff ``x`` is a square matrix and is equal to
    its transpose. Every square diagonal matrix is a symmetric matrix.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.symmetric(X*Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(X + Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(Y))
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_matrix

    """
    # TODO: Add handlers to make these keys work with
    # actual matrices and add more examples in the docstring.
    name = 'symmetric'
    handler = Dispatcher("SymmetricHandler", doc="Handler for Q.symmetric.")


class InvertiblePredicate(Predicate):
    """
    Invertible matrix predicate.

    Explanation
    ===========

    ``Q.invertible(x)`` is true iff ``x`` is an invertible matrix.
    A square matrix is called invertible only if its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.invertible(X*Y), Q.invertible(X))
    False
    >>> ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z))
    True
    >>> ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Invertible_matrix

    """
    name = 'invertible'
    handler = Dispatcher("InvertibleHandler", doc="Handler for Q.invertible.")


class OrthogonalPredicate(Predicate):
    """
    Orthogonal matrix predicate.

    Explanation
    ===========

    ``Q.orthogonal(x)`` is true iff ``x`` is an orthogonal matrix.
    A square matrix ``M`` is an orthogonal matrix if it satisfies
    ``M^TM = MM^T = I`` where ``M^T`` is the transpose matrix of
    ``M`` and ``I`` is an identity matrix. Note that an orthogonal
    matrix is necessarily invertible.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.orthogonal(Y))
    False
    >>> ask(Q.orthogonal(X*Z*X), Q.orthogonal(X) & Q.orthogonal(Z))
    True
    >>> ask(Q.orthogonal(Identity(3)))
    True
    >>> ask(Q.invertible(X), Q.orthogonal(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_matrix

    """
    name = 'orthogonal'
    handler = Dispatcher("OrthogonalHandler", doc="Handler for key 'orthogonal'.")


class UnitaryPredicate(Predicate):
    """
    Unitary matrix predicate.

    Explanation
    ===========

    ``Q.unitary(x)`` is true iff ``x`` is a unitary matrix.
    Unitary matrix is an analogue to orthogonal matrix. A square
    matrix ``M`` with complex elements is unitary if :math:``M^TM = MM^T= I``
    where :math:``M^T`` is the conjugate transpose matrix of ``M``.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.unitary(Y))
    False
    >>> ask(Q.unitary(X*Z*X), Q.unitary(X) & Q.unitary(Z))
    True
    >>> ask(Q.unitary(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_matrix

    """
    name = 'unitary'
    handler = Dispatcher("UnitaryHandler", doc="Handler for key 'unitary'.")


class FullRankPredicate(Predicate):
    """
    Fullrank matrix predicate.

    Explanation
    ===========

    ``Q.fullrank(x)`` is true iff ``x`` is a full rank matrix.
    A matrix is full rank if all rows and columns of the matrix
    are linearly independent. A square matrix is full rank iff
    its determinant is nonzero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.fullrank(X.T), Q.fullrank(X))
    True
    >>> ask(Q.fullrank(ZeroMatrix(3, 3)))
    False
    >>> ask(Q.fullrank(Identity(3)))
    True

    """
    name = 'fullrank'
    handler = Dispatcher("FullRankHandler", doc="Handler for key 'fullrank'.")


class PositiveDefinitePredicate(Predicate):
    r"""
    Positive definite matrix predicate.

    Explanation
    ===========

    If $M$ is a :math:`n \times n` symmetric real matrix, it is said
    to be positive definite if :math:`Z^TMZ` is positive for
    every non-zero column vector $Z$ of $n$ real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.positive_definite(Y))
    False
    >>> ask(Q.positive_definite(Identity(3)))
    True
    >>> ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
    ...     Q.positive_definite(Z))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Positive-definite_matrix

    """
    name = "positive_definite"
    handler = Dispatcher("PositiveDefiniteHandler", doc="Handler for key 'positive_definite'.")


class UpperTriangularPredicate(Predicate):
    """
    Upper triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called upper triangular matrix if :math:`M_{ij}=0`
    for :math:`i<j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.upper_triangular(Identity(3)))
    True
    >>> ask(Q.upper_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UpperTriangularMatrix.html

    """
    name = "upper_triangular"
    handler = Dispatcher("UpperTriangularHandler", doc="Handler for key 'upper_triangular'.")


class LowerTriangularPredicate(Predicate):
    """
    Lower triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called lower triangular matrix if :math:`M_{ij}=0`
    for :math:`i>j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.lower_triangular(Identity(3)))
    True
    >>> ask(Q.lower_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LowerTriangularMatrix.html

    """
    name = "lower_triangular"
    handler = Dispatcher("LowerTriangularHandler", doc="Handler for key 'lower_triangular'.")


class DiagonalPredicate(Predicate):
    """
    Diagonal matrix predicate.

    Explanation
    ===========

    ``Q.diagonal(x)`` is true iff ``x`` is a diagonal matrix. A diagonal
    matrix is a matrix in which the entries outside the main diagonal
    are all zero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.diagonal(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.diagonal(X), Q.lower_triangular(X) &
    ...     Q.upper_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Diagonal_matrix

    """
    name = "diagonal"
    handler = Dispatcher("DiagonalHandler", doc="Handler for key 'diagonal'.")


class IntegerElementsPredicate(Predicate):
    """
    Integer elements matrix predicate.

    Explanation
    ===========

    ``Q.integer_elements(x)`` is true iff all the elements of ``x``
    are integers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    True

    """
    name = "integer_elements"
    handler = Dispatcher("IntegerElementsHandler", doc="Handler for key 'integer_elements'.")


class RealElementsPredicate(Predicate):
    """
    Real elements matrix predicate.

    Explanation
    ===========

    ``Q.real_elements(x)`` is true iff all the elements of ``x``
    are real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
    True

    """
    name = "real_elements"
    handler = Dispatcher("RealElementsHandler", doc="Handler for key 'real_elements'.")


class ComplexElementsPredicate(Predicate):
    """
    Complex elements matrix predicate.

    Explanation
    ===========

    ``Q.complex_elements(x)`` is true iff all the elements of ``x``
    are complex numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    True
    >>> ask(Q.complex_elements(X), Q.integer_elements(X))
    True

    """
    name = "complex_elements"
    handler = Dispatcher("ComplexElementsHandler", doc="Handler for key 'complex_elements'.")


class SingularPredicate(Predicate):
    """
    Singular matrix predicate.

    A matrix is singular iff the value of its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.singular(X), Q.invertible(X))
    False
    >>> ask(Q.singular(X), ~Q.invertible(X))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/SingularMatrix.html

    """
    name = "singular"
    handler = Dispatcher("SingularHandler", doc="Predicate fore key 'singular'.")


class NormalPredicate(Predicate):
    """
    Normal matrix predicate.

    A matrix is normal if it commutes with its conjugate transpose.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.normal(X), Q.unitary(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_matrix

    """
    name = "normal"
    handler = Dispatcher("NormalHandler", doc="Predicate fore key 'normal'.")


class TriangularPredicate(Predicate):
    """
    Triangular matrix predicate.

    Explanation
    ===========

    ``Q.triangular(X)`` is true if ``X`` is one that is either lower
    triangular or upper triangular.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.upper_triangular(X))
    True
    >>> ask(Q.triangular(X), Q.lower_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_matrix

    """
    name = "triangular"
    handler = Dispatcher("TriangularHandler", doc="Predicate fore key 'triangular'.")


class UnitTriangularPredicate(Predicate):
    """
    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    """
    name = "unit_triangular"
    handler = Dispatcher("UnitTriangularHandler", doc="Predicate fore key 'unit_triangular'.")
