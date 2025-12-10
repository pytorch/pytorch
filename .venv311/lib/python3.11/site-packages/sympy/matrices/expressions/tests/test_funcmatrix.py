from sympy.core import symbols, Lambda
from sympy.core.sympify import SympifyError
from sympy.functions import KroneckerDelta
from sympy.matrices import Matrix
from sympy.matrices.expressions import FunctionMatrix, MatrixExpr, Identity
from sympy.testing.pytest import raises


def test_funcmatrix_creation():
    i, j, k = symbols('i j k')
    assert FunctionMatrix(2, 2, Lambda((i, j), 0))
    assert FunctionMatrix(0, 0, Lambda((i, j), 0))

    raises(ValueError, lambda: FunctionMatrix(-1, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(2.0, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(2j, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, -1, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, 2.0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, 2j, Lambda((i, j), 0)))

    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda(i, 0)))
    raises(SympifyError, lambda: FunctionMatrix(2, 2, lambda i, j: 0))
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i,), 0)))
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i, j, k), 0)))
    raises(ValueError, lambda: FunctionMatrix(2, 2, i+j))
    assert FunctionMatrix(2, 2, "lambda i, j: 0") == \
        FunctionMatrix(2, 2, Lambda((i, j), 0))

    m = FunctionMatrix(2, 2, KroneckerDelta)
    assert m.as_explicit() == Identity(2).as_explicit()
    assert m.args[2].dummy_eq(Lambda((i, j), KroneckerDelta(i, j)))

    n = symbols('n')
    assert FunctionMatrix(n, n, Lambda((i, j), 0))
    n = symbols('n', integer=False)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))


def test_funcmatrix():
    i, j = symbols('i,j')
    X = FunctionMatrix(3, 3, Lambda((i, j), i - j))
    assert X[1, 1] == 0
    assert X[1, 2] == -1
    assert X.shape == (3, 3)
    assert X.rows == X.cols == 3
    assert Matrix(X) == Matrix(3, 3, lambda i, j: i - j)
    assert isinstance(X*X + X, MatrixExpr)


def test_replace_issue():
    X = FunctionMatrix(3, 3, KroneckerDelta)
    assert X.replace(lambda x: True, lambda x: x) == X
