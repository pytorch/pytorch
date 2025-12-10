"""Tests for the ``sympy.simplify._cse_diff.py`` module."""

import pytest

from sympy.core.symbol import (Symbol, symbols)
from sympy.core.numbers import Integer
from sympy.core.function import Function
from sympy.core import Derivative
from sympy.functions.elementary.exponential import exp
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.physics.mechanics import dynamicsymbols
from sympy.simplify._cse_diff import (_forward_jacobian,
                                      _remove_cse_from_derivative,
                                      _forward_jacobian_cse,
                                      _forward_jacobian_norm_in_cse_out)
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix, eye

from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.simplify.trigsimp import trigsimp

from sympy import cse


w = Symbol('w')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

q1, q2, q3 = dynamicsymbols('q1 q2 q3')

# Define the custom functions
k = Function('k')(x, y)
f = Function('f')(k, z)

zero = Integer(0)
one = Integer(1)
two = Integer(2)
neg_one = Integer(-1)


@pytest.mark.parametrize(
    'expr, wrt',
    [
        ([zero], [x]),
        ([one], [x]),
        ([two], [x]),
        ([neg_one], [x]),
        ([x], [x]),
        ([y], [x]),
        ([x + y], [x]),
        ([x*y], [x]),
        ([x**2], [x]),
        ([x**y], [x]),
        ([exp(x)], [x]),
        ([sin(x)], [x]),
        ([tan(x)], [x]),
        ([zero, one, x, y, x*y, x + y], [x, y]),
        ([((x/y) + sin(x/y) - exp(y))*((x/y) - exp(y))], [x, y]),
        ([w*tan(y*z)/(x - tan(y*z)), w*x*tan(y*z)/(x - tan(y*z))], [w, x, y, z]),
        ([q1**2 + q2, q2**2 + q3, q3**2 + q1], [q1, q2, q3]),
        ([f + Derivative(f, x) + k + 2*x], [x])
    ]
)


def test_forward_jacobian(expr, wrt):
    expr = ImmutableDenseMatrix([expr]).T
    wrt = ImmutableDenseMatrix([wrt]).T
    jacobian = _forward_jacobian(expr, wrt)
    zeros = ImmutableDenseMatrix.zeros(*jacobian.shape)
    assert simplify(jacobian - expr.jacobian(wrt)) == zeros


def test_process_cse():
    x, y, z = symbols('x y z')
    f = Function('f')
    k = Function('k')
    expr = Matrix([f(k(x,y), z) + Derivative(f(k(x,y), z), x) + k(x,y) + 2*x])
    repl, reduced = cse(expr)
    p_repl, p_reduced = _remove_cse_from_derivative(repl, reduced)

    x0 = symbols('x0')
    x1 = symbols('x1')

    expected_output = (
        [(x0, k(x, y)), (x1, f(x0, z))],
        [Matrix([2 * x + x0 + x1 + Derivative(f(k(x, y), z), x)])]
    )

    assert p_repl == expected_output[0], f"Expected {expected_output[0]}, but got {p_repl}"
    assert p_reduced == expected_output[1], f"Expected {expected_output[1]}, but got {p_reduced}"


def test_io_matrix_type():
    x, y, z = symbols('x y z')
    expr = ImmutableDenseMatrix([
        x * y + y * z + x * y * z,
        x ** 2 + y ** 2 + z ** 2,
        x * y + x * z + y * z
    ])
    wrt = ImmutableDenseMatrix([x, y, z])

    replacements, reduced_expr = cse(expr)

    # Test _forward_jacobian_core
    replacements_core, jacobian_core, precomputed_fs_core = _forward_jacobian_cse(replacements, reduced_expr, wrt)
    assert isinstance(jacobian_core[0], type(reduced_expr[0])), "Jacobian should be a Matrix of the same type as the input"

    # Test _forward_jacobian_norm_in_dag_out
    replacements_norm, jacobian_norm, precomputed_fs_norm = _forward_jacobian_norm_in_cse_out(
        expr, wrt)
    assert isinstance(jacobian_norm[0], type(reduced_expr[0])), "Jacobian should be a Matrix of the same type as the input"

    # Test _forward_jacobian
    jacobian = _forward_jacobian(expr, wrt)
    assert isinstance(jacobian, type(expr)), "Jacobian should be a Matrix of the same type as the input"


def test_forward_jacobian_input_output():
    x, y, z = symbols('x y z')
    expr = Matrix([
        x * y + y * z + x * y * z,
        x ** 2 + y ** 2 + z ** 2,
        x * y + x * z + y * z
    ])
    wrt = Matrix([x, y, z])

    replacements, reduced_expr = cse(expr)

    # Test _forward_jacobian_core
    replacements_core, jacobian_core, precomputed_fs_core = _forward_jacobian_cse(replacements, reduced_expr, wrt)
    assert isinstance(replacements_core, type(replacements)), "Replacements should be a list"
    assert isinstance(jacobian_core, type(reduced_expr)), "Jacobian should be a list"
    assert isinstance(precomputed_fs_core, list), "Precomputed free symbols should be a list"
    assert len(replacements_core) == len(replacements), "Length of replacements does not match"
    assert len(jacobian_core) == 1, "Jacobian should have one element"
    assert len(precomputed_fs_core) == len(replacements), "Length of precomputed free symbols does not match"

    # Test _forward_jacobian_norm_in_dag_out
    replacements_norm, jacobian_norm, precomputed_fs_norm = _forward_jacobian_norm_in_cse_out(expr, wrt)
    assert isinstance(replacements_norm, type(replacements)), "Replacements should be a list"
    assert isinstance(jacobian_norm, type(reduced_expr)), "Jacobian should be a list"
    assert isinstance(precomputed_fs_norm, list), "Precomputed free symbols should be a list"
    assert len(replacements_norm) == len(replacements), "Length of replacements does not match"
    assert len(jacobian_norm) == 1, "Jacobian should have one element"
    assert len(precomputed_fs_norm) == len(replacements), "Length of precomputed free symbols does not match"


def test_jacobian_hessian():
    L = Matrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    assert _forward_jacobian(L, syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    L = Matrix(1, 2, [x, x**2*y**3])
    assert _forward_jacobian(L, syms) == Matrix([[1, 0], [2*x*y**3, x**2*3*y**2]])


def test_jacobian_metrics():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho * cos(phi), rho * sin(phi)])
    Y = Matrix([rho, phi])
    J = _forward_jacobian(X, Y)
    assert J == X.jacobian(Y.T)
    assert J == (X.T).jacobian(Y)
    assert J == (X.T).jacobian(Y.T)
    g = J.T * eye(J.shape[0]) * J
    g = g.applyfunc(trigsimp)
    assert g == Matrix([[1, 0], [0, rho ** 2]])


def test_jacobian2():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho * cos(phi), rho * sin(phi), rho ** 2])
    Y = Matrix([rho, phi])
    J = Matrix([
        [cos(phi), -rho * sin(phi)],
        [sin(phi), rho * cos(phi)],
        [2 * rho, 0],
    ])
    assert _forward_jacobian(X, Y) == J


def test_issue_4564():
    X = Matrix([exp(x + y + z), exp(x + y + z), exp(x + y + z)])
    Y = Matrix([x, y, z])
    for i in range(1, 3):
        for j in range(1, 3):
            X_slice = X[:i, :]
            Y_slice = Y[:j, :]
            J = _forward_jacobian(X_slice, Y_slice)
            assert J.rows == i
            assert J.cols == j
            for k in range(j):
                assert J[:, k] == X_slice


def test_nonvectorJacobian():
    X = Matrix([[exp(x + y + z), exp(x + y + z)],
                [exp(x + y + z), exp(x + y + z)]])
    raises(TypeError, lambda: _forward_jacobian(X, Matrix([x, y, z])))
    X = X[0, :]
    Y = Matrix([[x, y], [x, z]])
    raises(TypeError, lambda: _forward_jacobian(X, Y))
    raises(TypeError, lambda: _forward_jacobian(X, Matrix([[x, y], [x, z]])))
