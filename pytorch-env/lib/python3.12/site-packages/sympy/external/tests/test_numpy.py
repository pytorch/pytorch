# This testfile tests SymPy <-> NumPy compatibility

# Don't test any SymPy features here. Just pure interaction with NumPy.
# Always write regular SymPy tests for anything, that can be tested in pure
# Python (without numpy). Here we test everything, that a user may need when
# using SymPy with NumPy
from sympy.external.importtools import version_tuple
from sympy.external import import_module

numpy = import_module('numpy')
if numpy:
    array, matrix, ndarray = numpy.array, numpy.matrix, numpy.ndarray
else:
    #bin/test will not execute any tests now
    disabled = True


from sympy.core.numbers import (Float, Integer, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import (Matrix, list2numpy, matrix2numpy, symarray)
from sympy.utilities.lambdify import lambdify
import sympy

import mpmath
from sympy.abc import x, y, z
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import raises


# first, systematically check, that all operations are implemented and don't
# raise an exception


def test_systematic_basic():
    def s(sympy_object, numpy_array):
        _ = [sympy_object + numpy_array,
        numpy_array + sympy_object,
        sympy_object - numpy_array,
        numpy_array - sympy_object,
        sympy_object * numpy_array,
        numpy_array * sympy_object,
        sympy_object / numpy_array,
        numpy_array / sympy_object,
        sympy_object ** numpy_array,
        numpy_array ** sympy_object]
    x = Symbol("x")
    y = Symbol("y")
    sympy_objs = [
        Rational(2, 3),
        Float("1.3"),
        x,
        y,
        pow(x, y)*y,
        Integer(5),
        Float(5.5),
    ]
    numpy_objs = [
        array([1]),
        array([3, 8, -1]),
        array([x, x**2, Rational(5)]),
        array([x/y*sin(y), 5, Rational(5)]),
    ]
    for x in sympy_objs:
        for y in numpy_objs:
            s(x, y)


# now some random tests, that test particular problems and that also
# check that the results of the operations are correct

def test_basics():
    one = Rational(1)
    zero = Rational(0)
    assert array(1) == array(one)
    assert array([one]) == array([one])
    assert array([x]) == array([x])
    assert array(x) == array(Symbol("x"))
    assert array(one + x) == array(1 + x)

    X = array([one, zero, zero])
    assert (X == array([one, zero, zero])).all()
    assert (X == array([one, 0, 0])).all()


def test_arrays():
    one = Rational(1)
    zero = Rational(0)
    X = array([one, zero, zero])
    Y = one*X
    X = array([Symbol("a") + Rational(1, 2)])
    Y = X + X
    assert Y == array([1 + 2*Symbol("a")])
    Y = Y + 1
    assert Y == array([2 + 2*Symbol("a")])
    Y = X - X
    assert Y == array([0])


def test_conversion1():
    a = list2numpy([x**2, x])
    #looks like an array?
    assert isinstance(a, ndarray)
    assert a[0] == x**2
    assert a[1] == x
    assert len(a) == 2
    #yes, it's the array


def test_conversion2():
    a = 2*list2numpy([x**2, x])
    b = list2numpy([2*x**2, 2*x])
    assert (a == b).all()

    one = Rational(1)
    zero = Rational(0)
    X = list2numpy([one, zero, zero])
    Y = one*X
    X = list2numpy([Symbol("a") + Rational(1, 2)])
    Y = X + X
    assert Y == array([1 + 2*Symbol("a")])
    Y = Y + 1
    assert Y == array([2 + 2*Symbol("a")])
    Y = X - X
    assert Y == array([0])


def test_list2numpy():
    assert (array([x**2, x]) == list2numpy([x**2, x])).all()


def test_Matrix1():
    m = Matrix([[x, x**2], [5, 2/x]])
    assert (array(m.subs(x, 2)) == array([[2, 4], [5, 1]])).all()
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    assert (array(m.subs(x, 2)) == array([[sin(2), 4], [5, 1]])).all()


def test_Matrix2():
    m = Matrix([[x, x**2], [5, 2/x]])
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[2, 4], [5, 1]])).all()
    m = Matrix([[sin(x), x**2], [5, 2/x]])
    with ignore_warnings(PendingDeprecationWarning):
        assert (matrix(m.subs(x, 2)) == matrix([[sin(2), 4], [5, 1]])).all()


def test_Matrix3():
    a = array([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    a = array([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix4():
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    with ignore_warnings(PendingDeprecationWarning):
        a = matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])


def test_Matrix_sum():
    M = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 3, 4], [x, 5, 6], [x, y, z**2]])
    assert M + m == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert m + M == Matrix([[3, 5, 7], [2*x, y + 5, x + 6], [2*y + x, y - 50, z*x + z**2]])
    assert M + m == M.add(m)


def test_Matrix_mul():
    M = Matrix([[1, 2, 3], [x, y, x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 4], [x, 6], [x, z**2]])
    assert M*m == Matrix([
        [         2 + 5*x,        16 + 3*z**2],
        [2*x + x*y + x**2, 4*x + 6*y + x*z**2],
    ])

    assert m*M == Matrix([
        [   2 + 4*x,      4 + 4*y,      6 + 4*x],
        [       7*x,    2*x + 6*y,          9*x],
        [x + x*z**2, 2*x + y*z**2, 3*x + x*z**2],
    ])
    a = array([2])
    assert a[0] * M == 2 * M
    assert M * a[0] == 2 * M


def test_Matrix_array():
    class matarray:
        def __array__(self, dtype=object, copy=None):
            if copy is not None and not copy:
                raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
            from numpy import array
            return array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matarr = matarray()
    assert Matrix(matarr) == Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_matrix2numpy():
    a = matrix2numpy(Matrix([[1, x**2], [3*sin(x), 0]]))
    assert isinstance(a, ndarray)
    assert a.shape == (2, 2)
    assert a[0, 0] == 1
    assert a[0, 1] == x**2
    assert a[1, 0] == 3*sin(x)
    assert a[1, 1] == 0


def test_matrix2numpy_conversion():
    a = Matrix([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    b = array([[1, 2, sin(x)], [x**2, x, Rational(1, 2)]])
    assert (matrix2numpy(a) == b).all()
    assert matrix2numpy(a).dtype == numpy.dtype('object')

    c = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='int8')
    d = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='float64')
    assert c.dtype == numpy.dtype('int8')
    assert d.dtype == numpy.dtype('float64')


def test_issue_3728():
    assert (Rational(1, 2)*array([2*x, 0]) == array([x, 0])).all()
    assert (Rational(1, 2) + array(
        [2*x, 0]) == array([2*x + Rational(1, 2), Rational(1, 2)])).all()
    assert (Float("0.5")*array([2*x, 0]) == array([Float("1.0")*x, 0])).all()
    assert (Float("0.5") + array(
        [2*x, 0]) == array([2*x + Float("0.5"), Float("0.5")])).all()


@conserve_mpmath_dps
def test_lambdify():
    mpmath.mp.dps = 16
    sin02 = mpmath.mpf("0.198669330795061215459412627")
    f = lambdify(x, sin(x), "numpy")
    prec = 1e-15
    assert -prec < f(0.2) - sin02 < prec

    # if this succeeds, it can't be a numpy function

    if version_tuple(numpy.__version__) >= version_tuple('1.17'):
        with raises(TypeError):
            f(x)
    else:
        with raises(AttributeError):
            f(x)


def test_lambdify_matrix():
    f = lambdify(x, Matrix([[x, 2*x], [1, 2]]), [{'ImmutableMatrix': numpy.array}, "numpy"])
    assert (f(1) == array([[1, 2], [1, 2]])).all()


def test_lambdify_matrix_multi_input():
    M = sympy.Matrix([[x**2, x*y, x*z],
                      [y*x, y**2, y*z],
                      [z*x, z*y, z**2]])
    f = lambdify((x, y, z), M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    xh, yh, zh = 1.0, 2.0, 3.0
    expected = array([[xh**2, xh*yh, xh*zh],
                      [yh*xh, yh**2, yh*zh],
                      [zh*xh, zh*yh, zh**2]])
    actual = f(xh, yh, zh)
    assert numpy.allclose(actual, expected)


def test_lambdify_matrix_vec_input():
    X = sympy.DeferredVector('X')
    M = Matrix([
        [X[0]**2, X[0]*X[1], X[0]*X[2]],
        [X[1]*X[0], X[1]**2, X[1]*X[2]],
        [X[2]*X[0], X[2]*X[1], X[2]**2]])
    f = lambdify(X, M, [{'ImmutableMatrix': numpy.array}, "numpy"])

    Xh = array([1.0, 2.0, 3.0])
    expected = array([[Xh[0]**2, Xh[0]*Xh[1], Xh[0]*Xh[2]],
                      [Xh[1]*Xh[0], Xh[1]**2, Xh[1]*Xh[2]],
                      [Xh[2]*Xh[0], Xh[2]*Xh[1], Xh[2]**2]])
    actual = f(Xh)
    assert numpy.allclose(actual, expected)


def test_lambdify_transl():
    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, mat in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in numpy.__dict__


def test_symarray():
    """Test creation of numpy arrays of SymPy symbols."""

    import numpy as np
    import numpy.testing as npt

    syms = symbols('_0,_1,_2')
    s1 = symarray("", 3)
    s2 = symarray("", 3)
    npt.assert_array_equal(s1, np.array(syms, dtype=object))
    assert s1[0] == s2[0]

    a = symarray('a', 3)
    b = symarray('b', 3)
    assert not(a[0] == b[0])

    asyms = symbols('a_0,a_1,a_2')
    npt.assert_array_equal(a, np.array(asyms, dtype=object))

    # Multidimensional checks
    a2d = symarray('a', (2, 3))
    assert a2d.shape == (2, 3)
    a00, a12 = symbols('a_0_0,a_1_2')
    assert a2d[0, 0] == a00
    assert a2d[1, 2] == a12

    a3d = symarray('a', (2, 3, 2))
    assert a3d.shape == (2, 3, 2)
    a000, a120, a121 = symbols('a_0_0_0,a_1_2_0,a_1_2_1')
    assert a3d[0, 0, 0] == a000
    assert a3d[1, 2, 0] == a120
    assert a3d[1, 2, 1] == a121


def test_vectorize():
    assert (numpy.vectorize(
        sin)([1, 2, 3]) == numpy.array([sin(1), sin(2), sin(3)])).all()
