from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)

from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor, _simple_dens, denoms

from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf

from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn

from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


def test_swap_back():
    f, g = map(Function, 'fg')
    fx, gx = f(x), g(x)
    assert solve([fx + y - 2, fx - gx - 5], fx, y, gx) == \
        {fx: gx + 5, y: -gx - 3}
    assert solve(fx + gx*x - 2, [fx, gx], dict=True) == [{fx: 2, gx: 0}]
    assert solve(fx + gx**2*x - y, [fx, gx], dict=True) == [{fx: y, gx: 0}]
    assert solve([f(1) - 2, x + 2], dict=True) == [{x: -2, f(1): 2}]


def guess_solve_strategy(eq, symbol):
    try:
        solve(eq, symbol)
        return True
    except (TypeError, NotImplementedError):
        return False


def test_guess_poly():
    # polynomial equations
    assert guess_solve_strategy( S(4), x )  # == GS_POLY
    assert guess_solve_strategy( x, x )  # == GS_POLY
    assert guess_solve_strategy( x + a, x )  # == GS_POLY
    assert guess_solve_strategy( 2*x, x )  # == GS_POLY
    assert guess_solve_strategy( x + sqrt(2), x)  # == GS_POLY
    assert guess_solve_strategy( x + 2**Rational(1, 4), x)  # == GS_POLY
    assert guess_solve_strategy( x**2 + 1, x )  # == GS_POLY
    assert guess_solve_strategy( x**2 - 1, x )  # == GS_POLY
    assert guess_solve_strategy( x*y + y, x )  # == GS_POLY
    assert guess_solve_strategy( x*exp(y) + y, x)  # == GS_POLY
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), x)  # == GS_POLY


def test_guess_poly_cv():
    # polynomial equations via a change of variable
    assert guess_solve_strategy( sqrt(x) + 1, x )  # == GS_POLY_CV_1
    assert guess_solve_strategy(
        x**Rational(1, 3) + sqrt(x) + 1, x )  # == GS_POLY_CV_1
    assert guess_solve_strategy( 4*x*(1 - sqrt(x)), x )  # == GS_POLY_CV_1

    # polynomial equation multiplying both sides by x**n
    assert guess_solve_strategy( x + 1/x + y, x )  # == GS_POLY_CV_2


def test_guess_rational_cv():
    # rational functions
    assert guess_solve_strategy( (x + 1)/(x**2 + 2), x)  # == GS_RATIONAL
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), y)  # == GS_RATIONAL_CV_1

    # rational functions via the change of variable y -> x**n
    assert guess_solve_strategy( (sqrt(x) + 1)/(x**Rational(1, 3) + sqrt(x) + 1), x ) \
        #== GS_RATIONAL_CV_1


def test_guess_transcendental():
    #transcendental functions
    assert guess_solve_strategy( exp(x) + 1, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy( 2*cos(x) - y, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(
        exp(x) + exp(-x) - y, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(3**x - 10, x)  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(-3**x + 10, x)  # == GS_TRANSCENDENTAL

    assert guess_solve_strategy(a*x**b - y, x)  # == GS_TRANSCENDENTAL


def test_solve_args():
    # equation container, issue 5113
    ans = {x: -3, y: 1}
    eqs = (x + 5*y - 2, -3*x + 6*y - 15)
    assert all(solve(container(eqs), x, y) == ans for container in
        (tuple, list, set, frozenset))
    assert solve(Tuple(*eqs), x, y) == ans
    # implicit symbol to solve for
    assert set(solve(x**2 - 4)) == {S(2), -S(2)}
    assert solve([x + y - 3, x - y - 5]) == {x: 4, y: -1}
    assert solve(x - exp(x), x, implicit=True) == [exp(x)]
    # no symbol to solve for
    assert solve(42) == solve(42, x) == []
    assert solve([1, 2]) == []
    assert solve([sqrt(2)],[x]) == []
    # duplicate symbols raises
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    # no error in exclude
    assert solve(x, x, exclude=[y, y]) == [0]
    # duplicate symbols raises
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    # no error in exclude
    assert solve(x, x, exclude=[y, y]) == [0]
    # unordered symbols
    # only 1
    assert solve(y - 3, {y}) == [3]
    # more than 1
    assert solve(y - 3, {x, y}) == [{y: 3}]
    # multiple symbols: take the first linear solution+
    # - return as tuple with values for all requested symbols
    assert solve(x + y - 3, [x, y]) == [(3 - y, y)]
    # - unless dict is True
    assert solve(x + y - 3, [x, y], dict=True) == [{x: 3 - y}]
    # - or no symbols are given
    assert solve(x + y - 3) == [{x: 3 - y}]
    # multiple symbols might represent an undetermined coefficients system
    assert solve(a + b*x - 2, [a, b]) == {a: 2, b: 0}
    assert solve((a + b)*x + b - c, [a, b]) == {a: -c, b: c}
    eq = a*x**2 + b*x + c - ((x - h)**2 + 4*p*k)/4/p
    # - check that flags are obeyed
    sol = solve(eq, [h, p, k], exclude=[a, b, c])
    assert sol == {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    assert solve(eq, [h, p, k], dict=True) == [sol]
    assert solve(eq, [h, p, k], set=True) == \
        ([h, p, k], {(-b/(2*a), 1/(4*a), (4*a*c - b**2)/(4*a))})
    # issue 23889 - polysys not simplified
    assert solve(eq, [h, p, k], exclude=[a, b, c], simplify=False) == \
        {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    # but this only happens when system has a single solution
    args = (a + b)*x - b**2 + 2, a, b
    assert solve(*args) == [((b**2 - b*x - 2)/x, b)]
    # and if the system has a solution; the following doesn't so
    # an algebraic solution is returned
    assert solve(a*x + b**2/(x + 4) - 3*x - 4/x, a, b, dict=True) == \
        [{a: (-b**2*x + 3*x**3 + 12*x**2 + 4*x + 16)/(x**2*(x + 4))}]
    # failed single equation
    assert solve(1/(1/x - y + exp(y))) == []
    raises(
        NotImplementedError, lambda: solve(exp(x) + sin(x) + exp(y) + sin(y)))
    # failed system
    # --  when no symbols given, 1 fails
    assert solve([y, exp(x) + x]) == [{x: -LambertW(1), y: 0}]
    #     both fail
    assert solve(
        (exp(x) - x, exp(y) - y)) == [{x: -LambertW(-1), y: -LambertW(-1)}]
    # --  when symbols given
    assert solve([y, exp(x) + x], x, y) == [(-LambertW(1), 0)]
    # symbol is a number
    assert solve(x**2 - pi, pi) == [x**2]
    # no equations
    assert solve([], [x]) == []
    # nonlinear system
    assert solve((x**2 - 4, y - 2), x, y) == [(-2, 2), (2, 2)]
    assert solve((x**2 - 4, y - 2), y, x) == [(2, -2), (2, 2)]
    assert solve((x**2 - 4 + z, y - 2 - z), a, z, y, x, set=True
        ) == ([a, z, y, x], {
        (a, z, z + 2, -sqrt(4 - z)),
        (a, z, z + 2, sqrt(4 - z))})
    # overdetermined system
    # - nonlinear
    assert solve([(x + y)**2 - 4, x + y - 2]) == [{x: -y + 2}]
    # - linear
    assert solve((x + y - 2, 2*x + 2*y - 4)) == {x: -y + 2}
    # When one or more args are Boolean
    assert solve(Eq(x**2, 0.0)) == [0.0]  # issue 19048
    assert solve([True, Eq(x, 0)], [x], dict=True) == [{x: 0}]
    assert solve([Eq(x, x), Eq(x, 0), Eq(x, x+1)], [x], dict=True) == []
    assert not solve([Eq(x, x+1), x < 2], x)
    assert solve([Eq(x, 0), x+1<2]) == Eq(x, 0)
    assert solve([Eq(x, x), Eq(x, x+1)], x) == []
    assert solve(True, x) == []
    assert solve([x - 1, False], [x], set=True) == ([], set())
    assert solve([-y*(x + y - 1)/2, (y - 1)/x/y + 1/y],
        set=True, check=False) == ([x, y], {(1 - y, y), (x, 0)})
    # ordering should be canonical, fastest to order by keys instead
    # of by size
    assert list(solve((y - 1, x - sqrt(3)*z)).keys()) == [x, y]
    # as set always returns as symbols, set even if no solution
    assert solve([x - 1, x], (y, x), set=True) == ([y, x], set())
    assert solve([x - 1, x], {y, x}, set=True) == ([x, y], set())


def test_solve_polynomial1():
    assert solve(3*x - 2, x) == [Rational(2, 3)]
    assert solve(Eq(3*x, 2), x) == [Rational(2, 3)]

    assert set(solve(x**2 - 1, x)) == {-S.One, S.One}
    assert set(solve(Eq(x**2, 1), x)) == {-S.One, S.One}

    assert solve(x - y**3, x) == [y**3]
    rx = root(x, 3)
    assert solve(x - y**3, y) == [
        rx, -rx/2 - sqrt(3)*I*rx/2, -rx/2 +  sqrt(3)*I*rx/2]
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y) == \
        {
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
            y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
        }

    solution = {x: S.Zero, y: S.Zero}

    assert solve((x - y, x + y), x, y ) == solution
    assert solve((x - y, x + y), (x, y)) == solution
    assert solve((x - y, x + y), [x, y]) == solution

    assert set(solve(x**3 - 15*x - 4, x)) == {
        -2 + 3**S.Half,
        S(4),
        -2 - 3**S.Half
    }

    assert set(solve((x**2 - 1)**2 - a, x)) == \
        {sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))}


def test_solve_polynomial2():
    assert solve(4, x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to a polynomial equation
    using the change of variable y -> x**Rational(p, q)
    """
    assert solve( sqrt(x) - 1, x) == [1]
    assert solve( sqrt(x) - 2, x) == [4]
    assert solve( x**Rational(1, 4) - 2, x) == [16]
    assert solve( x**Rational(1, 3) - 3, x) == [27]
    assert solve(sqrt(x) + x**Rational(1, 3) + x**Rational(1, 4), x) == [0]


def test_solve_polynomial_cv_1b():
    assert set(solve(4*x*(1 - a*sqrt(x)), x)) == {S.Zero, 1/a**2}
    assert set(solve(x*(root(x, 3) - 3), x)) == {S.Zero, S(27)}


def test_solve_polynomial_cv_2():
    """
    Test for solving on equations that can be converted to a polynomial equation
    multiplying both sides of the equation by x**m
    """
    assert solve(x + 1/x - 1, x) in \
        [[ S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2],
         [ S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]]


def test_quintics_1():
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solve(f, check=False)
    for r in s:
        res = f.subs(x, r.n()).n()
        assert tn(res, 0)

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = solve(f)
    for r in s:
        assert r.func == CRootOf

    # if one uses solve to get the roots of a polynomial that has a CRootOf
    # solution, make sure that the use of nfloat during the solve process
    # doesn't fail. Note: if you want numerical solutions to a polynomial
    # it is *much* faster to use nroots to get them than to solve the
    # equation only to get RootOf solutions which are then numerically
    # evaluated. So for eq = x**5 + 3*x + 7 do Poly(eq).nroots() rather
    # than [i.n() for i in solve(eq)] to get the numerical roots of eq.
    assert nfloat(solve(x**5 + 3*x**3 + 7)[0], exponent=False) == \
        CRootOf(x**5 + 3*x**3 + 7, 0).n()


def test_quintics_2():
    f = x**5 + 15*x + 12
    s = solve(f, check=False)
    for r in s:
        res = f.subs(x, r.n()).n()
        assert tn(res, 0)

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = solve(f)
    for r in s:
        assert r.func == CRootOf

    assert solve(x**5 - 6*x**3 - 6*x**2 + x - 6) == [
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 0),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 1),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 2),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 3),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 4)]


def test_quintics_3():
    y = x**5 + x**3 - 2**Rational(1, 3)
    assert solve(y) == solve(-y) == []


def test_highorder_poly():
    # just testing that the uniq generator is unpacked
    sol = solve(x**6 - 2*x + 2)
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6


def test_solve_rational():
    """Test solve for rational functions"""
    assert solve( ( x - y**3 )/( (y**2)*sqrt(1 - y**2) ), x) == [y**3]


def test_solve_conjugate():
    """Test solve for simple conjugate functions"""
    assert solve(conjugate(x) -3 + I) == [3 + I]


def test_solve_nonlinear():
    assert solve(x**2 - y**2, x, y, dict=True) == [{x: -y}, {x: y}]
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: -x*sqrt(exp(x))},
                                                          {y: x*sqrt(exp(x))}]


def test_issue_8666():
    x = symbols('x')
    assert solve(Eq(x**2 - 1/(x**2 - 4), 4 - 1/(x**2 - 4)), x) == []
    assert solve(Eq(x + 1/x, 1/x), x) == []


def test_issue_7228():
    assert solve(4**(2*(x**2) + 2*x) - 8, x) == [Rational(-3, 2), S.Half]


def test_issue_7190():
    assert solve(log(x-3) + log(x+3), x) == [sqrt(10)]


def test_issue_21004():
    x = symbols('x')
    f = x/sqrt(x**2+1)
    f_diff = f.diff(x)
    assert solve(f_diff, x) == []


def test_issue_24650():
    x = symbols('x')
    r = solve(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0))
    assert r == [0]
    r = checksol(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0), x, sol=0)
    assert r is True


def test_linear_system():
    x, y, z, t, n = symbols('x, y, z, t, n')

    assert solve([x - 1, x - y, x - 2*y, y - 1], [x, y]) == []

    assert solve([x - 1, x - y, x - 2*y, x - 1], [x, y]) == []
    assert solve([x - 1, x - 1, x - y, x - 2*y], [x, y]) == []

    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == {x: -3, y: 1}

    M = Matrix([[0, 0, n*(n + 1), (n + 1)**2, 0],
                [n + 1, n + 1, -2*n - 1, -(n + 1), 0],
                [-1, 0, 1, 0, 0]])

    assert solve_linear_system(M, x, y, z, t) == \
        {x: t*(-n-1)/n, y: 0, z: t*(-n-1)/n}

    assert solve([x + y + z + t, -z - t], x, y, z, t) == {x: -y, z: -t}


@XFAIL
def test_linear_system_xfail():
    # https://github.com/sympy/sympy/issues/6420
    M = Matrix([[0,    15.0, 10.0, 700.0],
                [1,    1,    1,    100.0],
                [0,    10.0, 5.0,  200.0],
                [-5.0, 0,    0,    0    ]])

    assert solve_linear_system(M, x, y, z) == {x: 0, y: -60.0, z: 160.0}


def test_linear_system_function():
    a = Function('a')
    assert solve([a(0, 0) + a(0, 1) + a(1, 0) + a(1, 1), -a(1, 0) - a(1, 1)],
        a(0, 0), a(0, 1), a(1, 0), a(1, 1)) == {a(1, 0): -a(1, 1), a(0, 0): -a(0, 1)}


def test_linear_system_symbols_doesnt_hang_1():

    def _mk_eqs(wy):
        # Equations for fitting a wy*2 - 1 degree polynomial between two points,
        # at end points derivatives are known up to order: wy - 1
        order = 2*wy - 1
        x, x0, x1 = symbols('x, x0, x1', real=True)
        y0s = symbols('y0_:{}'.format(wy), real=True)
        y1s = symbols('y1_:{}'.format(wy), real=True)
        c = symbols('c_:{}'.format(order+1), real=True)

        expr = sum(coeff*x**o for o, coeff in enumerate(c))
        eqs = []
        for i in range(wy):
            eqs.append(expr.diff(x, i).subs({x: x0}) - y0s[i])
            eqs.append(expr.diff(x, i).subs({x: x1}) - y1s[i])
        return eqs, c

    #
    # The purpose of this test is just to see that these calls don't hang. The
    # expressions returned are complicated so are not included here. Testing
    # their correctness takes longer than solving the system.
    #

    for n in range(1, 7+1):
        eqs, c = _mk_eqs(n)
        solve(eqs, c)


def test_linear_system_symbols_doesnt_hang_2():

    M = Matrix([
        [66, 24, 39, 50, 88, 40, 37, 96, 16, 65, 31, 11, 37, 72, 16, 19, 55, 37, 28, 76],
        [10, 93, 34, 98, 59, 44, 67, 74, 74, 94, 71, 61, 60, 23,  6,  2, 57,  8, 29, 78],
        [19, 91, 57, 13, 64, 65, 24, 53, 77, 34, 85, 58, 87, 39, 39,  7, 36, 67, 91,  3],
        [74, 70, 15, 53, 68, 43, 86, 83, 81, 72, 25, 46, 67, 17, 59, 25, 78, 39, 63,  6],
        [69, 40, 67, 21, 67, 40, 17, 13, 93, 44, 46, 89, 62, 31, 30, 38, 18, 20, 12, 81],
        [50, 22, 74, 76, 34, 45, 19, 76, 28, 28, 11, 99, 97, 82,  8, 46, 99, 57, 68, 35],
        [58, 18, 45, 88, 10, 64,  9, 34, 90, 82, 17, 41, 43, 81, 45, 83, 22, 88, 24, 39],
        [42, 21, 70, 68,  6, 33, 64, 81, 83, 15, 86, 75, 86, 17, 77, 34, 62, 72, 20, 24],
        [ 7,  8,  2, 72, 71, 52, 96,  5, 32, 51, 31, 36, 79, 88, 25, 77, 29, 26, 33, 13],
        [19, 31, 30, 85, 81, 39, 63, 28, 19, 12, 16, 49, 37, 66, 38, 13,  3, 71, 61, 51],
        [29, 82, 80, 49, 26, 85,  1, 37,  2, 74, 54, 82, 26, 47, 54,  9, 35,  0, 99, 40],
        [15, 49, 82, 91, 93, 57, 45, 25, 45, 97, 15, 98, 48, 52, 66, 24, 62, 54, 97, 37],
        [62, 23, 73, 53, 52, 86, 28, 38,  0, 74, 92, 38, 97, 70, 71, 29, 26, 90, 67, 45],
        [ 2, 32, 23, 24, 71, 37, 25, 71,  5, 41, 97, 65, 93, 13, 65, 45, 25, 88, 69, 50],
        [40, 56,  1, 29, 79, 98, 79, 62, 37, 28, 45, 47,  3,  1, 32, 74, 98, 35, 84, 32],
        [33, 15, 87, 79, 65,  9, 14, 63, 24, 19, 46, 28, 74, 20, 29, 96, 84, 91, 93,  1],
        [97, 18, 12, 52,  1,  2, 50, 14, 52, 76, 19, 82, 41, 73, 51, 79, 13,  3, 82, 96],
        [40, 28, 52, 10, 10, 71, 56, 78, 82,  5, 29, 48,  1, 26, 16, 18, 50, 76, 86, 52],
        [38, 89, 83, 43, 29, 52, 90, 77, 57,  0, 67, 20, 81, 88, 48, 96, 88, 58, 14,  3]])

    syms = x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18 = symbols('x:19')

    sol = {
        x0:  -S(1967374186044955317099186851240896179)/3166636564687820453598895768302256588,
        x1:  -S(84268280268757263347292368432053826)/791659141171955113399723942075564147,
        x2:  -S(229962957341664730974463872411844965)/1583318282343910226799447884151128294,
        x3:   S(990156781744251750886760432229180537)/6333273129375640907197791536604513176,
        x4:  -S(2169830351210066092046760299593096265)/18999819388126922721593374609813539528,
        x5:   S(4680868883477577389628494526618745355)/9499909694063461360796687304906769764,
        x6:  -S(1590820774344371990683178396480879213)/3166636564687820453598895768302256588,
        x7:  -S(54104723404825537735226491634383072)/339282489073695048599881689460956063,
        x8:   S(3182076494196560075964847771774733847)/6333273129375640907197791536604513176,
        x9:  -S(10870817431029210431989147852497539675)/18999819388126922721593374609813539528,
        x10: -S(13118019242576506476316318268573312603)/18999819388126922721593374609813539528,
        x11: -S(5173852969886775824855781403820641259)/4749954847031730680398343652453384882,
        x12:  S(4261112042731942783763341580651820563)/4749954847031730680398343652453384882,
        x13: -S(821833082694661608993818117038209051)/6333273129375640907197791536604513176,
        x14:  S(906881575107250690508618713632090559)/904753304196520129599684505229216168,
        x15: -S(732162528717458388995329317371283987)/6333273129375640907197791536604513176,
        x16:  S(4524215476705983545537087360959896817)/9499909694063461360796687304906769764,
        x17: -S(3898571347562055611881270844646055217)/6333273129375640907197791536604513176,
        x18:  S(7513502486176995632751685137907442269)/18999819388126922721593374609813539528
    }

    eqs = list(M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol

    y = Symbol('y')
    eqs = list(y * M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol


def test_linear_systemLU():
    n = Symbol('n')

    M = Matrix([[1, 2, 0, 1], [1, 3, 2*n, 1], [4, -1, n**2, 1]])

    assert solve_linear_system_LU(M, [x, y, z]) == {z: -3/(n**2 + 18*n),
                                                  x: 1 - 12*n/(n**2 + 18*n),
                                                  y: 6*n/(n**2 + 18*n)}

# Note: multiple solutions exist for some of these equations, so the tests
# should be expected to break if the implementation of the solver changes
# in such a way that a different branch is chosen

@slow
def test_solve_transcendental():
    from sympy.abc import a, b

    assert solve(exp(x) - 3, x) == [log(3)]
    assert set(solve((a*x + b)*(exp(x) - 3), x)) == {-b/a, log(3)}
    assert solve(cos(x) - y, x) == [-acos(y) + 2*pi, acos(y)]
    assert solve(2*cos(x) - y, x) == [-acos(y/2) + 2*pi, acos(y/2)]
    assert solve(Eq(cos(x), sin(x)), x) == [pi/4]

    assert set(solve(exp(x) + exp(-x) - y, x)) in [{
        log(y/2 - sqrt(y**2 - 4)/2),
        log(y/2 + sqrt(y**2 - 4)/2),
    }, {
        log(y - sqrt(y**2 - 4)) - log(2),
        log(y + sqrt(y**2 - 4)) - log(2)},
    {
        log(y/2 - sqrt((y - 2)*(y + 2))/2),
        log(y/2 + sqrt((y - 2)*(y + 2))/2)}]
    assert solve(exp(x) - 3, x) == [log(3)]
    assert solve(Eq(exp(x), 3), x) == [log(3)]
    assert solve(log(x) - 3, x) == [exp(3)]
    assert solve(sqrt(3*x) - 4, x) == [Rational(16, 3)]
    assert solve(3**(x + 2), x) == []
    assert solve(3**(2 - x), x) == []
    assert solve(x + 2**x, x) == [-LambertW(log(2))/log(2)]
    assert solve(2*x + 5 + log(3*x - 2), x) == \
        [Rational(2, 3) + LambertW(2*exp(Rational(-19, 3))/3)/2]
    assert solve(3*x + log(4*x), x) == [LambertW(Rational(3, 4))/3]
    assert set(solve((2*x + 8)*(8 + exp(x)), x)) == {S(-4), log(8) + pi*I}
    eq = 2*exp(3*x + 4) - 3
    ans = solve(eq, x)  # this generated a failure in flatten
    assert len(ans) == 3 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve(2*log(3*x + 4) - 3, x) == [(exp(Rational(3, 2)) - 4)/3]
    assert solve(exp(x) + 1, x) == [pi*I]

    eq = 2*(3*x + 4)**5 - 6*7**(3*x + 9)
    result = solve(eq, x)
    x0 = -log(2401)
    x1 = 3**Rational(1, 5)
    x2 = log(7**(7*x1/20))
    x3 = sqrt(2)
    x4 = sqrt(5)
    x5 = x3*sqrt(x4 - 5)
    x6 = x4 + 1
    x7 = 1/(3*log(7))
    x8 = -x4
    x9 = x3*sqrt(x8 - 5)
    x10 = x8 + 1
    ans = [x7*(x0 - 5*LambertW(x2*(-x5 + x6))),
           x7*(x0 - 5*LambertW(x2*(x5 + x6))),
           x7*(x0 - 5*LambertW(x2*(x10 - x9))),
           x7*(x0 - 5*LambertW(x2*(x10 + x9))),
           x7*(x0 - 5*LambertW(-log(7**(7*x1/5))))]
    assert result == ans, result
    # it works if expanded, too
    assert solve(eq.expand(), x) == result

    assert solve(z*cos(x) - y, x) == [-acos(y/z) + 2*pi, acos(y/z)]
    assert solve(z*cos(2*x) - y, x) == [-acos(y/z)/2 + pi, acos(y/z)/2]
    assert solve(z*cos(sin(x)) - y, x) == [
        pi - asin(acos(y/z)), asin(acos(y/z) - 2*pi) + pi,
        -asin(acos(y/z) - 2*pi), asin(acos(y/z))]

    assert solve(z*cos(x), x) == [pi/2, pi*Rational(3, 2)]

    # issue 4508
    assert solve(y - b*x/(a + x), x) in [[-a*y/(y - b)], [a*y/(b - y)]]
    assert solve(y - b*exp(a/x), x) == [a/log(y/b)]
    # issue 4507
    assert solve(y - b/(1 + a*x), x) in [[(b - y)/(a*y)], [-((y - b)/(a*y))]]
    # issue 4506
    assert solve(y - a*x**b, x) == [(y/a)**(1/b)]
    # issue 4505
    assert solve(z**x - y, x) == [log(y)/log(z)]
    # issue 4504
    assert solve(2**x - 10, x) == [1 + log(5)/log(2)]
    # issue 6744
    assert solve(x*y) == [{x: 0}, {y: 0}]
    assert solve([x*y]) == [{x: 0}, {y: 0}]
    assert solve(x**y - 1) == [{x: 1}, {y: 0}]
    assert solve([x**y - 1]) == [{x: 1}, {y: 0}]
    assert solve(x*y*(x**2 - y**2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve([x*y*(x**2 - y**2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    # issue 4739
    assert solve(exp(log(5)*x) - 2**x, x) == [0]
    # issue 14791
    assert solve(exp(log(5)*x) - exp(log(2)*x), x) == [0]
    f = Function('f')
    assert solve(y*f(log(5)*x) - y*f(log(2)*x), x) == [0]
    assert solve(f(x) - f(0), x) == [0]
    assert solve(f(x) - f(2 - x), x) == [1]
    raises(NotImplementedError, lambda: solve(f(x, y) - f(1, 2), x))
    raises(NotImplementedError, lambda: solve(f(x, y) - f(2 - x, 2), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1 - x), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1), x))

    # misc
    # make sure that the right variables is picked up in tsolve
    # shouldn't generate a GeneratorsNeeded error in _tsolve when the NaN is generated
    # for eq_down. Actual answers, as determined numerically are approx. +/- 0.83
    raises(NotImplementedError, lambda:
        solve(sinh(x)*sinh(sinh(x)) + cosh(x)*cosh(sinh(x)) - 3))

    # watch out for recursive loop in tsolve
    raises(NotImplementedError, lambda: solve((x + 2)**y*x - 3, x))

    # issue 7245
    assert solve(sin(sqrt(x))) == [0, pi**2]

    # issue 7602
    a, b = symbols('a, b', real=True, negative=False)
    assert str(solve(Eq(a, 0.5 - cos(pi*b)/2), b)) == \
        '[2.0 - 0.318309886183791*acos(1.0 - 2.0*a), 0.318309886183791*acos(1.0 - 2.0*a)]'

    # issue 15325
    assert solve(y**(1/x) - z, x) == [log(y)/log(z)]

    # issue 25685 (basic trig identies should give simple solutions)
    for yi in [cos(2*x),sin(2*x),cos(x - pi/3)]:
        sol = solve([cos(x) - S(3)/5, yi - y])
        assert (sol[0][y] + sol[1][y]).is_Rational, (yi,sol)
    # don't allow massive expansion
    assert solve(cos(1000*x) - S.Half) == [pi/3000, pi/600]
    assert solve(cos(x - 1000*y) - 1, x) == [1000*y, 1000*y + 2*pi]
    assert solve(cos(x + y + z) - 1, x) == [-y - z, -y - z + 2*pi]

    # issue 26008
    assert solve(sin(x + pi/6)) == [-pi/6, 5*pi/6]


def test_solve_for_functions_derivatives():
    t = Symbol('t')
    x = Function('x')(t)
    y = Function('y')(t)
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    soln = solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y)
    assert soln == {
        x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
        y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
    }

    assert solve(x - 1, x) == [1]
    assert solve(3*x - 2, x) == [Rational(2, 3)]

    soln = solve([a11*x.diff(t) + a12*y.diff(t) - b1, a21*x.diff(t) +
            a22*y.diff(t) - b2], x.diff(t), y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x.diff(t): (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    assert solve(x.diff(t) - 1, x.diff(t)) == [1]
    assert solve(3*x.diff(t) - 2, x.diff(t)) == [Rational(2, 3)]

    eqns = {3*x - 1, 2*y - 4}
    assert solve(eqns, {x, y}) == { x: Rational(1, 3), y: 2 }
    x = Symbol('x')
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    assert solve(F.diff(x), diff(f(x), x)) == [(-x + 2)/f(x)]

    # Mixed cased with a Symbol and a Function
    x = Symbol('x')
    y = Function('y')(t)

    soln = solve([a11*x + a12*y.diff(t) - b1, a21*x +
            a22*y.diff(t) - b2], x, y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    # issue 13263
    x = Symbol('x')
    f = Function('f')
    soln = solve([f(x).diff(x) + f(x).diff(x, 2) - 1, f(x).diff(x) - f(x).diff(x, 2)],
            f(x).diff(x), f(x).diff(x, 2))
    assert soln == { f(x).diff(x, 2): S(1)/2, f(x).diff(x): S(1)/2 }

    soln = solve([f(x).diff(x, 2) + f(x).diff(x, 3) - 1, 1 - f(x).diff(x, 2) -
            f(x).diff(x, 3), 1 - f(x).diff(x,3)], f(x).diff(x, 2), f(x).diff(x, 3))
    assert soln == { f(x).diff(x, 2): 0, f(x).diff(x, 3): 1 }


def test_issue_3725():
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    e = F.diff(x)
    assert solve(e, f(x).diff(x)) in [[(2 - x)/f(x)], [-((x - 2)/f(x))]]


def test_issue_3870():
    a, b, c, d = symbols('a b c d')
    A = Matrix(2, 2, [a, b, c, d])
    B = Matrix(2, 2, [0, 2, -3, 0])
    C = Matrix(2, 2, [1, 2, 3, 4])

    assert solve(A*B - C, [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve([A*B - C], [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve(Eq(A*B, C), [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}

    assert solve([A*B - B*A], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([A*C - C*A], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([A*B - B*A, A*C - C*A], [a, b, c, d]) == {a: d, b: 0, c: 0}

    assert solve([Eq(A*B, B*A)], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([Eq(A*C, C*A)], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([Eq(A*B, B*A), Eq(A*C, C*A)], [a, b, c, d]) == {a: d, b: 0, c: 0}


def test_solve_linear():
    w = Wild('w')
    assert solve_linear(x, x) == (0, 1)
    assert solve_linear(x, exclude=[x]) == (0, 1)
    assert solve_linear(x, symbols=[w]) == (0, 1)
    assert solve_linear(x, y - 2*x) in [(x, y/3), (y, 3*x)]
    assert solve_linear(x, y - 2*x, exclude=[x]) == (y, 3*x)
    assert solve_linear(3*x - y, 0) in [(x, y/3), (y, 3*x)]
    assert solve_linear(3*x - y, 0, [x]) == (x, y/3)
    assert solve_linear(3*x - y, 0, [y]) == (y, 3*x)
    assert solve_linear(x**2/y, 1) == (y, x**2)
    assert solve_linear(w, x) in [(w, x), (x, w)]
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y) == \
        (y, -2 - cos(x)**2 - sin(x)**2)
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y, symbols=[x]) == (0, 1)
    assert solve_linear(Eq(x, 3)) == (x, 3)
    assert solve_linear(1/(1/x - 2)) == (0, 0)
    assert solve_linear((x + 1)*exp(-x), symbols=[x]) == (x, -1)
    assert solve_linear((x + 1)*exp(x), symbols=[x]) == ((x + 1)*exp(x), 1)
    assert solve_linear(x*exp(-x**2), symbols=[x]) == (x, 0)
    assert solve_linear(0**x - 1) == (0**x - 1, 1)
    assert solve_linear(1 + 1/(x - 1)) == (x, 0)
    eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    assert solve_linear(eq) == (0, 1)
    eq = cos(x)**2 + sin(x)**2  # = 1
    assert solve_linear(eq) == (0, 1)
    raises(ValueError, lambda: solve_linear(Eq(x, 3), 3))


def test_solve_undetermined_coeffs():
    assert solve_undetermined_coeffs(
        a*x**2 + b*x**2 + b*x + 2*c*x + c + 1, [a, b, c], x
        ) == {a: -2, b: 2, c: -1}
    # Test that rational functions work
    assert solve_undetermined_coeffs(a/x + b/(x + 1)
        - (2*x + 1)/(x**2 + x), [a, b], x) == {a: 1, b: 1}
    # Test cancellation in rational functions
    assert solve_undetermined_coeffs(
        ((c + 1)*a*x**2 + (c + 1)*b*x**2 +
        (c + 1)*b*x + (c + 1)*2*c*x + (c + 1)**2)/(c + 1),
        [a, b, c], x) == \
        {a: -2, b: 2, c: -1}
    # multivariate
    X, Y, Z = y, x**y, y*x**y
    eq = a*X + b*Y + c*Z - X - 2*Y - 3*Z
    coeffs = a, b, c
    syms = x, y
    assert solve_undetermined_coeffs(eq, coeffs) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, syms) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, *syms) == {
        a: 1, b: 2, c: 3}
    # check output format
    assert solve_undetermined_coeffs(a*x + a - 2, [a]) == []
    assert solve_undetermined_coeffs(a**2*x - 4*x, [a]) == [
        {a: -2}, {a: 2}]
    assert solve_undetermined_coeffs(0, [a]) == []
    assert solve_undetermined_coeffs(0, [a], dict=True) == []
    assert solve_undetermined_coeffs(0, [a], set=True) == ([], {})
    assert solve_undetermined_coeffs(1, [a]) == []
    abeq = a*x - 2*x + b - 3
    s = {b, a}
    assert solve_undetermined_coeffs(abeq, s, x) == {a: 2, b: 3}
    assert solve_undetermined_coeffs(abeq, s, x, set=True) == ([a, b], {(2, 3)})
    assert solve_undetermined_coeffs(sin(a*x) - sin(2*x), (a,)) is None
    assert solve_undetermined_coeffs(a*x + b*x - 2*x, (a, b)) == {a: 2 - b}


def test_solve_inequalities():
    x = Symbol('x')
    sol = And(S.Zero < x, x < oo)
    assert solve(x + 1 > 1) == sol
    assert solve([x + 1 > 1]) == sol
    assert solve([x + 1 > 1], x) == sol
    assert solve([x + 1 > 1], [x]) == sol

    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    assert solve(system) == \
        And(Or(And(Lt(-sqrt(2), x), Lt(x, -1)),
               And(Lt(1, x), Lt(x, sqrt(2)))), Eq(0, 0))

    x = Symbol('x', real=True)
    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    assert solve(system) == \
        Or(And(Lt(-sqrt(2), x), Lt(x, -1)), And(Lt(1, x), Lt(x, sqrt(2))))

    # issues 6627, 3448
    assert solve((x - 3)/(x - 2) < 0, x) == And(Lt(2, x), Lt(x, 3))
    assert solve(x/(x + 1) > 1, x) == And(Lt(-oo, x), Lt(x, -1))

    assert solve(sin(x) > S.Half) == And(pi/6 < x, x < pi*Rational(5, 6))

    assert solve(Eq(False, x < 1)) == (S.One <= x) & (x < oo)
    assert solve(Eq(True, x < 1)) == (-oo < x) & (x < 1)
    assert solve(Eq(x < 1, False)) == (S.One <= x) & (x < oo)
    assert solve(Eq(x < 1, True)) == (-oo < x) & (x < 1)

    assert solve(Eq(False, x)) == False
    assert solve(Eq(0, x)) == [0]
    assert solve(Eq(True, x)) == True
    assert solve(Eq(1, x)) == [1]
    assert solve(Eq(False, ~x)) == True
    assert solve(Eq(True, ~x)) == False
    assert solve(Ne(True, x)) == False
    assert solve(Ne(1, x)) == (x > -oo) & (x < oo) & Ne(x, 1)


def test_issue_4793():
    assert solve(1/x) == []
    assert solve(x*(1 - 5/x)) == [5]
    assert solve(x + sqrt(x) - 2) == [1]
    assert solve(-(1 + x)/(2 + x)**2 + 1/(2 + x)) == []
    assert solve(-x**2 - 2*x + (x + 1)**2 - 1) == []
    assert solve((x/(x + 1) + 3)**(-2)) == []
    assert solve(x/sqrt(x**2 + 1), x) == [0]
    assert solve(exp(x) - y, x) == [log(y)]
    assert solve(exp(x)) == []
    assert solve(x**2 + x + sin(y)**2 + cos(y)**2 - 1, x) in [[0, -1], [-1, 0]]
    eq = 4*3**(5*x + 2) - 7
    ans = solve(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve(log(x**2) - y**2/exp(x), x, y, set=True) == (
        [x, y],
        {(x, sqrt(exp(x) * log(x ** 2))), (x, -sqrt(exp(x) * log(x ** 2)))})
    assert solve(x**2*z**2 - z**2*y**2) == [{x: -y}, {x: y}, {z: 0}]
    assert solve((x - 1)/(1 + 1/(x - 1))) == []
    assert solve(x**(y*z) - x, x) == [1]
    raises(NotImplementedError, lambda: solve(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve(2**x - exp(x) - 3))


def test_PR1964():
    # issue 5171
    assert solve(sqrt(x)) == solve(sqrt(x**3)) == [0]
    assert solve(sqrt(x - 1)) == [1]
    # issue 4462
    a = Symbol('a')
    assert solve(-3*a/sqrt(x), x) == []
    # issue 4486
    assert solve(2*x/(x + 2) - 1, x) == [2]
    # issue 4496
    assert set(solve((x**2/(7 - x)).diff(x))) == {S.Zero, S(14)}
    # issue 4695
    f = Function('f')
    assert solve((3 - 5*x/f(x))*f(x), f(x)) == [x*Rational(5, 3)]
    # issue 4497
    assert solve(1/root(5 + x, 5) - 9, x) == [Rational(-295244, 59049)]

    assert solve(sqrt(x) + sqrt(sqrt(x)) - 4) == [(Rational(-1, 2) + sqrt(17)/2)**4]
    assert set(solve(Poly(sqrt(exp(x)) + sqrt(exp(-x)) - 4))) in \
        [
            {log((-sqrt(3) + 2)**2), log((sqrt(3) + 2)**2)},
            {2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2)},
            {log(-4*sqrt(3) + 7), log(4*sqrt(3) + 7)},
        ]
    assert set(solve(Poly(exp(x) + exp(-x) - 4))) == \
        {log(-sqrt(3) + 2), log(sqrt(3) + 2)}
    assert set(solve(x**y + x**(2*y) - 1, x)) == \
        {(Rational(-1, 2) + sqrt(5)/2)**(1/y), (Rational(-1, 2) - sqrt(5)/2)**(1/y)}

    assert solve(exp(x/y)*exp(-z/y) - 2, y) == [(x - z)/log(2)]
    assert solve(
        x**z*y**z - 2, z) in [[log(2)/(log(x) + log(y))], [log(2)/(log(x*y))]]
    # if you do inversion too soon then multiple roots (as for the following)
    # will be missed, e.g. if exp(3*x) = exp(3) -> 3*x = 3
    E = S.Exp1
    assert solve(exp(3*x) - exp(3), x) in [
        [1, log(E*(Rational(-1, 2) - sqrt(3)*I/2)), log(E*(Rational(-1, 2) + sqrt(3)*I/2))],
        [1, log(-E/2 - sqrt(3)*E*I/2), log(-E/2 + sqrt(3)*E*I/2)],
        ]

    # coverage test
    p = Symbol('p', positive=True)
    assert solve((1/p + 1)**(p + 1)) == []


def test_issue_5197():
    x = Symbol('x', real=True)
    assert solve(x**2 + 1, x) == []
    n = Symbol('n', integer=True, positive=True)
    assert solve((n - 1)*(n + 2)*(2*n - 1), n) == [1]
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == []
                 # not {x: -3, y: 1} b/c x is positive
    # The solution following should not contain (-sqrt(2), sqrt(2))
    assert solve([(x + y), 2 - y**2], x, y) == [(sqrt(2), -sqrt(2))]
    y = Symbol('y', positive=True)
    # The solution following should not contain {y: -x*exp(x/2)}
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: x*exp(x/2)}]
    x, y, z = symbols('x y z', positive=True)
    assert solve(z**2*x**2 - z**2*y**2/exp(x), y, x, z, dict=True) == [{y: x*exp(x/2)}]


def test_checking():
    assert set(
        solve(x*(x - y/x), x, check=False)) == {sqrt(y), S.Zero, -sqrt(y)}
    assert set(solve(x*(x - y/x), x, check=True)) == {sqrt(y), -sqrt(y)}
    # {x: 0, y: 4} sets denominator to 0 in the following so system should return None
    assert solve((1/(1/x + 2), 1/(y - 3) - 1)) == []
    # 0 sets denominator of 1/x to zero so None is returned
    assert solve(1/(1/x + 2)) == []


def test_issue_4671_4463_4467():
    assert solve(sqrt(x**2 - 1) - 2) in ([sqrt(5), -sqrt(5)],
                                           [-sqrt(5), sqrt(5)])
    assert solve((2**exp(y**2/x) + 2)/(x**2 + 15), y) == [
        -sqrt(x*log(1 + I*pi/log(2))), sqrt(x*log(1 + I*pi/log(2)))]

    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solve(C1 + C2/x**2 - exp(-f(x)), f(x)) == [log(x**2/(C1*x**2 + C2))]
    a = Symbol('a')
    E = S.Exp1
    assert solve(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2]
    )
    assert solve(log(a**(-3) - x**2)/a, x) in (
        [-sqrt(-1 + a**(-3)), sqrt(-1 + a**(-3))],
        [sqrt(-1 + a**(-3)), -sqrt(-1 + a**(-3))],)
    assert solve(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2],)
    assert solve((a**2 + 1)*(sin(a*x) + cos(a*x)), x) == [-pi/(4*a)]
    assert solve(3 - (sinh(a*x) + cosh(a*x)), x) == [log(3)/a]
    assert set(solve(3 - (sinh(a*x) + cosh(a*x)**2), x)) == \
        {log(-2 + sqrt(5))/a, log(-sqrt(2) + 1)/a,
        log(-sqrt(5) - 2)/a, log(1 + sqrt(2))/a}
    assert solve(atan(x) - 1) == [tan(1)]


def test_issue_5132():
    r, t = symbols('r,t')
    assert set(solve([r - x**2 - y**2, tan(t) - y/x], [x, y])) == \
        {(
            -sqrt(r*cos(t)**2), -1*sqrt(r*cos(t)**2)*tan(t)),
            (sqrt(r*cos(t)**2), sqrt(r*cos(t)**2)*tan(t))}
    assert solve([exp(x) - sin(y), 1/y - 3], [x, y]) == \
        [(log(sin(Rational(1, 3))), Rational(1, 3))]
    assert solve([exp(x) - sin(y), 1/exp(y) - 3], [x, y]) == \
        [(log(-sin(log(3))), -log(3))]
    assert set(solve([exp(x) - sin(y), y**2 - 4], [x, y])) == \
        {(log(-sin(2)), -S(2)), (log(sin(2)), S(2))}
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    assert solve(eqs, set=True) == \
        ([y, z], {
        (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
        (-log(3), -sqrt(-exp(2*x) - sin(log(3))))})
    assert solve(eqs, x, z, set=True) == (
        [x, z],
        {(x, sqrt(-exp(2*x) + sin(y))), (x, -sqrt(-exp(2*x) + sin(y)))})
    assert set(solve(eqs, x, y)) == \
        {
            (log(-sqrt(-z**2 - sin(log(3)))), -log(3)),
        (log(-z**2 - sin(log(3)))/2, -log(3))}
    assert set(solve(eqs, y, z)) == \
        {
            (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
        (-log(3), sqrt(-exp(2*x) - sin(log(3))))}
    eqs = [exp(x)**2 - sin(y) + z, 1/exp(y) - 3]
    assert solve(eqs, set=True) == ([y, z], {
        (-log(3), -exp(2*x) - sin(log(3)))})
    assert solve(eqs, x, z, set=True) == (
        [x, z], {(x, -exp(2*x) + sin(y))})
    assert set(solve(eqs, x, y)) == {
            (log(-sqrt(-z - sin(log(3)))), -log(3)),
            (log(-z - sin(log(3)))/2, -log(3))}
    assert solve(eqs, z, y) == \
        [(-exp(2*x) - sin(log(3)), -log(3))]
    assert solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), set=True) == (
        [x, y], {(S.One, S(3)), (S(3), S.One)})
    assert set(solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), x, y)) == \
        {(S.One, S(3)), (S(3), S.One)}


def test_issue_5335():
    lam, a0, conc = symbols('lam a0 conc')
    a = 0.005
    b = 0.743436700916726
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    sym = [x, y, a0]
    # there are 4 solutions obtained manually but only two are valid
    assert len(solve(eqs, sym, manual=True, minimal=True)) == 2
    assert len(solve(eqs, sym)) == 2  # cf below with rational=False


@SKIP("Hangs")
def _test_issue_5335_float():
    # gives ZeroDivisionError: polynomial division
    lam, a0, conc = symbols('lam a0 conc')
    a = 0.005
    b = 0.743436700916726
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    sym = [x, y, a0]
    assert len(solve(eqs, sym, rational=False)) == 2


def test_issue_5767():
    assert set(solve([x**2 + y + 4], [x])) == \
        {(-sqrt(-y - 4),), (sqrt(-y - 4),)}


def _make_example_24609():
    D, R, H, B_g, V, D_c = symbols("D, R, H, B_g, V, D_c", real=True, positive=True)
    Sigma_f, Sigma_a, nu = symbols("Sigma_f, Sigma_a, nu", real=True, positive=True)
    x = symbols("x", real=True, positive=True)
    eq = (
        2**(S(2)/3)*pi**(S(2)/3)*D_c*(S(231361)/10000 + pi**2/x**2)
        /(6*V**(S(2)/3)*x**(S(1)/3))
      - 2**(S(2)/3)*pi**(S(8)/3)*D_c/(2*V**(S(2)/3)*x**(S(7)/3))
    )
    expected = 100*sqrt(2)*pi/481
    return eq, expected, x


def test_issue_24609():
    # https://github.com/sympy/sympy/issues/24609
    eq, expected, x = _make_example_24609()
    assert solve(eq, x, simplify=True) == [expected]
    [solapprox] = solve(eq.n(), x)
    assert abs(solapprox - expected.n()) < 1e-14


@XFAIL
def test_issue_24609_xfail():
    #
    # This returns 5 solutions when it should be 1 (with x positive).
    # Simplification reveals all solutions to be equivalent. It is expected
    # that solve without simplify=True returns duplicate solutions in some
    # cases but the core of this equation is a simple quadratic that can easily
    # be solved without introducing any redundant solutions:
    #
    #     >>> print(factor_terms(eq.as_numer_denom()[0]))
    #     2**(2/3)*pi**(2/3)*D_c*V**(2/3)*x**(7/3)*(231361*x**2 - 20000*pi**2)
    #
    eq, expected, x = _make_example_24609()
    assert len(solve(eq, x)) == [expected]
    #
    # We do not want to pass this test just by using simplify so if the above
    # passes then uncomment the additional test below:
    #
    # assert len(solve(eq, x, simplify=False)) == 1


def test_polysys():
    assert set(solve([x**2 + 2/y - 2, x + y - 3], [x, y])) == \
        {(S.One, S(2)), (1 + sqrt(5), 2 - sqrt(5)),
        (1 - sqrt(5), 2 + sqrt(5))}
    assert solve([x**2 + y - 2, x**2 + y]) == []
    # the ordering should be whatever the user requested
    assert solve([x**2 + y - 3, x - y - 4], (x, y)) != solve([x**2 +
                 y - 3, x - y - 4], (y, x))


@slow
def test_unrad1():
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x)) + 3))
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + (x + 1)**Rational(1, 3) + 2*sqrt(y)))

    s = symbols('s', cls=Dummy)

    # checkers to deal with possibility of answer coming
    # back with a sign change (cf issue 5203)
    def check(rv, ans):
        assert bool(rv[1]) == bool(ans[1])
        if ans[1]:
            return s_check(rv, ans)
        e = rv[0].expand()
        a = ans[0].expand()
        return e in [a, -a] and rv[1] == ans[1]

    def s_check(rv, ans):
        # get the dummy
        rv = list(rv)
        d = rv[0].atoms(Dummy)
        reps = list(zip(d, [s]*len(d)))
        # replace s with this dummy
        rv = (rv[0].subs(reps).expand(), [rv[1][0].subs(reps), rv[1][1].subs(reps)])
        ans = (ans[0].subs(reps).expand(), [ans[1][0].subs(reps), ans[1][1].subs(reps)])
        return str(rv[0]) in [str(ans[0]), str(-ans[0])] and \
            str(rv[1]) == str(ans[1])

    assert unrad(1) is None
    assert check(unrad(sqrt(x)),
        (x, []))
    assert check(unrad(sqrt(x) + 1),
        (x - 1, []))
    assert check(unrad(sqrt(x) + root(x, 3) + 2),
        (s**3 + s**2 + 2, [s, s**6 - x]))
    assert check(unrad(sqrt(x)*root(x, 3) + 2),
        (x**5 - 64, []))
    assert check(unrad(sqrt(x) + (x + 1)**Rational(1, 3)),
        (x**3 - (x + 1)**2, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(2*x)),
        (-2*sqrt(2)*x - 2*x + 1, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + 2),
        (16*x - 9, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - x)),
        (5*x**2 - 4*x, []))
    assert check(unrad(a*sqrt(x) + b*sqrt(x) + c*sqrt(y) + d*sqrt(y)),
        ((a*sqrt(x) + b*sqrt(x))**2 - (c*sqrt(y) + d*sqrt(y))**2, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x)),
        (2*x - 1, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x) - 3),
        (x**2 - x + 16, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x)),
        (5*x**2 - 2*x + 1, []))
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - 3) in [
        (25*x**4 + 376*x**3 + 1256*x**2 - 2272*x + 784, []),
        (25*x**8 - 476*x**6 + 2534*x**4 - 1468*x**2 + 169, [])]
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - sqrt(1 - 2*x)) == \
        (41*x**4 + 40*x**3 + 232*x**2 - 160*x + 16, [])  # orig root at 0.487
    assert check(unrad(sqrt(x) + sqrt(x + 1)), (S.One, []))

    eq = sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x))
    assert check(unrad(eq),
        (16*x**2 - 9*x, []))
    assert set(solve(eq, check=False)) == {S.Zero, Rational(9, 16)}
    assert solve(eq) == []
    # but this one really does have those solutions
    assert set(solve(sqrt(x) - sqrt(x + 1) + sqrt(1 - sqrt(x)))) == \
        {S.Zero, Rational(9, 16)}

    assert check(unrad(sqrt(x) + root(x + 1, 3) + 2*sqrt(y), y),
        (S('2*sqrt(x)*(x + 1)**(1/3) + x - 4*y + (x + 1)**(2/3)'), []))
    assert check(unrad(sqrt(x/(1 - x)) + (x + 1)**Rational(1, 3)),
        (x**5 - x**4 - x**3 + 2*x**2 + x - 1, []))
    assert check(unrad(sqrt(x/(1 - x)) + 2*sqrt(y), y),
        (4*x*y + x - 4*y, []))
    assert check(unrad(sqrt(x)*sqrt(1 - x) + 2, x),
        (x**2 - x + 4, []))

    # http://tutorial.math.lamar.edu/
    #        Classes/Alg/SolveRadicalEqns.aspx#Solve_Rad_Ex2_a
    assert solve(Eq(x, sqrt(x + 6))) == [3]
    assert solve(Eq(x + sqrt(x - 4), 4)) == [4]
    assert solve(Eq(1, x + sqrt(2*x - 3))) == []
    assert set(solve(Eq(sqrt(5*x + 6) - 2, x))) == {-S.One, S(2)}
    assert set(solve(Eq(sqrt(2*x - 1) - sqrt(x - 4), 2))) == {S(5), S(13)}
    assert solve(Eq(sqrt(x + 7) + 2, sqrt(3 - x))) == [-6]
    # http://www.purplemath.com/modules/solverad.htm
    assert solve((2*x - 5)**Rational(1, 3) - 3) == [16]
    assert set(solve(x + 1 - root(x**4 + 4*x**3 - x, 4))) == \
        {Rational(-1, 2), Rational(-1, 3)}
    assert set(solve(sqrt(2*x**2 - 7) - (3 - x))) == {-S(8), S(2)}
    assert solve(sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)) == [0]
    assert solve(sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)) == [5]
    assert solve(sqrt(x)*sqrt(x - 7) - 12) == [16]
    assert solve(sqrt(x - 3) + sqrt(x) - 3) == [4]
    assert solve(sqrt(9*x**2 + 4) - (3*x + 2)) == [0]
    assert solve(sqrt(x) - 2 - 5) == [49]
    assert solve(sqrt(x - 3) - sqrt(x) - 3) == []
    assert solve(sqrt(x - 1) - x + 7) == [10]
    assert solve(sqrt(x - 2) - 5) == [27]
    assert solve(sqrt(17*x - sqrt(x**2 - 5)) - 7) == [3]
    assert solve(sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))) == []

    # don't posify the expression in unrad and do use _mexpand
    z = sqrt(2*x + 1)/sqrt(x) - sqrt(2 + 1/x)
    p = posify(z)[0]
    assert solve(p) == []
    assert solve(z) == []
    assert solve(z + 6*I) == [Rational(-1, 11)]
    assert solve(p + 6*I) == []
    # issue 8622
    assert unrad(root(x + 1, 5) - root(x, 3)) == (
        -(x**5 - x**3 - 3*x**2 - 3*x - 1), [])
    # issue #8679
    assert check(unrad(x + root(x, 3) + root(x, 3)**2 + sqrt(y), x),
        (s**3 + s**2 + s + sqrt(y), [s, s**3 - x]))

    # for coverage
    assert check(unrad(sqrt(x) + root(x, 3) + y),
        (s**3 + s**2 + y, [s, s**6 - x]))
    assert solve(sqrt(x) + root(x, 3) - 2) == [1]
    raises(NotImplementedError, lambda:
        solve(sqrt(x) + root(x, 3) + root(x + 1, 5) - 2))
    # fails through a different code path
    raises(NotImplementedError, lambda: solve(-sqrt(2) + cosh(x)/x))
    # unrad some
    assert solve(sqrt(x + root(x, 3))+root(x - y, 5), y) == [
        x + (x**Rational(1, 3) + x)**Rational(5, 2)]
    assert check(unrad(sqrt(x) - root(x + 1, 3)*sqrt(x + 2) + 2),
        (s**10 + 8*s**8 + 24*s**6 - 12*s**5 - 22*s**4 - 160*s**3 - 212*s**2 -
        192*s - 56, [s, s**2 - x]))
    e = root(x + 1, 3) + root(x, 3)
    assert unrad(e) == (2*x + 1, [])
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    assert check(unrad(eq),
        (15625*x**4 + 173000*x**3 + 355600*x**2 - 817920*x + 331776, []))
    assert check(unrad(root(x, 4) + root(x, 4)**3 - 1),
        (s**3 + s - 1, [s, s**4 - x]))
    assert check(unrad(root(x, 2) + root(x, 2)**3 - 1),
        (x**3 + 2*x**2 + x - 1, []))
    assert unrad(x**0.5) is None
    assert check(unrad(t + root(x + y, 5) + root(x + y, 5)**3),
        (s**3 + s + t, [s, s**5 - x - y]))
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, y),
        (s**3 + s + x, [s, s**5 - x - y]))
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, x),
        (s**5 + s**3 + s - y, [s, s**5 - x - y]))
    assert check(unrad(root(x - 1, 3) + root(x + 1, 5) + root(2, 5)),
        (s**5 + 5*2**Rational(1, 5)*s**4 + s**3 + 10*2**Rational(2, 5)*s**3 +
        10*2**Rational(3, 5)*s**2 + 5*2**Rational(4, 5)*s + 4, [s, s**3 - x + 1]))
    raises(NotImplementedError, lambda:
        unrad((root(x, 2) + root(x, 3) + root(x, 4)).subs(x, x**5 - x + 1)))

    # the simplify flag should be reset to False for unrad results;
    # if it's not then this next test will take a long time
    assert solve(root(x, 3) + root(x, 5) - 2) == [1]
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    assert check(unrad(eq),
        ((5*x - 4)*(3125*x**3 + 37100*x**2 + 100800*x - 82944), []))
    ans = S('''
        [4/5, -1484/375 + 172564/(140625*(114*sqrt(12657)/78125 +
        12459439/52734375)**(1/3)) +
        4*(114*sqrt(12657)/78125 + 12459439/52734375)**(1/3)]''')
    assert solve(eq) == ans
    # duplicate radical handling
    assert check(unrad(sqrt(x + root(x + 1, 3)) - root(x + 1, 3) - 2),
        (s**3 - s**2 - 3*s - 5, [s, s**3 - x - 1]))
    # cov post-processing
    e = root(x**2 + 1, 3) - root(x**2 - 1, 5) - 2
    assert check(unrad(e),
        (s**5 - 10*s**4 + 39*s**3 - 80*s**2 + 80*s - 30,
        [s, s**3 - x**2 - 1]))

    e = sqrt(x + root(x + 1, 2)) - root(x + 1, 3) - 2
    assert check(unrad(e),
        (s**6 - 2*s**5 - 7*s**4 - 3*s**3 + 26*s**2 + 40*s + 25,
        [s, s**3 - x - 1]))
    assert check(unrad(e, _reverse=True),
        (s**6 - 14*s**5 + 73*s**4 - 187*s**3 + 276*s**2 - 228*s + 89,
        [s, s**2 - x - sqrt(x + 1)]))
    # this one needs r0, r1 reversal to work
    assert check(unrad(sqrt(x + sqrt(root(x, 3) - 1)) - root(x, 6) - 2),
        (s**12 - 2*s**8 - 8*s**7 - 8*s**6 + s**4 + 8*s**3 + 23*s**2 +
        32*s + 17, [s, s**6 - x]))

    # why does this pass
    assert unrad(root(cosh(x), 3)/x*root(x + 1, 5) - 1) == (
        -(x**15 - x**3*cosh(x)**5 - 3*x**2*cosh(x)**5 - 3*x*cosh(x)**5
        - cosh(x)**5), [])
    # and this fail?
    #assert unrad(sqrt(cosh(x)/x) + root(x + 1, 3)*sqrt(x) - 1) == (
    #    -s**6 + 6*s**5 - 15*s**4 + 20*s**3 - 15*s**2 + 6*s + x**5 +
    #    2*x**4 + x**3 - 1, [s, s**2 - cosh(x)/x])

    # watch for symbols in exponents
    assert unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1')) is None
    assert check(unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1'), x),
        (s**(2*y) + s + 1, [s, s**3 - x - y]))
    # should _Q be so lenient?
    assert unrad(x**(S.Half/y) + y, x) == (x**(1/y) - y**2, [])

    # This tests two things: that if full unrad is attempted and fails
    # the solution should still be found; also it tests that the use of
    # composite
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3

    # watch out for when the cov doesn't involve the symbol of interest
    eq = S('-x + (7*y/8 - (27*x/2 + 27*sqrt(x**2)/2)**(1/3)/3)**3 - 1')
    assert solve(eq, y) == [
        2**(S(2)/3)*(27*x + 27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 - sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 + sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 + S(512)/343)**(S(1)/3)]

    eq = root(x + 1, 3) - (root(x, 3) + root(x, 5))
    assert check(unrad(eq),
        (3*s**13 + 3*s**11 + s**9 - 1, [s, s**15 - x]))
    assert check(unrad(eq - 2),
        (3*s**13 + 3*s**11 + 6*s**10 + s**9 + 12*s**8 + 6*s**6 + 12*s**5 +
        12*s**3 + 7, [s, s**15 - x]))
    assert check(unrad(root(x, 3) - root(x + 1, 4)/2 + root(x + 2, 3)),
        (s*(4096*s**9 + 960*s**8 + 48*s**7 - s**6 - 1728),
        [s, s**4 - x - 1]))  # orig expr has two real roots: -1, -.389
    assert check(unrad(root(x, 3) + root(x + 1, 4) - root(x + 2, 3)/2),
        (343*s**13 + 2904*s**12 + 1344*s**11 + 512*s**10 - 1323*s**9 -
        3024*s**8 - 1728*s**7 + 1701*s**5 + 216*s**4 - 729*s, [s, s**4 - x -
        1]))  # orig expr has one real root: -0.048
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3)),
        (729*s**13 - 216*s**12 + 1728*s**11 - 512*s**10 + 1701*s**9 -
        3024*s**8 + 1344*s**7 + 1323*s**5 - 2904*s**4 + 343*s, [s, s**4 - x -
        1]))  # orig expr has 2 real roots: -0.91, -0.15
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3) - 2),
        (729*s**13 + 1242*s**12 + 18496*s**10 + 129701*s**9 + 388602*s**8 +
        453312*s**7 - 612864*s**6 - 3337173*s**5 - 6332418*s**4 - 7134912*s**3
        - 5064768*s**2 - 2111913*s - 398034, [s, s**4 - x - 1]))
        # orig expr has 1 real root: 19.53

    ans = solve(sqrt(x) + sqrt(x + 1) -
                sqrt(1 - x) - sqrt(2 + x))
    assert len(ans) == 1 and NS(ans[0])[:4] == '0.73'
    # the fence optimization problem
    # https://github.com/sympy/sympy/issues/4793#issuecomment-36994519
    F = Symbol('F')
    eq = F - (2*x + 2*y + sqrt(x**2 + y**2))
    ans = F*Rational(2, 7) - sqrt(2)*F/14
    X = solve(eq, x, check=False)
    for xi in reversed(X):  # reverse since currently, ans is the 2nd one
        Y = solve((x*y).subs(x, xi).diff(y), y, simplify=False, check=False)
        if any((a - ans).expand().is_zero for a in Y):
            break
    else:
        assert None  # no answer was found
    assert solve(sqrt(x + 1) + root(x, 3) - 2) == S('''
        [(-11/(9*(47/54 + sqrt(93)/6)**(1/3)) + 1/3 + (47/54 +
        sqrt(93)/6)**(1/3))**3]''')
    assert solve(sqrt(sqrt(x + 1)) + x**Rational(1, 3) - 2) == S('''
        [(-sqrt(-2*(-1/16 + sqrt(6913)/16)**(1/3) + 6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/2 + 121/(4*sqrt(-6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 2*(-1/16 + sqrt(6913)/16)**(1/3) + 17/4)))/2 +
        sqrt(-6/(-1/16 + sqrt(6913)/16)**(1/3) + 2*(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/4)/2 + 9/4)**3]''')
    assert solve(sqrt(x) + root(sqrt(x) + 1, 3) - 2) == S('''
        [(-(81/2 + 3*sqrt(741)/2)**(1/3)/3 + (81/2 + 3*sqrt(741)/2)**(-1/3) +
        2)**2]''')
    eq = S('''
        -x + (1/2 - sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3
        + x*(3*x**2 - 34) + 90)**2/4 - 39304/27) - 45)**(1/3) + 34/(3*(1/2 -
        sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3 + x*(3*x**2
        - 34) + 90)**2/4 - 39304/27) - 45)**(1/3))''')
    assert check(unrad(eq),
        (s*-(-s**6 + sqrt(3)*s**6*I - 153*2**Rational(2, 3)*3**Rational(1, 3)*s**4 +
        51*12**Rational(1, 3)*s**4 - 102*2**Rational(2, 3)*3**Rational(5, 6)*s**4*I - 1620*s**3 +
        1620*sqrt(3)*s**3*I + 13872*18**Rational(1, 3)*s**2 - 471648 +
        471648*sqrt(3)*I), [s, s**3 - 306*x - sqrt(3)*sqrt(31212*x**2 -
        165240*x + 61484) + 810]))

    assert solve(eq) == [] # not other code errors
    eq = root(x, 3) - root(y, 3) + root(x, 5)
    assert check(unrad(eq),
           (s**15 + 3*s**13 + 3*s**11 + s**9 - y, [s, s**15 - x]))
    eq = root(x, 3) + root(y, 3) + root(x*y, 4)
    assert check(unrad(eq),
                 (s*y*(-s**12 - 3*s**11*y - 3*s**10*y**2 - s**9*y**3 -
                       3*s**8*y**2 + 21*s**7*y**3 - 3*s**6*y**4 - 3*s**4*y**4 -
                       3*s**3*y**5 - y**6), [s, s**4 - x*y]))
    raises(NotImplementedError,
           lambda: unrad(root(x, 3) + root(y, 3) + root(x*y, 5)))

    # Test unrad with an Equality
    eq = Eq(-x**(S(1)/5) + x**(S(1)/3), -3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5))
    assert check(unrad(eq),
        (-s**5 + s**3 - 3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5), [s, s**15 - x]))

    # make sure buried radicals are exposed
    s = sqrt(x) - 1
    assert unrad(s**2 - s**3) == (x**3 - 6*x**2 + 9*x - 4, [])
    # make sure numerators which are already polynomial are rejected
    assert unrad((x/(x + 1) + 3)**(-2), x) is None

    # https://github.com/sympy/sympy/issues/23707
    eq = sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))
    assert solve(eq, y) == [x - 1]
    assert unrad(eq) is None


@slow
def test_unrad_slow():
    # this has roots with multiplicity > 1; there should be no
    # repeats in roots obtained, however
    eq = (sqrt(1 + sqrt(1 - 4*x**2)) - x*(1 + sqrt(1 + 2*sqrt(1 - 4*x**2))))
    assert solve(eq) == [S.Half]


@XFAIL
def test_unrad_fail():
    # this only works if we check real_root(eq.subs(x, Rational(1, 3)))
    # but checksol doesn't work like that
    assert solve(root(x**3 - 3*x**2, 3) + 1 - x) == [Rational(1, 3)]
    assert solve(root(x + 1, 3) + root(x**2 - 2, 5) + 1) == [
        -1, -1 + CRootOf(x**5 + x**4 + 5*x**3 + 8*x**2 + 10*x + 5, 0)**3]


def test_checksol():
    x, y, r, t = symbols('x, y, r, t')
    eq = r - x**2 - y**2
    dict_var_soln = {y: - sqrt(r) / sqrt(tan(t)**2 + 1),
        x: -sqrt(r)*tan(t)/sqrt(tan(t)**2 + 1)}
    assert checksol(eq, dict_var_soln) == True
    assert checksol(Eq(x, False), {x: False}) is True
    assert checksol(Ne(x, False), {x: False}) is False
    assert checksol(Eq(x < 1, True), {x: 0}) is True
    assert checksol(Eq(x < 1, True), {x: 1}) is False
    assert checksol(Eq(x < 1, False), {x: 1}) is True
    assert checksol(Eq(x < 1, False), {x: 0}) is False
    assert checksol(Eq(x + 1, x**2 + 1), {x: 1}) is True
    assert checksol([x - 1, x**2 - 1], x, 1) is True
    assert checksol([x - 1, x**2 - 2], x, 1) is False
    assert checksol(Poly(x**2 - 1), x, 1) is True
    assert checksol(0, {}) is True
    assert checksol([1e-10, x - 2], x, 2) is False
    assert checksol([0.5, 0, x], x, 0) is False
    assert checksol(y, x, 2) is False
    assert checksol(x+1e-10, x, 0, numerical=True) is True
    assert checksol(x+1e-10, x, 0, numerical=False) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2)}) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2) + I*pi}) is False
    assert checksol(1/x**5, x, 1000) is False
    raises(ValueError, lambda: checksol(x, 1))
    raises(ValueError, lambda: checksol([], x, 1))


def test__invert():
    assert _invert(x - 2) == (2, x)
    assert _invert(2) == (2, 0)
    assert _invert(exp(1/x) - 3, x) == (1/log(3), x)
    assert _invert(exp(1/x + a/x) - 3, x) == ((a + 1)/log(3), x)
    assert _invert(a, x) == (a, 0)


def test_issue_4463():
    assert solve(-a*x + 2*x*log(x), x) == [exp(a/2)]
    assert solve(x**x) == []
    assert solve(x**x - 2) == [exp(LambertW(log(2)))]
    assert solve(((x - 3)*(x - 2))**((x - 3)*(x - 4))) == [2]

@slow
def test_issue_5114_solvers():
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('a:r')

    # there is no 'a' in the equation set but this is how the
    # problem was originally posed
    syms = a, b, c, f, h, k, n
    eqs = [b + r/d - c/d,
    c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p]
    assert len(solve(eqs, syms, manual=True, check=False, simplify=False)) == 1


def test_issue_5849():
    #
    # XXX: This system does not have a solution for most values of the
    # parameters. Generally solve returns the empty set for systems that are
    # generically inconsistent.
    #
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    e = (
        I1 - I2 - I3,
        I3 - I4 - I5,
        I4 + I5 - I6,
        -I1 + I2 + I6,
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,
        -I4 + dQ4,
        -I2 + dQ2,
        2*I3 + 2*I5 + 3*I6 - Q2,
        I4 - 2*I5 + 2*Q4 + dI4
    )

    ans = [{
    I1: I2 + I3,
    dI1: -4*I2 - 8*I3 - 4*I5 - 6*I6 + 24,
    I4: I3 - I5,
    dQ4: I3 - I5,
    Q4: -I3/2 + 3*I5/2 - dI4/2,
    dQ2: I2,
    Q2: 2*I3 + 2*I5 + 3*I6}]

    v = I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4
    assert solve(e, *v, manual=True, check=False, dict=True) == ans
    assert solve(e, *v, manual=True, check=False) == [
        tuple([a.get(i, i) for i in v]) for a in ans]
    assert solve(e, *v, manual=True) == []
    assert solve(e, *v) == []

    # the matrix solver (tested below) doesn't like this because it produces
    # a zero row in the matrix. Is this related to issue 4551?
    assert [ei.subs(
        ans[0]) for ei in e] == [0, 0, I3 - I6, -I3 + I6, 0, 0, 0, 0, 0]


def test_issue_5849_matrix():
    '''Same as test_issue_5849 but solved with the matrix solver.

    A solution only exists if I3 == I6 which is not generically true,
    but `solve` does not return conditions under which the solution is
    valid, only a solution that is canonical and consistent with the input.
    '''
    # a simple example with the same issue
    # assert solve([x+y+z, x+y], [x, y]) == {x: y}
    # the longer example
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    e = (
        I1 - I2 - I3,
        I3 - I4 - I5,
        I4 + I5 - I6,
        -I1 + I2 + I6,
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,
        -I4 + dQ4,
        -I2 + dQ2,
        2*I3 + 2*I5 + 3*I6 - Q2,
        I4 - 2*I5 + 2*Q4 + dI4
    )
    assert solve(e, I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4) == []


def test_issue_21882():

    a, b, c, d, f, g, k = unknowns = symbols('a, b, c, d, f, g, k')

    equations = [
        -k*a + b + 5*f/6 + 2*c/9 + 5*d/6 + 4*a/3,
        -k*f + 4*f/3 + d/2,
        -k*d + f/6 + d,
        13*b/18 + 13*c/18 + 13*a/18,
        -k*c + b/2 + 20*c/9 + a,
        -k*b + b + c/18 + a/6,
        5*b/3 + c/3 + a,
        2*b/3 + 2*c + 4*a/3,
        -g,
    ]

    answer = [
        {a: 0, f: 0, b: 0, d: 0, c: 0, g: 0},
        {a: 0, f: -d, b: 0, k: S(5)/6, c: 0, g: 0},
        {a: -2*c, f: 0, b: c, d: 0, k: S(13)/18, g: 0}]
    # but not {a: 0, f: 0, b: 0, k: S(3)/2, c: 0, d: 0, g: 0}
    # since this is already covered by the first solution
    got = solve(equations, unknowns, dict=True)
    assert got == answer, (got,answer)


def test_issue_5901():
    f, g, h = map(Function, 'fgh')
    a = Symbol('a')
    D = Derivative(f(x), x)
    G = Derivative(g(a), a)
    assert solve(f(x) + f(x).diff(x), f(x)) == \
        [-D]
    assert solve(f(x) - 3, f(x)) == \
        [3]
    assert solve(f(x) - 3*f(x).diff(x), f(x)) == \
        [3*D]
    assert solve([f(x) - 3*f(x).diff(x)], f(x)) == \
        {f(x): 3*D}
    assert solve([f(x) - 3*f(x).diff(x), f(x)**2 - y + 4], f(x), y) == \
        [(3*D, 9*D**2 + 4)]
    assert solve(-f(a)**2*g(a)**2 + f(a)**2*h(a)**2 + g(a).diff(a),
                h(a), g(a), set=True) == \
        ([h(a), g(a)], {
        (-sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a)),
        (sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a))}), solve(-f(a)**2*g(a)**2 + f(a)**2*h(a)**2 + g(a).diff(a),
                h(a), g(a), set=True)
    args = [[f(x).diff(x, 2)*(f(x) + g(x)), 2 - g(x)**2], f(x), g(x)]
    assert solve(*args, set=True)[1] == \
        {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))}
    eqs = [f(x)**2 + g(x) - 2*f(x).diff(x), g(x)**2 - 4]
    assert solve(eqs, f(x), g(x), set=True) == \
        ([f(x), g(x)], {
        (-sqrt(2*D - 2), S(2)),
        (sqrt(2*D - 2), S(2)),
        (-sqrt(2*D + 2), -S(2)),
        (sqrt(2*D + 2), -S(2))})

    # the underlying problem was in solve_linear that was not masking off
    # anything but a Mul or Add; it now raises an error if it gets anything
    # but a symbol and solve handles the substitutions necessary so solve_linear
    # won't make this error
    raises(
        ValueError, lambda: solve_linear(f(x) + f(x).diff(x), symbols=[f(x)]))
    assert solve_linear(f(x) + f(x).diff(x), symbols=[x]) == \
        (f(x) + Derivative(f(x), x), 1)
    assert solve_linear(f(x) + Integral(x, (x, y)), symbols=[x]) == \
        (f(x) + Integral(x, (x, y)), 1)
    assert solve_linear(f(x) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x + f(x) + Integral(x, (x, y)), 1)
    assert solve_linear(f(y) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x, -f(y) - Integral(x, (x, y)))
    assert solve_linear(x - f(x)/a + (f(x) - 1)/a, symbols=[x]) == \
        (x, 1/a)
    assert solve_linear(x + Derivative(2*x, x)) == \
        (x, -2)
    assert solve_linear(x + Integral(x, y), symbols=[x]) == \
        (x, 0)
    assert solve_linear(x + Integral(x, y) - 2, symbols=[x]) == \
        (x, 2/(y + 1))

    assert set(solve(x + exp(x)**2, exp(x))) == \
        {-sqrt(-x), sqrt(-x)}
    assert solve(x + exp(x), x, implicit=True) == \
        [-exp(x)]
    assert solve(cos(x) - sin(x), x, implicit=True) == []
    assert solve(x - sin(x), x, implicit=True) == \
        [sin(x)]
    assert solve(x**2 + x - 3, x, implicit=True) == \
        [-x**2 + 3]
    assert solve(x**2 + x - 3, x**2, implicit=True) == \
        [-x + 3]


def test_issue_5912():
    assert set(solve(x**2 - x - 0.1, rational=True)) == \
        {S.Half + sqrt(35)/10, -sqrt(35)/10 + S.Half}
    ans = solve(x**2 - x - 0.1, rational=False)
    assert len(ans) == 2 and all(a.is_Number for a in ans)
    ans = solve(x**2 - x - 0.1)
    assert len(ans) == 2 and all(a.is_Number for a in ans)


def test_float_handling():
    def test(e1, e2):
        return len(e1.atoms(Float)) == len(e2.atoms(Float))
    assert solve(x - 0.5, rational=True)[0].is_Rational
    assert solve(x - 0.5, rational=False)[0].is_Float
    assert solve(x - S.Half, rational=False)[0].is_Rational
    assert solve(x - 0.5, rational=None)[0].is_Float
    assert solve(x - S.Half, rational=None)[0].is_Rational
    assert test(nfloat(1 + 2*x), 1.0 + 2.0*x)
    for contain in [list, tuple, set]:
        ans = nfloat(contain([1 + 2*x]))
        assert type(ans) is contain and test(list(ans)[0], 1.0 + 2.0*x)
    k, v = list(nfloat({2*x: [1 + 2*x]}).items())[0]
    assert test(k, 2*x) and test(v[0], 1.0 + 2.0*x)
    assert test(nfloat(cos(2*x)), cos(2.0*x))
    assert test(nfloat(3*x**2), 3.0*x**2)
    assert test(nfloat(3*x**2, exponent=True), 3.0*x**2.0)
    assert test(nfloat(exp(2*x)), exp(2.0*x))
    assert test(nfloat(x/3), x/3.0)
    assert test(nfloat(x**4 + 2*x + cos(Rational(1, 3)) + 1),
            x**4 + 2.0*x + 1.94495694631474)
    # don't call nfloat if there is no solution
    tot = 100 + c + z + t
    assert solve(((.7 + c)/tot - .6, (.2 + z)/tot - .3, t/tot - .1)) == []


def test_check_assumptions():
    x = symbols('x', positive=True)
    assert solve(x**2 - 1) == [1]


def test_issue_6056():
    assert solve(tanh(x + 3)*tanh(x - 3) - 1) == []
    assert solve(tanh(x - 1)*tanh(x + 1) + 1) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]
    assert solve((tanh(x + 3)*tanh(x - 3) + 1)**2) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]


def test_issue_5673():
    eq = -x + exp(exp(LambertW(log(x)))*LambertW(log(x)))
    assert checksol(eq, x, 2) is True
    assert checksol(eq, x, 2, numerical=False) is None


def test_exclude():
    R, C, Ri, Vout, V1, Vminus, Vplus, s = \
        symbols('R, C, Ri, Vout, V1, Vminus, Vplus, s')
    Rf = symbols('Rf', positive=True)  # to eliminate Rf = 0 soln
    eqs = [C*V1*s + Vplus*(-2*C*s - 1/R),
           Vminus*(-1/Ri - 1/Rf) + Vout/Rf,
           C*Vplus*s + V1*(-C*s - 1/R) + Vout/R,
           -Vminus + Vplus]
    assert solve(eqs, exclude=s*C*R) == [
        {
            Rf: Ri*(C*R*s + 1)**2/(C*R*s),
            Vminus: Vplus,
            V1: 2*Vplus + Vplus/(C*R*s),
            Vout: C*R*Vplus*s + 3*Vplus + Vplus/(C*R*s)},
        {
            Vplus: 0,
            Vminus: 0,
            V1: 0,
            Vout: 0},
    ]

    # TODO: Investigate why currently solution [0] is preferred over [1].
    assert solve(eqs, exclude=[Vplus, s, C]) in [[{
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 + sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus - sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }, {
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 - sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus + sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }], [{
        Vminus: Vplus,
        Vout: (V1**2 - V1*Vplus - Vplus**2)/(V1 - 2*Vplus),
        Rf: Ri*(V1 - Vplus)**2/(Vplus*(V1 - 2*Vplus)),
        R: Vplus/(C*s*(V1 - 2*Vplus)),
    }]]


def test_high_order_roots():
    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    assert set(solve(s)) == set(Poly(s*4, domain='ZZ').all_roots())


def test_minsolve_linear_system():
    pqt = {"quick": True, "particular": True}
    pqf = {"quick": False, "particular": True}
    assert solve([x + y - 5, 2*x - y - 1], **pqt) == {x: 2, y: 3}
    assert solve([x + y - 5, 2*x - y - 1], **pqf) == {x: 2, y: 3}
    def count(dic):
        return len([x for x in dic.values() if x == 0])
    assert count(solve([x + y + z, y + z + a + t], **pqt)) == 3
    assert count(solve([x + y + z, y + z + a + t], **pqf)) == 3
    assert count(solve([x + y + z, y + z + a], **pqt)) == 1
    assert count(solve([x + y + z, y + z + a], **pqf)) == 2
    # issue 22718
    A = Matrix([
        [ 1,  1,  1,  0,  1,  1,  0,  1,  0,  0,  1,  1,  1,  0],
        [ 1,  1,  0,  1,  1,  0,  1,  0,  1,  0, -1, -1,  0,  0],
        [-1, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  1,  0,  1],
        [ 1,  0,  1,  1,  0,  1,  1,  0,  0,  1, -1,  0, -1,  0],
        [-1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  1,  0,  1,  1],
        [-1,  0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1],
        [ 0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0, -1, -1,  0],
        [ 0, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  1,  1,  1],
        [ 0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1],
        [ 0,  0, -1, -1,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1],
        [ 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1, -1,  0, -1,  0,  0,  0,  0,  0,  0]])
    v = Matrix(symbols("v:14", integer=True))
    B = Matrix([[2], [-2], [0], [0], [0], [0], [0], [0], [0],
        [0], [0], [0]])
    eqs = A@v-B
    assert solve(eqs) == []
    assert solve(eqs, particular=True) == []  # assumption violated
    assert all(v for v in solve([x + y + z, y + z + a]).values())
    for _q in (True, False):
        assert not all(v for v in solve(
            [x + y + z, y + z + a], quick=_q,
            particular=True).values())
        # raise error if quick used w/o particular=True
        raises(ValueError, lambda: solve([x + 1], quick=_q))
        raises(ValueError, lambda: solve([x + 1], quick=_q, particular=False))
    # and give a good error message if someone tries to use
    # particular with a single equation
    raises(ValueError, lambda: solve(x + 1, particular=True))


def test_real_roots():
    # cf. issue 6650
    x = Symbol('x', real=True)
    assert len(solve(x**5 + x**3 + 1)) == 1


def test_issue_6528():
    eqs = [
        327600995*x**2 - 37869137*x + 1809975124*y**2 - 9998905626,
        895613949*x**2 - 273830224*x*y + 530506983*y**2 - 10000000000]
    # two expressions encountered are > 1400 ops long so if this hangs
    # it is likely because simplification is being done
    assert len(solve(eqs, y, x, check=False)) == 4


def test_overdetermined():
    x = symbols('x', real=True)
    eqs = [Abs(4*x - 7) - 5, Abs(3 - 8*x) - 1]
    assert solve(eqs, x) == [(S.Half,)]
    assert solve(eqs, x, manual=True) == [(S.Half,)]
    assert solve(eqs, x, manual=True, check=False) == [(S.Half,), (S(3),)]


def test_issue_6605():
    x = symbols('x')
    assert solve(4**(x/2) - 2**(x/3)) == [0, 3*I*pi/log(2)]
    # while the first one passed, this one failed
    x = symbols('x', real=True)
    assert solve(5**(x/2) - 2**(x/3)) == [0]
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    assert solve(5**(x/2) - 2**(3/x)) == [-b, b]


def test__ispow():
    assert _ispow(x**2)
    assert not _ispow(x)
    assert not _ispow(True)


def test_issue_6644():
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    sol = solve(eq, q, simplify=False, check=False)
    assert len(sol) == 5


def test_issue_6752():
    assert solve([a**2 + a, a - b], [a, b]) == [(-1, -1), (0, 0)]
    assert solve([a**2 + a*c, a - b], [a, b]) == [(0, 0), (-c, -c)]


def test_issue_6792():
    assert solve(x*(x - 1)**2*(x + 1)*(x**6 - x + 1)) == [
        -1, 0, 1, CRootOf(x**6 - x + 1, 0), CRootOf(x**6 - x + 1, 1),
         CRootOf(x**6 - x + 1, 2), CRootOf(x**6 - x + 1, 3),
         CRootOf(x**6 - x + 1, 4), CRootOf(x**6 - x + 1, 5)]


def test_issues_6819_6820_6821_6248_8692_25777_25779():
    # issue 6821
    x, y = symbols('x y', real=True)
    assert solve(abs(x + 3) - 2*abs(x - 3)) == [1, 9]
    assert solve([abs(x) - 2, arg(x) - pi], x) == [(-2,)]
    assert set(solve(abs(x - 7) - 8)) == {-S.One, S(15)}

    # issue 8692
    assert solve(Eq(Abs(x + 1) + Abs(x**2 - 7), 9), x) == [
        Rational(-1, 2) + sqrt(61)/2, -sqrt(69)/2 + S.Half]

    # issue 7145
    assert solve(2*abs(x) - abs(x - 1)) == [-1, Rational(1, 3)]

    # 25777
    assert solve(abs(x**3 + x + 2)/(x + 1)) == []

    # 25779
    assert solve(abs(x)) == [0]
    assert solve(Eq(abs(x**2 - 2*x), 4), x) == [
        1 - sqrt(5), 1 + sqrt(5)]
    nn = symbols('nn', nonnegative=True)
    assert solve(abs(sqrt(nn))) == [0]
    nz = symbols('nz', nonzero=True)
    assert solve(Eq(Abs(4 + 1 / (4*nz)), 0)) == [-Rational(1, 16)]

    x = symbols('x')
    assert solve([re(x) - 1, im(x) - 2], x) == [
        {x: 1 + 2*I, re(x): 1, im(x): 2}]

    # check for 'dict' handling of solution
    eq = sqrt(re(x)**2 + im(x)**2) - 3
    assert solve(eq) == solve(eq, x)

    i = symbols('i', imaginary=True)
    assert solve(abs(i) - 3) == [-3*I, 3*I]
    raises(NotImplementedError, lambda: solve(abs(x) - 3))

    w = symbols('w', integer=True)
    assert solve(2*x**w - 4*y**w, w) == solve((x/y)**w - 2, w)

    x, y = symbols('x y', real=True)
    assert solve(x + y*I + 3) == {y: 0, x: -3}
    # issue 2642
    assert solve(x*(1 + I)) == [0]

    x, y = symbols('x y', imaginary=True)
    assert solve(x + y*I + 3 + 2*I) == {x: -2*I, y: 3*I}

    x = symbols('x', real=True)
    assert solve(x + y + 3 + 2*I) == {x: -3, y: -2*I}

    # issue 6248
    f = Function('f')
    assert solve(f(x + 1) - f(2*x - 1)) == [2]
    assert solve(log(x + 1) - log(2*x - 1)) == [2]

    x = symbols('x')
    assert solve(2**x + 4**x) == [I*pi/log(2)]

def test_issue_17638():

    assert solve(((2-exp(2*x))*exp(x))/(exp(2*x)+2)**2 > 0, x) == (-oo < x) & (x < log(2)/2)
    assert solve(((2-exp(2*x)+2)*exp(x+2))/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < log(4)/2)
    assert solve((exp(x)+2+x**2)*exp(2*x+2)/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < oo)



def test_issue_14607():
    # issue 14607
    s, tau_c, tau_1, tau_2, phi, K = symbols(
        's, tau_c, tau_1, tau_2, phi, K')

    target = (s**2*tau_1*tau_2 + s*tau_1 + s*tau_2 + 1)/(K*s*(-phi + tau_c))

    K_C, tau_I, tau_D = symbols('K_C, tau_I, tau_D',
                                positive=True, nonzero=True)
    PID = K_C*(1 + 1/(tau_I*s) + tau_D*s)

    eq = (target - PID).together()
    eq *= denom(eq).simplify()
    eq = Poly(eq, s)
    c = eq.coeffs()

    vars = [K_C, tau_I, tau_D]
    s = solve(c, vars, dict=True)

    assert len(s) == 1

    knownsolution = {K_C: -(tau_1 + tau_2)/(K*(phi - tau_c)),
                     tau_I: tau_1 + tau_2,
                     tau_D: tau_1*tau_2/(tau_1 + tau_2)}

    for var in vars:
        assert s[0][var].simplify() == knownsolution[var].simplify()


def test_lambert_multivariate():
    from sympy.abc import x, y
    assert _filtered_gens(Poly(x + 1/x + exp(x) + y), x) == {x, exp(x)}
    assert _lambert(x, x) == []
    assert solve((x**2 - 2*x + 1).subs(x, log(x) + 3*x)) == [LambertW(3*S.Exp1)/3]
    assert solve((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1)) == \
          [LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3]
    assert solve((x**2 - 2*x - 2).subs(x, log(x) + 3*x)) == \
          [LambertW(3*exp(1 - sqrt(3)))/3, LambertW(3*exp(1 + sqrt(3)))/3]
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solve(eq) == [LambertW(3*exp(-LambertW(3)))]
    # coverage test
    raises(NotImplementedError, lambda: solve(x - sin(x)*log(y - x), x))
    ans = [3, -3*LambertW(-log(3)/3)/log(3)]  # 3 and 2.478...
    assert solve(x**3 - 3**x, x) == ans
    assert set(solve(3*log(x) - x*log(3))) == set(ans)
    assert solve(LambertW(2*x) - y, x) == [y*exp(y)/2]


@XFAIL
def test_other_lambert():
    assert solve(3*sin(x) - x*sin(3), x) == [3]
    assert set(solve(x**a - a**x), x) == {
        a, -a*LambertW(-log(a)/a)/log(a)}


@slow
def test_lambert_bivariate():
    # tests passing current implementation
    assert solve((x**2 + x)*exp(x**2 + x) - 1) == [
        Rational(-1, 2) + sqrt(1 + 4*LambertW(1))/2,
        Rational(-1, 2) - sqrt(1 + 4*LambertW(1))/2]
    assert solve((x**2 + x)*exp((x**2 + x)*2) - 1) == [
        Rational(-1, 2) + sqrt(1 + 2*LambertW(2))/2,
        Rational(-1, 2) - sqrt(1 + 2*LambertW(2))/2]
    assert solve(a/x + exp(x/2), x) == [2*LambertW(-a/2)]
    assert solve((a/x + exp(x/2)).diff(x), x) == \
            [4*LambertW(-sqrt(2)*sqrt(a)/4), 4*LambertW(sqrt(2)*sqrt(a)/4)]
    assert solve((1/x + exp(x/2)).diff(x), x) == \
        [4*LambertW(-sqrt(2)/4),
        4*LambertW(sqrt(2)/4),  # nsimplifies as 2*2**(141/299)*3**(206/299)*5**(205/299)*7**(37/299)/21
        4*LambertW(-sqrt(2)/4, -1)]
    assert solve(x*log(x) + 3*x + 1, x) == \
            [exp(-3 + LambertW(-exp(3)))]
    assert solve(-x**2 + 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]
    assert solve(x**2 - 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]
    ans = solve(3*x + 5 + 2**(-5*x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == \
        Rational(-5, 3) + LambertW(-10240*root(2, 3)*log(2)/3)/(5*log(2))
    assert solve(5*x - 1 + 3*exp(2 - 7*x), x) == \
        [Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7]
    assert solve((log(x) + x).subs(x, x**2 + 1)) == [
        -I*sqrt(-LambertW(1) + 1), sqrt(-1 + LambertW(1))]
    # check collection
    ax = a**(3*x + 5)
    ans = solve(3*log(ax) + b*log(ax) + ax, x)
    x0 = 1/log(a)
    x1 = sqrt(3)*I
    x2 = b + 3
    x3 = x2*LambertW(1/x2)/a**5
    x4 = x3**Rational(1, 3)/2
    assert ans == [
        x0*log(x4*(-x1 - 1)),
        x0*log(x4*(x1 - 1)),
        x0*log(x3)/3]
    x1 = LambertW(Rational(1, 3))
    x2 = a**(-5)
    x3 = -3**Rational(1, 3)
    x4 = 3**Rational(5, 6)*I
    x5 = x1**Rational(1, 3)*x2**Rational(1, 3)/2
    ans = solve(3*log(ax) + ax, x)
    assert ans == [
        x0*log(3*x1*x2)/3,
        x0*log(x5*(x3 - x4)),
        x0*log(x5*(x3 + x4))]
    # coverage
    p = symbols('p', positive=True)
    eq = 4*2**(2*p + 3) - 2*p - 3
    assert _solve_lambert(eq, p, _filtered_gens(Poly(eq), p)) == [
        Rational(-3, 2) - LambertW(-4*log(2))/(2*log(2))]
    assert set(solve(3**cos(x) - cos(x)**3)) == {
        acos(3), acos(-3*LambertW(-log(3)/3)/log(3))}
    # should give only one solution after using `uniq`
    assert solve(2*log(x) - 2*log(z) + log(z + log(x) + log(z)), x) == [
        exp(-z + LambertW(2*z**4*exp(2*z))/2)/z]
    # cases when p != S.One
    # issue 4271
    ans = solve((a/x + exp(x/2)).diff(x, 2), x)
    x0 = (-a)**Rational(1, 3)
    x1 = sqrt(3)*I
    x2 = x0/6
    assert ans == [
        6*LambertW(x0/3),
        6*LambertW(x2*(-x1 - 1)),
        6*LambertW(x2*(x1 - 1))]
    assert solve((1/x + exp(x/2)).diff(x, 2), x) == \
                [6*LambertW(Rational(-1, 3)), 6*LambertW(Rational(1, 6) - sqrt(3)*I/6), \
                6*LambertW(Rational(1, 6) + sqrt(3)*I/6), 6*LambertW(Rational(-1, 3), -1)]
    assert solve(x**2 - y**2/exp(x), x, y, dict=True) == \
                [{x: 2*LambertW(-y/2)}, {x: 2*LambertW(y/2)}]
    # this is slow but not exceedingly slow
    assert solve((x**3)**(x/2) + pi/2, x) == [
        exp(LambertW(-2*log(2)/3 + 2*log(pi)/3 + I*pi*Rational(2, 3)))]

    # issue 23253
    assert solve((1/log(sqrt(x) + 2)**2 - 1/x)) == [
        (LambertW(-exp(-2), -1) + 2)**2]
    assert solve((1/log(1/sqrt(x) + 2)**2 - x)) == [
        (LambertW(-exp(-2), -1) + 2)**-2]
    assert solve((1/log(x**2 + 2)**2 - x**-4)) == [
        -I*sqrt(2 - LambertW(exp(2))),
        -I*sqrt(LambertW(-exp(-2)) + 2),
        sqrt(-2 - LambertW(-exp(-2))),
        sqrt(-2 + LambertW(exp(2))),
        -sqrt(-2 - LambertW(-exp(-2), -1)),
        sqrt(-2 - LambertW(-exp(-2), -1))]


def test_rewrite_trig():
    assert solve(sin(x) + tan(x)) == [0, -pi, pi, 2*pi]
    assert solve(sin(x) + sec(x)) == [
        -2*atan(Rational(-1, 2) + sqrt(2)*sqrt(1 - sqrt(3)*I)/2 + sqrt(3)*I/2),
        2*atan(S.Half - sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half
        + sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half -
        sqrt(3)*I/2 + sqrt(2)*sqrt(1 - sqrt(3)*I)/2)]
    assert solve(sinh(x) + tanh(x)) == [0, I*pi]

    # issue 6157
    assert solve(2*sin(x) - cos(x), x) == [atan(S.Half)]


@XFAIL
def test_rewrite_trigh():
    # if this import passes then the test below should also pass
    from sympy.functions.elementary.hyperbolic import sech
    assert solve(sinh(x) + sech(x)) == [
        2*atanh(Rational(-1, 2) + sqrt(5)/2 - sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(Rational(-1, 2) + sqrt(5)/2 + sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(-sqrt(5)/2 - S.Half + sqrt(2 + 2*sqrt(5))/2),
        2*atanh(-sqrt(2 + 2*sqrt(5))/2 - sqrt(5)/2 - S.Half)]


def test_uselogcombine():
    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solve(eq, x, force=True) == [-sqrt(y*(y - exp(z))), sqrt(y*(y - exp(z)))]
    assert solve(log(x + 3) + log(1 + 3/x) - 3) in [
        [-3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2],
        [-3 + sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2,
        -3 - sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2],
        ]
    assert solve(log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)) == []


def test_atan2():
    assert solve(atan2(x, 2) - pi/3, x) == [2*sqrt(3)]


def test_errorinverses():
    assert solve(erf(x) - y, x) == [erfinv(y)]
    assert solve(erfinv(x) - y, x) == [erf(y)]
    assert solve(erfc(x) - y, x) == [erfcinv(y)]
    assert solve(erfcinv(x) - y, x) == [erfc(y)]


def test_issue_2725():
    R = Symbol('R')
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    sol = solve(eq, R, set=True)[1]
    assert sol == {(Rational(5, 3) + (Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3) + 40/(9*((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3))),), (Rational(5, 3) + 40/(9*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3)) + (Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3),)}


def test_issue_5114_6611():
    # See that it doesn't hang; this solves in about 2 seconds.
    # Also check that the solution is relatively small.
    # Note: the system in issue 6611 solves in about 5 seconds and has
    # an op-count of 138336 (with simplify=False).
    b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('b:r')
    eqs = Matrix([
        [b - c/d + r/d], [c*(1/g + 1/e + 1/d) - f/g - r/d],
        [-c/g + f*(1/j + 1/i + 1/g) - h/i], [-f/i + h*(1/m + 1/l + 1/i) - k/m],
        [-h/m + k*(1/p + 1/o + 1/m) - n/p], [-k/p + n*(1/q + 1/p)]])
    v = Matrix([f, h, k, n, b, c])
    ans = solve(list(eqs), list(v), simplify=False)
    # If time is taken to simplify then then 2617 below becomes
    # 1168 and the time is about 50 seconds instead of 2.
    assert sum(s.count_ops() for s in ans.values()) <= 3270


def test_det_quick():
    m = Matrix(3, 3, symbols('a:9'))
    assert m.det() == det_quick(m)  # calls det_perm
    m[0, 0] = 1
    assert m.det() == det_quick(m)  # calls det_minor
    m = Matrix(3, 3, list(range(9)))
    assert m.det() == det_quick(m)  # defaults to .det()
    # make sure they work with Sparse
    s = SparseMatrix(2, 2, (1, 2, 1, 4))
    assert det_perm(s) == det_minor(s) == s.det()


def test_real_imag_splitting():
    a, b = symbols('a b', real=True)
    assert solve(sqrt(a**2 + b**2) - 3, a) == \
        [-sqrt(-b**2 + 9), sqrt(-b**2 + 9)]
    a, b = symbols('a b', imaginary=True)
    assert solve(sqrt(a**2 + b**2) - 3, a) == []


def test_issue_7110():
    y = -2*x**3 + 4*x**2 - 2*x + 5
    assert any(ask(Q.real(i)) for i in solve(y))


def test_units():
    assert solve(1/x - 1/(2*cm)) == [2*cm]


def test_issue_7547():
    A, B, V = symbols('A,B,V')
    eq1 = Eq(630.26*(V - 39.0)*V*(V + 39) - A + B, 0)
    eq2 = Eq(B, 1.36*10**8*(V - 39))
    eq3 = Eq(A, 5.75*10**5*V*(V + 39.0))
    sol = Matrix(nsolve(Tuple(eq1, eq2, eq3), [A, B, V], (0, 0, 0)))
    assert str(sol) == str(Matrix(
        [['4442890172.68209'],
         ['4289299466.1432'],
         ['70.5389666628177']]))


def test_issue_7895():
    r = symbols('r', real=True)
    assert solve(sqrt(r) - 2) == [4]


def test_issue_2777():
    # the equations represent two circles
    x, y = symbols('x y', real=True)
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = [(a, -b), (a, b)]
    assert solve((e1, e2), (x, y)) == ans
    assert solve((e1, e2/(x - a)), (x, y)) == []
    # make the 2nd circle's radius be -3
    e2 += 6
    assert solve((e1, e2), (x, y)) == []
    assert solve((e1, e2), (x, y), check=False) == ans


def test_issue_7322():
    number = 5.62527e-35
    assert solve(x - number, x)[0] == number


def test_nsolve():
    raises(ValueError, lambda: nsolve(x, (-1, 1), method='bisect'))
    raises(TypeError, lambda: nsolve((x - y + 3,x + y,z - y),(x,y,z),(-50,50)))
    raises(TypeError, lambda: nsolve((x + y, x - y), (0, 1)))
    raises(TypeError, lambda: nsolve(x < 0.5, x, 1))


@slow
def test_high_order_multivariate():
    assert len(solve(a*x**3 - x + 1, x)) == 3
    assert len(solve(a*x**4 - x + 1, x)) == 4
    assert solve(a*x**5 - x + 1, x) == []  # incomplete solution allowed
    raises(NotImplementedError, lambda:
        solve(a*x**5 - x + 1, x, incomplete=False))

    # result checking must always consider the denominator and CRootOf
    # must be checked, too
    d = x**5 - x + 1
    assert solve(d*(1 + 1/d)) == [CRootOf(d + 1, i) for i in range(5)]
    d = x - 1
    assert solve(d*(2 + 1/d)) == [S.Half]


def test_base_0_exp_0():
    assert solve(0**x - 1) == [0]
    assert solve(0**(x - 2) - 1) == [2]
    assert solve(S('x*(1/x**0 - x)', evaluate=False)) == \
        [0, 1]


def test__simple_dens():
    assert _simple_dens(1/x**0, [x]) == set()
    assert _simple_dens(1/x**y, [x]) == {x**y}
    assert _simple_dens(1/root(x, 3), [x]) == {x}


def test_issue_8755():
    # This tests two things: that if full unrad is attempted and fails
    # the solution should still be found; also it tests the use of
    # keyword `composite`.
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3


@slow
def test_issue_8828():
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = x, y, z

    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x - x2)**2 + (y - y2)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = f1,f2,f3

    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = g1,g2,g3

    A = solve(F, v)
    B = solve(G, v)
    C = solve(G, v, manual=True)

    p, q, r = [{tuple(i.evalf(2) for i in j) for j in R} for R in [A, B, C]]
    assert p == q == r


def test_issue_2840_8155():
    # with parameter-free solutions (i.e. no `n`), we want to avoid
    # excessive periodic solutions
    assert solve(sin(3*x) + sin(6*x)) == [0, -2*pi/9, 2*pi/9]
    assert solve(sin(300*x) + sin(600*x)) == [0, -pi/450, pi/450]
    assert solve(2*sin(x) - 2*sin(2*x)) == [0, -pi/3, pi/3]


def test_issue_9567():
    assert solve(1 + 1/(x - 1)) == [0]


def test_issue_11538():
    assert solve(x + E) == [-E]
    assert solve(x**2 + E) == [-I*sqrt(E), I*sqrt(E)]
    assert solve(x**3 + 2*E) == [
        -cbrt(2 * E),
        cbrt(2)*cbrt(E)/2 - cbrt(2)*sqrt(3)*I*cbrt(E)/2,
        cbrt(2)*cbrt(E)/2 + cbrt(2)*sqrt(3)*I*cbrt(E)/2]
    assert solve([x + 4, y + E], x, y) == {x: -4, y: -E}
    assert solve([x**2 + 4, y + E], x, y) == [
        (-2*I, -E), (2*I, -E)]

    e1 = x - y**3 + 4
    e2 = x + y + 4 + 4 * E
    assert len(solve([e1, e2], x, y)) == 3


@slow
def test_issue_12114():
    a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g')
    terms = [1 + a*b + d*e, 1 + a*c + d*f, 1 + b*c + e*f,
             g - a**2 - d**2, g - b**2 - e**2, g - c**2 - f**2]
    sol = solve(terms, [a, b, c, d, e, f, g], dict=True)
    s = sqrt(-f**2 - 1)
    s2 = sqrt(2 - f**2)
    s3 = sqrt(6 - 3*f**2)
    s4 = sqrt(3)*f
    s5 = sqrt(3)*s2
    assert sol == [
        {a: -s, b: -s, c: -s, d: f, e: f, g: -1},
        {a: s, b: s, c: s, d: f, e: f, g: -1},
        {a: -s4/2 - s2/2, b: s4/2 - s2/2, c: s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2},
        {a: -s4/2 + s2/2, b: s4/2 + s2/2, c: -s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 - s2/2, b: -s4/2 - s2/2, c: s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 + s2/2, b: -s4/2 + s2/2, c: -s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2}]


def test_inf():
    assert solve(1 - oo*x) == []
    assert solve(oo*x, x) == []
    assert solve(oo*x - oo, x) == []


def test_issue_12448():
    f = Function('f')
    fun = [f(i) for i in range(15)]
    sym = symbols('x:15')
    reps = dict(zip(fun, sym))

    (x, y, z), c = sym[:3], sym[3:]
    ssym = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))

    (x, y, z), c = fun[:3], fun[3:]
    sfun = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))

    assert sfun[fun[0]].xreplace(reps).count_ops() == \
        ssym[sym[0]].count_ops()


def test_denoms():
    assert denoms(x/2 + 1/y) == {2, y}
    assert denoms(x/2 + 1/y, y) == {y}
    assert denoms(x/2 + 1/y, [y]) == {y}
    assert denoms(1/x + 1/y + 1/z, [x, y]) == {x, y}
    assert denoms(1/x + 1/y + 1/z, x, y) == {x, y}
    assert denoms(1/x + 1/y + 1/z, {x, y}) == {x, y}


def test_issue_12476():
    x0, x1, x2, x3, x4, x5 = symbols('x0 x1 x2 x3 x4 x5')
    eqns = [x0**2 - x0, x0*x1 - x1, x0*x2 - x2, x0*x3 - x3, x0*x4 - x4, x0*x5 - x5,
            x0*x1 - x1, -x0/3 + x1**2 - 2*x2/3, x1*x2 - x1/3 - x2/3 - x3/3,
            x1*x3 - x2/3 - x3/3 - x4/3, x1*x4 - 2*x3/3 - x5/3, x1*x5 - x4, x0*x2 - x2,
            x1*x2 - x1/3 - x2/3 - x3/3, -x0/6 - x1/6 + x2**2 - x2/6 - x3/3 - x4/6,
            -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6, x2*x4 - x2/3 - x3/3 - x4/3,
            x2*x5 - x3, x0*x3 - x3, x1*x3 - x2/3 - x3/3 - x4/3,
            -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6,
            -x0/6 - x1/6 - x2/6 + x3**2 - x3/3 - x4/6, -x1/3 - x2/3 + x3*x4 - x3/3,
            -x2 + x3*x5, x0*x4 - x4, x1*x4 - 2*x3/3 - x5/3, x2*x4 - x2/3 - x3/3 - x4/3,
            -x1/3 - x2/3 + x3*x4 - x3/3, -x0/3 - 2*x2/3 + x4**2, -x1 + x4*x5, x0*x5 - x5,
            x1*x5 - x4, x2*x5 - x3, -x2 + x3*x5, -x1 + x4*x5, -x0 + x5**2, x0 - 1]
    sols = [{x0: 1, x3: Rational(1, 6), x2: Rational(1, 6), x4: Rational(-2, 3), x1: Rational(-2, 3), x5: 1},
            {x0: 1, x3: S.Half, x2: Rational(-1, 2), x4: 0, x1: 0, x5: -1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(-1, 3), x4: Rational(1, 3), x1: Rational(1, 3), x5: 1},
            {x0: 1, x3: 1, x2: 1, x4: 1, x1: 1, x5: 1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: sqrt(5)/3, x1: -sqrt(5)/3, x5: -1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: -sqrt(5)/3, x1: sqrt(5)/3, x5: -1}]

    assert solve(eqns) == sols


def test_issue_13849():
    t = symbols('t')
    assert solve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) == []


def test_issue_14860():
    from sympy.physics.units import newton, kilo
    assert solve(8*kilo*newton + x + y, x) == [-8000*newton - y]


def test_issue_14721():
    k, h, a, b = symbols(':4')
    assert solve([
        -1 + (-k + 1)**2/b**2 + (-h - 1)**2/a**2,
        -1 + (-k + 1)**2/b**2 + (-h + 1)**2/a**2,
        h, k + 2], h, k, a, b) == [
        (0, -2, -b*sqrt(1/(b**2 - 9)), b),
        (0, -2, b*sqrt(1/(b**2 - 9)), b)]
    assert solve([
        h, h/a + 1/b**2 - 2, -h/2 + 1/b**2 - 2], a, h, b) == [
        (a, 0, -sqrt(2)/2), (a, 0, sqrt(2)/2)]
    assert solve((a + b**2 - 1, a + b**2 - 2)) == []


def test_issue_14779():
    x = symbols('x', real=True)
    assert solve(sqrt(x**4 - 130*x**2 + 1089) + sqrt(x**4 - 130*x**2
                 + 3969) - 96*Abs(x)/x,x) == [sqrt(130)]


def test_issue_15307():
    assert solve((y - 2, Mul(x + 3,x - 2, evaluate=False))) == \
        [{x: -3, y: 2}, {x: 2, y: 2}]
    assert solve((y - 2, Mul(3, x - 2, evaluate=False))) == \
        {x: 2, y: 2}
    assert solve((y - 2, Add(x + 4, x - 2, evaluate=False))) == \
        {x: -1, y: 2}
    eq1 = Eq(12513*x + 2*y - 219093, -5726*x - y)
    eq2 = Eq(-2*x + 8, 2*x - 40)
    assert solve([eq1, eq2]) == {x:12, y:75}


def test_issue_15415():
    assert solve(x - 3, x) == [3]
    assert solve([x - 3], x) == {x:3}
    assert solve(Eq(y + 3*x**2/2, y + 3*x), y) == []
    assert solve([Eq(y + 3*x**2/2, y + 3*x)], y) == []
    assert solve([Eq(y + 3*x**2/2, y + 3*x), Eq(x, 1)], y) == []


@slow
def test_issue_15731():
    # f(x)**g(x)=c
    assert solve(Eq((x**2 - 7*x + 11)**(x**2 - 13*x + 42), 1)) == [2, 3, 4, 5, 6, 7]
    assert solve((x)**(x + 4) - 4) == [-2]
    assert solve((-x)**(-x + 4) - 4) == [2]
    assert solve((x**2 - 6)**(x**2 - 2) - 4) == [-2, 2]
    assert solve((x**2 - 2*x - 1)**(x**2 - 3) - 1/(1 - 2*sqrt(2))) == [sqrt(2)]
    assert solve(x**(x + S.Half) - 4*sqrt(2)) == [S(2)]
    assert solve((x**2 + 1)**x - 25) == [2]
    assert solve(x**(2/x) - 2) == [2, 4]
    assert solve((x/2)**(2/x) - sqrt(2)) == [4, 8]
    assert solve(x**(x + S.Half) - Rational(9, 4)) == [Rational(3, 2)]
    # a**g(x)=c
    assert solve((-sqrt(sqrt(2)))**x - 2) == [4, log(2)/(log(2**Rational(1, 4)) + I*pi)]
    assert solve((sqrt(2))**x - sqrt(sqrt(2))) == [S.Half]
    assert solve((-sqrt(2))**x + 2*(sqrt(2))) == [3,
            (3*log(2)**2 + 4*pi**2 - 4*I*pi*log(2))/(log(2)**2 + 4*pi**2)]
    assert solve((sqrt(2))**x - 2*(sqrt(2))) == [3]
    assert solve(I**x + 1) == [2]
    assert solve((1 + I)**x - 2*I) == [2]
    assert solve((sqrt(2) + sqrt(3))**x - (2*sqrt(6) + 5)**Rational(1, 3)) == [Rational(2, 3)]
    # bases of both sides are equal
    b = Symbol('b')
    assert solve(b**x - b**2, x) == [2]
    assert solve(b**x - 1/b, x) == [-1]
    assert solve(b**x - b, x) == [1]
    b = Symbol('b', positive=True)
    assert solve(b**x - b**2, x) == [2]
    assert solve(b**x - 1/b, x) == [-1]


def test_issue_10933():
    assert solve(x**4 + y*(x + 0.1), x)  # doesn't fail
    assert solve(I*x**4 + x**3 + x**2 + 1.)  # doesn't fail


def test_Abs_handling():
    x = symbols('x', real=True)
    assert solve(abs(x/y), x) == [0]


def test_issue_7982():
    x = Symbol('x')
    # Test that no exception happens
    assert solve([2*x**2 + 5*x + 20 <= 0, x >= 1.5], x) is S.false
    # From #8040
    assert solve([x**3 - 8.08*x**2 - 56.48*x/5 - 106 >= 0, x - 1 <= 0], [x]) is S.false


def test_issue_14645():
    x, y = symbols('x y')
    assert solve([x*y - x - y, x*y - x - y], [x, y]) == [(y/(y - 1), y)]


def test_issue_12024():
    x, y = symbols('x y')
    assert solve(Piecewise((0.0, x < 0.1), (x, x >= 0.1)) - y) == \
        [{y: Piecewise((0.0, x < 0.1), (x, True))}]


def test_issue_17452():
    assert solve((7**x)**x + pi, x) == [-sqrt(log(pi) + I*pi)/sqrt(log(7)),
                                        sqrt(log(pi) + I*pi)/sqrt(log(7))]
    assert solve(x**(x/11) + pi/11, x) == [exp(LambertW(-11*log(11) + 11*log(pi) + 11*I*pi))]


def test_issue_17799():
    assert solve(-erf(x**(S(1)/3))**pi + I, x) == []


def test_issue_17650():
    x = Symbol('x', real=True)
    assert solve(abs(abs(x**2 - 1) - x) - x) == [1, -1 + sqrt(2), 1 + sqrt(2)]


def test_issue_17882():
    eq = -8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3))
    assert unrad(eq) is None


def test_issue_17949():
    assert solve(exp(+x+x**2), x) == []
    assert solve(exp(-x+x**2), x) == []
    assert solve(exp(+x-x**2), x) == []
    assert solve(exp(-x-x**2), x) == []


def test_issue_10993():
    assert solve(Eq(binomial(x, 2), 3)) == [-2, 3]
    assert solve(Eq(pow(x, 2) + binomial(x, 3), x)) == [-4, 0, 1]
    assert solve(Eq(binomial(x, 2), 0)) == [0, 1]
    assert solve(a+binomial(x, 3), a) == [-binomial(x, 3)]
    assert solve(x-binomial(a, 3) + binomial(y, 2) + sin(a), x) == [-sin(a) + binomial(a, 3) - binomial(y, 2)]
    assert solve((x+1)-binomial(x+1, 3), x) == [-2, -1, 3]


def test_issue_11553():
    eq1 = x + y + 1
    eq2 = x + GoldenRatio
    assert solve([eq1, eq2], x, y) == {x: -GoldenRatio, y: -1 + GoldenRatio}
    eq3 = x + 2 + TribonacciConstant
    assert solve([eq1, eq3], x, y) == {x: -2 - TribonacciConstant, y: 1 + TribonacciConstant}


def test_issue_19113_19102():
    t = S(1)/3
    solve(cos(x)**5-sin(x)**5)
    assert solve(4*cos(x)**3 - 2*sin(x)**3) == [
        atan(2**(t)), -atan(2**(t)*(1 - sqrt(3)*I)/2),
        -atan(2**(t)*(1 + sqrt(3)*I)/2)]
    h = S.Half
    assert solve(cos(x)**2 + sin(x)) == [
        2*atan(-h + sqrt(5)/2 + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(h + sqrt(5)/2 + sqrt(2)*sqrt(1 + sqrt(5))/2),
        -2*atan(-sqrt(5)/2 + h + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(-sqrt(2)*sqrt(1 + sqrt(5))/2 + h + sqrt(5)/2)]
    assert solve(3*cos(x) - sin(x)) == [atan(3)]


def test_issue_19509():
    a = S(3)/4
    b = S(5)/8
    c = sqrt(5)/8
    d = sqrt(5)/4
    assert solve(1/(x -1)**5 - 1) == [2,
        -d + a - sqrt(-b + c),
        -d + a + sqrt(-b + c),
        d + a - sqrt(-b - c),
        d + a + sqrt(-b - c)]

def test_issue_20747():
    THT, HT, DBH, dib, c0, c1, c2, c3, c4  = symbols('THT HT DBH dib c0 c1 c2 c3 c4')
    f = DBH*c3 + THT*c4 + c2
    rhs = 1 - ((HT - 1)/(THT - 1))**c1*(1 - exp(c0/f))
    eq = dib - DBH*(c0 - f*log(rhs))
    term = ((1 - exp((DBH*c0 - dib)/(DBH*(DBH*c3 + THT*c4 + c2))))
            / (1 - exp(c0/(DBH*c3 + THT*c4 + c2))))
    sol = [THT*term**(1/c1) - term**(1/c1) + 1]
    assert solve(eq, HT) == sol


def test_issue_20902():
    f = (t / ((1 + t) ** 2))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)
    assert solve(f.subs({t: 3 * x + 3}).diff(x) > 0, x) == (S(-4)/3 < x) & (x < S(-2)/3)
    assert solve(f.subs({t: 3 * x + 4}).diff(x) > 0, x) == (S(-5)/3 < x) & (x < S(-1))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)


def test_issue_21034():
    a = symbols('a', real=True)
    system = [x - cosh(cos(4)), y - sinh(cos(a)), z - tanh(x)]
    # constants inside hyperbolic functions should not be rewritten in terms of exp
    assert solve(system, x, y, z) == [(cosh(cos(4)), sinh(cos(a)), tanh(cosh(cos(4))))]
    # but if the variable of interest is present in a hyperbolic function,
    # then it should be rewritten in terms of exp and solved further
    newsystem = [(exp(x) - exp(-x)) - tanh(x)*(exp(x) + exp(-x)) + x - 5]
    assert solve(newsystem, x) == {x: 5}


def test_issue_4886():
    z = a*sqrt(R**2*a**2 + R**2*b**2 - c**2)/(a**2 + b**2)
    t = b*c/(a**2 + b**2)
    sol = [((b*(t - z) - c)/(-a), t - z), ((b*(t + z) - c)/(-a), t + z)]
    assert solve([x**2 + y**2 - R**2, a*x + b*y - c], x, y) == sol


def test_issue_6819():
    a, b, c, d = symbols('a b c d', positive=True)
    assert solve(a*b**x - c*d**x, x) == [log(c/a)/log(b/d)]


def test_issue_17454():
    x = Symbol('x')
    assert solve((1 - x - I)**4, x) == [1 - I]


def test_issue_21852():
    solution = [21 - 21*sqrt(2)/2]
    assert solve(2*x + sqrt(2*x**2) - 21) == solution


def test_issue_21942():
    eq = -d + (a*c**(1 - e) + b**(1 - e)*(1 - a))**(1/(1 - e))
    sol = solve(eq, c, simplify=False, check=False)
    assert sol == [((a*b**(1 - e) - b**(1 - e) +
        d**(1 - e))/a)**(1/(1 - e))]


def test_solver_flags():
    root = solve(x**5 + x**2 - x - 1, cubics=False)
    rad = solve(x**5 + x**2 - x - 1, cubics=True)
    assert root != rad


def test_issue_22768():
    eq = 2*x**3 - 16*(y - 1)**6*z**3
    assert solve(eq.expand(), x, simplify=False
        ) == [2*z*(y - 1)**2, z*(-1 + sqrt(3)*I)*(y - 1)**2,
        -z*(1 + sqrt(3)*I)*(y - 1)**2]


def test_issue_22717():
    assert solve((-y**2 + log(y**2/x) + 2, -2*x*y + 2*x/y)) == [
        {y: -1, x: E}, {y: 1, x: E}]


def test_issue_25176():
    eq = (x - 5)**-8 - 3
    sol = solve(eq)
    assert not any(eq.subs(x, i) for i in sol)


def test_issue_10169():
    eq = S(-8*a - x**5*(a + b + c + e) - x**4*(4*a - 2**Rational(3,4)*c + 4*c +
        d + 2**Rational(3,4)*e + 4*e + k) - x**3*(-4*2**Rational(3,4)*c + sqrt(2)*c -
        2**Rational(3,4)*d + 4*d + sqrt(2)*e + 4*2**Rational(3,4)*e + 2**Rational(3,4)*k + 4*k) -
        x**2*(4*sqrt(2)*c - 4*2**Rational(3,4)*d + sqrt(2)*d + 4*sqrt(2)*e +
        sqrt(2)*k + 4*2**Rational(3,4)*k) - x*(2*a + 2*b + 4*sqrt(2)*d +
        4*sqrt(2)*k) + 5)
    assert solve_undetermined_coeffs(eq, [a, b, c, d, e, k], x) == {
        a: Rational(5,8),
        b: Rational(-5,1032),
        c: Rational(-40,129) - 5*2**Rational(3,4)/129 + 5*2**Rational(1,4)/1032,
        d: -20*2**Rational(3,4)/129 - 10*sqrt(2)/129 - 5*2**Rational(1,4)/258,
        e: Rational(-40,129) - 5*2**Rational(1,4)/1032 + 5*2**Rational(3,4)/129,
        k: -10*sqrt(2)/129 + 5*2**Rational(1,4)/258 + 20*2**Rational(3,4)/129
    }


def test_solve_undetermined_coeffs_issue_23927():
    A, B, r, phi = symbols('A, B, r, phi')
    e = Eq(A*sin(t) + B*cos(t), r*sin(t - phi))
    eq = (e.lhs - e.rhs).expand(trig=True)
    soln = solve_undetermined_coeffs(eq, (r, phi), t)
    assert soln == [{
        phi: 2*atan((A - sqrt(A**2 + B**2))/B),
        r: (-A**2 + A*sqrt(A**2 + B**2) - B**2)/(A - sqrt(A**2 + B**2))
        }, {
        phi: 2*atan((A + sqrt(A**2 + B**2))/B),
        r: (A**2 + A*sqrt(A**2 + B**2) + B**2)/(A + sqrt(A**2 + B**2))/-1
        }]

def test_issue_24368():
    # Ideally these would produce a solution, but for now just check that they
    # don't fail with a RuntimeError
    raises(NotImplementedError, lambda: solve(Mod(x**2, 49), x))
    s2 = Symbol('s2', integer=True, positive=True)
    f = floor(s2/2 - S(1)/2)
    raises(NotImplementedError, lambda: solve((Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1))*f + Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1), s2))


def test_solve_Piecewise():
    assert [S(10)/3] == solve(3*Piecewise(
        (S.NaN, x <= 0),
        (20*x - 3*(x - 6)**2/2 - 176, (x >= 0) & (x >= 2) & (x>= 4) & (x >= 6) & (x < 10)),
        (100 - 26*x, (x >= 0) & (x >= 2) & (x >= 4) & (x < 10)),
        (16*x - 3*(x - 6)**2/2 - 176, (x >= 2) & (x >= 4) & (x >= 6) & (x < 10)),
        (100 - 30*x, (x >= 2) & (x >= 4) & (x < 10)),
        (30*x - 3*(x - 6)**2/2 - 196, (x>= 0) & (x >= 4) & (x >= 6) & (x < 10)),
        (80 - 16*x, (x >= 0) & (x >= 4) & (x < 10)),
        (26*x - 3*(x - 6)**2/2 - 196, (x >= 4) & (x >= 6) & (x < 10)),
        (80 - 20*x, (x >= 4) & (x < 10)),
        (40*x - 3*(x - 6)**2/2 - 256, (x >= 0) & (x >= 2) & (x >= 6) & (x < 10)),
        (20 - 6*x, (x >= 0) & (x >= 2) & (x < 10)),
        (36*x - 3*(x - 6)**2/2 - 256, (x >= 2) & (x >= 6) & (x < 10)),
        (20 - 10*x, (x >= 2) & (x < 10)),
        (50*x - 3*(x - 6)**2/2 - 276, (x >= 0) & (x >= 6) & (x < 10)),
        (4*x, (x >= 0) & (x < 10)),
        (46*x - 3*(x - 6)**2/2 - 276, (x >= 6) & (x < 10)),
        (0, x < 10),  # this will simplify away
        (S.NaN,True)))
