from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.ntheory.factor_ import factorint
from sympy.simplify.powsimp import powsimp
from sympy.core.function import _mexpand
from sympy.core.sorting import default_sort_key, ordered
from sympy.functions.elementary.trigonometric import sin
from sympy.solvers.diophantine import diophantine
from sympy.solvers.diophantine.diophantine import (diop_DN,
    diop_solve, diop_ternary_quadratic_normal,
    diop_general_pythagorean, diop_ternary_quadratic, diop_linear,
    diop_quadratic, diop_general_sum_of_squares, diop_general_sum_of_even_powers,
    descent, diop_bf_DN, divisible, equivalent, find_DN, ldescent, length,
    reconstruct, partition, power_representation,
    prime_as_sum_of_two_squares, square_factor, sum_of_four_squares,
    sum_of_three_squares, transformation_to_DN, transformation_to_normal,
    classify_diop, base_solution_linear, cornacchia, sqf_normal, gaussian_reduce, holzer,
    check_param, parametrize_ternary_quadratic, sum_of_powers, sum_of_squares,
    _diop_ternary_quadratic_normal, _nint_or_floor,
    _odd, _even, _remove_gcd, _can_do_sum_of_squares, DiophantineSolutionSet, GeneralPythagorean,
    BinaryQuadratic)

from sympy.testing.pytest import slow, raises, XFAIL
from sympy.utilities.iterables import (
        signed_permutations)

a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z = symbols(
    "a, b, c, d, p, q, x, y, z, w, t, u, v, X, Y, Z", integer=True)
t_0, t_1, t_2, t_3, t_4, t_5, t_6 = symbols("t_:7", integer=True)
m1, m2, m3 = symbols('m1:4', integer=True)
n1 = symbols('n1', integer=True)


def diop_simplify(eq):
    return _mexpand(powsimp(_mexpand(eq)))


def test_input_format():
    raises(TypeError, lambda: diophantine(sin(x)))
    raises(TypeError, lambda: diophantine(x/pi - 3))


def test_nosols():
    # diophantine should sympify eq so that these are equivalent
    assert diophantine(3) == set()
    assert diophantine(S(3)) == set()


def test_univariate():
    assert diop_solve((x - 1)*(x - 2)**2) == {(1,), (2,)}
    assert diop_solve((x - 1)*(x - 2)) == {(1,), (2,)}


def test_classify_diop():
    raises(TypeError, lambda: classify_diop(x**2/3 - 1))
    raises(ValueError, lambda: classify_diop(1))
    raises(NotImplementedError, lambda: classify_diop(w*x*y*z - 1))
    raises(NotImplementedError, lambda: classify_diop(x**3 + y**3 + z**4 - 90))
    assert classify_diop(14*x**2 + 15*x - 42) == (
        [x], {1: -42, x: 15, x**2: 14}, 'univariate')
    assert classify_diop(x*y + z) == (
        [x, y, z], {x*y: 1, z: 1}, 'inhomogeneous_ternary_quadratic')
    assert classify_diop(x*y + z + w + x**2) == (
        [w, x, y, z], {x*y: 1, w: 1, x**2: 1, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + x*z + x**2 + 1) == (
        [x, y, z], {x*y: 1, x*z: 1, x**2: 1, 1: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + z + w + 42) == (
        [w, x, y, z], {x*y: 1, w: 1, 1: 42, z: 1}, 'inhomogeneous_general_quadratic')
    assert classify_diop(x*y + z*w) == (
        [w, x, y, z], {x*y: 1, w*z: 1}, 'homogeneous_general_quadratic')
    assert classify_diop(x*y**2 + 1) == (
        [x, y], {x*y**2: 1, 1: 1}, 'cubic_thue')
    assert classify_diop(x**4 + y**4 + z**4 - (1 + 16 + 81)) == (
        [x, y, z], {1: -98, x**4: 1, z**4: 1, y**4: 1}, 'general_sum_of_even_powers')
    assert classify_diop(x**2 + y**2 + z**2) == (
        [x, y, z], {x**2: 1, y**2: 1, z**2: 1}, 'homogeneous_ternary_quadratic_normal')


def test_linear():
    assert diop_solve(x) == (0,)
    assert diop_solve(1*x) == (0,)
    assert diop_solve(3*x) == (0,)
    assert diop_solve(x + 1) == (-1,)
    assert diop_solve(2*x + 1) == (None,)
    assert diop_solve(2*x + 4) == (-2,)
    assert diop_solve(y + x) == (t_0, -t_0)
    assert diop_solve(y + x + 0) == (t_0, -t_0)
    assert diop_solve(y + x - 0) == (t_0, -t_0)
    assert diop_solve(0*x - y - 5) == (-5,)
    assert diop_solve(3*y + 2*x - 5) == (3*t_0 - 5, -2*t_0 + 5)
    assert diop_solve(2*x - 3*y - 5) == (3*t_0 - 5, 2*t_0 - 5)
    assert diop_solve(-2*x - 3*y - 5) == (3*t_0 + 5, -2*t_0 - 5)
    assert diop_solve(7*x + 5*y) == (5*t_0, -7*t_0)
    assert diop_solve(2*x + 4*y) == (-2*t_0, t_0)
    assert diop_solve(4*x + 6*y - 4) == (3*t_0 - 2, -2*t_0 + 2)
    assert diop_solve(4*x + 6*y - 3) == (None, None)
    assert diop_solve(0*x + 3*y - 4*z + 5) == (4*t_0 + 5, 3*t_0 + 5)
    assert diop_solve(4*x + 3*y - 4*z + 5) == (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    assert diop_solve(4*x + 3*y - 4*z + 5, None) == (0, 5, 5)
    assert diop_solve(4*x + 2*y + 8*z - 5) == (None, None, None)
    assert diop_solve(5*x + 7*y - 2*z - 6) == (t_0, -3*t_0 + 2*t_1 + 6, -8*t_0 + 7*t_1 + 18)
    assert diop_solve(3*x - 6*y + 12*z - 9) == (2*t_0 + 3, t_0 + 2*t_1, t_1)
    assert diop_solve(6*w + 9*x + 20*y - z) == (t_0, t_1, t_1 + t_2, 6*t_0 + 29*t_1 + 20*t_2)

    # to ignore constant factors, use diophantine
    raises(TypeError, lambda: diop_solve(x/2))


def test_quadratic_simple_hyperbolic_case():
    # Simple Hyperbolic case: A = C = 0 and B != 0
    assert diop_solve(3*x*y + 34*x - 12*y + 1) == \
           {(-133, -11), (5, -57)}
    assert diop_solve(6*x*y + 2*x + 3*y + 1) == set()
    assert diop_solve(-13*x*y + 2*x - 4*y - 54) == {(27, 0)}
    assert diop_solve(-27*x*y - 30*x - 12*y - 54) == {(-14, -1)}
    assert diop_solve(2*x*y + 5*x + 56*y + 7) == {(-161, -3), (-47, -6), (-35, -12),
                                                  (-29, -69), (-27, 64), (-21, 7),
                                                  (-9, 1), (105, -2)}
    assert diop_solve(6*x*y + 9*x + 2*y + 3) == set()
    assert diop_solve(x*y + x + y + 1) == {(-1, t), (t, -1)}
    assert diophantine(48*x*y)


def test_quadratic_elliptical_case():
    # Elliptical case: B**2 - 4AC < 0

    assert diop_solve(42*x**2 + 8*x*y + 15*y**2 + 23*x + 17*y - 4915) == {(-11, -1)}
    assert diop_solve(4*x**2 + 3*y**2 + 5*x - 11*y + 12) == set()
    assert diop_solve(x**2 + y**2 + 2*x + 2*y + 2) == {(-1, -1)}
    assert diop_solve(15*x**2 - 9*x*y + 14*y**2 - 23*x - 14*y - 4950) == {(-15, 6)}
    assert diop_solve(10*x**2 + 12*x*y + 12*y**2 - 34) == \
           {(-1, -1), (-1, 2), (1, -2), (1, 1)}


def test_quadratic_parabolic_case():
    # Parabolic case: B**2 - 4AC = 0
    assert check_solutions(8*x**2 - 24*x*y + 18*y**2 + 5*x + 7*y + 16)
    assert check_solutions(8*x**2 - 24*x*y + 18*y**2 + 6*x + 12*y - 6)
    assert check_solutions(8*x**2 + 24*x*y + 18*y**2 + 4*x + 6*y - 7)
    assert check_solutions(-4*x**2 + 4*x*y - y**2 + 2*x - 3)
    assert check_solutions(x**2 + 2*x*y + y**2 + 2*x + 2*y + 1)
    assert check_solutions(x**2 - 2*x*y + y**2 + 2*x + 2*y + 1)
    assert check_solutions(y**2 - 41*x + 40)


def test_quadratic_perfect_square():
    # B**2 - 4*A*C > 0
    # B**2 - 4*A*C is a perfect square
    assert check_solutions(48*x*y)
    assert check_solutions(4*x**2 - 5*x*y + y**2 + 2)
    assert check_solutions(-2*x**2 - 3*x*y + 2*y**2 -2*x - 17*y + 25)
    assert check_solutions(12*x**2 + 13*x*y + 3*y**2 - 2*x + 3*y - 12)
    assert check_solutions(8*x**2 + 10*x*y + 2*y**2 - 32*x - 13*y - 23)
    assert check_solutions(4*x**2 - 4*x*y - 3*y- 8*x - 3)
    assert check_solutions(- 4*x*y - 4*y**2 - 3*y- 5*x - 10)
    assert check_solutions(x**2 - y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert check_solutions(4*x**2 - 9*y**2 - 4*x - 12*y - 3)


def test_quadratic_non_perfect_square():
    # B**2 - 4*A*C is not a perfect square
    # Used check_solutions() since the solutions are complex expressions involving
    # square roots and exponents
    assert check_solutions(x**2 - 2*x - 5*y**2)
    assert check_solutions(3*x**2 - 2*y**2 - 2*x - 2*y)
    assert check_solutions(x**2 - x*y - y**2 - 3*y)
    assert check_solutions(x**2 - 9*y**2 - 2*x - 6*y)
    assert BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2).solve() == {(-1, -1)}


def test_issue_9106():
    eq = -48 - 2*x*(3*x - 1) + y*(3*y - 1)
    v = (x, y)
    for sol in diophantine(eq):
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


def test_issue_18138():
    eq = x**2 - x - y**2
    v = (x, y)
    for sol in diophantine(eq):
        assert not diop_simplify(eq.xreplace(dict(zip(v, sol))))


@slow
def test_quadratic_non_perfect_slow():
    assert check_solutions(8*x**2 + 10*x*y - 2*y**2 - 32*x - 13*y - 23)
    # This leads to very large numbers.
    # assert check_solutions(5*x**2 - 13*x*y + y**2 - 4*x - 4*y - 15)
    assert check_solutions(-3*x**2 - 2*x*y + 7*y**2 - 5*x - 7)
    assert check_solutions(-4 - x + 4*x**2 - y - 3*x*y - 4*y**2)
    assert check_solutions(1 + 2*x + 2*x**2 + 2*y + x*y - 2*y**2)


def test_DN():
    # Most of the test cases were adapted from,
    # Solving the generalized Pell equation x**2 - D*y**2 = N, John P. Robertson, July 31, 2004.
    # https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    # others are verified using Wolfram Alpha.

    # Covers cases where D <= 0 or D > 0 and D is a square or N = 0
    # Solutions are straightforward in these cases.
    assert diop_DN(3, 0) == [(0, 0)]
    assert diop_DN(-17, -5) == []
    assert diop_DN(-19, 23) == [(2, 1)]
    assert diop_DN(-13, 17) == [(2, 1)]
    assert diop_DN(-15, 13) == []
    assert diop_DN(0, 5) == []
    assert diop_DN(0, 9) == [(3, t)]
    assert diop_DN(9, 0) == [(3*t, t)]
    assert diop_DN(16, 24) == []
    assert diop_DN(9, 180) == [(18, 4)]
    assert diop_DN(9, -180) == [(12, 6)]
    assert diop_DN(7, 0) == [(0, 0)]

    # When equation is x**2 + y**2 = N
    # Solutions are interchangeable
    assert diop_DN(-1, 5) == [(2, 1), (1, 2)]
    assert diop_DN(-1, 169) == [(12, 5), (5, 12), (13, 0), (0, 13)]

    # D > 0 and D is not a square

    # N = 1
    assert diop_DN(13, 1) == [(649, 180)]
    assert diop_DN(980, 1) == [(51841, 1656)]
    assert diop_DN(981, 1) == [(158070671986249, 5046808151700)]
    assert diop_DN(986, 1) == [(49299, 1570)]
    assert diop_DN(991, 1) == [(379516400906811930638014896080, 12055735790331359447442538767)]
    assert diop_DN(17, 1) == [(33, 8)]
    assert diop_DN(19, 1) == [(170, 39)]

    # N = -1
    assert diop_DN(13, -1) == [(18, 5)]
    assert diop_DN(991, -1) == []
    assert diop_DN(41, -1) == [(32, 5)]
    assert diop_DN(290, -1) == [(17, 1)]
    assert diop_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_DN(32, -1) == []

    # |N| > 1
    # Some tests were created using calculator at
    # http://www.numbertheory.org/php/patz.html

    assert diop_DN(13, -4) == [(3, 1), (393, 109), (36, 10)]
    # Source I referred returned (3, 1), (393, 109) and (-3, 1) as fundamental solutions
    # So (-3, 1) and (393, 109) should be in the same equivalent class
    assert equivalent(-3, 1, 393, 109, 13, -4) == True

    assert diop_DN(13, 27) == [(220, 61), (40, 11), (768, 213), (12, 3)]
    assert set(diop_DN(157, 12)) == {(13, 1), (10663, 851), (579160, 46222),
                                     (483790960, 38610722), (26277068347, 2097138361),
                                     (21950079635497, 1751807067011)}
    assert diop_DN(13, 25) == [(3245, 900)]
    assert diop_DN(192, 18) == []
    assert diop_DN(23, 13) == [(-6, 1), (6, 1)]
    assert diop_DN(167, 2) == [(13, 1)]
    assert diop_DN(167, -2) == []

    assert diop_DN(123, -2) == [(11, 1)]
    # One calculator returned [(11, 1), (-11, 1)] but both of these are in
    # the same equivalence class
    assert equivalent(11, 1, -11, 1, 123, -2)

    assert diop_DN(123, -23) == [(-10, 1), (10, 1)]

    assert diop_DN(0, 0, t) == [(0, t)]
    assert diop_DN(0, -1, t) == []


def test_bf_pell():
    assert diop_bf_DN(13, -4) == [(3, 1), (-3, 1), (36, 10)]
    assert diop_bf_DN(13, 27) == [(12, 3), (-12, 3), (40, 11), (-40, 11)]
    assert diop_bf_DN(167, -2) == []
    assert diop_bf_DN(1729, 1) == [(44611924489705, 1072885712316)]
    assert diop_bf_DN(89, -8) == [(9, 1), (-9, 1)]
    assert diop_bf_DN(21257, -1) == [(13913102721304, 95427381109)]
    assert diop_bf_DN(340, -4) == [(756, 41)]
    assert diop_bf_DN(-1, 0, t) == [(0, 0)]
    assert diop_bf_DN(0, 0, t) == [(0, t)]
    assert diop_bf_DN(4, 0, t) == [(2*t, t), (-2*t, t)]
    assert diop_bf_DN(3, 0, t) == [(0, 0)]
    assert diop_bf_DN(1, -2, t) == []


def test_length():
    assert length(2, 1, 0) == 1
    assert length(-2, 4, 5) == 3
    assert length(-5, 4, 17) == 4
    assert length(0, 4, 13) == 6
    assert length(7, 13, 11) == 23
    assert length(1, 6, 4) == 2


def is_pell_transformation_ok(eq):
    """
    Test whether X*Y, X, or Y terms are present in the equation
    after transforming the equation using the transformation returned
    by transformation_to_pell(). If they are not present we are good.
    Moreover, coefficient of X**2 should be a divisor of coefficient of
    Y**2 and the constant term.
    """
    A, B = transformation_to_DN(eq)
    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    simplified = diop_simplify(eq.subs(zip((x, y), (u, v))))

    coeff = dict([reversed(t.as_independent(*[X, Y])) for t in simplified.args])

    for term in [X*Y, X, Y]:
        if term in coeff.keys():
            return False

    for term in [X**2, Y**2, 1]:
        if term not in coeff.keys():
            coeff[term] = 0

    if coeff[X**2] != 0:
        return divisible(coeff[Y**2], coeff[X**2]) and \
        divisible(coeff[1], coeff[X**2])

    return True


def test_transformation_to_pell():
    assert is_pell_transformation_ok(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y - 14)
    assert is_pell_transformation_ok(-17*x**2 + 19*x*y - 7*y**2 - 5*x - 13*y - 23)
    assert is_pell_transformation_ok(x**2 - y**2 + 17)
    assert is_pell_transformation_ok(-x**2 + 7*y**2 - 23)
    assert is_pell_transformation_ok(25*x**2 - 45*x*y + 5*y**2 - 5*x - 10*y + 5)
    assert is_pell_transformation_ok(190*x**2 + 30*x*y + y**2 - 3*y - 170*x - 130)
    assert is_pell_transformation_ok(x**2 - 2*x*y -190*y**2 - 7*y - 23*x - 89)
    assert is_pell_transformation_ok(15*x**2 - 9*x*y + 14*y**2 - 23*x - 14*y - 4950)


def test_find_DN():
    assert find_DN(x**2 - 2*x - y**2) == (1, 1)
    assert find_DN(x**2 - 3*y**2 - 5) == (3, 5)
    assert find_DN(x**2 - 2*x*y - 4*y**2 - 7) == (5, 7)
    assert find_DN(4*x**2 - 8*x*y - y**2 - 9) == (20, 36)
    assert find_DN(7*x**2 - 2*x*y - y**2 - 12) == (8, 84)
    assert find_DN(-3*x**2 + 4*x*y -y**2) == (1, 0)
    assert find_DN(-13*x**2 - 7*x*y + y**2 + 2*x - 2*y -14) == (101, -7825480)


def test_ldescent():
    # Equations which have solutions
    u = ([(13, 23), (3, -11), (41, -113), (4, -7), (-7, 4), (91, -3), (1, 1), (1, -1),
        (4, 32), (17, 13), (123689, 1), (19, -570)])
    for a, b in u:
        w, x, y = ldescent(a, b)
        assert a*x**2 + b*y**2 == w**2
    assert ldescent(-1, -1) is None
    assert ldescent(2, 6) is None


def test_diop_ternary_quadratic_normal():
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(23*x**2 + 616*y**2 - z**2)
    assert check_solutions(5*x**2 + 4*y**2 - z**2)
    assert check_solutions(3*x**2 + 6*y**2 - 3*z**2)
    assert check_solutions(x**2 + 3*y**2 - z**2)
    assert check_solutions(4*x**2 + 5*y**2 - z**2)
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(16*x**2 + y**2 - 25*z**2)
    assert check_solutions(6*x**2 - y**2 + 10*z**2)
    assert check_solutions(213*x**2 + 12*y**2 - 9*z**2)
    assert check_solutions(34*x**2 - 3*y**2 - 301*z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)


def is_normal_transformation_ok(eq):
    A = transformation_to_normal(eq)
    X, Y, Z = A*Matrix([x, y, z])
    simplified = diop_simplify(eq.subs(zip((x, y, z), (X, Y, Z))))

    coeff = dict([reversed(t.as_independent(*[X, Y, Z])) for t in simplified.args])
    for term in [X*Y, Y*Z, X*Z]:
        if term in coeff.keys():
            return False

    return True


def test_transformation_to_normal():
    assert is_normal_transformation_ok(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert is_normal_transformation_ok(x**2 + 3*y**2 - 100*z**2)
    assert is_normal_transformation_ok(x**2 + 23*y*z)
    assert is_normal_transformation_ok(3*y**2 - 100*z**2 - 12*x*y)
    assert is_normal_transformation_ok(x**2 + 23*x*y - 34*y*z + 12*x*z)
    assert is_normal_transformation_ok(z**2 + 34*x*y - 23*y*z + x*z)
    assert is_normal_transformation_ok(x**2 + y**2 + z**2 - x*y - y*z - x*z)
    assert is_normal_transformation_ok(x**2 + 2*y*z + 3*z**2)
    assert is_normal_transformation_ok(x*y + 2*x*z + 3*y*z)
    assert is_normal_transformation_ok(2*x*z + 3*y*z)


def test_diop_ternary_quadratic():
    assert check_solutions(2*x**2 + z**2 + y**2 - 4*x*y)
    assert check_solutions(x**2 - y**2 - z**2 - x*y - y*z)
    assert check_solutions(3*x**2 - x*y - y*z - x*z)
    assert check_solutions(x**2 - y*z - x*z)
    assert check_solutions(5*x**2 - 3*x*y - x*z)
    assert check_solutions(4*x**2 - 5*y**2 - x*z)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(8*x**2 - 12*y*z)
    assert check_solutions(45*x**2 - 7*y**2 - 8*x*y - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 17*y*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - x*y - 16*y*z + 12*x*z)
    assert check_solutions(x**2 + 3*y**2 + z**2 - 13*x*y - 16*y*z + 12*x*z)
    assert check_solutions(x*y - 7*y*z + 13*x*z)

    assert diop_ternary_quadratic_normal(x**2 + y**2 + z**2) == (None, None, None)
    assert diop_ternary_quadratic_normal(x**2 + y**2) is None
    raises(ValueError, lambda:
        _diop_ternary_quadratic_normal((x, y, z),
        {x*y: 1, x**2: 2, y**2: 3, z**2: 0}))
    eq = -2*x*y - 6*x*z + 7*y**2 - 3*y*z + 4*z**2
    assert diop_ternary_quadratic(eq) == (7, 2, 0)
    assert diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2) == \
        (1, 0, 2)
    assert diop_ternary_quadratic(x*y + 2*y*z) == \
        (-2, 0, n1)
    eq = -5*x*y - 8*x*z - 3*y*z + 8*z**2
    assert parametrize_ternary_quadratic(eq) == \
        (8*p**2 - 3*p*q, -8*p*q + 8*q**2, 5*p*q)
    # this cannot be tested with diophantine because it will
    # factor into a product
    assert diop_solve(x*y + 2*y*z) == (-2*p*q, -n1*p**2 + p**2, p*q)


def test_square_factor():
    assert square_factor(1) == square_factor(-1) == 1
    assert square_factor(0) == 1
    assert square_factor(5) == square_factor(-5) == 1
    assert square_factor(4) == square_factor(-4) == 2
    assert square_factor(12) == square_factor(-12) == 2
    assert square_factor(6) == 1
    assert square_factor(18) == 3
    assert square_factor(52) == 2
    assert square_factor(49) == 7
    assert square_factor(392) == 14
    assert square_factor(factorint(-12)) == 2


def test_parametrize_ternary_quadratic():
    assert check_solutions(x**2 + y**2 - z**2)
    assert check_solutions(x**2 + 2*x*y + z**2)
    assert check_solutions(234*x**2 - 65601*y**2 - z**2)
    assert check_solutions(3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z)
    assert check_solutions(x**2 - y**2 - z**2)
    assert check_solutions(x**2 - 49*y**2 - z**2 + 13*z*y - 8*x*y)
    assert check_solutions(8*x*y + z**2)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)
    assert check_solutions(236*x**2 - 225*y**2 - 11*x*y - 13*y*z - 17*x*z)
    assert check_solutions(90*x**2 + 3*y**2 + 5*x*y + 2*z*y + 5*x*z)
    assert check_solutions(124*x**2 - 30*y**2 - 7729*z**2)


def test_no_square_ternary_quadratic():
    assert check_solutions(2*x*y + y*z - 3*x*z)
    assert check_solutions(189*x*y - 345*y*z - 12*x*z)
    assert check_solutions(23*x*y + 34*y*z)
    assert check_solutions(x*y + y*z + z*x)
    assert check_solutions(23*x*y + 23*y*z + 23*x*z)


def test_descent():

    u = ([(13, 23), (3, -11), (41, -113), (91, -3), (1, 1), (1, -1), (17, 13), (123689, 1), (19, -570)])
    for a, b in u:
        w, x, y = descent(a, b)
        assert a*x**2 + b*y**2 == w**2
    # the docstring warns against bad input, so these are expected results
    # - can't both be negative
    raises(TypeError, lambda: descent(-1, -3))
    # A can't be zero unless B != 1
    raises(ZeroDivisionError, lambda: descent(0, 3))
    # supposed to be square-free
    raises(TypeError, lambda: descent(4, 3))


def test_diophantine():
    assert check_solutions((x - y)*(y - z)*(z - x))
    assert check_solutions((x - y)*(x**2 + y**2 - z**2))
    assert check_solutions((x - 3*y + 7*z)*(x**2 + y**2 - z**2))
    assert check_solutions(x**2 - 3*y**2 - 1)
    assert check_solutions(y**2 + 7*x*y)
    assert check_solutions(x**2 - 3*x*y + y**2)
    assert check_solutions(z*(x**2 - y**2 - 15))
    assert check_solutions(x*(2*y - 2*z + 5))
    assert check_solutions((x**2 - 3*y**2 - 1)*(x**2 - y**2 - 15))
    assert check_solutions((x**2 - 3*y**2 - 1)*(y - 7*z))
    assert check_solutions((x**2 + y**2 - z**2)*(x - 7*y - 3*z + 4*w))
    # Following test case caused problems in parametric representation
    # But this can be solved by factoring out y.
    # No need to use methods for ternary quadratic equations.
    assert check_solutions(y**2 - 7*x*y + 4*y*z)
    assert check_solutions(x**2 - 2*x + 1)

    assert diophantine(x - y) == diophantine(Eq(x, y))
    # 18196
    eq = x**4 + y**4 - 97
    assert diophantine(eq, permute=True) == diophantine(-eq, permute=True)
    assert diophantine(3*x*pi - 2*y*pi) == {(2*t_0, 3*t_0)}
    eq = x**2 + y**2 + z**2 - 14
    base_sol = {(1, 2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln

    assert diophantine(x**2 + x*Rational(15, 14) - 3) == set()
    # test issue 11049
    eq = 92*x**2 - 99*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(9, 7, 51)}
    assert diophantine(eq) == {(
        891*p**2 + 9*q**2, -693*p**2 - 102*p*q + 7*q**2,
        5049*p**2 - 1386*p*q - 51*q**2)}
    eq = 2*x**2 + 2*y**2 - z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(1, 1, 2)}
    assert diophantine(eq) == {(
        2*p**2 - q**2, -2*p**2 + 4*p*q - q**2,
        4*p**2 - 4*p*q + 2*q**2)}
    eq = 411*x**2+57*y**2-221*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(2021, 2645, 3066)}
    assert diophantine(eq) == \
           {(115197*p**2 - 446641*q**2, -150765*p**2 + 1355172*p*q -
             584545*q**2, 174762*p**2 - 301530*p*q + 677586*q**2)}
    eq = 573*x**2+267*y**2-984*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(49, 233, 127)}
    assert diophantine(eq) == \
           {(4361*p**2 - 16072*q**2, -20737*p**2 + 83312*p*q - 76424*q**2,
             11303*p**2 - 41474*p*q + 41656*q**2)}
    # this produces factors during reconstruction
    eq = x**2 + 3*y**2 - 12*z**2
    coeff = eq.as_coefficients_dict()
    assert _diop_ternary_quadratic_normal((x, y, z), coeff) == \
           {(0, 2, 1)}
    assert diophantine(eq) == \
           {(24*p*q, 2*p**2 - 24*q**2, p**2 + 12*q**2)}
    # solvers have not been written for every type
    raises(NotImplementedError, lambda: diophantine(x*y**2 + 1))

    # rational expressions
    assert diophantine(1/x) == set()
    assert diophantine(1/x + 1/y - S.Half) == {(6, 3), (-2, 1), (4, 4), (1, -2), (3, 6)}
    assert diophantine(x**2 + y**2 +3*x- 5, permute=True) == \
           {(-1, 1), (-4, -1), (1, -1), (1, 1), (-4, 1), (-1, -1), (4, 1), (4, -1)}


    #test issue 18186
    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(x, y), permute=True) == \
           {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}
    assert diophantine(y**4 + x**4 - 2**4 - 3**4, syms=(y, x), permute=True) == \
           {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

    # issue 18122
    assert check_solutions(x**2 - y)
    assert check_solutions(y**2 - x)
    assert diophantine((x**2 - y), t) == {(t, t**2)}
    assert diophantine((y**2 - x), t) == {(t**2, t)}


def test_general_pythagorean():
    from sympy.abc import a, b, c, d, e

    assert check_solutions(a**2 + b**2 + c**2 - d**2)
    assert check_solutions(a**2 + 4*b**2 + 4*c**2 - d**2)
    assert check_solutions(9*a**2 + 4*b**2 + 4*c**2 - d**2)
    assert check_solutions(9*a**2 + 4*b**2 - 25*d**2 + 4*c**2 )
    assert check_solutions(9*a**2 - 16*d**2 + 4*b**2 + 4*c**2)
    assert check_solutions(-e**2 + 9*a**2 + 4*b**2 + 4*c**2 + 25*d**2)
    assert check_solutions(16*a**2 - b**2 + 9*c**2 + d**2 + 25*e**2)

    assert GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve(parameters=[x, y, z]) == \
           {(x**2 + y**2 - z**2, 2*x*z, 2*y*z, x**2 + y**2 + z**2)}


def test_diop_general_sum_of_squares_quick():
    for i in range(3, 10):
        assert check_solutions(sum(i**2 for i in symbols(':%i' % i)) - i)

    assert diop_general_sum_of_squares(x**2 + y**2 - 2) is None
    assert diop_general_sum_of_squares(x**2 + y**2 + z**2 + 2) == set()
    eq = x**2 + y**2 + z**2 - (1 + 4 + 9)
    assert diop_general_sum_of_squares(eq) == \
           {(1, 2, 3)}
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 1313
    assert len(diop_general_sum_of_squares(eq, 3)) == 3
    # issue 11016
    var = symbols(':5') + (symbols('6', negative=True),)
    eq = Add(*[i**2 for i in var]) - 112

    base_soln = {(0, 1, 1, 5, 6, -7), (1, 1, 1, 3, 6, -8), (2, 3, 3, 4, 5, -7), (0, 1, 1, 1, 3, -10),
                 (0, 0, 4, 4, 4, -8), (1, 2, 3, 3, 5, -8), (0, 1, 2, 3, 7, -7), (2, 2, 4, 4, 6, -6),
                 (1, 1, 3, 4, 6, -7), (0, 2, 3, 3, 3, -9), (0, 0, 2, 2, 2, -10), (1, 1, 2, 3, 4, -9),
                 (0, 1, 1, 2, 5, -9), (0, 0, 2, 6, 6, -6), (1, 3, 4, 5, 5, -6), (0, 2, 2, 2, 6, -8),
                 (0, 3, 3, 3, 6, -7), (0, 2, 3, 5, 5, -7), (0, 1, 5, 5, 5, -6)}
    assert diophantine(eq) == base_soln
    assert len(diophantine(eq, permute=True)) == 196800

    # handle negated squares with signsimp
    assert diophantine(12 - x**2 - y**2 - z**2) == {(2, 2, 2)}
    # diophantine handles simplification, so classify_diop should
    # not have to look for additional patterns that are removed
    # by diophantine
    eq = a**2 + b**2 + c**2 + d**2 - 4
    raises(NotImplementedError, lambda: classify_diop(-eq))


def test_issue_23807():
    # fixes recursion error
    eq = x**2 + y**2 + z**2 - 1000000
    base_soln = {(0, 0, 1000), (0, 352, 936), (480, 600, 640), (24, 640, 768), (192, 640, 744),
                 (192, 480, 856), (168, 224, 960), (0, 600, 800), (280, 576, 768), (152, 480, 864),
                 (0, 280, 960), (352, 360, 864), (424, 480, 768), (360, 480, 800), (224, 600, 768),
                 (96, 360, 928), (168, 576, 800), (96, 480, 872)}

    assert diophantine(eq) == base_soln


def test_diop_partition():
    for n in [8, 10]:
        for k in range(1, 8):
            for p in partition(n, k):
                assert len(p) == k
    assert list(partition(3, 5)) == []
    assert [list(p) for p in partition(3, 5, 1)] == [
        [0, 0, 0, 0, 3], [0, 0, 0, 1, 2], [0, 0, 1, 1, 1]]
    assert list(partition(0)) == [()]
    assert list(partition(1, 0)) == [()]
    assert [list(i) for i in partition(3)] == [[1, 1, 1], [1, 2], [3]]


def test_prime_as_sum_of_two_squares():
    for i in [5, 13, 17, 29, 37, 41, 2341, 3557, 34841, 64601]:
        a, b = prime_as_sum_of_two_squares(i)
        assert a**2 + b**2 == i
    assert prime_as_sum_of_two_squares(7) is None
    ans = prime_as_sum_of_two_squares(800029)
    assert ans == (450, 773) and type(ans[0]) is int


def test_sum_of_three_squares():
    for i in [0, 1, 2, 34, 123, 34304595905, 34304595905394941, 343045959052344,
              800, 801, 802, 803, 804, 805, 806]:
        a, b, c = sum_of_three_squares(i)
        assert a**2 + b**2 + c**2 == i
        assert a >= 0

    # error
    raises(ValueError, lambda: sum_of_three_squares(-1))

    assert sum_of_three_squares(7) is None
    assert sum_of_three_squares((4**5)*15) is None
    # if there are two zeros, there might be a solution
    # with only one zero, e.g. 25 => (0, 3, 4) or
    # with no zeros, e.g. 49 => (2, 3, 6)
    assert sum_of_three_squares(25) == (0, 0, 5)
    assert sum_of_three_squares(4) == (0, 0, 2)


def test_sum_of_four_squares():
    from sympy.core.random import randint

    # this should never fail
    n = randint(1, 100000000000000)
    assert sum(i**2 for i in sum_of_four_squares(n)) == n

    # error
    raises(ValueError, lambda: sum_of_four_squares(-1))

    for n in range(1000):
        result = sum_of_four_squares(n)
        assert len(result) == 4
        assert all(r >= 0 for r in result)
        assert sum(r**2 for r in result) == n
        assert list(result) == sorted(result)


def test_power_representation():
    tests = [(1729, 3, 2), (234, 2, 4), (2, 1, 2), (3, 1, 3), (5, 2, 2), (12352, 2, 4),
             (32760, 2, 3)]

    for test in tests:
        n, p, k = test
        f = power_representation(n, p, k)

        while True:
            try:
                l = next(f)
                assert len(l) == k

                chk_sum = 0
                for l_i in l:
                    chk_sum = chk_sum + l_i**p
                assert chk_sum == n

            except StopIteration:
                break

    assert list(power_representation(20, 2, 4, True)) == \
        [(1, 1, 3, 3), (0, 0, 2, 4)]
    raises(ValueError, lambda: list(power_representation(1.2, 2, 2)))
    raises(ValueError, lambda: list(power_representation(2, 0, 2)))
    raises(ValueError, lambda: list(power_representation(2, 2, 0)))
    assert list(power_representation(-1, 2, 2)) == []
    assert list(power_representation(1, 1, 1)) == [(1,)]
    assert list(power_representation(3, 2, 1)) == []
    assert list(power_representation(4, 2, 1)) == [(2,)]
    assert list(power_representation(3**4, 4, 6, zeros=True)) == \
        [(1, 2, 2, 2, 2, 2), (0, 0, 0, 0, 0, 3)]
    assert list(power_representation(3**4, 4, 5, zeros=False)) == []
    assert list(power_representation(-2, 3, 2)) == [(-1, -1)]
    assert list(power_representation(-2, 4, 2)) == []
    assert list(power_representation(0, 3, 2, True)) == [(0, 0)]
    assert list(power_representation(0, 3, 2, False)) == []
    # when we are dealing with squares, do feasibility checks
    assert len(list(power_representation(4**10*(8*10 + 7), 2, 3))) == 0
    # there will be a recursion error if these aren't recognized
    big = 2**30
    for i in [13, 10, 7, 5, 4, 2, 1]:
        assert list(sum_of_powers(big, 2, big - i)) == []


def test_assumptions():
    """
    Test whether diophantine respects the assumptions.
    """
    #Test case taken from the below so question regarding assumptions in diophantine module
    #https://stackoverflow.com/questions/23301941/how-can-i-declare-natural-symbols-with-sympy
    m, n = symbols('m n', integer=True, positive=True)
    diof = diophantine(n**2 + m*n - 500)
    assert diof == {(5, 20), (40, 10), (95, 5), (121, 4), (248, 2), (499, 1)}

    a, b = symbols('a b', integer=True, positive=False)
    diof = diophantine(a*b + 2*a + 3*b - 6)
    assert diof == {(-15, -3), (-9, -4), (-7, -5), (-6, -6), (-5, -8), (-4, -14)}

    x, y = symbols('x y', integer=True)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == {(1, -5), (-3, 5), (0, 0)}

    x, y = symbols('x y', integer=True, positive=True)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == set()

    x, y = symbols('x y', integer=True, negative=False)
    diof = diophantine(10*x**2 + 5*x*y - 3*y)
    assert diof == {(0, 0)}


def check_solutions(eq):
    """
    Determines whether solutions returned by diophantine() satisfy the original
    equation. Hope to generalize this so we can remove functions like check_ternay_quadratic,
    check_solutions_normal, check_solutions()
    """
    s = diophantine(eq)

    factors = Mul.make_args(eq)

    var = list(eq.free_symbols)
    var.sort(key=default_sort_key)

    while s:
        solution = s.pop()
        for f in factors:
            if diop_simplify(f.subs(zip(var, solution))) == 0:
                break
        else:
            return False
    return True


def test_diopcoverage():
    eq = (2*x + y + 1)**2
    assert diop_solve(eq) == {(t_0, -2*t_0 - 1)}
    eq = 2*x**2 + 6*x*y + 12*x + 4*y**2 + 18*y + 18
    assert diop_solve(eq) == {(t, -t - 3), (-2*t - 3, t)}
    assert diop_quadratic(x + y**2 - 3) == {(-t**2 + 3, t)}

    assert diop_linear(x + y - 3) == (t_0, 3 - t_0)

    assert base_solution_linear(0, 1, 2, t=None) == (0, 0)
    ans = (3*t - 1, -2*t + 1)
    assert base_solution_linear(4, 8, 12, t) == ans
    assert base_solution_linear(4, 8, 12, t=None) == tuple(_.subs(t, 0) for _ in ans)

    assert cornacchia(1, 1, 20) == set()
    assert cornacchia(1, 1, 5) == {(2, 1)}
    assert cornacchia(1, 2, 17) == {(3, 2)}

    raises(ValueError, lambda: reconstruct(4, 20, 1))

    assert gaussian_reduce(4, 1, 3) == (1, 1)
    eq = -w**2 - x**2 - y**2 + z**2

    assert diop_general_pythagorean(eq) == \
        diop_general_pythagorean(-eq) == \
            (m1**2 + m2**2 - m3**2, 2*m1*m3,
            2*m2*m3, m1**2 + m2**2 + m3**2)

    assert len(check_param(S(3) + x/3, S(4) + x/2, S(2), [x])) == 0
    assert len(check_param(Rational(3, 2), S(4) + x, S(2), [x])) == 0
    assert len(check_param(S(4) + x, Rational(3, 2), S(2), [x])) == 0

    assert _nint_or_floor(16, 10) == 2
    assert _odd(1) == (not _even(1)) == True
    assert _odd(0) == (not _even(0)) == False
    assert _remove_gcd(2, 4, 6) == (1, 2, 3)
    raises(TypeError, lambda: _remove_gcd((2, 4, 6)))
    assert sqf_normal(2*3**2*5, 2*5*11, 2*7**2*11)  == \
        (11, 1, 5)

    # it's ok if these pass some day when the solvers are implemented
    raises(NotImplementedError, lambda: diophantine(x**2 + y**2 + x*y + 2*y*z - 12))
    raises(NotImplementedError, lambda: diophantine(x**3 + y**2))
    assert diop_quadratic(x**2 + y**2 - 1**2 - 3**4) == \
           {(-9, -1), (-9, 1), (-1, -9), (-1, 9), (1, -9), (1, 9), (9, -1), (9, 1)}


def test_holzer():
    # if the input is good, don't let it diverge in holzer()
    # (but see test_fail_holzer below)
    assert holzer(2, 7, 13, 4, 79, 23) == (2, 7, 13)

    # None in uv condition met; solution is not Holzer reduced
    # so this will hopefully change but is here for coverage
    assert holzer(2, 6, 2, 1, 1, 10) == (2, 6, 2)

    raises(ValueError, lambda: holzer(2, 7, 14, 4, 79, 23))


@XFAIL
def test_fail_holzer():
    eq = lambda x, y, z: a*x**2 + b*y**2 - c*z**2
    a, b, c = 4, 79, 23
    x, y, z = xyz = 26, 1, 11
    X, Y, Z = ans = 2, 7, 13
    assert eq(*xyz) == 0
    assert eq(*ans) == 0
    assert max(a*x**2, b*y**2, c*z**2) <= a*b*c
    assert max(a*X**2, b*Y**2, c*Z**2) <= a*b*c
    h = holzer(x, y, z, a, b, c)
    assert h == ans  # it would be nice to get the smaller soln


def test_issue_9539():
    assert diophantine(6*w + 9*y + 20*x - z) == \
           {(t_0, t_1, t_1 + t_2, 6*t_0 + 29*t_1 + 9*t_2)}


def test_issue_8943():
    assert diophantine(
        3*(x**2 + y**2 + z**2) - 14*(x*y + y*z + z*x)) == \
           {(0, 0, 0)}


def test_diop_sum_of_even_powers():
    eq = x**4 + y**4 + z**4 - 2673
    assert diop_solve(eq) == {(3, 6, 6), (2, 4, 7)}
    assert diop_general_sum_of_even_powers(eq, 2) == {(3, 6, 6), (2, 4, 7)}
    raises(NotImplementedError, lambda: diop_general_sum_of_even_powers(-eq, 2))
    neg = symbols('neg', negative=True)
    eq = x**4 + y**4 + neg**4 - 2673
    assert diop_general_sum_of_even_powers(eq) == {(-3, 6, 6)}
    assert diophantine(x**4 + y**4 + 2) == set()
    assert diop_general_sum_of_even_powers(x**4 + y**4 - 2, limit=0) == set()


def test_sum_of_squares_powers():
    tru = {(0, 0, 1, 1, 11), (0, 0, 5, 7, 7), (0, 1, 3, 7, 8), (0, 1, 4, 5, 9), (0, 3, 4, 7, 7), (0, 3, 5, 5, 8),
           (1, 1, 2, 6, 9), (1, 1, 6, 6, 7), (1, 2, 3, 3, 10), (1, 3, 4, 4, 9), (1, 5, 5, 6, 6), (2, 2, 3, 5, 9),
           (2, 3, 5, 6, 7), (3, 3, 4, 5, 8)}
    eq = u**2 + v**2 + x**2 + y**2 + z**2 - 123
    ans = diop_general_sum_of_squares(eq, oo)  # allow oo to be used
    assert len(ans) == 14
    assert ans == tru

    raises(ValueError, lambda: list(sum_of_squares(10, -1)))
    assert list(sum_of_squares(1, 1)) == [(1,)]
    assert list(sum_of_squares(1, 2)) == []
    assert list(sum_of_squares(1, 2, True)) == [(0, 1)]
    assert list(sum_of_squares(-10, 2)) == []
    assert list(sum_of_squares(2, 3)) == []
    assert list(sum_of_squares(0, 3, True)) == [(0, 0, 0)]
    assert list(sum_of_squares(0, 3)) == []
    assert list(sum_of_squares(4, 1)) == [(2,)]
    assert list(sum_of_squares(5, 1)) == []
    assert list(sum_of_squares(50, 2)) == [(5, 5), (1, 7)]
    assert list(sum_of_squares(11, 5, True)) == [
        (1, 1, 1, 2, 2), (0, 0, 1, 1, 3)]
    assert list(sum_of_squares(8, 8)) == [(1, 1, 1, 1, 1, 1, 1, 1)]

    assert [len(list(sum_of_squares(i, 5, True))) for i in range(30)] == [
        1, 1, 1, 1, 2,
        2, 1, 1, 2, 2,
        2, 2, 2, 3, 2,
        1, 3, 3, 3, 3,
        4, 3, 3, 2, 2,
        4, 4, 4, 4, 5]
    assert [len(list(sum_of_squares(i, 5))) for i in range(30)] == [
        0, 0, 0, 0, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 1,
        0, 1, 1, 0, 1,
        2, 1, 1, 1, 1,
        1, 1, 1, 1, 3]
    for i in range(30):
        s1 = set(sum_of_squares(i, 5, True))
        assert not s1 or all(sum(j**2 for j in t) == i for t in s1)
        s2 = set(sum_of_squares(i, 5))
        assert all(sum(j**2 for j in t) == i for t in s2)

    raises(ValueError, lambda: list(sum_of_powers(2, -1, 1)))
    raises(ValueError, lambda: list(sum_of_powers(2, 1, -1)))
    assert list(sum_of_powers(-2, 3, 2)) == [(-1, -1)]
    assert list(sum_of_powers(-2, 4, 2)) == []
    assert list(sum_of_powers(2, 1, 1)) == [(2,)]
    assert list(sum_of_powers(2, 1, 3, True)) == [(0, 0, 2), (0, 1, 1)]
    assert list(sum_of_powers(5, 1, 2, True)) == [(0, 5), (1, 4), (2, 3)]
    assert list(sum_of_powers(6, 2, 2)) == []
    assert list(sum_of_powers(3**5, 3, 1)) == []
    assert list(sum_of_powers(3**6, 3, 1)) == [(9,)] and (9**3 == 3**6)
    assert list(sum_of_powers(2**1000, 5, 2)) == []


def test__can_do_sum_of_squares():
    assert _can_do_sum_of_squares(3, -1) is False
    assert _can_do_sum_of_squares(-3, 1) is False
    assert _can_do_sum_of_squares(0, 1)
    assert _can_do_sum_of_squares(4, 1)
    assert _can_do_sum_of_squares(1, 2)
    assert _can_do_sum_of_squares(2, 2)
    assert _can_do_sum_of_squares(3, 2) is False


def test_diophantine_permute_sign():
    from sympy.abc import a, b, c, d, e
    eq = a**4 + b**4 - (2**4 + 3**4)
    base_sol = {(2, 3)}
    assert diophantine(eq) == base_sol
    complete_soln = set(signed_permutations(base_sol.pop()))
    assert diophantine(eq, permute=True) == complete_soln

    eq = a**2 + b**2 + c**2 + d**2 + e**2 - 234
    assert len(diophantine(eq)) == 35
    assert len(diophantine(eq, permute=True)) == 62000
    soln = {(-1, -1), (-1, 2), (1, -2), (1, 1)}
    assert diophantine(10*x**2 + 12*x*y + 12*y**2 - 34, permute=True) == soln


@XFAIL
def test_not_implemented():
    eq = x**2 + y**4 - 1**2 - 3**4
    assert diophantine(eq, syms=[x, y]) == {(9, 1), (1, 3)}


def test_issue_9538():
    eq = x - 3*y + 2
    assert diophantine(eq, syms=[y,x]) == {(t_0, 3*t_0 - 2)}
    raises(TypeError, lambda: diophantine(eq, syms={y, x}))


def test_ternary_quadratic():
    # solution with 3 parameters
    s = diophantine(2*x**2 + y**2 - 2*z**2)
    p, q, r = ordered(S(s).free_symbols)
    assert s == {(
        p**2 - 2*q**2,
        -2*p**2 + 4*p*q - 4*p*r - 4*q**2,
        p**2 - 4*p*q + 2*q**2 - 4*q*r)}
    # solution with Mul in solution
    s = diophantine(x**2 + 2*y**2 - 2*z**2)
    assert s == {(4*p*q, p**2 - 2*q**2, p**2 + 2*q**2)}
    # solution with no Mul in solution
    s = diophantine(2*x**2 + 2*y**2 - z**2)
    assert s == {(2*p**2 - q**2, -2*p**2 + 4*p*q - q**2,
        4*p**2 - 4*p*q + 2*q**2)}
    # reduced form when parametrized
    s = diophantine(3*x**2 + 72*y**2 - 27*z**2)
    assert s == {(24*p**2 - 9*q**2, 6*p*q, 8*p**2 + 3*q**2)}
    assert parametrize_ternary_quadratic(
        3*x**2 + 2*y**2 - z**2 - 2*x*y + 5*y*z - 7*y*z) == (
        2*p**2 - 2*p*q - q**2, 2*p**2 + 2*p*q - q**2, 2*p**2 -
        2*p*q + 3*q**2)
    assert parametrize_ternary_quadratic(
        124*x**2 - 30*y**2 - 7729*z**2) == (
        -1410*p**2 - 363263*q**2, 2700*p**2 + 30916*p*q -
        695610*q**2, -60*p**2 + 5400*p*q + 15458*q**2)


def test_diophantine_solution_set():
    s1 = DiophantineSolutionSet([], [])
    assert set(s1) == set()
    assert s1.symbols == ()
    assert s1.parameters == ()
    raises(ValueError, lambda: s1.add((x,)))
    assert list(s1.dict_iterator()) == []

    s2 = DiophantineSolutionSet([x, y], [t, u])
    assert s2.symbols == (x, y)
    assert s2.parameters == (t, u)
    raises(ValueError, lambda: s2.add((1,)))
    s2.add((3, 4))
    assert set(s2) == {(3, 4)}
    s2.update((3, 4), (-1, u))
    assert set(s2) == {(3, 4), (-1, u)}
    raises(ValueError, lambda: s1.update(s2))
    assert list(s2.dict_iterator()) == [{x: -1, y: u}, {x: 3, y: 4}]

    s3 = DiophantineSolutionSet([x, y, z], [t, u])
    assert len(s3.parameters) == 2
    s3.add((t**2 + u, t - u, 1))
    assert set(s3) == {(t**2 + u, t - u, 1)}
    assert s3.subs(t, 2) == {(u + 4, 2 - u, 1)}
    assert s3(2) == {(u + 4, 2 - u, 1)}
    assert s3.subs({t: 7, u: 8}) == {(57, -1, 1)}
    assert s3(7, 8) == {(57, -1, 1)}
    assert s3.subs({t: 5}) == {(u + 25, 5 - u, 1)}
    assert s3(5) == {(u + 25, 5 - u, 1)}
    assert s3.subs(u, -3) == {(t**2 - 3, t + 3, 1)}
    assert s3(None, -3) == {(t**2 - 3, t + 3, 1)}
    assert s3.subs({t: 2, u: 8}) == {(12, -6, 1)}
    assert s3(2, 8) == {(12, -6, 1)}
    assert s3.subs({t: 5, u: -3}) == {(22, 8, 1)}
    assert s3(5, -3) == {(22, 8, 1)}
    raises(TypeError, lambda: s3.subs(x=1))
    raises(TypeError, lambda: s3.subs(1, 2, 3))
    raises(ValueError, lambda: s3.add(()))
    raises(ValueError, lambda: s3.add((1, 2, 3, 4)))
    raises(ValueError, lambda: s3.add((1, 2)))
    raises(ValueError, lambda: s3(1, 2, 3))
    raises(TypeError, lambda: s3(t=1))

    s4 = DiophantineSolutionSet([x, y], [t, u])
    s4.add((t, 11*t))
    s4.add((-t, 22*t))
    assert s4(0, 0) == {(0, 0)}


def test_quadratic_parameter_passing():
    eq = -33*x*y + 3*y**2
    solution = BinaryQuadratic(eq).solve(parameters=[t, u])
    # test that parameters are passed all the way to the final solution
    assert solution == {(t, 11*t), (t, -22*t)}
    assert solution(0, 0) == {(0, 0)}

def test_issue_18628():
    eq1 = x**2 - 15*x + y**2 - 8*y
    sol = diophantine(eq1)
    assert sol == {(15, 0), (15, 8), (-1, 4), (0, 0), (0, 8), (16, 4)}
    eq2 = 2*x**2 - 9*x + 4*y**2 - 8*y + 14
    sol = diophantine(eq2)
    assert sol == {(2, 1)}
