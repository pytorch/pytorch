from itertools import product
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.printing.repr import srepr
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2, minimum, maximum, amax, amin
from sympy.testing.pytest import raises

x, y, z = symbols('x y z')

def test_logaddexp():
    lae_xy = logaddexp(x, y)
    ref_xy = log(exp(x) + exp(y))
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            lae_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(log).simplify() == 0

    one_third_e = 1*exp(1)/3
    two_thirds_e = 2*exp(1)/3
    logThirdE = log(one_third_e)
    logTwoThirdsE = log(two_thirds_e)
    lae_sum_to_e = logaddexp(logThirdE, logTwoThirdsE)
    assert lae_sum_to_e.rewrite(log) == 1
    assert lae_sum_to_e.simplify() == 1
    was = logaddexp(2, 3)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with 2, 3


def test_logaddexp2():
    lae2_xy = logaddexp2(x, y)
    ref2_xy = log(2**x + 2**y)/log(2)
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            lae2_xy.diff(wrt, deriv_order) -
            ref2_xy.diff(wrt, deriv_order)
        ).rewrite(log).cancel() == 0

    def lb(x):
        return log(x)/log(2)

    two_thirds = S.One*2/3
    four_thirds = 2*two_thirds
    lbTwoThirds = lb(two_thirds)
    lbFourThirds = lb(four_thirds)
    lae2_sum_to_2 = logaddexp2(lbTwoThirds, lbFourThirds)
    assert lae2_sum_to_2.rewrite(log) == 1
    assert lae2_sum_to_2.simplify() == 1
    was = logaddexp2(x, y)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with x, y


def test_minimum_maximum():
    for MM, mm in zip([Min, Max], [minimum, maximum]):
        ref = MM(x, y, z)
        m = mm(x, y, z)
        assert m != ref
        assert m.rewrite(MM) == ref


def test_amin_amax():
    for am in [amin, amax]:
        assert am(x).array == x
        assert am(x).axis == None
        assert am(x, axis=3).axis == 3
        with raises(ValueError):
            am(x, y, z)
