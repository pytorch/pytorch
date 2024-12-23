from sympy.core.add import Add
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O, Order
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
from sympy.abc import w, x, y, z


def test_caching_bug():
    #needs to be a first test, so that all caches are clean
    #cache it
    O(w)
    #and test that this won't raise an exception
    O(w**(-1/x/log(3)*log(5)), w)


def test_free_symbols():
    assert Order(1).free_symbols == set()
    assert Order(x).free_symbols == {x}
    assert Order(1, x).free_symbols == {x}
    assert Order(x*y).free_symbols == {x, y}
    assert Order(x, x, y).free_symbols == {x, y}


def test_simple_1():
    o = Rational(0)
    assert Order(2*x) == Order(x)
    assert Order(x)*3 == Order(x)
    assert -28*Order(x) == Order(x)
    assert Order(Order(x)) == Order(x)
    assert Order(Order(x), y) == Order(Order(x), x, y)
    assert Order(-23) == Order(1)
    assert Order(exp(x)) == Order(1, x)
    assert Order(exp(1/x)).expr == exp(1/x)
    assert Order(x*exp(1/x)).expr == x*exp(1/x)
    assert Order(x**(o/3)).expr == x**(o/3)
    assert Order(x**(o*Rational(5, 3))).expr == x**(o*Rational(5, 3))
    assert Order(x**2 + x + y, x) == O(1, x)
    assert Order(x**2 + x + y, y) == O(1, y)
    raises(ValueError, lambda: Order(exp(x), x, x))
    raises(TypeError, lambda: Order(x, 2 - x))


def test_simple_2():
    assert Order(2*x)*x == Order(x**2)
    assert Order(2*x)/x == Order(1, x)
    assert Order(2*x)*x*exp(1/x) == Order(x**2*exp(1/x))
    assert (Order(2*x)*x*exp(1/x)/log(x)**3).expr == x**2*exp(1/x)*log(x)**-3


def test_simple_3():
    assert Order(x) + x == Order(x)
    assert Order(x) + 2 == 2 + Order(x)
    assert Order(x) + x**2 == Order(x)
    assert Order(x) + 1/x == 1/x + Order(x)
    assert Order(1/x) + 1/x**2 == 1/x**2 + Order(1/x)
    assert Order(x) + exp(1/x) == Order(x) + exp(1/x)


def test_simple_4():
    assert Order(x)**2 == Order(x**2)


def test_simple_5():
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(x) + Order(x**-2) == Order(x**-2)
    assert Order(x) + Order(1/x) == Order(1/x)


def test_simple_6():
    assert Order(x) - Order(x) == Order(x)
    assert Order(x) + Order(1) == Order(1)
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(1/x) + Order(1) == Order(1/x)
    assert Order(x) + Order(exp(1/x)) == Order(exp(1/x))
    assert Order(x**3) + Order(exp(2/x)) == Order(exp(2/x))
    assert Order(x**-3) + Order(exp(2/x)) == Order(exp(2/x))


def test_simple_7():
    assert 1 + O(1) == O(1)
    assert 2 + O(1) == O(1)
    assert x + O(1) == O(1)
    assert 1/x + O(1) == 1/x + O(1)


def test_simple_8():
    assert O(sqrt(-x)) == O(sqrt(x))
    assert O(x**2*sqrt(x)) == O(x**Rational(5, 2))
    assert O(x**3*sqrt(-(-x)**3)) == O(x**Rational(9, 2))
    assert O(x**Rational(3, 2)*sqrt((-x)**3)) == O(x**3)
    assert O(x*(-2*x)**(I/2)) == O(x*(-x)**(I/2))


def test_as_expr_variables():
    assert Order(x).as_expr_variables(None) == (x, ((x, 0),))
    assert Order(x).as_expr_variables(((x, 0),)) == (x, ((x, 0),))
    assert Order(y).as_expr_variables(((x, 0),)) == (y, ((x, 0), (y, 0)))
    assert Order(y).as_expr_variables(((x, 0), (y, 0))) == (y, ((x, 0), (y, 0)))


def test_contains_0():
    assert Order(1, x).contains(Order(1, x))
    assert Order(1, x).contains(Order(1))
    assert Order(1).contains(Order(1, x)) is False


def test_contains_1():
    assert Order(x).contains(Order(x))
    assert Order(x).contains(Order(x**2))
    assert not Order(x**2).contains(Order(x))
    assert not Order(x).contains(Order(1/x))
    assert not Order(1/x).contains(Order(exp(1/x)))
    assert not Order(x).contains(Order(exp(1/x)))
    assert Order(1/x).contains(Order(x))
    assert Order(exp(1/x)).contains(Order(x))
    assert Order(exp(1/x)).contains(Order(1/x))
    assert Order(exp(1/x)).contains(Order(exp(1/x)))
    assert Order(exp(2/x)).contains(Order(exp(1/x)))
    assert not Order(exp(1/x)).contains(Order(exp(2/x)))


def test_contains_2():
    assert Order(x).contains(Order(y)) is None
    assert Order(x).contains(Order(y*x))
    assert Order(y*x).contains(Order(x))
    assert Order(y).contains(Order(x*y))
    assert Order(x).contains(Order(y**2*x))


def test_contains_3():
    assert Order(x*y**2).contains(Order(x**2*y)) is None
    assert Order(x**2*y).contains(Order(x*y**2)) is None


def test_contains_4():
    assert Order(sin(1/x**2)).contains(Order(cos(1/x**2))) is True
    assert Order(cos(1/x**2)).contains(Order(sin(1/x**2))) is True


def test_contains():
    assert Order(1, x) not in Order(1)
    assert Order(1) in Order(1, x)
    raises(TypeError, lambda: Order(x*y**2) in Order(x**2*y))


def test_add_1():
    assert Order(x + x) == Order(x)
    assert Order(3*x - 2*x**2) == Order(x)
    assert Order(1 + x) == Order(1, x)
    assert Order(1 + 1/x) == Order(1/x)
    # TODO : A better output for Order(log(x) + 1/log(x))
    # could be Order(log(x)). Currently Order for expressions
    # where all arguments would involve a log term would fall
    # in this category and outputs for these should be improved.
    assert Order(log(x) + 1/log(x)) == Order((log(x)**2 + 1)/log(x))
    assert Order(exp(1/x) + x) == Order(exp(1/x))
    assert Order(exp(1/x) + 1/x**20) == Order(exp(1/x))


def test_ln_args():
    assert O(log(x)) + O(log(2*x)) == O(log(x))
    assert O(log(x)) + O(log(x**3)) == O(log(x))
    assert O(log(x*y)) + O(log(x) + log(y)) == O(log(x) + log(y), x, y)


def test_multivar_0():
    assert Order(x*y).expr == x*y
    assert Order(x*y**2).expr == x*y**2
    assert Order(x*y, x).expr == x
    assert Order(x*y**2, y).expr == y**2
    assert Order(x*y*z).expr == x*y*z
    assert Order(x/y).expr == x/y
    assert Order(x*exp(1/y)).expr == x*exp(1/y)
    assert Order(exp(x)*exp(1/y)).expr == exp(x)*exp(1/y)


def test_multivar_0a():
    assert Order(exp(1/x)*exp(1/y)).expr == exp(1/x)*exp(1/y)


def test_multivar_1():
    assert Order(x + y).expr == x + y
    assert Order(x + 2*y).expr == x + y
    assert (Order(x + y) + x).expr == (x + y)
    assert (Order(x + y) + x**2) == Order(x + y)
    assert (Order(x + y) + 1/x) == 1/x + Order(x + y)
    assert Order(x**2 + y*x).expr == x**2 + y*x


def test_multivar_2():
    assert Order(x**2*y + y**2*x, x, y).expr == x**2*y + y**2*x


def test_multivar_mul_1():
    assert Order(x + y)*x == Order(x**2 + y*x, x, y)


def test_multivar_3():
    assert (Order(x) + Order(y)).args in [
        (Order(x), Order(y)),
        (Order(y), Order(x))]
    assert Order(x) + Order(y) + Order(x + y) == Order(x + y)
    assert (Order(x**2*y) + Order(y**2*x)).args in [
        (Order(x*y**2), Order(y*x**2)),
        (Order(y*x**2), Order(x*y**2))]
    assert (Order(x**2*y) + Order(y*x)) == Order(x*y)


def test_issue_3468():
    y = Symbol('y', negative=True)
    z = Symbol('z', complex=True)

    # check that Order does not modify assumptions about symbols
    Order(x)
    Order(y)
    Order(z)

    assert x.is_positive is None
    assert y.is_positive is False
    assert z.is_positive is None


def test_leading_order():
    assert (x + 1 + 1/x**5).extract_leading_order(x) == ((1/x**5, O(1/x**5)),)
    assert (1 + 1/x).extract_leading_order(x) == ((1/x, O(1/x)),)
    assert (1 + x).extract_leading_order(x) == ((1, O(1, x)),)
    assert (1 + x**2).extract_leading_order(x) == ((1, O(1, x)),)
    assert (2 + x**2).extract_leading_order(x) == ((2, O(1, x)),)
    assert (x + x**2).extract_leading_order(x) == ((x, O(x)),)


def test_leading_order2():
    assert set((2 + pi + x**2).extract_leading_order(x)) == {(pi, O(1, x)),
            (S(2), O(1, x))}
    assert set((2*x + pi*x + x**2).extract_leading_order(x)) == {(2*x, O(x)),
            (x*pi, O(x))}


def test_order_leadterm():
    assert O(x**2)._eval_as_leading_term(x) == O(x**2)


def test_order_symbols():
    e = x*y*sin(x)*Integral(x, (x, 1, 2))
    assert O(e) == O(x**2*y, x, y)
    assert O(e, x) == O(x**2)


def test_nan():
    assert O(nan) is nan
    assert not O(x).contains(nan)


def test_O1():
    assert O(1, x) * x == O(x)
    assert O(1, y) * x == O(1, y)


def test_getn():
    # other lines are tested incidentally by the suite
    assert O(x).getn() == 1
    assert O(x/log(x)).getn() == 1
    assert O(x**2/log(x)**2).getn() == 2
    assert O(x*log(x)).getn() == 1
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())


def test_diff():
    assert O(x**2).diff(x) == O(x)


def test_getO():
    assert (x).getO() is None
    assert (x).removeO() == x
    assert (O(x)).getO() == O(x)
    assert (O(x)).removeO() == 0
    assert (z + O(x) + O(y)).getO() == O(x) + O(y)
    assert (z + O(x) + O(y)).removeO() == z
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())


def test_leading_term():
    from sympy.functions.special.gamma_functions import digamma
    assert O(1/digamma(1/x)) == O(1/log(x))


def test_eval():
    assert Order(x).subs(Order(x), 1) == 1
    assert Order(x).subs(x, y) == Order(y)
    assert Order(x).subs(y, x) == Order(x)
    assert Order(x).subs(x, x + y) == Order(x + y, (x, -y))
    assert (O(1)**x).is_Pow


def test_issue_4279():
    a, b = symbols('a b')
    assert O(a, a, b) + O(1, a, b) == O(1, a, b)
    assert O(b, a, b) + O(1, a, b) == O(1, a, b)
    assert O(a + b, a, b) + O(1, a, b) == O(1, a, b)
    assert O(1, a, b) + O(a, a, b) == O(1, a, b)
    assert O(1, a, b) + O(b, a, b) == O(1, a, b)
    assert O(1, a, b) + O(a + b, a, b) == O(1, a, b)


def test_issue_4855():
    assert 1/O(1) != O(1)
    assert 1/O(x) != O(1/x)
    assert 1/O(x, (x, oo)) != O(1/x, (x, oo))

    f = Function('f')
    assert 1/O(f(x)) != O(1/x)


def test_order_conjugate_transpose():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    assert conjugate(Order(x)) == Order(conjugate(x))
    assert conjugate(Order(y)) == Order(conjugate(y))
    assert conjugate(Order(x**2)) == Order(conjugate(x)**2)
    assert conjugate(Order(y**2)) == Order(conjugate(y)**2)
    assert transpose(Order(x)) == Order(transpose(x))
    assert transpose(Order(y)) == Order(transpose(y))
    assert transpose(Order(x**2)) == Order(transpose(x)**2)
    assert transpose(Order(y**2)) == Order(transpose(y)**2)


def test_order_noncommutative():
    A = Symbol('A', commutative=False)
    assert Order(A + A*x, x) == Order(1, x)
    assert (A + A*x)*Order(x) == Order(x)
    assert (A*x)*Order(x) == Order(x**2, x)
    assert expand((1 + Order(x))*A*A*x) == A*A*x + Order(x**2, x)
    assert expand((A*A + Order(x))*x) == A*A*x + Order(x**2, x)
    assert expand((A + Order(x))*A*x) == A*A*x + Order(x**2, x)


def test_issue_6753():
    assert (1 + x**2)**10000*O(x) == O(x)


def test_order_at_infinity():
    assert Order(1 + x, (x, oo)) == Order(x, (x, oo))
    assert Order(3*x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo))*3 == Order(x, (x, oo))
    assert -28*Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (x, oo)) == Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (y, oo)) == Order(x, (x, oo), (y, oo))
    assert Order(3, (x, oo)) == Order(1, (x, oo))
    assert Order(x**2 + x + y, (x, oo)) == O(x**2, (x, oo))
    assert Order(x**2 + x + y, (y, oo)) == O(y, (y, oo))

    assert Order(2*x, (x, oo))*x == Order(x**2, (x, oo))
    assert Order(2*x, (x, oo))/x == Order(1, (x, oo))
    assert Order(2*x, (x, oo))*x*exp(1/x) == Order(x**2*exp(1/x), (x, oo))
    assert Order(2*x, (x, oo))*x*exp(1/x)/log(x)**3 == Order(x**2*exp(1/x)*log(x)**-3, (x, oo))

    assert Order(x, (x, oo)) + 1/x == 1/x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + 1 == 1 + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x == x + Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + x**2 == x**2 + Order(x, (x, oo))
    assert Order(1/x, (x, oo)) + 1/x**2 == 1/x**2 + Order(1/x, (x, oo)) == Order(1/x, (x, oo))
    assert Order(x, (x, oo)) + exp(1/x) == exp(1/x) + Order(x, (x, oo))

    assert Order(x, (x, oo))**2 == Order(x**2, (x, oo))

    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    assert Order(x, (x, oo)) + Order(x**-2, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1/x, (x, oo)) == Order(x, (x, oo))

    assert Order(x, (x, oo)) - Order(x, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(1, (x, oo)) == Order(x, (x, oo))
    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    assert Order(1/x, (x, oo)) + Order(1, (x, oo)) == Order(1, (x, oo))
    assert Order(x, (x, oo)) + Order(exp(1/x), (x, oo)) == Order(x, (x, oo))
    assert Order(x**3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(x**3, (x, oo))
    assert Order(x**-3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(exp(2/x), (x, oo))

    # issue 7207
    assert Order(exp(x), (x, oo)).expr == Order(2*exp(x), (x, oo)).expr == exp(x)
    assert Order(y**x, (x, oo)).expr == Order(2*y**x, (x, oo)).expr == exp(x*log(y))

    # issue 19545
    assert Order(1/x - 3/(3*x + 2), (x, oo)).expr == x**(-2)

def test_mixing_order_at_zero_and_infinity():
    assert (Order(x, (x, 0)) + Order(x, (x, oo))).is_Add
    assert Order(x, (x, 0)) + Order(x, (x, oo)) == Order(x, (x, oo)) + Order(x, (x, 0))
    assert Order(Order(x, (x, oo))) == Order(x, (x, oo))

    # not supported (yet)
    raises(NotImplementedError, lambda: Order(x, (x, 0))*Order(x, (x, oo)))
    raises(NotImplementedError, lambda: Order(x, (x, oo))*Order(x, (x, 0)))
    raises(NotImplementedError, lambda: Order(Order(x, (x, oo)), y))
    raises(NotImplementedError, lambda: Order(Order(x), (x, oo)))


def test_order_at_some_point():
    assert Order(x, (x, 1)) == Order(1, (x, 1))
    assert Order(2*x - 2, (x, 1)) == Order(x - 1, (x, 1))
    assert Order(-x + 1, (x, 1)) == Order(x - 1, (x, 1))
    assert Order(x - 1, (x, 1))**2 == Order((x - 1)**2, (x, 1))
    assert Order(x - 2, (x, 2)) - O(x - 2, (x, 2)) == Order(x - 2, (x, 2))


def test_order_subs_limits():
    # issue 3333
    assert (1 + Order(x)).subs(x, 1/x) == 1 + Order(1/x, (x, oo))
    assert (1 + Order(x)).limit(x, 0) == 1
    # issue 5769
    assert ((x + Order(x**2))/x).limit(x, 0) == 1

    assert Order(x**2).subs(x, y - 1) == Order((y - 1)**2, (y, 1))
    assert Order(10*x**2, (x, 2)).subs(x, y - 1) == Order(1, (y, 3))


def test_issue_9351():
    assert exp(x).series(x, 10, 1) == exp(10) + Order(x - 10, (x, 10))


def test_issue_9192():
    assert O(1)*O(1) == O(1)
    assert O(1)**O(1) == O(1)


def test_issue_9910():
    assert O(x*log(x) + sin(x), (x, oo)) == O(x*log(x), (x, oo))


def test_performance_of_adding_order():
    l = [x**i for i in range(1000)]
    l.append(O(x**1001))
    assert Add(*l).subs(x,1) == O(1)

def test_issue_14622():
    assert (x**(-4) + x**(-3) + x**(-1) + O(x**(-6), (x, oo))).as_numer_denom() == (
        x**4 + x**5 + x**7 + O(x**2, (x, oo)), x**8)
    assert (x**3 + O(x**2, (x, oo))).is_Add
    assert O(x**2, (x, oo)).contains(x**3) is False
    assert O(x, (x, oo)).contains(O(x, (x, 0))) is None
    assert O(x, (x, 0)).contains(O(x, (x, oo))) is None
    raises(NotImplementedError, lambda: O(x**3).contains(x**w))


def test_issue_15539():
    assert O(1/x**2 + 1/x**4, (x, -oo)) == O(1/x**2, (x, -oo))
    assert O(1/x**4 + exp(x), (x, -oo)) == O(1/x**4, (x, -oo))
    assert O(1/x**4 + exp(-x), (x, -oo)) == O(exp(-x), (x, -oo))
    assert O(1/x, (x, oo)).subs(x, -x) == O(-1/x, (x, -oo))

def test_issue_18606():
    assert unchanged(Order, 0)


def test_issue_22165():
    assert O(log(x)).contains(2)


def test_issue_23231():
    # This test checks Order for expressions having
    # arguments containing variables in exponents/powers.
    assert O(x**x + 2**x, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(x**x + x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(x**x + 1/x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
    assert O(2**x + 3**x , (x, oo)) == O(exp(x*log(3)), (x, oo))


def test_issue_9917():
    assert O(x*sin(x) + 1, (x, oo)) == O(x, (x, oo))
