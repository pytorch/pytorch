from sympy.core.symbol import symbols, Symbol
from sympy.functions import Max
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.plotting.intervalmath.interval_arithmetic import \
    interval, intervalMembership


# Tests for exception handling in experimental_lambdify
def test_experimental_lambify():
    x = Symbol('x')
    f = experimental_lambdify([x], Max(x, 5))
    # XXX should f be tested? If f(2) is attempted, an
    # error is raised because a complex produced during wrapping of the arg
    # is being compared with an int.
    assert Max(2, 5) == 5
    assert Max(5, 7) == 7

    x = Symbol('x-3')
    f = experimental_lambdify([x], x + 1)
    assert f(1) == 2


def test_composite_boolean_region():
    x, y = symbols('x y')

    r1 = (x - 1)**2 + y**2 < 2
    r2 = (x + 1)**2 + y**2 < 2

    f = experimental_lambdify((x, y), r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), r1 | r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), ~r1 & r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(True, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(False, True)

    f = experimental_lambdify((x, y), ~r1 & ~r2)
    a = (interval(-0.1, 0.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-1.1, -0.9), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(0.9, 1.1), interval(-0.1, 0.1))
    assert f(*a) == intervalMembership(False, True)
    a = (interval(-0.1, 0.1), interval(1.9, 2.1))
    assert f(*a) == intervalMembership(True, True)
