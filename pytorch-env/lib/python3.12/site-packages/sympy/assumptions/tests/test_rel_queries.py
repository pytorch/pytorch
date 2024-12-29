from sympy.assumptions.lra_satask import lra_satask
from sympy.logic.algorithms.lra_theory import UnhandledInput
from sympy.assumptions.ask import Q, ask

from sympy.core import symbols, Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.numbers import I

from sympy.testing.pytest import raises, XFAIL
x, y, z = symbols("x y z", real=True)

def test_lra_satask():
    im = Symbol('im', imaginary=True)

    # test preprocessing of unequalities is working correctly
    assert lra_satask(Q.eq(x, 1), ~Q.ne(x, 0)) is False
    assert lra_satask(Q.eq(x, 0), ~Q.ne(x, 0)) is True
    assert lra_satask(~Q.ne(x, 0), Q.eq(x, 0)) is True
    assert lra_satask(~Q.eq(x, 0), Q.eq(x, 0)) is False
    assert lra_satask(Q.ne(x, 0), Q.eq(x, 0)) is False

    # basic tests
    assert lra_satask(Q.ne(x, x)) is False
    assert lra_satask(Q.eq(x, x)) is True
    assert lra_satask(Q.gt(x, 0), Q.gt(x, 1)) is True

    # check that True/False are handled
    assert lra_satask(Q.gt(x, 0), True) is None
    assert raises(ValueError, lambda: lra_satask(Q.gt(x, 0), False))

    # check imaginary numbers are correctly handled
    # (im * I).is_real returns True so this is an edge case
    raises(UnhandledInput, lambda: lra_satask(Q.gt(im * I, 0), Q.gt(im * I, 0)))

    # check matrix inputs
    X = MatrixSymbol("X", 2, 2)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(X, 2) & Q.gt(X, 3)))


def test_old_assumptions():
    # test unhandled old assumptions
    w = symbols("w")
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", rational=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", odd=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", even=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", prime=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", composite=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", integer=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", integer=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))

    # test handled
    w = symbols("w", positive=True, real=True)
    assert lra_satask(Q.le(w, 0)) is False
    assert lra_satask(Q.gt(w, 0)) is True
    w = symbols("w", negative=True, real=True)
    assert lra_satask(Q.lt(w, 0)) is True
    assert lra_satask(Q.ge(w, 0)) is False
    w = symbols("w", zero=True, real=True)
    assert lra_satask(Q.eq(w, 0)) is True
    assert lra_satask(Q.ne(w, 0)) is False
    w = symbols("w", nonzero=True, real=True)
    assert lra_satask(Q.ne(w, 0)) is True
    assert lra_satask(Q.eq(w, 1)) is None
    w = symbols("w", nonpositive=True, real=True)
    assert lra_satask(Q.le(w, 0)) is True
    assert lra_satask(Q.gt(w, 0)) is False
    w = symbols("w", nonnegative=True, real=True)
    assert lra_satask(Q.ge(w, 0)) is True
    assert lra_satask(Q.lt(w, 0)) is False


def test_rel_queries():
    assert ask(Q.lt(x, 2) & Q.gt(x, 3)) is False
    assert ask(Q.positive(x - z), (x > y) & (y > z)) is True
    assert ask(x + y > 2, (x < 0) & (y <0)) is False
    assert ask(x > z, (x > y) & (y > z)) is True


def test_unhandled_queries():
    X = MatrixSymbol("X", 2, 2)
    assert ask(Q.lt(X, 2) & Q.gt(X, 3)) is None


def test_all_pred():
    # test usable pred
    assert lra_satask(Q.extended_positive(x), (x > 2)) is True
    assert lra_satask(Q.positive_infinite(x)) is False
    assert lra_satask(Q.negative_infinite(x)) is False

    # test disallowed pred
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.prime(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.composite(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.odd(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.even(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.integer(x)))


def test_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # Transitivity
    # If a <= b and b <= c, then a <= c.
    assert ask(a <= c, (a <= b) & (b <= c)) is True
    # If a <= b and b < c, then a < c.
    assert ask(a < c, (a <= b) & (b < c)) is True
    # If a < b and b <= c, then a < c.
    assert ask(a < c, (a < b) & (b <= c)) is True

    # Addition and subtraction
    # If a <= b, then a + c <= b + c and a - c <= b - c.
    assert ask(a + c <= b + c, a <= b) is True
    assert ask(a - c <= b - c, a <= b) is True


@XFAIL
def test_failing_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # Multiplication and division
    # If a <= b and c > 0, then ac <= bc and a/c <= b/c. (True for non-zero c)
    assert ask(a*c <= b*c, (a <= b) & (c > 0) & ~ Q.zero(c)) is True
    assert ask(a/c <= b/c, (a <= b) & (c > 0) & ~ Q.zero(c)) is True
    # If a <= b and c < 0, then ac >= bc and a/c >= b/c. (True for non-zero c)
    assert ask(a*c >= b*c, (a <= b) & (c < 0) & ~ Q.zero(c)) is True
    assert ask(a/c >= b/c, (a <= b) & (c < 0) & ~ Q.zero(c)) is True

    # Additive inverse
    # If a <= b, then -a >= -b.
    assert ask(-a >= -b, a <= b) is True

    # Multiplicative inverse
    # For a, b that are both negative or both positive:
    # If a <= b, then 1/a >= 1/b .
    assert ask(1/a >= 1/b, (a <= b) & Q.positive(x) & Q.positive(b)) is True
    assert ask(1/a >= 1/b, (a <= b) & Q.negative(x) & Q.negative(b)) is True


def test_equality():
    # test symetry and reflexivity
    assert ask(Q.eq(x, x)) is True
    assert ask(Q.eq(y, x), Q.eq(x, y)) is True
    assert ask(Q.eq(y, x), ~Q.eq(z, z) | Q.eq(x, y)) is True

    # test transitivity
    assert ask(Q.eq(x,z), Q.eq(x,y) & Q.eq(y,z)) is True


@XFAIL
def test_equality_failing():
    # Note that implementing the substitution property of equality
    # most likely requires a redesign of the new assumptions.
    # See issue #25485 for why this is the case and general ideas
    # about how things could be redesigned.

    # test substitution property
    assert ask(Q.prime(x), Q.eq(x, y) & Q.prime(y)) is True
    assert ask(Q.real(x), Q.eq(x, y) & Q.real(y)) is True
    assert ask(Q.imaginary(x), Q.eq(x, y) & Q.imaginary(y)) is True
