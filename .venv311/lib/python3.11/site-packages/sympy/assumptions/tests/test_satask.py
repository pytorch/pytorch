from sympy.assumptions.ask import Q
from sympy.assumptions.assume import assuming
from sympy.core.numbers import (I, pi)
from sympy.core.relational import (Eq, Gt)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import Implies
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.satask import (satask, extract_predargs,
    get_relevant_clsfacts)

from sympy.testing.pytest import raises, XFAIL


x, y, z = symbols('x y z')


def test_satask():
    # No relevant facts
    assert satask(Q.real(x), Q.real(x)) is True
    assert satask(Q.real(x), ~Q.real(x)) is False
    assert satask(Q.real(x)) is None

    assert satask(Q.real(x), Q.positive(x)) is True
    assert satask(Q.positive(x), Q.real(x)) is None
    assert satask(Q.real(x), ~Q.positive(x)) is None
    assert satask(Q.positive(x), ~Q.real(x)) is False

    raises(ValueError, lambda: satask(Q.real(x), Q.real(x) & ~Q.real(x)))

    with assuming(Q.positive(x)):
        assert satask(Q.real(x)) is True
        assert satask(~Q.positive(x)) is False
        raises(ValueError, lambda: satask(Q.real(x), ~Q.positive(x)))

    assert satask(Q.zero(x), Q.nonzero(x)) is False
    assert satask(Q.positive(x), Q.zero(x)) is False
    assert satask(Q.real(x), Q.zero(x)) is True
    assert satask(Q.zero(x), Q.zero(x*y)) is None
    assert satask(Q.zero(x*y), Q.zero(x))


def test_zero():
    """
    Everything in this test doesn't work with the ask handlers, and most
    things would be very difficult or impossible to make work under that
    model.

    """
    assert satask(Q.zero(x) | Q.zero(y), Q.zero(x*y)) is True
    assert satask(Q.zero(x*y), Q.zero(x) | Q.zero(y)) is True

    assert satask(Implies(Q.zero(x), Q.zero(x*y))) is True

    # This one in particular requires computing the fixed-point of the
    # relevant facts, because going from Q.nonzero(x*y) -> ~Q.zero(x*y) and
    # Q.zero(x*y) -> Equivalent(Q.zero(x*y), Q.zero(x) | Q.zero(y)) takes two
    # steps.
    assert satask(Q.zero(x) | Q.zero(y), Q.nonzero(x*y)) is False

    assert satask(Q.zero(x), Q.zero(x**2)) is True


def test_zero_positive():
    assert satask(Q.zero(x + y), Q.positive(x) & Q.positive(y)) is False
    assert satask(Q.positive(x) & Q.positive(y), Q.zero(x + y)) is False
    assert satask(Q.nonzero(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(x) & Q.positive(y), Q.nonzero(x + y)) is None

    # This one requires several levels of forward chaining
    assert satask(Q.zero(x*(x + y)), Q.positive(x) & Q.positive(y)) is False

    assert satask(Q.positive(pi*x*y + 1), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.positive(pi*x*y - 5), Q.positive(x) & Q.positive(y)) is None


def test_zero_pow():
    assert satask(Q.zero(x**y), Q.zero(x) & Q.positive(y)) is True
    assert satask(Q.zero(x**y), Q.nonzero(x) & Q.zero(y)) is False

    assert satask(Q.zero(x), Q.zero(x**y)) is True

    assert satask(Q.zero(x**y), Q.zero(x)) is None


@XFAIL
# Requires correct Q.square calculation first
def test_invertible():
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    assert satask(Q.invertible(A*B), Q.invertible(A) & Q.invertible(B)) is True
    assert satask(Q.invertible(A), Q.invertible(A*B)) is True
    assert satask(Q.invertible(A) & Q.invertible(B), Q.invertible(A*B)) is True


def test_prime():
    assert satask(Q.prime(5)) is True
    assert satask(Q.prime(6)) is False
    assert satask(Q.prime(-5)) is False

    assert satask(Q.prime(x*y), Q.integer(x) & Q.integer(y)) is None
    assert satask(Q.prime(x*y), Q.prime(x) & Q.prime(y)) is False


def test_old_assump():
    assert satask(Q.positive(1)) is True
    assert satask(Q.positive(-1)) is False
    assert satask(Q.positive(0)) is False
    assert satask(Q.positive(I)) is False
    assert satask(Q.positive(pi)) is True

    assert satask(Q.negative(1)) is False
    assert satask(Q.negative(-1)) is True
    assert satask(Q.negative(0)) is False
    assert satask(Q.negative(I)) is False
    assert satask(Q.negative(pi)) is False

    assert satask(Q.zero(1)) is False
    assert satask(Q.zero(-1)) is False
    assert satask(Q.zero(0)) is True
    assert satask(Q.zero(I)) is False
    assert satask(Q.zero(pi)) is False

    assert satask(Q.nonzero(1)) is True
    assert satask(Q.nonzero(-1)) is True
    assert satask(Q.nonzero(0)) is False
    assert satask(Q.nonzero(I)) is False
    assert satask(Q.nonzero(pi)) is True

    assert satask(Q.nonpositive(1)) is False
    assert satask(Q.nonpositive(-1)) is True
    assert satask(Q.nonpositive(0)) is True
    assert satask(Q.nonpositive(I)) is False
    assert satask(Q.nonpositive(pi)) is False

    assert satask(Q.nonnegative(1)) is True
    assert satask(Q.nonnegative(-1)) is False
    assert satask(Q.nonnegative(0)) is True
    assert satask(Q.nonnegative(I)) is False
    assert satask(Q.nonnegative(pi)) is True


def test_rational_irrational():
    assert satask(Q.irrational(2)) is False
    assert satask(Q.rational(2)) is True
    assert satask(Q.irrational(pi)) is True
    assert satask(Q.rational(pi)) is False
    assert satask(Q.irrational(I)) is False
    assert satask(Q.rational(I)) is False

    assert satask(Q.irrational(x*y*z), Q.irrational(x) & Q.irrational(y) &
        Q.rational(z)) is None
    assert satask(Q.irrational(x*y*z), Q.irrational(x) & Q.rational(y) &
        Q.rational(z)) is True
    assert satask(Q.irrational(pi*x*y), Q.rational(x) & Q.rational(y)) is True

    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.irrational(y) &
        Q.rational(z)) is None
    assert satask(Q.irrational(x + y + z), Q.irrational(x) & Q.rational(y) &
        Q.rational(z)) is True
    assert satask(Q.irrational(pi + x + y), Q.rational(x) & Q.rational(y)) is True

    assert satask(Q.irrational(x*y*z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is False
    assert satask(Q.rational(x*y*z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is True

    assert satask(Q.irrational(x + y + z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is False
    assert satask(Q.rational(x + y + z), Q.rational(x) & Q.rational(y) &
        Q.rational(z)) is True


def test_even_satask():
    assert satask(Q.even(2)) is True
    assert satask(Q.even(3)) is False

    assert satask(Q.even(x*y), Q.even(x) & Q.odd(y)) is True
    assert satask(Q.even(x*y), Q.even(x) & Q.integer(y)) is True
    assert satask(Q.even(x*y), Q.even(x) & Q.even(y)) is True
    assert satask(Q.even(x*y), Q.odd(x) & Q.odd(y)) is False
    assert satask(Q.even(x*y), Q.even(x)) is None
    assert satask(Q.even(x*y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.even(x*y), Q.odd(x) & Q.odd(y)) is False

    assert satask(Q.even(abs(x)), Q.even(x)) is True
    assert satask(Q.even(abs(x)), Q.odd(x)) is False
    assert satask(Q.even(x), Q.even(abs(x))) is None # x could be complex


def test_odd_satask():
    assert satask(Q.odd(2)) is False
    assert satask(Q.odd(3)) is True

    assert satask(Q.odd(x*y), Q.even(x) & Q.odd(y)) is False
    assert satask(Q.odd(x*y), Q.even(x) & Q.integer(y)) is False
    assert satask(Q.odd(x*y), Q.even(x) & Q.even(y)) is False
    assert satask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) is True
    assert satask(Q.odd(x*y), Q.even(x)) is None
    assert satask(Q.odd(x*y), Q.odd(x) & Q.integer(y)) is None
    assert satask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) is True

    assert satask(Q.odd(abs(x)), Q.even(x)) is False
    assert satask(Q.odd(abs(x)), Q.odd(x)) is True
    assert satask(Q.odd(x), Q.odd(abs(x))) is None # x could be complex


def test_integer():
    assert satask(Q.integer(1)) is True
    assert satask(Q.integer(S.Half)) is False

    assert satask(Q.integer(x + y), Q.integer(x) & Q.integer(y)) is True
    assert satask(Q.integer(x + y), Q.integer(x)) is None

    assert satask(Q.integer(x + y), Q.integer(x) & ~Q.integer(y)) is False
    assert satask(Q.integer(x + y + z), Q.integer(x) & Q.integer(y) &
        ~Q.integer(z)) is False
    assert satask(Q.integer(x + y + z), Q.integer(x) & ~Q.integer(y) &
        ~Q.integer(z)) is None
    assert satask(Q.integer(x + y + z), Q.integer(x) & ~Q.integer(y)) is None
    assert satask(Q.integer(x + y), Q.integer(x) & Q.irrational(y)) is False

    assert satask(Q.integer(x*y), Q.integer(x) & Q.integer(y)) is True
    assert satask(Q.integer(x*y), Q.integer(x)) is None

    assert satask(Q.integer(x*y), Q.integer(x) & ~Q.integer(y)) is None
    assert satask(Q.integer(x*y), Q.integer(x) & ~Q.rational(y)) is False
    assert satask(Q.integer(x*y*z), Q.integer(x) & Q.integer(y) &
        ~Q.rational(z)) is False
    assert satask(Q.integer(x*y*z), Q.integer(x) & ~Q.rational(y) &
        ~Q.rational(z)) is None
    assert satask(Q.integer(x*y*z), Q.integer(x) & ~Q.rational(y)) is None
    assert satask(Q.integer(x*y), Q.integer(x) & Q.irrational(y)) is False


def test_abs():
    assert satask(Q.nonnegative(abs(x))) is True
    assert satask(Q.positive(abs(x)), ~Q.zero(x)) is True
    assert satask(Q.zero(x), ~Q.zero(abs(x))) is False
    assert satask(Q.zero(x), Q.zero(abs(x))) is True
    assert satask(Q.nonzero(x), ~Q.zero(abs(x))) is None # x could be complex
    assert satask(Q.zero(abs(x)), Q.zero(x)) is True


def test_imaginary():
    assert satask(Q.imaginary(2*I)) is True
    assert satask(Q.imaginary(x*y), Q.imaginary(x)) is None
    assert satask(Q.imaginary(x*y), Q.imaginary(x) & Q.real(y)) is True
    assert satask(Q.imaginary(x), Q.real(x)) is False
    assert satask(Q.imaginary(1)) is False
    assert satask(Q.imaginary(x*y), Q.real(x) & Q.real(y)) is False
    assert satask(Q.imaginary(x + y), Q.real(x) & Q.real(y)) is False


def test_real():
    assert satask(Q.real(x*y), Q.real(x) & Q.real(y)) is True
    assert satask(Q.real(x + y), Q.real(x) & Q.real(y)) is True
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y)) is None
    assert satask(Q.real(x*y*z), Q.real(x) & Q.real(y) & Q.imaginary(z)) is False
    assert satask(Q.real(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    assert satask(Q.real(x + y + z), Q.real(x) & Q.real(y)) is None


def test_pos_neg():
    assert satask(~Q.positive(x), Q.negative(x)) is True
    assert satask(~Q.negative(x), Q.positive(x)) is True
    assert satask(Q.positive(x + y), Q.positive(x) & Q.positive(y)) is True
    assert satask(Q.negative(x + y), Q.negative(x) & Q.negative(y)) is True
    assert satask(Q.positive(x + y), Q.negative(x) & Q.negative(y)) is False
    assert satask(Q.negative(x + y), Q.positive(x) & Q.positive(y)) is False


def test_pow_pos_neg():
    assert satask(Q.nonnegative(x**2), Q.positive(x)) is True
    assert satask(Q.nonpositive(x**2), Q.positive(x)) is False
    assert satask(Q.positive(x**2), Q.positive(x)) is True
    assert satask(Q.negative(x**2), Q.positive(x)) is False
    assert satask(Q.real(x**2), Q.positive(x)) is True

    assert satask(Q.nonnegative(x**2), Q.negative(x)) is True
    assert satask(Q.nonpositive(x**2), Q.negative(x)) is False
    assert satask(Q.positive(x**2), Q.negative(x)) is True
    assert satask(Q.negative(x**2), Q.negative(x)) is False
    assert satask(Q.real(x**2), Q.negative(x)) is True

    assert satask(Q.nonnegative(x**2), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x**2), Q.nonnegative(x)) is None
    assert satask(Q.positive(x**2), Q.nonnegative(x)) is None
    assert satask(Q.negative(x**2), Q.nonnegative(x)) is False
    assert satask(Q.real(x**2), Q.nonnegative(x)) is True

    assert satask(Q.nonnegative(x**2), Q.nonpositive(x)) is True
    assert satask(Q.nonpositive(x**2), Q.nonpositive(x)) is None
    assert satask(Q.positive(x**2), Q.nonpositive(x)) is None
    assert satask(Q.negative(x**2), Q.nonpositive(x)) is False
    assert satask(Q.real(x**2), Q.nonpositive(x)) is True

    assert satask(Q.nonnegative(x**3), Q.positive(x)) is True
    assert satask(Q.nonpositive(x**3), Q.positive(x)) is False
    assert satask(Q.positive(x**3), Q.positive(x)) is True
    assert satask(Q.negative(x**3), Q.positive(x)) is False
    assert satask(Q.real(x**3), Q.positive(x)) is True

    assert satask(Q.nonnegative(x**3), Q.negative(x)) is False
    assert satask(Q.nonpositive(x**3), Q.negative(x)) is True
    assert satask(Q.positive(x**3), Q.negative(x)) is False
    assert satask(Q.negative(x**3), Q.negative(x)) is True
    assert satask(Q.real(x**3), Q.negative(x)) is True

    assert satask(Q.nonnegative(x**3), Q.nonnegative(x)) is True
    assert satask(Q.nonpositive(x**3), Q.nonnegative(x)) is None
    assert satask(Q.positive(x**3), Q.nonnegative(x)) is None
    assert satask(Q.negative(x**3), Q.nonnegative(x)) is False
    assert satask(Q.real(x**3), Q.nonnegative(x)) is True

    assert satask(Q.nonnegative(x**3), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x**3), Q.nonpositive(x)) is True
    assert satask(Q.positive(x**3), Q.nonpositive(x)) is False
    assert satask(Q.negative(x**3), Q.nonpositive(x)) is None
    assert satask(Q.real(x**3), Q.nonpositive(x)) is True

    # If x is zero, x**negative is not real.
    assert satask(Q.nonnegative(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.nonpositive(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.positive(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.negative(x**-2), Q.nonpositive(x)) is None
    assert satask(Q.real(x**-2), Q.nonpositive(x)) is None

    # We could deduce things for negative powers if x is nonzero, but it
    # isn't implemented yet.


def test_prime_composite():
    assert satask(Q.prime(x), Q.composite(x)) is False
    assert satask(Q.composite(x), Q.prime(x)) is False
    assert satask(Q.composite(x), ~Q.prime(x)) is None
    assert satask(Q.prime(x), ~Q.composite(x)) is None
    # since 1 is neither prime nor composite the following should hold
    assert satask(Q.prime(x), Q.integer(x) & Q.positive(x) & ~Q.composite(x)) is None
    assert satask(Q.prime(2)) is True
    assert satask(Q.prime(4)) is False
    assert satask(Q.prime(1)) is False
    assert satask(Q.composite(1)) is False


def test_extract_predargs():
    props = CNF.from_prop(Q.zero(Abs(x*y)) & Q.zero(x*y))
    assump = CNF.from_prop(Q.zero(x))
    context = CNF.from_prop(Q.zero(y))
    assert extract_predargs(props) == {Abs(x*y), x*y}
    assert extract_predargs(props, assump) == {Abs(x*y), x*y, x}
    assert extract_predargs(props, assump, context) == {Abs(x*y), x*y, x, y}

    props = CNF.from_prop(Eq(x, y))
    assump = CNF.from_prop(Gt(y, z))
    assert extract_predargs(props, assump) == {x, y, z}


def test_get_relevant_clsfacts():
    exprs = {Abs(x*y)}
    exprs, facts = get_relevant_clsfacts(exprs)
    assert exprs == {x*y}
    assert facts.clauses == \
        {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),
        frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),
        frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),
        frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),
        frozenset({Literal(Q.even(Abs(x*y)), False),
                    Literal(Q.odd(Abs(x*y)), False),
                    Literal(Q.odd(x*y), True)}),
        frozenset({Literal(Q.even(Abs(x*y)), False),
                    Literal(Q.even(x*y), True),
                    Literal(Q.odd(Abs(x*y)), False)}),
        frozenset({Literal(Q.positive(Abs(x*y)), False),
                    Literal(Q.zero(Abs(x*y)), False)})}
