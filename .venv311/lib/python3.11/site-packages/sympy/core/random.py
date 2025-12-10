"""
When you need to use random numbers in SymPy library code, import from here
so there is only one generator working for SymPy. Imports from here should
behave the same as if they were being imported from Python's random module.
But only the routines currently used in SymPy are included here. To use others
import ``rng`` and access the method directly. For example, to capture the
current state of the generator use ``rng.getstate()``.

There is intentionally no Random to import from here. If you want
to control the state of the generator, import ``seed`` and call it
with or without an argument to set the state.

Examples
========

>>> from sympy.core.random import random, seed
>>> assert random() < 1
>>> seed(1); a = random()
>>> b = random()
>>> seed(1); c = random()
>>> assert a == c
>>> assert a != b  # remote possibility this will fail

"""
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int

import random as _random
rng = _random.Random()

choice = rng.choice
random = rng.random
randint = rng.randint
randrange = rng.randrange
sample = rng.sample
# seed = rng.seed
shuffle = rng.shuffle
uniform = rng.uniform

_assumptions_rng = _random.Random()
_assumptions_shuffle = _assumptions_rng.shuffle


def seed(a=None, version=2):
    rng.seed(a=a, version=version)
    _assumptions_rng.seed(a=a, version=version)


def random_complex_number(a=2, b=-1, c=3, d=1, rational=False, tolerance=None):
    """
    Return a random complex number.

    To reduce chance of hitting branch cuts or anything, we guarantee
    b <= Im z <= d, a <= Re z <= c

    When rational is True, a rational approximation to a random number
    is obtained within specified tolerance, if any.
    """
    from sympy.core.numbers import I
    from sympy.simplify.simplify import nsimplify
    A, B = uniform(a, c), uniform(b, d)
    if not rational:
        return A + I*B
    return (nsimplify(A, rational=True, tolerance=tolerance) +
        I*nsimplify(B, rational=True, tolerance=tolerance))


def verify_numerically(f, g, z=None, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that f and g agree when evaluated in the argument z.

    If z is None, all symbols will be tested. This routine does not test
    whether there are Floats present with precision higher than 15 digits
    so if there are, your results may not be what you expect due to round-
    off errors.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.abc import x
    >>> from sympy.core.random import verify_numerically as tn
    >>> tn(sin(x)**2 + cos(x)**2, 1, x)
    True
    """
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.core.numbers import comp
    f, g = (sympify(i) for i in (f, g))
    if z is None:
        z = f.free_symbols | g.free_symbols
    elif isinstance(z, Symbol):
        z = [z]
    reps = list(zip(z, [random_complex_number(a, b, c, d) for _ in z]))
    z1 = f.subs(reps).n()
    z2 = g.subs(reps).n()
    return comp(z1, z2, tol)


def test_derivative_numerically(f, z, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that the symbolically computed derivative of f
    with respect to z is correct.

    This routine does not test whether there are Floats present with
    precision higher than 15 digits so if there are, your results may
    not be what you expect due to round-off errors.

    Examples
    ========

    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> from sympy.core.random import test_derivative_numerically as td
    >>> td(sin(x), x)
    True
    """
    from sympy.core.numbers import comp
    from sympy.core.function import Derivative
    z0 = random_complex_number(a, b, c, d)
    f1 = f.diff(z).subs(z, z0)
    f2 = Derivative(f, z).doit_numerically(z0)
    return comp(f1.n(), f2.n(), tol)


def _randrange(seed=None):
    """Return a randrange generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randrange
    >>> rr = _randrange()
    >>> rr(1000) # doctest: +SKIP
    999
    >>> rr = _randrange(3)
    >>> rr(1000) # doctest: +SKIP
    238
    >>> rr = _randrange([0, 5, 1, 3, 4])
    >>> rr(3), rr(3)
    (0, 1)
    """
    if seed is None:
        return randrange
    elif isinstance(seed, int):
        rng.seed(seed)
        return randrange
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b=None, seq=seed):
            if b is None:
                a, b = 0, a
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 1:
                raise ValueError('_randrange got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randrange sequence was too short')
            if a <= x < b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randrange got an unexpected seed')


def _randint(seed=None):
    """Return a randint generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randint
    >>> ri = _randint()
    >>> ri(1, 1000) # doctest: +SKIP
    999
    >>> ri = _randint(3)
    >>> ri(1, 1000) # doctest: +SKIP
    238
    >>> ri = _randint([0, 5, 1, 2, 4])
    >>> ri(1, 3), ri(1, 3)
    (1, 2)
    """
    if seed is None:
        return randint
    elif isinstance(seed, int):
        rng.seed(seed)
        return randint
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b, seq=seed):
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 0:
                raise ValueError('_randint got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randint sequence was too short')
            if a <= x <= b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randint got an unexpected seed')
