from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'
    handler = Dispatcher(
        "IntegerHandler",
        doc=("Handler for Q.integer.\n\n"
        "Test that an expression belongs to the field of integer numbers.")
    )


class NonIntegerPredicate(Predicate):
    """
    Non-integer extended real predicate.
    """
    name = 'noninteger'
    handler = Dispatcher(
        "NonIntegerHandler",
        doc=("Handler for Q.noninteger.\n\n"
        "Test that an expression is a non-integer extended real number.")
    )


class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'
    handler = Dispatcher(
        "RationalHandler",
        doc=("Handler for Q.rational.\n\n"
        "Test that an expression belongs to the field of rational numbers.")
    )


class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'
    handler = Dispatcher(
        "IrrationalHandler",
        doc=("Handler for Q.irrational.\n\n"
        "Test that an expression is irrational numbers.")
    )


class RealPredicate(Predicate):
    r"""
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\infty, \infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.

    - Every real number is also complex.

    - Every real number is finite.

    - Every real number is either rational or irrational.

    - Every real number is either algebraic or transcendental.

    - The facts ``Q.negative``, ``Q.zero``, ``Q.positive``,
        ``Q.nonnegative``, ``Q.nonpositive``, ``Q.nonzero``,
        ``Q.integer``, ``Q.rational``, and ``Q.irrational`` all imply
        ``Q.real``, as do all facts that imply those facts.

    - The facts ``Q.algebraic``, and ``Q.transcendental`` do not imply
        ``Q.real``; they imply ``Q.complex``. An algebraic or
        transcendental number may or may not be real.

    - The "non" facts (i.e., ``Q.nonnegative``, ``Q.nonzero``,
        ``Q.nonpositive`` and ``Q.noninteger``) are not equivalent to
        not the fact, but rather, not the fact *and* ``Q.real``.
        For example, ``Q.nonnegative`` means ``~Q.negative & Q.real``.
        So for example, ``I`` is not nonnegative, nonzero, or
        nonpositive.

    Examples
    ========

    >>> from sympy import Q, ask, symbols
    >>> x = symbols('x')
    >>> ask(Q.real(x), Q.positive(x))
    True
    >>> ask(Q.real(0))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Real_number

    """
    name = 'real'
    handler = Dispatcher(
        "RealHandler",
        doc=("Handler for Q.real.\n\n"
        "Test that an expression belongs to the field of real numbers.")
    )


class ExtendedRealPredicate(Predicate):
    r"""
    Extended real predicate.

    Explanation
    ===========

    ``Q.extended_real(x)`` is true iff ``x`` is a real number or
    `\{-\infty, \infty\}`.

    See documentation of ``Q.real`` for more information about related
    facts.

    Examples
    ========

    >>> from sympy import ask, Q, oo, I
    >>> ask(Q.extended_real(1))
    True
    >>> ask(Q.extended_real(I))
    False
    >>> ask(Q.extended_real(oo))
    True

    """
    name = 'extended_real'
    handler = Dispatcher(
        "ExtendedRealHandler",
        doc=("Handler for Q.extended_real.\n\n"
        "Test that an expression belongs to the field of extended real\n"
        "numbers, that is real numbers union {Infinity, -Infinity}.")
    )


class HermitianPredicate(Predicate):
    """
    Hermitian predicate.

    Explanation
    ===========

    ``ask(Q.hermitian(x))`` is true iff ``x`` belongs to the set of
    Hermitian operators.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'hermitian'
    handler = Dispatcher(
        "HermitianHandler",
        doc=("Handler for Q.hermitian.\n\n"
        "Test that an expression belongs to the field of Hermitian operators.")
    )


class ComplexPredicate(Predicate):
    """
    Complex number predicate.

    Explanation
    ===========

    ``Q.complex(x)`` is true iff ``x`` belongs to the set of complex
    numbers. Note that every complex number is finite.

    Examples
    ========

    >>> from sympy import Q, Symbol, ask, I, oo
    >>> x = Symbol('x')
    >>> ask(Q.complex(0))
    True
    >>> ask(Q.complex(2 + 3*I))
    True
    >>> ask(Q.complex(oo))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_number

    """
    name = 'complex'
    handler = Dispatcher(
        "ComplexHandler",
        doc=("Handler for Q.complex.\n\n"
        "Test that an expression belongs to the field of complex numbers.")
    )


class ImaginaryPredicate(Predicate):
    """
    Imaginary number predicate.

    Explanation
    ===========

    ``Q.imaginary(x)`` is true iff ``x`` can be written as a real
    number multiplied by the imaginary unit ``I``. Please note that ``0``
    is not considered to be an imaginary number.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.imaginary(3*I))
    True
    >>> ask(Q.imaginary(2 + 3*I))
    False
    >>> ask(Q.imaginary(0))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_number

    """
    name = 'imaginary'
    handler = Dispatcher(
        "ImaginaryHandler",
        doc=("Handler for Q.imaginary.\n\n"
        "Test that an expression belongs to the field of imaginary numbers,\n"
        "that is, numbers in the form x*I, where x is real.")
    )


class AntihermitianPredicate(Predicate):
    """
    Antihermitian predicate.

    Explanation
    ===========

    ``Q.antihermitian(x)`` is true iff ``x`` belongs to the field of
    antihermitian operators, i.e., operators in the form ``x*I``, where
    ``x`` is Hermitian.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'antihermitian'
    handler = Dispatcher(
        "AntiHermitianHandler",
        doc=("Handler for Q.antihermitian.\n\n"
        "Test that an expression belongs to the field of anti-Hermitian\n"
        "operators, that is, operators in the form x*I, where x is Hermitian.")
    )


class AlgebraicPredicate(Predicate):
    r"""
    Algebraic number predicate.

    Explanation
    ===========

    ``Q.algebraic(x)`` is true iff ``x`` belongs to the set of
    algebraic numbers. ``x`` is algebraic if there is some polynomial
    in ``p(x)\in \mathbb\{Q\}[x]`` such that ``p(x) = 0``.

    Examples
    ========

    >>> from sympy import ask, Q, sqrt, I, pi
    >>> ask(Q.algebraic(sqrt(2)))
    True
    >>> ask(Q.algebraic(I))
    True
    >>> ask(Q.algebraic(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Algebraic_number

    """
    name = 'algebraic'
    AlgebraicHandler = Dispatcher(
        "AlgebraicHandler",
        doc="""Handler for Q.algebraic key."""
    )


class TranscendentalPredicate(Predicate):
    """
    Transcedental number predicate.

    Explanation
    ===========

    ``Q.transcendental(x)`` is true iff ``x`` belongs to the set of
    transcendental numbers. A transcendental number is a real
    or complex number that is not algebraic.

    """
    # TODO: Add examples
    name = 'transcendental'
    handler = Dispatcher(
        "Transcendental",
        doc="""Handler for Q.transcendental key."""
    )
