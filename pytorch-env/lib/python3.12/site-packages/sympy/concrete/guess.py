"""Various algorithms for helping identifying numbers and sequences."""


from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public


@public
def find_simple_recurrence_vector(l):
    """
    This function is used internally by other functions from the
    sympy.concrete.guess module. While most users may want to rather use the
    function find_simple_recurrence when looking for recurrence relations
    among rational numbers, the current function may still be useful when
    some post-processing has to be done.

    Explanation
    ===========

    The function returns a vector of length n when a recurrence relation of
    order n is detected in the sequence of rational numbers v.

    If the returned vector has a length 1, then the returned value is always
    the list [0], which means that no relation has been found.

    While the functions is intended to be used with rational numbers, it should
    work for other kinds of real numbers except for some cases involving
    quadratic numbers; for that reason it should be used with some caution when
    the argument is not a list of rational numbers.

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence_vector
    >>> from sympy import fibonacci
    >>> find_simple_recurrence_vector([fibonacci(k) for k in range(12)])
    [1, -1, -1]

    See Also
    ========

    See the function sympy.concrete.guess.find_simple_recurrence which is more
    user-friendly.

    """
    q1 = [0]
    q2 = [1]
    b, z = 0, len(l) >> 1
    while len(q2) <= z:
        while l[b]==0:
            b += 1
            if b == len(l):
                c = 1
                for x in q2:
                    c = lcm(c, denom(x))
                if q2[0]*c < 0: c = -c
                for k in range(len(q2)):
                    q2[k] = int(q2[k]*c)
                return q2
        a = S.One/l[b]
        m = [a]
        for k in range(b+1, len(l)):
            m.append(-sum(l[j+1]*m[b-j-1] for j in range(b, k))*a)
        l, m = m, [0] * max(len(q2), b+len(q1))
        for k, q in enumerate(q2):
            m[k] = a*q
        for k, q in enumerate(q1):
            m[k+b] += q
        while m[-1]==0: m.pop() # because trailing zeros can occur
        q1, q2, b = q2, m, 1
    return [0]

@public
def find_simple_recurrence(v, A=Function('a'), N=Symbol('n')):
    """
    Detects and returns a recurrence relation from a sequence of several integer
    (or rational) terms. The name of the function in the returned expression is
    'a' by default; the main variable is 'n' by default. The smallest index in
    the returned expression is always n (and never n-1, n-2, etc.).

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence
    >>> from sympy import fibonacci
    >>> find_simple_recurrence([fibonacci(k) for k in range(12)])
    -a(n) - a(n + 1) + a(n + 2)

    >>> from sympy import Function, Symbol
    >>> a = [1, 1, 1]
    >>> for k in range(15): a.append(5*a[-1]-3*a[-2]+8*a[-3])
    >>> find_simple_recurrence(a, A=Function('f'), N=Symbol('i'))
    -8*f(i) + 3*f(i + 1) - 5*f(i + 2) + f(i + 3)

    """
    p = find_simple_recurrence_vector(v)
    n = len(p)
    if n <= 1: return S.Zero

    return Add(*[A(N+n-1-k)*p[k] for k in range(n)])


@public
def rationalize(x, maxcoeff=10000):
    """
    Helps identifying a rational number from a float (or mpmath.mpf) value by
    using a continued fraction. The algorithm stops as soon as a large partial
    quotient is detected (greater than 10000 by default).

    Examples
    ========

    >>> from sympy.concrete.guess import rationalize
    >>> from mpmath import cos, pi
    >>> rationalize(cos(pi/3))
    1/2

    >>> from mpmath import mpf
    >>> rationalize(mpf("0.333333333333333"))
    1/3

    While the function is rather intended to help 'identifying' rational
    values, it may be used in some cases for approximating real numbers.
    (Though other functions may be more relevant in that case.)

    >>> rationalize(pi, maxcoeff = 250)
    355/113

    See Also
    ========

    Several other methods can approximate a real number as a rational, like:

      * fractions.Fraction.from_decimal
      * fractions.Fraction.from_float
      * mpmath.identify
      * mpmath.pslq by using the following syntax: mpmath.pslq([x, 1])
      * mpmath.findpoly by using the following syntax: mpmath.findpoly(x, 1)
      * sympy.simplify.nsimplify (which is a more general function)

    The main difference between the current function and all these variants is
    that control focuses on magnitude of partial quotients here rather than on
    global precision of the approximation. If the real is "known to be" a
    rational number, the current function should be able to detect it correctly
    with the default settings even when denominator is great (unless its
    expansion contains unusually big partial quotients) which may occur
    when studying sequences of increasing numbers. If the user cares more
    on getting simple fractions, other methods may be more convenient.

    """
    p0, p1 = 0, 1
    q0, q1 = 1, 0
    a = floor(x)
    while a < maxcoeff or q1==0:
        p = a*p1 + p0
        q = a*q1 + q0
        p0, p1 = p1, p
        q0, q1 = q1, q
        if x==a: break
        x = 1/(x-a)
        a = floor(x)
    return sympify(p) / q


@public
def guess_generating_function_rational(v, X=Symbol('x')):
    """
    Tries to "guess" a rational generating function for a sequence of rational
    numbers v.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function_rational
    >>> from sympy import fibonacci
    >>> l = [fibonacci(k) for k in range(5,15)]
    >>> guess_generating_function_rational(l)
    (3*x + 5)/(-x**2 - x + 1)

    See Also
    ========

    sympy.series.approximants
    mpmath.pade

    """
    #   a) compute the denominator as q
    q = find_simple_recurrence_vector(v)
    n = len(q)
    if n <= 1: return None
    #   b) compute the numerator as p
    p = [sum(v[i-k]*q[k] for k in range(min(i+1, n)))
            for i in range(len(v)>>1)]
    return (sum(p[k]*X**k for k in range(len(p)))
            / sum(q[k]*X**k for k in range(n)))


@public
def guess_generating_function(v, X=Symbol('x'), types=['all'], maxsqrtn=2):
    """
    Tries to "guess" a generating function for a sequence of rational numbers v.
    Only a few patterns are implemented yet.

    Explanation
    ===========

    The function returns a dictionary where keys are the name of a given type of
    generating function. Six types are currently implemented:

         type  |  formal definition
        -------+----------------------------------------------------------------
        ogf    | f(x) = Sum(            a_k * x^k       ,  k: 0..infinity )
        egf    | f(x) = Sum(            a_k * x^k / k!  ,  k: 0..infinity )
        lgf    | f(x) = Sum( (-1)^(k+1) a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        hlgf   | f(x) = Sum(            a_k * x^k / k   ,  k: 1..infinity )
               |        (with initial index being hold as 1 rather than 0)
        lgdogf | f(x) = derivate( log(Sum( a_k * x^k, k: 0..infinity )), x)
        lgdegf | f(x) = derivate( log(Sum( a_k * x^k / k!, k: 0..infinity )), x)

    In order to spare time, the user can select only some types of generating
    functions (default being ['all']). While forgetting to use a list in the
    case of a single type may seem to work most of the time as in: types='ogf'
    this (convenient) syntax may lead to unexpected extra results in some cases.

    Discarding a type when calling the function does not mean that the type will
    not be present in the returned dictionary; it only means that no extra
    computation will be performed for that type, but the function may still add
    it in the result when it can be easily converted from another type.

    Two generating functions (lgdogf and lgdegf) are not even computed if the
    initial term of the sequence is 0; it may be useful in that case to try
    again after having removed the leading zeros.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function as ggf
    >>> ggf([k+1 for k in range(12)], types=['ogf', 'lgf', 'hlgf'])
    {'hlgf': 1/(1 - x), 'lgf': 1/(x + 1), 'ogf': 1/(x**2 - 2*x + 1)}

    >>> from sympy import sympify
    >>> l = sympify("[3/2, 11/2, 0, -121/2, -363/2, 121]")
    >>> ggf(l)
    {'ogf': (x + 3/2)/(11*x**2 - 3*x + 1)}

    >>> from sympy import fibonacci
    >>> ggf([fibonacci(k) for k in range(5, 15)], types=['ogf'])
    {'ogf': (3*x + 5)/(-x**2 - x + 1)}

    >>> from sympy import factorial
    >>> ggf([factorial(k) for k in range(12)], types=['ogf', 'egf', 'lgf'])
    {'egf': 1/(1 - x)}

    >>> ggf([k+1 for k in range(12)], types=['egf'])
    {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}

    N-th root of a rational function can also be detected (below is an example
    coming from the sequence A108626 from https://oeis.org).
    The greatest n-th root to be tested is specified as maxsqrtn (default 2).

    >>> ggf([1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])['ogf']
    sqrt(1/(x**4 + 2*x**2 - 4*x + 1))

    References
    ==========

    .. [1] "Concrete Mathematics", R.L. Graham, D.E. Knuth, O. Patashnik
    .. [2] https://oeis.org/wiki/Generating_functions

    """
    # List of all types of all g.f. known by the algorithm
    if 'all' in types:
        types = ('ogf', 'egf', 'lgf', 'hlgf', 'lgdogf', 'lgdegf')

    result = {}

    # Ordinary Generating Function (ogf)
    if 'ogf' in types:
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(v) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*v[i] for i in range(n+1)) for n in range(len(v))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['ogf'] = g**Rational(1, d+1)
                break

    # Exponential Generating Function (egf)
    if 'egf' in types:
        # Transform sequence (division by factorial)
        w, f = [], S.One
        for i, k in enumerate(v):
            f *= i if i else 1
            w.append(k/f)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['egf'] = g**Rational(1, d+1)
                break

    # Logarithmic Generating Function (lgf)
    if 'lgf' in types:
        # Transform sequence (multiplication by (-1)^(n+1) / n)
        w, f = [], S.NegativeOne
        for i, k in enumerate(v):
            f = -f
            w.append(f*k/Integer(i+1))
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgf'] = g**Rational(1, d+1)
                break

    # Hyperbolic logarithmic Generating Function (hlgf)
    if 'hlgf' in types:
        # Transform sequence (division by n+1)
        w = []
        for i, k in enumerate(v):
            w.append(k/Integer(i+1))
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['hlgf'] = g**Rational(1, d+1)
                break

    # Logarithmic derivative of ordinary generating Function (lgdogf)
    if v[0] != 0 and ('lgdogf' in types
                       or ('ogf' in types and 'ogf' not in result)):
        # Transform sequence by computing f'(x)/f(x)
        # because log(f(x)) = integrate( f'(x)/f(x) )
        a, w = sympify(v[0]), []
        for n in range(len(v)-1):
            w.append(
               (v[n+1]*(n+1) - sum(w[-i-1]*v[i+1] for i in range(n)))/a)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdogf'] = g**Rational(1, d+1)
                if 'ogf' not in result:
                    result['ogf'] = exp(integrate(result['lgdogf'], X))
                break

    # Logarithmic derivative of exponential generating Function (lgdegf)
    if v[0] != 0 and ('lgdegf' in types
                       or ('egf' in types and 'egf' not in result)):
        # Transform sequence / step 1 (division by factorial)
        z, f = [], S.One
        for i, k in enumerate(v):
            f *= i if i else 1
            z.append(k/f)
        # Transform sequence / step 2 by computing f'(x)/f(x)
        # because log(f(x)) = integrate( f'(x)/f(x) )
        a, w = z[0], []
        for n in range(len(z)-1):
            w.append(
               (z[n+1]*(n+1) - sum(w[-i-1]*z[i+1] for i in range(n)))/a)
        # Perform some convolutions of the sequence with itself
        t = [1] + [0]*(len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum(t[n-i]*w[i] for i in range(n+1)) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdegf'] = g**Rational(1, d+1)
                if 'egf' not in result:
                    result['egf'] = exp(integrate(result['lgdegf'], X))
                break

    return result


@public
def guess(l, all=False, evaluate=True, niter=2, variables=None):
    """
    This function is adapted from the Rate.m package for Mathematica
    written by Christian Krattenthaler.
    It tries to guess a formula from a given sequence of rational numbers.

    Explanation
    ===========

    In order to speed up the process, the 'all' variable is set to False by
    default, stopping the computation as some results are returned during an
    iteration; the variable can be set to True if more iterations are needed
    (other formulas may be found; however they may be equivalent to the first
    ones).

    Another option is the 'evaluate' variable (default is True); setting it
    to False will leave the involved products unevaluated.

    By default, the number of iterations is set to 2 but a greater value (up
    to len(l)-1) can be specified with the optional 'niter' variable.
    More and more convoluted results are found when the order of the
    iteration gets higher:

      * first iteration returns polynomial or rational functions;
      * second iteration returns products of rising factorials and their
        inverses;
      * third iteration returns products of products of rising factorials
        and their inverses;
      * etc.

    The returned formulas contain symbols i0, i1, i2, ... where the main
    variables is i0 (and auxiliary variables are i1, i2, ...). A list of
    other symbols can be provided in the 'variables' option; the length of
    the least should be the value of 'niter' (more is acceptable but only
    the first symbols will be used); in this case, the main variable will be
    the first symbol in the list.

    Examples
    ========

    >>> from sympy.concrete.guess import guess
    >>> guess([1,2,6,24,120], evaluate=False)
    [Product(i1 + 1, (i1, 1, i0 - 1))]

    >>> from sympy import symbols
    >>> r = guess([1,2,7,42,429,7436,218348,10850216], niter=4)
    >>> i0 = symbols("i0")
    >>> [r[0].subs(i0,n).doit() for n in range(1,10)]
    [1, 2, 7, 42, 429, 7436, 218348, 10850216, 911835460]
    """
    if any(a==0 for a in l[:-1]):
        return []
    N = len(l)
    niter = min(N-1, niter)
    myprod = product if evaluate else Product
    g = []
    res = []
    if variables is None:
        symb = symbols('i:'+str(niter))
    else:
        symb = variables
    for k, s in enumerate(symb):
        g.append(l)
        n, r = len(l), []
        for i in range(n-2-1, -1, -1):
            ri = rinterp(enumerate(g[k][:-1], start=1), i, X=s)
            if ((denom(ri).subs({s:n}) != 0)
                    and (ri.subs({s:n}) - g[k][-1] == 0)
                    and ri not in r):
              r.append(ri)
        if r:
            for i in range(k-1, -1, -1):
                r = [g[i][0]
                      * myprod(v, (symb[i+1], 1, symb[i]-1)) for v in r]
            if not all: return r
            res += r
        l = [Rational(l[i+1], l[i]) for i in range(N-k-1)]
    return res
