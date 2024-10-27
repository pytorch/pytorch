from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polytools import lcm
from sympy.utilities import public

@public
def approximants(l, X=Symbol('x'), simplify=False):
    """
    Return a generator for consecutive Pade approximants for a series.
    It can also be used for computing the rational generating function of a
    series when possible, since the last approximant returned by the generator
    will be the generating function (if any).

    Explanation
    ===========

    The input list can contain more complex expressions than integer or rational
    numbers; symbols may also be involved in the computation. An example below
    show how to compute the generating function of the whole Pascal triangle.

    The generator can be asked to apply the sympy.simplify function on each
    generated term, which will make the computation slower; however it may be
    useful when symbols are involved in the expressions.

    Examples
    ========

    >>> from sympy.series import approximants
    >>> from sympy import lucas, fibonacci, symbols, binomial
    >>> g = [lucas(k) for k in range(16)]
    >>> [e for e in approximants(g)]
    [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)]

    >>> h = [fibonacci(k) for k in range(16)]
    >>> [e for e in approximants(h)]
    [x, -x/(x - 1), (x**2 - x)/(2*x - 1), -x/(x**2 + x - 1)]

    >>> x, t = symbols("x,t")
    >>> p=[sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    >>> y = approximants(p, t)
    >>> for k in range(3): print(next(y))
    1
    (x + 1)/((-x - 1)*(t*(x + 1) + (x + 1)/(-x - 1)))
    nan

    >>> y = approximants(p, t, simplify=True)
    >>> for k in range(3): print(next(y))
    1
    -1/(t*(x + 1) - 1)
    nan

    See Also
    ========

    sympy.concrete.guess.guess_generating_function_rational
    mpmath.pade
    """
    from sympy.simplify import simplify as simp
    from sympy.simplify.radsimp import denom
    p1, q1 = [S.One], [S.Zero]
    p2, q2 = [S.Zero], [S.One]
    while len(l):
        b = 0
        while l[b]==0:
            b += 1
            if b == len(l):
                return
        m = [S.One/l[b]]
        for k in range(b+1, len(l)):
            s = 0
            for j in range(b, k):
                s -= l[j+1] * m[b-j-1]
            m.append(s/l[b])
        l = m
        a, l[0] = l[0], 0
        p = [0] * max(len(p2), b+len(p1))
        q = [0] * max(len(q2), b+len(q1))
        for k in range(len(p2)):
            p[k] = a*p2[k]
        for k in range(b, b+len(p1)):
            p[k] += p1[k-b]
        for k in range(len(q2)):
            q[k] = a*q2[k]
        for k in range(b, b+len(q1)):
            q[k] += q1[k-b]
        while p[-1]==0: p.pop()
        while q[-1]==0: q.pop()
        p1, p2 = p2, p
        q1, q2 = q2, q

        # yield result
        c = 1
        for x in p:
            c = lcm(c, denom(x))
        for x in q:
            c = lcm(c, denom(x))
        out = ( sum(c*e*X**k for k, e in enumerate(p))
              / sum(c*e*X**k for k, e in enumerate(q)) )
        if simplify:
            yield(simp(out))
        else:
            yield out
    return
