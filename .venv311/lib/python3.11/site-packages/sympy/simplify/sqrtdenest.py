from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError


def is_sqrt(expr):
    """Return True if expr is a sqrt, otherwise False."""

    return expr.is_Pow and expr.exp.is_Rational and abs(expr.exp) is S.Half


def sqrt_depth(p) -> int:
    """Return the maximum depth of any square root argument of p.

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrt_depth

    Neither of these square roots contains any other square roots
    so the depth is 1:

    >>> sqrt_depth(1 + sqrt(2)*(1 + sqrt(3)))
    1

    The sqrt(3) is contained within a square root so the depth is
    2:

    >>> sqrt_depth(1 + sqrt(2)*sqrt(1 + sqrt(3)))
    2
    """
    if p is S.ImaginaryUnit:
        return 1
    if p.is_Atom:
        return 0
    if p.is_Add or p.is_Mul:
        return max(sqrt_depth(x) for x in p.args)
    if is_sqrt(p):
        return sqrt_depth(p.base) + 1
    return 0


def is_algebraic(p):
    """Return True if p is comprised of only Rationals or square roots
    of Rationals and algebraic operations.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import is_algebraic
    >>> from sympy import cos
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*sqrt(2))))
    True
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*cos(2))))
    False
    """

    if p.is_Rational:
        return True
    elif p.is_Atom:
        return False
    elif is_sqrt(p) or p.is_Pow and p.exp.is_Integer:
        return is_algebraic(p.base)
    elif p.is_Add or p.is_Mul:
        return all(is_algebraic(x) for x in p.args)
    else:
        return False


def _subsets(n):
    """
    Returns all possible subsets of the set (0, 1, ..., n-1) except the
    empty set, listed in reversed lexicographical order according to binary
    representation, so that the case of the fourth root is treated last.

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _subsets
    >>> _subsets(2)
    [[1, 0], [0, 1], [1, 1]]

    """
    if n == 1:
        a = [[1]]
    elif n == 2:
        a = [[1, 0], [0, 1], [1, 1]]
    elif n == 3:
        a = [[1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    else:
        b = _subsets(n - 1)
        a0 = [x + [0] for x in b]
        a1 = [x + [1] for x in b]
        a = a0 + [[0]*(n - 1) + [1]] + a1
    return a


def sqrtdenest(expr, max_iter=3):
    """Denests sqrts in an expression that contain other square roots
    if possible, otherwise returns the expr unchanged. This is based on the
    algorithms of [1].

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> from sympy import sqrt
    >>> sqrtdenest(sqrt(5 + 2 * sqrt(6)))
    sqrt(2) + sqrt(3)

    See Also
    ========

    sympy.solvers.solvers.unrad

    References
    ==========

    .. [1] https://web.archive.org/web/20210806201615/https://researcher.watson.ibm.com/researcher/files/us-fagin/symb85.pdf

    .. [2] D. J. Jeffrey and A. D. Rich, 'Symplifying Square Roots of Square Roots
           by Denesting' (available at https://www.cybertester.com/data/denest.pdf)

    """
    expr = expand_mul(expr)
    for i in range(max_iter):
        z = _sqrtdenest0(expr)
        if expr == z:
            return expr
        expr = z
    return expr


def _sqrt_match(p):
    """Return [a, b, r] for p.match(a + b*sqrt(r)) where, in addition to
    matching, sqrt(r) also has then maximal sqrt_depth among addends of p.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match
    >>> _sqrt_match(1 + sqrt(2) + sqrt(2)*sqrt(3) +  2*sqrt(1+sqrt(5)))
    [1 + sqrt(2) + sqrt(6), 2, 1 + sqrt(5)]
    """
    from sympy.simplify.radsimp import split_surds

    p = _mexpand(p)
    if p.is_Number:
        res = (p, S.Zero, S.Zero)
    elif p.is_Add:
        pargs = sorted(p.args, key=default_sort_key)
        sqargs = [x**2 for x in pargs]
        if all(sq.is_Rational and sq.is_positive for sq in sqargs):
            r, b, a = split_surds(p)
            res = a, b, r
            return list(res)
        # to make the process canonical, the argument is included in the tuple
        # so when the max is selected, it will be the largest arg having a
        # given depth
        v = [(sqrt_depth(x), x, i) for i, x in enumerate(pargs)]
        nmax = max(v, key=default_sort_key)
        if nmax[0] == 0:
            res = []
        else:
            # select r
            depth, _, i = nmax
            r = pargs.pop(i)
            v.pop(i)
            b = S.One
            if r.is_Mul:
                bv = []
                rv = []
                for x in r.args:
                    if sqrt_depth(x) < depth:
                        bv.append(x)
                    else:
                        rv.append(x)
                b = Mul._from_args(bv)
                r = Mul._from_args(rv)
            # collect terms containing r
            a1 = []
            b1 = [b]
            for x in v:
                if x[0] < depth:
                    a1.append(x[1])
                else:
                    x1 = x[1]
                    if x1 == r:
                        b1.append(1)
                    else:
                        if x1.is_Mul:
                            x1args = list(x1.args)
                            if r in x1args:
                                x1args.remove(r)
                                b1.append(Mul(*x1args))
                            else:
                                a1.append(x[1])
                        else:
                            a1.append(x[1])
            a = Add(*a1)
            b = Add(*b1)
            res = (a, b, r**2)
    else:
        b, r = p.as_coeff_Mul()
        if is_sqrt(r):
            res = (S.Zero, b, r**2)
        else:
            res = []
    return list(res)


class SqrtdenestStopIteration(StopIteration):
    pass


def _sqrtdenest0(expr):
    """Returns expr after denesting its arguments."""

    if is_sqrt(expr):
        n, d = expr.as_numer_denom()
        if d is S.One:  # n is a square root
            if n.base.is_Add:
                args = sorted(n.base.args, key=default_sort_key)
                if len(args) > 2 and all((x**2).is_Integer for x in args):
                    try:
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                expr = sqrt(_mexpand(Add(*[_sqrtdenest0(x) for x in args])))
            return _sqrtdenest1(expr)
        else:
            n, d = [_sqrtdenest0(i) for i in (n, d)]
            return n/d

    if isinstance(expr, Add):
        cs = []
        args = []
        for arg in expr.args:
            c, a = arg.as_coeff_Mul()
            cs.append(c)
            args.append(a)

        if all(c.is_Rational for c in cs) and all(is_sqrt(arg) for arg in args):
            return _sqrt_ratcomb(cs, args)

    if isinstance(expr, Expr):
        args = expr.args
        if args:
            return expr.func(*[_sqrtdenest0(a) for a in args])
    return expr


def _sqrtdenest_rec(expr):
    """Helper that denests the square root of three or more surds.

    Explanation
    ===========

    It returns the denested expression; if it cannot be denested it
    throws SqrtdenestStopIteration

    Algorithm: expr.base is in the extension Q_m = Q(sqrt(r_1),..,sqrt(r_k));
    split expr.base = a + b*sqrt(r_k), where `a` and `b` are on
    Q_(m-1) = Q(sqrt(r_1),..,sqrt(r_(k-1))); then a**2 - b**2*r_k is
    on Q_(m-1); denest sqrt(a**2 - b**2*r_k) and so on.
    See [1], section 6.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrtdenest_rec
    >>> _sqrtdenest_rec(sqrt(-72*sqrt(2) + 158*sqrt(5) + 498))
    -sqrt(10) + sqrt(2) + 9 + 9*sqrt(5)
    >>> w=-6*sqrt(55)-6*sqrt(35)-2*sqrt(22)-2*sqrt(14)+2*sqrt(77)+6*sqrt(10)+65
    >>> _sqrtdenest_rec(sqrt(w))
    -sqrt(11) - sqrt(7) + sqrt(2) + 3*sqrt(5)
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize, split_surds
    if not expr.is_Pow:
        return sqrtdenest(expr)
    if expr.base < 0:
        return sqrt(-1)*_sqrtdenest_rec(sqrt(-expr.base))
    g, a, b = split_surds(expr.base)
    a = a*sqrt(g)
    if a < b:
        a, b = b, a
    c2 = _mexpand(a**2 - b**2)
    if len(c2.args) > 2:
        g, a1, b1 = split_surds(c2)
        a1 = a1*sqrt(g)
        if a1 < b1:
            a1, b1 = b1, a1
        c2_1 = _mexpand(a1**2 - b1**2)
        c_1 = _sqrtdenest_rec(sqrt(c2_1))
        d_1 = _sqrtdenest_rec(sqrt(a1 + c_1))
        num, den = rad_rationalize(b1, d_1)
        c = _mexpand(d_1/sqrt(2) + num/(den*sqrt(2)))
    else:
        c = _sqrtdenest1(sqrt(c2))

    if sqrt_depth(c) > 1:
        raise SqrtdenestStopIteration
    ac = a + c
    if len(ac.args) >= len(expr.args):
        if count_ops(ac) >= count_ops(expr.base):
            raise SqrtdenestStopIteration
    d = sqrtdenest(sqrt(ac))
    if sqrt_depth(d) > 1:
        raise SqrtdenestStopIteration
    num, den = rad_rationalize(b, d)
    r = d/sqrt(2) + num/(den*sqrt(2))
    r = radsimp(r)
    return _mexpand(r)


def _sqrtdenest1(expr, denester=True):
    """Return denested expr after denesting with simpler methods or, that
    failing, using the denester."""

    from sympy.simplify.simplify import radsimp

    if not is_sqrt(expr):
        return expr

    a = expr.base
    if a.is_Atom:
        return expr
    val = _sqrt_match(a)
    if not val:
        return expr

    a, b, r = val
    # try a quick numeric denesting
    d2 = _mexpand(a**2 - b**2*r)
    if d2.is_Rational:
        if d2.is_positive:
            z = _sqrt_numeric_denest(a, b, r, d2)
            if z is not None:
                return z
        else:
            # fourth root case
            # sqrtdenest(sqrt(3 + 2*sqrt(3))) =
            # sqrt(2)*3**(1/4)/2 + sqrt(2)*3**(3/4)/2
            dr2 = _mexpand(-d2*r)
            dr = sqrt(dr2)
            if dr.is_Rational:
                z = _sqrt_numeric_denest(_mexpand(b*r), a, r, dr2)
                if z is not None:
                    return z/root(r, 4)

    else:
        z = _sqrt_symbolic_denest(a, b, r)
        if z is not None:
            return z

    if not denester or not is_algebraic(expr):
        return expr

    res = sqrt_biquadratic_denest(expr, a, b, r, d2)
    if res:
        return res

    # now call to the denester
    av0 = [a, b, r, d2]
    z = _denester([radsimp(expr**2)], av0, 0, sqrt_depth(expr))[0]
    if av0[1] is None:
        return expr
    if z is not None:
        if sqrt_depth(z) == sqrt_depth(expr) and count_ops(z) > count_ops(expr):
            return expr
        return z
    return expr


def _sqrt_symbolic_denest(a, b, r):
    """Given an expression, sqrt(a + b*sqrt(b)), return the denested
    expression or None.

    Explanation
    ===========

    If r = ra + rb*sqrt(rr), try replacing sqrt(rr) in ``a`` with
    (y**2 - ra)/rb, and if the result is a quadratic, ca*y**2 + cb*y + cc, and
    (cb + b)**2 - 4*ca*cc is 0, then sqrt(a + b*sqrt(r)) can be rewritten as
    sqrt(ca*(sqrt(r) + (cb + b)/(2*ca))**2).

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _sqrt_symbolic_denest, sqrtdenest
    >>> from sympy import sqrt, Symbol
    >>> from sympy.abc import x

    >>> a, b, r = 16 - 2*sqrt(29), 2, -10*sqrt(29) + 55
    >>> _sqrt_symbolic_denest(a, b, r)
    sqrt(11 - 2*sqrt(29)) + sqrt(5)

    If the expression is numeric, it will be simplified:

    >>> w = sqrt(sqrt(sqrt(3) + 1) + 1) + 1 + sqrt(2)
    >>> sqrtdenest(sqrt((w**2).expand()))
    1 + sqrt(2) + sqrt(1 + sqrt(1 + sqrt(3)))

    Otherwise, it will only be simplified if assumptions allow:

    >>> w = w.subs(sqrt(3), sqrt(x + 3))
    >>> sqrtdenest(sqrt((w**2).expand()))
    sqrt((sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2))**2)

    Notice that the argument of the sqrt is a square. If x is made positive
    then the sqrt of the square is resolved:

    >>> _.subs(x, Symbol('x', positive=True))
    sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2)
    """

    a, b, r = map(sympify, (a, b, r))
    rval = _sqrt_match(r)
    if not rval:
        return None
    ra, rb, rr = rval
    if rb:
        y = Dummy('y', positive=True)
        try:
            newa = Poly(a.subs(sqrt(rr), (y**2 - ra)/rb), y)
        except PolynomialError:
            return None
        if newa.degree() == 2:
            ca, cb, cc = newa.all_coeffs()
            cb += b
            if _mexpand(cb**2 - 4*ca*cc).equals(0):
                z = sqrt(ca*(sqrt(r) + cb/(2*ca))**2)
                if z.is_number:
                    z = _mexpand(Mul._from_args(z.as_content_primitive()))
                return z


def _sqrt_numeric_denest(a, b, r, d2):
    r"""Helper that denest
    $\sqrt{a + b \sqrt{r}}, d^2 = a^2 - b^2 r > 0$

    If it cannot be denested, it returns ``None``.
    """
    d = sqrt(d2)
    s = a + d
    # sqrt_depth(res) <= sqrt_depth(s) + 1
    # sqrt_depth(expr) = sqrt_depth(r) + 2
    # there is denesting if sqrt_depth(s) + 1 < sqrt_depth(r) + 2
    # if s**2 is Number there is a fourth root
    if sqrt_depth(s) < sqrt_depth(r) + 1 or (s**2).is_Rational:
        s1, s2 = sign(s), sign(b)
        if s1 == s2 == -1:
            s1 = s2 = 1
        res = (s1 * sqrt(a + d) + s2 * sqrt(a - d)) * sqrt(2) / 2
        return res.expand()


def sqrt_biquadratic_denest(expr, a, b, r, d2):
    """denest expr = sqrt(a + b*sqrt(r))
    where a, b, r are linear combinations of square roots of
    positive rationals on the rationals (SQRR) and r > 0, b != 0,
    d2 = a**2 - b**2*r > 0

    If it cannot denest it returns None.

    Explanation
    ===========

    Search for a solution A of type SQRR of the biquadratic equation
    4*A**4 - 4*a*A**2 + b**2*r = 0                               (1)
    sqd = sqrt(a**2 - b**2*r)
    Choosing the sqrt to be positive, the possible solutions are
    A = sqrt(a/2 +/- sqd/2)
    Since a, b, r are SQRR, then a**2 - b**2*r is a SQRR,
    so if sqd can be denested, it is done by
    _sqrtdenest_rec, and the result is a SQRR.
    Similarly for A.
    Examples of solutions (in both cases a and sqd are positive):

      Example of expr with solution sqrt(a/2 + sqd/2) but not
      solution sqrt(a/2 - sqd/2):
      expr = sqrt(-sqrt(15) - sqrt(2)*sqrt(-sqrt(5) + 5) - sqrt(3) + 8)
      a = -sqrt(15) - sqrt(3) + 8; sqd = -2*sqrt(5) - 2 + 4*sqrt(3)

      Example of expr with solution sqrt(a/2 - sqd/2) but not
      solution sqrt(a/2 + sqd/2):
      w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
      expr = sqrt((w**2).expand())
      a = 4*sqrt(6) + 8*sqrt(2) + 47 + 28*sqrt(3)
      sqd = 29 + 20*sqrt(3)

    Define B = b/2*A; eq.(1) implies a = A**2 + B**2*r; then
    expr**2 = a + b*sqrt(r) = (A + B*sqrt(r))**2

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match, sqrt_biquadratic_denest
    >>> z = sqrt((2*sqrt(2) + 4)*sqrt(2 + sqrt(2)) + 5*sqrt(2) + 8)
    >>> a, b, r = _sqrt_match(z**2)
    >>> d2 = a**2 - b**2*r
    >>> sqrt_biquadratic_denest(z, a, b, r, d2)
    sqrt(2) + sqrt(sqrt(2) + 2) + 2
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize
    if r <= 0 or d2 < 0 or not b or sqrt_depth(expr.base) < 2:
        return None
    for x in (a, b, r):
        for y in x.args:
            y2 = y**2
            if not y2.is_Integer or not y2.is_positive:
                return None
    sqd = _mexpand(sqrtdenest(sqrt(radsimp(d2))))
    if sqrt_depth(sqd) > 1:
        return None
    x1, x2 = [a/2 + sqd/2, a/2 - sqd/2]
    # look for a solution A with depth 1
    for x in (x1, x2):
        A = sqrtdenest(sqrt(x))
        if sqrt_depth(A) > 1:
            continue
        Bn, Bd = rad_rationalize(b, _mexpand(2*A))
        B = Bn/Bd
        z = A + B*sqrt(r)
        if z < 0:
            z = -z
        return _mexpand(z)
    return None


def _denester(nested, av0, h, max_depth_level):
    """Denests a list of expressions that contain nested square roots.

    Explanation
    ===========

    Algorithm based on <http://www.almaden.ibm.com/cs/people/fagin/symb85.pdf>.

    It is assumed that all of the elements of 'nested' share the same
    bottom-level radicand. (This is stated in the paper, on page 177, in
    the paragraph immediately preceding the algorithm.)

    When evaluating all of the arguments in parallel, the bottom-level
    radicand only needs to be denested once. This means that calling
    _denester with x arguments results in a recursive invocation with x+1
    arguments; hence _denester has polynomial complexity.

    However, if the arguments were evaluated separately, each call would
    result in two recursive invocations, and the algorithm would have
    exponential complexity.

    This is discussed in the paper in the middle paragraph of page 179.
    """
    from sympy.simplify.simplify import radsimp
    if h > max_depth_level:
        return None, None
    if av0[1] is None:
        return None, None
    if (av0[0] is None and
            all(n.is_Number for n in nested)):  # no arguments are nested
        for f in _subsets(len(nested)):  # test subset 'f' of nested
            p = _mexpand(Mul(*[nested[i] for i in range(len(f)) if f[i]]))
            if f.count(1) > 1 and f[-1]:
                p = -p
            sqp = sqrt(p)
            if sqp.is_Rational:
                return sqp, f  # got a perfect square so return its square root.
        # Otherwise, return the radicand from the previous invocation.
        return sqrt(nested[-1]), [0]*len(nested)
    else:
        R = None
        if av0[0] is not None:
            values = [av0[:2]]
            R = av0[2]
            nested2 = [av0[3], R]
            av0[0] = None
        else:
            values = list(filter(None, [_sqrt_match(expr) for expr in nested]))
            for v in values:
                if v[2]:  # Since if b=0, r is not defined
                    if R is not None:
                        if R != v[2]:
                            av0[1] = None
                            return None, None
                    else:
                        R = v[2]
            if R is None:
                # return the radicand from the previous invocation
                return sqrt(nested[-1]), [0]*len(nested)
            nested2 = [_mexpand(v[0]**2) -
                       _mexpand(R*v[1]**2) for v in values] + [R]
        d, f = _denester(nested2, av0, h + 1, max_depth_level)
        if not f:
            return None, None
        if not any(f[i] for i in range(len(nested))):
            v = values[-1]
            return sqrt(v[0] + _mexpand(v[1]*d)), f
        else:
            p = Mul(*[nested[i] for i in range(len(nested)) if f[i]])
            v = _sqrt_match(p)
            if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
                v[0] = -v[0]
                v[1] = -v[1]
            if not f[len(nested)]:  # Solution denests with square roots
                vad = _mexpand(v[0] + d)
                if vad <= 0:
                    # return the radicand from the previous invocation.
                    return sqrt(nested[-1]), [0]*len(nested)
                if not(sqrt_depth(vad) <= sqrt_depth(R) + 1 or
                       (vad**2).is_Number):
                    av0[1] = None
                    return None, None

                sqvad = _sqrtdenest1(sqrt(vad), denester=False)
                if not (sqrt_depth(sqvad) <= sqrt_depth(R) + 1):
                    av0[1] = None
                    return None, None
                sqvad1 = radsimp(1/sqvad)
                res = _mexpand(sqvad/sqrt(2) + (v[1]*sqrt(R)*sqvad1/sqrt(2)))
                return res, f

                      #          sign(v[1])*sqrt(_mexpand(v[1]**2*R*vad1/2))), f
            else:  # Solution requires a fourth root
                s2 = _mexpand(v[1]*R) + d
                if s2 <= 0:
                    return sqrt(nested[-1]), [0]*len(nested)
                FR, s = root(_mexpand(R), 4), sqrt(s2)
                return _mexpand(s/(sqrt(2)*FR) + v[0]*FR/(sqrt(2)*s)), f


def _sqrt_ratcomb(cs, args):
    """Denest rational combinations of radicals.

    Based on section 5 of [1].

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> z = sqrt(1+sqrt(3)) + sqrt(3+3*sqrt(3)) - sqrt(10+6*sqrt(3))
    >>> sqrtdenest(z)
    0
    """
    from sympy.simplify.radsimp import radsimp

    # check if there exists a pair of sqrt that can be denested
    def find(a):
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                s1 = a[i].base
                s2 = a[j].base
                p = _mexpand(s1 * s2)
                s = sqrtdenest(sqrt(p))
                if s != sqrt(p):
                    return s, i, j

    indices = find(args)
    if indices is None:
        return Add(*[c * arg for c, arg in zip(cs, args)])

    s, i1, i2 = indices

    c2 = cs.pop(i2)
    args.pop(i2)
    a1 = args[i1]

    # replace a2 by s/a1
    cs[i1] += radsimp(c2 * s / a1.base)

    return _sqrt_ratcomb(cs, args)
