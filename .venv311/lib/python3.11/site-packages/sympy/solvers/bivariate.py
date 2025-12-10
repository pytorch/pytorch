from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq


def _filtered_gens(poly, symbol):
    """process the generators of ``poly``, returning the set of generators that
    have ``symbol``.  If there are two generators that are inverses of each other,
    prefer the one that has no denominator.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _filtered_gens
    >>> from sympy import Poly, exp
    >>> from sympy.abc import x
    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)
    {x, exp(x)}

    """
    # TODO it would be good to pick the smallest divisible power
    # instead of the base for something like x**4 + x**2 -->
    # return x**2 not x
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        ag = 1/g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens


def _mostfunc(lhs, func, X=None):
    """Returns the term in lhs which contains the most of the
    func-type things e.g. log(log(x)) wins over log(x) if both terms appear.

    ``func`` can be a function (exp, log, etc...) or any other SymPy object,
    like Pow.

    If ``X`` is not ``None``, then the function returns the term composed with the
    most ``func`` having the specified variable.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _mostfunc
    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> _mostfunc(exp(x) + exp(exp(x) + 2), exp)
    exp(exp(x) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp)
    exp(exp(y) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp, x)
    exp(x)
    >>> _mostfunc(x, exp, x) is None
    True
    >>> _mostfunc(exp(x) + exp(x*y), exp, x)
    exp(x)
    """
    fterms = [tmp for tmp in lhs.atoms(func) if (not X or
        X.is_Symbol and X in tmp.free_symbols or
        not X.is_Symbol and tmp.has(X))]
    if len(fterms) == 1:
        return fterms[0]
    elif fterms:
        return max(list(ordered(fterms)), key=lambda x: x.count(func))
    return None


def _linab(arg, symbol):
    """Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``
    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are
    independent of ``symbol``.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _linab
    >>> from sympy.abc import x, y
    >>> from sympy import exp, S
    >>> _linab(S(2), x)
    (2, 0, 1)
    >>> _linab(2*x, x)
    (2, 0, x)
    >>> _linab(y + y*x + 2*x, x)
    (y + 2, y, x)
    >>> _linab(3 + 2*exp(x), x)
    (2, 3, exp(x))
    """
    arg = factor_terms(arg.expand())
    ind, dep = arg.as_independent(symbol)
    if arg.is_Mul and dep.is_Add:
        a, b, x = _linab(dep, symbol)
        return ind*a, ind*b, x
    if not arg.is_Add:
        b = 0
        a, x = ind, dep
    else:
        b = ind
        a, x = separatevars(dep).as_independent(symbol, as_Add=False)
    if x.could_extract_minus_sign():
        a = -a
        x = -x
    return a, b, x


def _lambert(eq, x):
    """
    Given an expression assumed to be in the form
        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``
    where X = g(x) and x = g^-1(X), return the Lambert solution,
        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.
    """
    eq = _mexpand(expand_log(eq))
    mainlog = _mostfunc(eq, log, x)
    if not mainlog:
        return []  # violated assumptions
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []  # violated assumptions
        other = -(-other).args[0]
        eq += other
    if x not in other.free_symbols:
        return [] # violated assumptions
    d, f, X2 = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []  # violated assumptions
    logarg = mainlog.args[0]
    b, c, X1 = _linab(logarg, x)
    if X1 != X2:
        return []  # violated assumptions

    # invert the generator X1 so we have x(u)
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)

    # There are infinitely many branches for LambertW
    # but only branches for k = -1 and 0 might be real. The k = 0
    # branch is real and the k = -1 branch is real if the LambertW argument
    # in in range [-1/e, 0]. Since `solve` does not return infinite
    # solutions we will only include the -1 branch if it tests as real.
    # Otherwise, inclusion of any LambertW in the solution indicates to
    #  the user that there are imaginary solutions corresponding to
    # different k values.
    lambert_real_branches = [-1, 0]
    sol = []

    # solution of the given Lambert equation is like
    # sol = -c/b + (a/d)*LambertW(arg, k),
    # where arg = d/(a*b)*exp((c*d-b*f)/a/b) and k in lambert_real_branches.
    # Instead of considering the single arg, `d/(a*b)*exp((c*d-b*f)/a/b)`,
    # the individual `p` roots obtained when writing `exp((c*d-b*f)/a/b)`
    # as `exp(A/p) = exp(A)**(1/p)`, where `p` is an Integer, are used.

    # calculating args for LambertW
    num, den = ((c*d-b*f)/a/b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num/den)
    t = Dummy('t')
    args = [d/(a*b)*t for t in roots(t**p - e, t).keys()]

    # calculating solutions from args
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            if k and not w.is_real:
                continue
            rhs = -c/b + (a/d)*w

            sol.extend(xu.subs(u, rhs) for xu in xusolns)
    return sol


def _solve_lambert(f, symbol, gens):
    """Return solution to ``f`` if it is a Lambert-type expression
    else raise NotImplementedError.

    For ``f(X, a..f) = a*log(b*X + c) + d*X - f = 0`` the solution
    for ``X`` is ``X = -c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(f/a))``.
    There are a variety of forms for `f(X, a..f)` as enumerated below:

    1a1)
      if B**B = R for R not in [0, 1] (since those cases would already
      be solved before getting here) then log of both sides gives
      log(B) + log(log(B)) = log(log(R)) and
      X = log(B), a = 1, b = 1, c = 0, d = 1, f = log(log(R))
    1a2)
      if B*(b*log(B) + c)**a = R then log of both sides gives
      log(B) + a*log(b*log(B) + c) = log(R) and
      X = log(B), d=1, f=log(R)
    1b)
      if a*log(b*B + c) + d*B = R and
      X = B, f = R
    2a)
      if (b*B + c)*exp(d*B + g) = R then log of both sides gives
      log(b*B + c) + d*B + g = log(R) and
      X = B, a = 1, f = log(R) - g
    2b)
      if g*exp(d*B + h) - b*B = c then the log form is
      log(g) + d*B + h - log(b*B + c) = 0 and
      X = B, a = -1, f = -h - log(g)
    3)
      if d*p**(a*B + g) - b*B = c then the log form is
      log(d) + (a*B + g)*log(p) - log(b*B + c) = 0 and
      X = B, a = -1, d = a*log(p), f = -log(d) - g*log(p)
    """

    def _solve_even_degree_expr(expr, t, symbol):
        """Return the unique solutions of equations derived from
        ``expr`` by replacing ``t`` with ``+/- symbol``.

        Parameters
        ==========

        expr : Expr
            The expression which includes a dummy variable t to be
            replaced with +symbol and -symbol.

        symbol : Symbol
            The symbol for which a solution is being sought.

        Returns
        =======

        List of unique solution of the two equations generated by
        replacing ``t`` with positive and negative ``symbol``.

        Notes
        =====

        If ``expr = 2*log(t) + x/2` then solutions for
        ``2*log(x) + x/2 = 0`` and ``2*log(-x) + x/2 = 0`` are
        returned by this function. Though this may seem
        counter-intuitive, one must note that the ``expr`` being
        solved here has been derived from a different expression. For
        an expression like ``eq = x**2*g(x) = 1``, if we take the
        log of both sides we obtain ``log(x**2) + log(g(x)) = 0``. If
        x is positive then this simplifies to
        ``2*log(x) + log(g(x)) = 0``; the Lambert-solving routines will
        return solutions for this, but we must also consider the
        solutions for  ``2*log(-x) + log(g(x))`` since those must also
        be a solution of ``eq`` which has the same value when the ``x``
        in ``x**2`` is negated. If `g(x)` does not have even powers of
        symbol then we do not want to replace the ``x`` there with
        ``-x``. So the role of the ``t`` in the expression received by
        this function is to mark where ``+/-x`` should be inserted
        before obtaining the Lambert solutions.

        """
        nlhs, plhs = [
            expr.xreplace({t: sgn*symbol}) for sgn in (-1, 1)]
        sols = _solve_lambert(nlhs, symbol, gens)
        if plhs != nlhs:
            sols.extend(_solve_lambert(plhs, symbol, gens))
        # uniq is needed for a case like
        # 2*log(t) - log(-z**2) + log(z + log(x) + log(z))
        # where substituting t with +/-x gives all the same solution;
        # uniq, rather than list(set()), is used to maintain canonical
        # order
        return list(uniq(sols))

    nrhs, lhs = f.as_independent(symbol, as_Add=True)
    rhs = -nrhs

    lamcheck = [tmp for tmp in gens
                if (tmp.func in [exp, log] or
                (tmp.is_Pow and symbol in tmp.exp.free_symbols))]
    if not lamcheck:
        raise NotImplementedError()

    if lhs.is_Add or lhs.is_Mul:
        # replacing all even_degrees of symbol with dummy variable t
        # since these will need special handling; non-Add/Mul do not
        # need this handling
        t = Dummy('t', **symbol.assumptions0)
        lhs = lhs.replace(
            lambda i:  # find symbol**even
                i.is_Pow and i.base == symbol and i.exp.is_even,
            lambda i:  # replace t**even
                t**i.exp)

        if lhs.is_Add and lhs.has(t):
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            if not t_term.is_Add and _rhs and not (
                    t_term.has(S.ComplexInfinity, S.NaN)):
                eq = expand_log(log(t_term) - log(_rhs))
                return _solve_even_degree_expr(eq, t, symbol)
        elif lhs.is_Mul and rhs:
            # this needs to happen whether t is present or not
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and lhs.is_Add:
                # it expanded from Mul to Add
                eq = lhs - rhs
                return _solve_even_degree_expr(eq, t, symbol)

        # restore symbol in lhs
        lhs = lhs.xreplace({t: symbol})

    lhs = powsimp(factor(lhs, deep=True))

    # make sure we have inverted as completely as possible
    r = Dummy()
    i, lhs = _invert(lhs - r, symbol)
    rhs = i.xreplace({r: rhs})

    # For the first forms:
    #
    # 1a1) B**B = R will arrive here as B*log(B) = log(R)
    #      lhs is Mul so take log of both sides:
    #        log(B) + log(log(B)) = log(log(R))
    # 1a2) B*(b*log(B) + c)**a = R will arrive unchanged so
    #      lhs is Mul, so take log of both sides:
    #        log(B) + a*log(b*log(B) + c) = log(R)
    # 1b) d*log(a*B + b) + c*B = R will arrive unchanged so
    #      lhs is Add, so isolate c*B and expand log of both sides:
    #        log(c) + log(B) = log(R - d*log(a*B + b))

    soln = []
    if not soln:
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(log(lhs) - log(rhs), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                if other and not other.is_Add and [
                        tmp for tmp in other.atoms(Pow)
                        if symbol in tmp.free_symbols]:
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    #it's ready to go
                    soln = _lambert(lhs - rhs, symbol)

    # For the next forms,
    #
    #     collect on main exp
    #     2a) (b*B + c)*exp(d*B + g) = R
    #         lhs is mul, so take log of both sides:
    #           log(b*B + c) + d*B = log(R) - g
    #     2b) g*exp(d*B + h) - b*B = R
    #         lhs is add, so add b*B to both sides,
    #         take the log of both sides and rearrange to give
    #           log(R + b*B) - d*B = log(g) + h

    if not soln:
        mainexp = _mostfunc(lhs, exp, symbol)
        if mainexp:
            lhs = collect(lhs, mainexp)
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainexp-containing term to rhs
                other = lhs.subs(mainexp, 0)
                mainterm = lhs - other
                rhs = rhs - other
                if (mainterm.could_extract_minus_sign() and
                    rhs.could_extract_minus_sign()):
                    mainterm *= -1
                    rhs *= -1
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    # For the last form:
    #
    #  3) d*p**(a*B + g) - b*B = c
    #     collect on main pow, add b*B to both sides,
    #     take log of both sides and rearrange to give
    #       a*B*log(p) - log(b*B + c) = -log(d) - g*log(p)
    if not soln:
        mainpow = _mostfunc(lhs, Pow, symbol)
        if mainpow and symbol in mainpow.exp.free_symbols:
            lhs = collect(lhs, mainpow)
            if lhs.is_Mul and rhs != 0:
                # b*B = 0
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainpow-containing term to rhs
                other = lhs.subs(mainpow, 0)
                mainterm = lhs - other
                rhs = rhs - other
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    if not soln:
        raise NotImplementedError('%s does not appear to have a solution in '
            'terms of LambertW' % f)

    return list(ordered(soln))


def bivariate_type(f, x, y, *, first=True):
    """Given an expression, f, 3 tests will be done to see what type
    of composite bivariate it might be, options for u(x, y) are::

        x*y
        x+y
        x*y+x
        x*y+y

    If it matches one of these types, ``u(x, y)``, ``P(u)`` and dummy
    variable ``u`` will be returned. Solving ``P(u)`` for ``u`` and
    equating the solutions to ``u(x, y)`` and then solving for ``x`` or
    ``y`` is equivalent to solving the original expression for ``x`` or
    ``y``. If ``x`` and ``y`` represent two functions in the same
    variable, e.g. ``x = g(t)`` and ``y = h(t)``, then if ``u(x, y) - p``
    can be solved for ``t`` then these represent the solutions to
    ``P(u) = 0`` when ``p`` are the solutions of ``P(u) = 0``.

    Only positive values of ``u`` are considered.

    Examples
    ========

    >>> from sympy import solve
    >>> from sympy.solvers.bivariate import bivariate_type
    >>> from sympy.abc import x, y
    >>> eq = (x**2 - 3).subs(x, x + y)
    >>> bivariate_type(eq, x, y)
    (x + y, _u**2 - 3, _u)
    >>> uxy, pu, u = _
    >>> usol = solve(pu, u); usol
    [sqrt(3)]
    >>> [solve(uxy - s) for s in solve(pu, u)]
    [[{x: -y + sqrt(3)}]]
    >>> all(eq.subs(s).equals(0) for sol in _ for s in sol)
    True

    """

    u = Dummy('u', positive=True)

    if first:
        p = Poly(f, x, y)
        f = p.as_expr()
        _x = Dummy()
        _y = Dummy()
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        if rv:
            reps = {_x: x, _y: y}
            return rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2]
        return

    p = f
    f = p.as_expr()

    # f(x*y)
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        a = _mexpand(a.subs(x, u/y))
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        return x*y, Add(*new), u

    def ok(f, v, c):
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if (x in free or y in free) else new

    # f(a*x + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x**d), d)
        b = root(p.coeff_monomial(y**d), d)
        new = ok(f, x, (u - b*y)/a)
        if new is not None:
            return a*x + b*y, new, u

    # f(a*x*y + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x**d*y**d), d)
            b = root(p.coeff_monomial(y**d), d)
            new = ok(f, x, (u - b*y)/a/y)
            if new is not None:
                return a*x*y + b*y, new, u
            x, y = y, x
