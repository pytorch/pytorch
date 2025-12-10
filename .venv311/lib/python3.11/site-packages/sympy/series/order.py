from sympy.core import S, sympify, Expr, Dummy, Add, Mul
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.function import Function, PoleError, expand_power_base, expand_log
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
from sympy.sets.sets import Complement
from sympy.utilities.iterables import uniq, is_sequence


class Order(Expr):
    r""" Represents the limiting behavior of some function.

    Explanation
    ===========

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if there
    exists a `\delta > 0` and an `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\limsup_{x \rightarrow a}
    |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, there is a `\delta > 0` and an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <https://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.

    """

    is_Order = True

    __slots__ = ()

    @cacheit
    def __new__(cls, expr, *args, **kwargs):
        expr = sympify(expr)

        if not args:
            if expr.is_Order:
                variables = expr.variables
                point = expr.point
            else:
                variables = list(expr.free_symbols)
                point = [S.Zero]*len(variables)
        else:
            args = list(args if is_sequence(args) else [args])
            variables, point = [], []
            if is_sequence(args[0]):
                for a in args:
                    v, p = list(map(sympify, a))
                    variables.append(v)
                    point.append(p)
            else:
                variables = list(map(sympify, args))
                point = [S.Zero]*len(variables)

        if not all(v.is_symbol for v in variables):
            raise TypeError('Variables are not symbols, got %s' % variables)

        if len(list(uniq(variables))) != len(variables):
            raise ValueError('Variables are supposed to be unique symbols, got %s' % variables)

        if expr.is_Order:
            expr_vp = dict(expr.args[1:])
            new_vp = dict(expr_vp)
            vp = dict(zip(variables, point))
            for v, p in vp.items():
                if v in new_vp.keys():
                    if p != new_vp[v]:
                        raise NotImplementedError(
                            "Mixing Order at different points is not supported.")
                else:
                    new_vp[v] = p
            if set(expr_vp.keys()) == set(new_vp.keys()):
                return expr
            else:
                variables = list(new_vp.keys())
                point = [new_vp[v] for v in variables]

        if expr is S.NaN:
            return S.NaN

        if any(x in p.free_symbols for x in variables for p in point):
            raise ValueError('Got %s as a point.' % point)

        if variables:
            if any(p != point[0] for p in point):
                raise NotImplementedError(
                    "Multivariable orders at different points are not supported.")
            if point[0] in (S.Infinity, S.Infinity*S.ImaginaryUnit):
                s = {k: 1/Dummy() for k in variables}
                rs = {1/v: 1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] in (S.NegativeInfinity, S.NegativeInfinity*S.ImaginaryUnit):
                s = {k: -1/Dummy() for k in variables}
                rs = {-1/v: -1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] is not S.Zero:
                s = {k: Dummy() + point[0] for k in variables}
                rs = {(v - point[0]).together(): k - point[0] for k, v in s.items()}
                ps = [S.Zero for p in point]
            else:
                s = ()
                rs = ()
                ps = list(point)

            expr = expr.subs(s)

            if expr.is_Add:
                expr = expr.factor()

            if s:
                args = tuple([r[0] for r in rs.items()])
            else:
                args = tuple(variables)

            if len(variables) > 1:
                # XXX: better way?  We need this expand() to
                # workaround e.g: expr = x*(x + y).
                # (x*(x + y)).as_leading_term(x, y) currently returns
                # x*y (wrong order term!).  That's why we want to deal with
                # expand()'ed expr (handled in "if expr.is_Add" branch below).
                expr = expr.expand()

            old_expr = None
            while old_expr != expr:
                old_expr = expr
                if expr.is_Add:
                    lst = expr.extract_leading_order(args)
                    expr = Add(*[f.expr for (e, f) in lst])

                elif expr:
                    try:
                        expr = expr.as_leading_term(*args)
                    except PoleError:
                        if isinstance(expr, Function) or\
                                all(isinstance(arg, Function) for arg in expr.args):
                            # It is not possible to simplify an expression
                            # containing only functions (which raise error on
                            # call to leading term) further
                            pass
                        else:
                            orders = []
                            pts = tuple(zip(args, ps))
                            for arg in expr.args:
                                try:
                                    lt = arg.as_leading_term(*args)
                                except PoleError:
                                    lt = arg
                                if lt not in args:
                                    order = Order(lt)
                                else:
                                    order = Order(lt, *pts)
                                orders.append(order)
                            if expr.is_Add:
                                new_expr = Order(Add(*orders), *pts)
                                if new_expr.is_Add:
                                    new_expr = Order(Add(*[a.expr for a in new_expr.args]), *pts)
                                expr = new_expr.expr
                            elif expr.is_Mul:
                                expr = Mul(*[a.expr for a in orders])
                            elif expr.is_Pow:
                                e = expr.exp
                                b = expr.base
                                expr = exp(e * log(b))

                    # It would probably be better to handle this somewhere
                    # else. This is needed for a testcase in which there is a
                    # symbol with the assumptions zero=True.
                    if expr.is_zero:
                        expr = S.Zero
                    else:
                        expr = expr.as_independent(*args, as_Add=False)[1]

                    expr = expand_power_base(expr)
                    expr = expand_log(expr)

                    if len(args) == 1:
                        # The definition of O(f(x)) symbol explicitly stated that
                        # the argument of f(x) is irrelevant.  That's why we can
                        # combine some power exponents (only "on top" of the
                        # expression tree for f(x)), e.g.:
                        # x**p * (-x)**q -> x**(p+q) for real p, q.
                        x = args[0]
                        margs = list(Mul.make_args(
                            expr.as_independent(x, as_Add=False)[1]))

                        for i, t in enumerate(margs):
                            if t.is_Pow:
                                b, q = t.args
                                if b in (x, -x) and q.is_real and not q.has(x):
                                    margs[i] = x**q
                                elif b.is_Pow and not b.exp.has(x):
                                    b, r = b.args
                                    if b in (x, -x) and r.is_real:
                                        margs[i] = x**(r*q)
                                elif b.is_Mul and b.args[0] is S.NegativeOne:
                                    b = -b
                                    if b.is_Pow and not b.exp.has(x):
                                        b, r = b.args
                                        if b in (x, -x) and r.is_real:
                                            margs[i] = x**(r*q)

                        expr = Mul(*margs)

            expr = expr.subs(rs)

        if expr.is_Order:
            expr = expr.expr

        if not expr.has(*variables) and not expr.is_zero:
            expr = S.One

        # create Order instance:
        vp = dict(zip(variables, point))
        variables.sort(key=default_sort_key)
        point = [vp[v] for v in variables]
        args = (expr,) + Tuple(*zip(variables, point))
        obj = Expr.__new__(cls, *args)
        return obj

    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        if self.args[1:]:
            return tuple(x[0] for x in self.args[1:])
        else:
            return ()

    @property
    def point(self):
        if self.args[1:]:
            return tuple(x[1] for x in self.args[1:])
        else:
            return ()

    @property
    def free_symbols(self):
        return self.expr.free_symbols | set(self.variables)

    def _eval_power(b, e):
        if e.is_Number and e.is_nonnegative:
            return b.func(b.expr ** e, *b.args[1:])
        if e == O(1):
            return b
        return

    def as_expr_variables(self, order_symbols):
        if order_symbols is None:
            order_symbols = self.args[1:]
        else:
            if (not all(o[1] == order_symbols[0][1] for o in order_symbols) and
                    not all(p == self.point[0] for p in self.point)):  # pragma: no cover
                raise NotImplementedError('Order at points other than 0 '
                    'or oo not supported, got %s as a point.' % self.point)
            if order_symbols and order_symbols[0][1] != self.point[0]:
                raise NotImplementedError(
                        "Multiplying Order at different points is not supported.")
            order_symbols = dict(order_symbols)
            for s, p in dict(self.args[1:]).items():
                if s not in order_symbols.keys():
                    order_symbols[s] = p
            order_symbols = sorted(order_symbols.items(), key=lambda x: default_sort_key(x[0]))
        return self.expr, tuple(order_symbols)

    def removeO(self):
        return S.Zero

    def getO(self):
        return self

    @cacheit
    def contains(self, expr):
        r"""
        Return True if expr belongs to Order(self.expr, \*self.variables).
        Return False if self belongs to expr.
        Return None if the inclusion relation cannot be determined
        (e.g. when self and expr have different symbols).
        """
        expr = sympify(expr)
        if expr.is_zero:
            return True
        if expr is S.NaN:
            return False
        point = self.point[0] if self.point else S.Zero
        if expr.is_Order:
            if (any(p != point for p in expr.point) or
                   any(p != point for p in self.point)):
                return None
            if expr.expr == self.expr:
                # O(1) + O(1), O(1) + O(1, x), etc.
                return all(x in self.args[1:] for x in expr.args[1:])
            if expr.expr.is_Add:
                return all(self.contains(x) for x in expr.expr.args)
            if self.expr.is_Add and point.is_zero:
                return any(self.func(x, *self.args[1:]).contains(expr)
                            for x in self.expr.args)
            if self.variables and expr.variables:
                common_symbols = tuple(
                    [s for s in self.variables if s in expr.variables])
            elif self.variables:
                common_symbols = self.variables
            else:
                common_symbols = expr.variables
            if not common_symbols:
                return None
            if (self.expr.is_Pow and len(self.variables) == 1
                and self.variables == expr.variables):
                    symbol = self.variables[0]
                    other = expr.expr.as_independent(symbol, as_Add=False)[1]
                    if (other.is_Pow and other.base == symbol and
                        self.expr.base == symbol):
                            if point.is_zero:
                                rv = (self.expr.exp - other.exp).is_nonpositive
                            if point.is_infinite:
                                rv = (self.expr.exp - other.exp).is_nonnegative
                            if rv is not None:
                                return rv

            from sympy.simplify.powsimp import powsimp
            r = None
            ratio = self.expr/expr.expr
            ratio = powsimp(ratio, deep=True, combine='exp')
            for s in common_symbols:
                from sympy.series.limits import Limit
                l = Limit(ratio, s, point).doit(heuristics=False)
                if not isinstance(l, Limit):
                    l = l != 0
                else:
                    l = None
                if r is None:
                    r = l
                else:
                    if r != l:
                        return
            return r

        if self.expr.is_Pow and len(self.variables) == 1:
            symbol = self.variables[0]
            other = expr.as_independent(symbol, as_Add=False)[1]
            if (other.is_Pow and other.base == symbol and
                self.expr.base == symbol):
                    if point.is_zero:
                        rv = (self.expr.exp - other.exp).is_nonpositive
                    if point.is_infinite:
                        rv = (self.expr.exp - other.exp).is_nonnegative
                    if rv is not None:
                        return rv

        obj = self.func(expr, *self.args[1:])
        return self.contains(obj)

    def __contains__(self, other):
        result = self.contains(other)
        if result is None:
            raise TypeError('contains did not evaluate to a bool')
        return result

    def _eval_subs(self, old, new):
        if old in self.variables:
            newexpr = self.expr.subs(old, new)
            i = self.variables.index(old)
            newvars = list(self.variables)
            newpt = list(self.point)
            if new.is_symbol:
                newvars[i] = new
            else:
                syms = new.free_symbols
                if len(syms) == 1 or old in syms:
                    if old in syms:
                        var = self.variables[i]
                    else:
                        var = syms.pop()
                    # First, try to substitute self.point in the "new"
                    # expr to see if this is a fixed point.
                    # E.g.  O(y).subs(y, sin(x))
                    from sympy import limit
                    if new.has(Order) and limit(new.getO().expr, var, new.getO().point[0]) == self.point[i]:
                        point = new.getO().point[0]
                        return Order(newexpr, *zip([var], [point]))
                    else:
                        point = new.subs(var, self.point[i])
                    if point != self.point[i]:
                        from sympy.solvers.solveset import solveset
                        d = Dummy()
                        sol = solveset(old - new.subs(var, d), d)
                        if isinstance(sol, Complement):
                            e1 = sol.args[0]
                            e2 = sol.args[1]
                            sol = set(e1) - set(e2)
                        res = [dict(zip((d, ), sol))]
                        point = d.subs(res[0]).limit(old, self.point[i])
                    newvars[i] = var
                    newpt[i] = point
                elif old not in syms:
                    del newvars[i], newpt[i]
                    if not syms and new == self.point[i]:
                        newvars.extend(syms)
                        newpt.extend([S.Zero]*len(syms))
                else:
                    return
            return Order(newexpr, *zip(newvars, newpt))

    def _eval_conjugate(self):
        expr = self.expr._eval_conjugate()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def _eval_derivative(self, x):
        return self.func(self.expr.diff(x), *self.args[1:]) or self

    def _eval_transpose(self):
        expr = self.expr._eval_transpose()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def __neg__(self):
        return self

O = Order
