from typing import Tuple as tTuple

from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.special.singularity_functions import Heaviside
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent


class Integral(AddWithLimits):
    """Represents unevaluated integral."""

    __slots__ = ()

    args: tTuple[Expr, Tuple]

    def __new__(cls, function, *symbols, **assumptions):
        """Create an unevaluated integral.

        Explanation
        ===========

        Arguments are an integrand followed by one or more limits.

        If no limits are given and there is only one free symbol in the
        expression, that symbol will be used, otherwise an error will be
        raised.

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x)
        Integral(x, x)
        >>> Integral(y)
        Integral(y, y)

        When limits are provided, they are interpreted as follows (using
        ``x`` as though it were the variable of integration):

            (x,) or x - indefinite integral
            (x, a) - "evaluate at" integral is an abstract antiderivative
            (x, a, b) - definite integral

        The ``as_dummy`` method can be used to see which symbols cannot be
        targeted by subs: those with a prepended underscore cannot be
        changed with ``subs``. (Also, the integration variables themselves --
        the first element of a limit -- can never be changed by subs.)

        >>> i = Integral(x, x)
        >>> at = Integral(x, (x, x))
        >>> i.as_dummy()
        Integral(x, x)
        >>> at.as_dummy()
        Integral(_0, (_0, x))

        """

        #This will help other classes define their own definitions
        #of behaviour with Integral.
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)

        if isinstance(function, Poly):
            sympy_deprecation_warning(
                """
                integrate(Poly) and Integral(Poly) are deprecated. Instead,
                use the Poly.integrate() method, or convert the Poly to an
                Expr first with the Poly.as_expr() method.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-integrate-poly")

        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def __getnewargs__(self):
        return (self.function,) + tuple([tuple(xab) for xab in self.limits])

    @property
    def free_symbols(self):
        """
        This method returns the symbols that will exist when the
        integral is evaluated. This is useful if one is trying to
        determine whether an integral depends on a certain
        symbol or not.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x, (x, y, 1)).free_symbols
        {y}

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.function
        sympy.concrete.expr_with_limits.ExprWithLimits.limits
        sympy.concrete.expr_with_limits.ExprWithLimits.variables
        """
        return super().free_symbols

    def _eval_is_zero(self):
        # This is a very naive and quick test, not intended to do the integral to
        # answer whether it is zero or not, e.g. Integral(sin(x), (x, 0, 2*pi))
        # is zero but this routine should return None for that case. But, like
        # Mul, there are trivial situations for which the integral will be
        # zero so we check for those.
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            # take integration symbol out of free since it will be replaced
            # with the free symbols in the limits
            free.discard(xab[0])
            # add in the new symbols
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False

    def transform(self, x, u):
        r"""
        Performs a change of variables from `x` to `u` using the relationship
        given by `x` and `u` which will define the transformations `f` and `F`
        (which are inverses of each other) as follows:

        1) If `x` is a Symbol (which is a variable of integration) then `u`
           will be interpreted as some function, f(u), with inverse F(u).
           This, in effect, just makes the substitution of x with f(x).

        2) If `u` is a Symbol then `x` will be interpreted as some function,
           F(x), with inverse f(u). This is commonly referred to as
           u-substitution.

        Once f and F have been identified, the transformation is made as
        follows:

        .. math:: \int_a^b x \mathrm{d}x \rightarrow \int_{F(a)}^{F(b)} f(x)
                  \frac{\mathrm{d}}{\mathrm{d}x}

        where `F(x)` is the inverse of `f(x)` and the limits and integrand have
        been corrected so as to retain the same value after integration.

        Notes
        =====

        The mappings, F(x) or f(u), must lead to a unique integral. Linear
        or rational linear expression, ``2*x``, ``1/x`` and ``sqrt(x)``, will
        always work; quadratic expressions like ``x**2 - 1`` are acceptable
        as long as the resulting integrand does not depend on the sign of
        the solutions (see examples).

        The integral will be returned unchanged if ``x`` is not a variable of
        integration.

        ``x`` must be (or contain) only one of of the integration variables. If
        ``u`` has more than one free symbol then it should be sent as a tuple
        (``u``, ``uvar``) where ``uvar`` identifies which variable is replacing
        the integration variable.
        XXX can it contain another integration variable?

        Examples
        ========

        >>> from sympy.abc import a, x, u
        >>> from sympy import Integral, cos, sqrt

        >>> i = Integral(x*cos(x**2 - 1), (x, 0, 1))

        transform can change the variable of integration

        >>> i.transform(x, u)
        Integral(u*cos(u**2 - 1), (u, 0, 1))

        transform can perform u-substitution as long as a unique
        integrand is obtained:

        >>> ui = i.transform(x**2 - 1, u)
        >>> ui
        Integral(cos(u)/2, (u, -1, 0))

        This attempt fails because x = +/-sqrt(u + 1) and the
        sign does not cancel out of the integrand:

        >>> Integral(cos(x**2 - 1), (x, 0, 1)).transform(x**2 - 1, u)
        Traceback (most recent call last):
        ...
        ValueError:
        The mapping between F(x) and f(u) did not give a unique integrand.

        transform can do a substitution. Here, the previous
        result is transformed back into the original expression
        using "u-substitution":

        >>> ui.transform(sqrt(u + 1), x) == i
        True

        We can accomplish the same with a regular substitution:

        >>> ui.transform(u, x**2 - 1) == i
        True

        If the `x` does not contain a symbol of integration then
        the integral will be returned unchanged. Integral `i` does
        not have an integration variable `a` so no change is made:

        >>> i.transform(a, x) == i
        True

        When `u` has more than one free symbol the symbol that is
        replacing `x` must be identified by passing `u` as a tuple:

        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, u))
        Integral(a + u, (u, -a, 1 - a))
        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, a))
        Integral(a + u, (a, -u, 1 - u))

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.variables : Lists the integration variables
        as_dummy : Replace integration variables with dummy ones
        """
        d = Dummy('d')

        xfree = x.free_symbols.intersection(self.variables)
        if len(xfree) > 1:
            raise ValueError(
                'F(x) can only contain one of: %s' % self.variables)
        xvar = xfree.pop() if xfree else d

        if xvar not in self.variables:
            return self

        u = sympify(u)
        if isinstance(u, Expr):
            ufree = u.free_symbols
            if len(ufree) == 0:
                raise ValueError(filldedent('''
                f(u) cannot be a constant'''))
            if len(ufree) > 1:
                raise ValueError(filldedent('''
                When f(u) has more than one free symbol, the one replacing x
                must be identified: pass f(u) as (f(u), u)'''))
            uvar = ufree.pop()
        else:
            u, uvar = u
            if uvar not in u.free_symbols:
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) where symbol identified
                a free symbol in expr, but symbol is not in expr's free
                symbols.'''))
            if not isinstance(uvar, Symbol):
                # This probably never evaluates to True
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) but didn't get
                a symbol; got %s''' % uvar))

        if x.is_Symbol and u.is_Symbol:
            return self.xreplace({x: u})

        if not x.is_Symbol and not u.is_Symbol:
            raise ValueError('either x or u must be a symbol')

        if uvar == xvar:
            return self.transform(x, (u.subs(uvar, d), d)).xreplace({d: uvar})

        if uvar in self.limits:
            raise ValueError(filldedent('''
            u must contain the same variable as in x
            or a variable that is not already an integration variable'''))

        from sympy.solvers.solvers import solve
        if not x.is_Symbol:
            F = [x.subs(xvar, d)]
            soln = solve(u - x, xvar, check=False)
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), x)')
            f = [fi.subs(uvar, d) for fi in soln]
        else:
            f = [u.subs(uvar, d)]
            from sympy.simplify.simplify import posify
            pdiff, reps = posify(u - x)
            puvar = uvar.subs([(v, k) for k, v in reps.items()])
            soln = [s.subs(reps) for s in solve(pdiff, puvar)]
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), u)')
            F = [fi.subs(xvar, d) for fi in soln]

        newfuncs = {(self.function.subs(xvar, fi)*fi.diff(d)
                        ).subs(d, uvar) for fi in f}
        if len(newfuncs) > 1:
            raise ValueError(filldedent('''
            The mapping between F(x) and f(u) did not give
            a unique integrand.'''))
        newfunc = newfuncs.pop()

        def _calc_limit_1(F, a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            wok = F.subs(d, a)
            if wok is S.NaN or wok.is_finite is False and a.is_finite:
                return limit(sign(b)*F, d, a)
            return wok

        def _calc_limit(a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            avals = list({_calc_limit_1(Fi, a, b) for Fi in F})
            if len(avals) > 1:
                raise ValueError(filldedent('''
                The mapping between F(x) and f(u) did not
                give a unique limit.'''))
            return avals[0]

        newlimits = []
        for xab in self.limits:
            sym = xab[0]
            if sym == xvar:
                if len(xab) == 3:
                    a, b = xab[1:]
                    a, b = _calc_limit(a, b), _calc_limit(b, a)
                    if fuzzy_bool(a - b > 0):
                        a, b = b, a
                        newfunc = -newfunc
                    newlimits.append((uvar, a, b))
                elif len(xab) == 2:
                    a = _calc_limit(xab[1], 1)
                    newlimits.append((uvar, a))
                else:
                    newlimits.append(uvar)
            else:
                newlimits.append(xab)

        return self.func(newfunc, *newlimits)

    def doit(self, **hints):
        """
        Perform the integration using any hints given.

        Examples
        ========

        >>> from sympy import Piecewise, S
        >>> from sympy.abc import x, t
        >>> p = x**2 + Piecewise((0, x/t < 0), (1, True))
        >>> p.integrate((t, S(4)/5, 1), (x, -1, 1))
        1/3

        See Also
        ========

        sympy.integrals.trigonometry.trigintegrate
        sympy.integrals.heurisch.heurisch
        sympy.integrals.rationaltools.ratint
        as_sum : Approximate the integral using a sum
        """
        if not hints.get('integrals', True):
            return self

        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        heurisch = hints.get('heurisch', None)
        manual = hints.get('manual', None)
        if len(list(filter(None, (manual, meijerg, risch, heurisch)))) > 1:
            raise ValueError("At most one of manual, meijerg, risch, heurisch can be True")
        elif manual:
            meijerg = risch = heurisch = False
        elif meijerg:
            manual = risch = heurisch = False
        elif risch:
            manual = meijerg = heurisch = False
        elif heurisch:
            manual = meijerg = risch = False
        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual, "heurisch": heurisch,
            "conds": conds}

        if conds not in ('separate', 'piecewise', 'none'):
            raise ValueError('conds must be one of "separate", "piecewise", '
                             '"none", got: %s' % conds)

        if risch and any(len(xab) > 1 for xab in self.limits):
            raise ValueError('risch=True is only allowed for indefinite integrals.')

        # check for the trivial zero
        if self.is_zero:
            return S.Zero

        # hacks to handle integrals of
        # nested summations
        from sympy.concrete.summations import Sum
        if isinstance(self.function, Sum):
            if any(v in self.function.limits[0] for v in self.variables):
                raise ValueError('Limit of the sum cannot be an integration variable.')
            if any(l.is_infinite for l in self.function.limits[0][1:]):
                return self
            _i = self
            _sum = self.function
            return _sum.func(_i.func(_sum.function, *_i.limits).doit(), *_sum.limits).doit()

        # now compute and check the function
        function = self.function

        # hack to use a consistent Heaviside(x, 1/2)
        function = function.replace(
            lambda x: isinstance(x, Heaviside) and x.args[1]*2 != 1,
            lambda x: Heaviside(x.args[0]))

        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero

        # hacks to handle special cases
        if isinstance(function, MatrixBase):
            return function.applyfunc(
                lambda f: self.func(f, *self.limits).doit(**hints))

        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)

        # There is no trivial answer and special handling
        # is done so continue

        # first make sure any definite limits have integration
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            if len(xab) != 3:
                # it makes sense to just make
                # all x real but in practice with the
                # current state of integration...this
                # doesn't work out well
                # x = xab[0]
                # if x not in reps and not x.is_real:
                #     reps[x] = Dummy(real=True)
                continue
            x, a, b = xab
            l = (a, b)
            if all(i.is_nonnegative for i in l) and not x.is_nonnegative:
                d = Dummy(positive=True)
            elif all(i.is_nonpositive for i in l) and not x.is_nonpositive:
                d = Dummy(negative=True)
            elif all(i.is_real for i in l) and not x.is_real:
                d = Dummy(real=True)
            else:
                d = None
            if d:
                reps[x] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        # continue with existing assumptions
        undone_limits = []
        # ulj = free symbols of any undone limits' upper and lower limits
        ulj = set()
        for xab in self.limits:
            # compute uli, the free symbols in the
            # Upper and Lower limits of limit I
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            # this integral can be done as long as there is no blocking
            # limit that has been undone. An undone limit is blocking if
            # it contains an integration variable that is in this limit's
            # upper or lower free symbols or vice versa
            if xab[0] in ulj or any(v[0] in uli for v in undone_limits):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*([function] + [xab]))
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue

            if function.has(Abs, sign) and (
                (len(xab) < 3 and all(x.is_extended_real for x in xab)) or
                (len(xab) == 3 and all(x.is_extended_real and not x.is_infinite for
                 x in xab[1:]))):
                    # some improper integrals are better off with Abs
                    xr = Dummy("xr", real=True)
                    function = (function.xreplace({xab[0]: xr})
                        .rewrite(Piecewise).xreplace({xr: xab[0]}))
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if (function.has(Piecewise) and
                not isinstance(function, Piecewise)):
                    function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0],
                        **eval_kwargs)
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
            else:
                # There are a number of tradeoffs in using the
                # Meijer G method. It can sometimes be a lot faster
                # than other methods, and sometimes slower. And
                # there are certain types of integrals for which it
                # is more likely to work than others. These
                # heuristics are incorporated in deciding what
                # integration methods to try, in what order. See the
                # integrate() docstring for details.
                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            _debug('NotImplementedError '
                                'from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                u = self.func(function, (x, a, b))
                                # if Piecewise modifies cond too
                                # much it may not be recognized by
                                # _condsimp pattern matching so just
                                # turn off all evaluation
                                return Piecewise((f, cond), (u, True),
                                    evaluate=False)
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('''
                                        conds=separate not supported in
                                        multiple integrals'''))
                                ret = f, cond
                            else:
                                ret = f
                    return ret

                meijerg1 = meijerg
                if (meijerg is not False and
                        len(xab) == 3 and xab[1].is_extended_real and xab[2].is_extended_real
                        and not function.is_Poly and
                        (xab[1].has(oo, -oo) or xab[2].has(oo, -oo))):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                # If the special meijerg code did not succeed in
                # finding a definite integral, then the code using
                # meijerint_indefinite will not either (it might
                # find an antiderivative, but the answer is likely
                # to be nonsensical). Thus if we are requested to
                # only use Meijer G-function methods, we give up at
                # this stage. Otherwise we just disable G-function
                # methods.
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue

            final = hints.get('final', True)
            # dotit may be iterated but floor terms making atan and acot
            # continuous should only be added in the final round
            if (final and not isinstance(antideriv, Integral) and
                antideriv is not None):
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        # The coefficient of `tan` should be constant
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a-pi/2)/pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        # The coefficient of `cot` should be constant
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a)/pi)))

            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*([function] + [xab])).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            else:
                if len(xab) == 1:
                    function = antideriv
                else:
                    if len(xab) == 3:
                        x, a, b = xab
                    elif len(xab) == 2:
                        x, b = xab
                        a = None
                    else:
                        raise NotImplementedError

                    if deep:
                        if isinstance(a, Basic):
                            a = a.doit(**hints)
                        if isinstance(b, Basic):
                            b = b.doit(**hints)

                    if antideriv.is_Poly:
                        gens = list(antideriv.gens)
                        gens.remove(x)

                        antideriv = antideriv.as_expr()

                        function = antideriv._eval_interval(x, a, b)
                        function = Poly(function, *gens)
                    else:
                        def is_indef_int(g, x):
                            return (isinstance(g, Integral) and
                                    any(i == (x,) for i in g.limits))

                        def eval_factored(f, x, a, b):
                            # _eval_interval for integrals with
                            # (constant) factors
                            # a single indefinite integral is assumed
                            args = []
                            for g in Mul.make_args(f):
                                if is_indef_int(g, x):
                                    args.append(g._eval_interval(x, a, b))
                                else:
                                    args.append(g)
                            return Mul(*args)

                        integrals, others, piecewises = [], [], []
                        for f in Add.make_args(antideriv):
                            if any(is_indef_int(g, x)
                                   for g in Mul.make_args(f)):
                                integrals.append(f)
                            elif any(isinstance(g, Piecewise)
                                     for g in Mul.make_args(f)):
                                piecewises.append(piecewise_fold(f))
                            else:
                                others.append(f)
                        uneval = Add(*[eval_factored(f, x, a, b)
                                       for f in integrals])
                        try:
                            evalued = Add(*others)._eval_interval(x, a, b)
                            evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                            function = uneval + evalued + evalued_pw
                        except NotImplementedError:
                            # This can happen if _eval_interval depends in a
                            # complicated way on limits that cannot be computed
                            undone_limits.append(xab)
                            function = self.func(*([function] + [xab]))
                            factored_function = function.factor()
                            if not isinstance(factored_function, Integral):
                                function = factored_function
        return function

    def _eval_derivative(self, sym):
        """Evaluate the derivative of the current Integral object by
        differentiating under the integral sign [1], using the Fundamental
        Theorem of Calculus [2] when possible.

        Explanation
        ===========

        Whenever an Integral is encountered that is equivalent to zero or
        has an integrand that is independent of the variable of integration
        those integrals are performed. All others are returned as Integral
        instances which can be resolved with doit() (provided they are integrable).

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Differentiation_under_the_integral_sign
        .. [2] https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> i = Integral(x + y, y, (y, 1, x))
        >>> i.diff(x)
        Integral(x + y, (y, x)) + Integral(1, y, (y, 1, x))
        >>> i.doit().diff(x) == i.diff(x).doit()
        True
        >>> i.diff(y)
        0

        The previous must be true since there is no y in the evaluated integral:

        >>> i.free_symbols
        {x}
        >>> i.doit()
        2*x**3/3 - x/2 - 1/6

        """

        # differentiate under the integral sign; we do not
        # check for regularity conditions (TODO), see issue 4215

        # get limits and the function
        f, limits = self.function, list(self.limits)

        # the order matters if variables of integration appear in the limits
        # so work our way in from the outside to the inside.
        limit = limits.pop(-1)
        if len(limit) == 3:
            x, a, b = limit
        elif len(limit) == 2:
            x, b = limit
            a = None
        else:
            a = b = None
            x = limit[0]

        if limits:  # f is the argument to an integral
            f = self.func(f, *tuple(limits))

        # assemble the pieces
        def _do(f, ab):
            dab_dsym = diff(ab, sym)
            if not dab_dsym:
                return S.Zero
            if isinstance(f, Integral):
                limits = [(x, x) if (len(l) == 1 and l[0] == x) else l
                          for l in f.limits]
                f = self.func(f.function, *limits)
            return f.subs(x, ab)*dab_dsym

        rv = S.Zero
        if b is not None:
            rv += _do(f, b)
        if a is not None:
            rv -= _do(f, a)
        if len(limit) == 1 and sym == x:
            # the dummy variable *is* also the real-world variable
            arg = f
            rv += arg
        else:
            # the dummy variable might match sym but it's
            # only a dummy and the actual variable is determined
            # by the limits, so mask off the variable of integration
            # while differentiating
            u = Dummy('u')
            arg = f.subs(x, u).diff(sym).subs(u, x)
            if arg:
                rv += self.func(arg, (x, a, b))
        return rv

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       heurisch=None, conds='piecewise',final=None):
        """
        Calculate the anti-derivative to the function f(x).

        Explanation
        ===========

        The following algorithms are applied (roughly in this order):

        1. Simple heuristics (based on pattern matching and integral table):

           - most frequently used functions (e.g. polynomials, products of
             trig functions)

        2. Integration of rational functions:

           - A complete algorithm for integrating rational functions is
             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm
             also uses the partial fraction decomposition algorithm
             implemented in apart() as a preprocessor to make this process
             faster.  Note that the integral of a rational function is always
             elementary, but in general, it may include a RootSum.

        3. Full Risch algorithm:

           - The Risch algorithm is a complete decision
             procedure for integrating elementary functions, which means that
             given any elementary function, it will either compute an
             elementary antiderivative, or else prove that none exists.
             Currently, part of transcendental case is implemented, meaning
             elementary integrals containing exponentials, logarithms, and
             (soon!) trigonometric functions can be computed.  The algebraic
             case, e.g., functions containing roots, is much more difficult
             and is not implemented yet.

           - If the routine fails (because the integrand is not elementary, or
             because a case is not implemented yet), it continues on to the
             next algorithms below.  If the routine proves that the integrals
             is nonelementary, it still moves on to the algorithms below,
             because we might be able to find a closed-form solution in terms
             of special functions.  If risch=True, however, it will stop here.

        4. The Meijer G-Function algorithm:

           - This algorithm works by first rewriting the integrand in terms of
             very general Meijer G-Function (meijerg in SymPy), integrating
             it, and then rewriting the result back, if possible.  This
             algorithm is particularly powerful for definite integrals (which
             is actually part of a different method of Integral), since it can
             compute closed-form solutions of definite integrals even when no
             closed-form indefinite integral exists.  But it also is capable
             of computing many indefinite integrals as well.

           - Another advantage of this method is that it can use some results
             about the Meijer G-Function to give a result in terms of a
             Piecewise expression, which allows to express conditionally
             convergent integrals.

           - Setting meijerg=True will cause integrate() to use only this
             method.

        5. The "manual integration" algorithm:

           - This algorithm tries to mimic how a person would find an
             antiderivative by hand, for example by looking for a
             substitution or applying integration by parts. This algorithm
             does not handle as many integrands but can return results in a
             more familiar form.

           - Sometimes this algorithm can evaluate parts of an integral; in
             this case integrate() will try to evaluate the rest of the
             integrand using the other methods here.

           - Setting manual=True will cause integrate() to use only this
             method.

        6. The Heuristic Risch algorithm:

           - This is a heuristic version of the Risch algorithm, meaning that
             it is not deterministic.  This is tried as a last resort because
             it can be very slow.  It is still used because not enough of the
             full Risch algorithm is implemented, so that there are still some
             integrals that can only be computed using this method.  The goal
             is to implement enough of the Risch and Meijer G-function methods
             so that this can be deleted.

             Setting heurisch=True will cause integrate() to use only this
             method. Set heurisch=False to not use it.

        """

        from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
        from sympy.integrals.manualintegrate import manualintegrate

        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None

        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass

        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual,
            "heurisch": heurisch, "conds": conds}

        # if it is a poly(x) then let the polynomial integrate itself (fast)
        #
        # It is important to make this check first, otherwise the other code
        # will return a SymPy expression instead of a Polynomial.
        #
        # see Polynomial for details.
        if isinstance(f, Poly) and not (manual or meijerg or risch):
            # Note: this is deprecated, but the deprecation warning is already
            # issued in the Integral constructor.
            return f.integrate(x)

        # Piecewise antiderivatives need to call special integrate.
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)

        # let's cut it short if `f` does not depend on `x`; if
        # x is only a dummy, that will be handled below
        if not f.has(x):
            return f*x

        # try to convert to poly(x) and then integrate if successful (fast)
        poly = f.as_poly(x)
        if poly is not None and not (manual or meijerg or risch):
            return poly.integrate().as_expr()

        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True,
                    conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    # There was a nonelementary integral. Try integrating it.

                    # if no part of the NonElementaryIntegral is integrated by
                    # the Risch algorithm, then use the original function to
                    # integrate, instead of re-written one
                    if result == 0:
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result

        # since Integral(f=g1+g2+...) == Integral(g1) + Integral(g2) + ...
        # we are going to handle Add terms separately,
        # if `f` is not Add -- we only have one term

        # Note that in general, this is a bad idea, because Integral(g1) +
        # Integral(g2) might not be computable, even if Integral(g1 + g2) is.
        # For example, Integral(x**x + x**x*log(x)).  But many heuristics only
        # work term-wise.  So we compute this step last, after trying
        # risch_integrate.  We also try risch_integrate again in this loop,
        # because maybe the integral is a sum of an elementary part and a
        # nonelementary part (like erf(x) + exp(x)).  risch_integrate() is
        # quite fast, so this is acceptable.
        from sympy.simplify.fu import sincos_to_sum
        parts = []
        args = Add.make_args(f)
        for g in args:
            coeff, g = g.as_independent(x)

            # g(x) = const
            if g is S.One and not meijerg:
                parts.append(coeff*x)
                continue

            # g(x) = expr + O(x**n)
            order_term = g.getO()

            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)

                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)

                    if h_order_expr is not None:
                        h_order_term = order_term.func(
                            h_order_expr, *order_term.variables)
                        parts.append(coeff*(h + h_order_term))
                        continue

                # NOTE: if there is O(x**n) and we fail to integrate then
                # there is no point in trying other methods because they
                # will fail, too.
                return None

            #               c
            # g(x) = (a*x+b)
            if g.is_Pow and not g.exp.has(x) and not meijerg:
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])

                M = g.base.match(a*x + b)

                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base**(g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base**(g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))

                    parts.append(coeff * h / M[a])
                    continue

            #        poly(x)
            # g(x) = -------
            #        poly(x)
            if g.is_rational_function(x) and not (manual or meijerg or risch):
                parts.append(coeff * ratint(g, x))
                continue

            if not (manual or meijerg or risch):
                # g(x) = Mul(trig)
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a DiracDelta term
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                from .singularityfunctions import singularityintegrate
                # g(x) has at least a Singularity Function term
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # Try risch again.
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x,
                            separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)

                        parts.append(coeff*h)
                        continue

                # fall back to heurisch
                if heurisch is not False:
                    from sympy.integrals.heurisch import (heurisch as heurisch_,
                                                          heurisch_wrapper)
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        # XXX: this exception means there is a bug in the
                        # implementation of heuristic Risch integration
                        # algorithm.
                        h = None
            else:
                h = None

            if meijerg is not False and h is None:
                # rewrite using G functions
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    _debug('NotImplementedError from meijerint_definite')
                if h is not None:
                    parts.append(coeff * h)
                    continue

            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and not isinstance(result, Integral):
                        if result.has(Integral) and not manual:
                            # Try to have other algorithms do the integrals
                            # manualintegrate can't handle,
                            # unless we were asked to use manual only.
                            # Keep the rest of eval_kwargs in case another
                            # method was set to False already
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs["manual"] = False
                            new_eval_kwargs["final"] = False
                            result = result.func(*[
                                arg.doit(**new_eval_kwargs) if
                                arg.has(Integral) else arg
                                for arg in result.args
                            ]).expand(multinomial=False,
                                      log=False,
                                      power_exp=False,
                                      power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    # can't handle some SymPy expressions
                    pass

            # if we failed maybe it was because we had
            # a product that could have been expanded,
            # so let's try an expansion of the whole
            # thing before giving up; we don't try this
            # at the outset because there are things
            # that cannot be solved unless they are
            # NOT expanded e.g., x**x*(1+log(x)). There
            # should probably be a checker somewhere in this
            # routine to look for such cases and try to do
            # collection on the expressions if they are already
            # in an expanded form
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    # Note: risch will be identical on the expanded
                    # expression, but maybe it will be able to pick out parts,
                    # like x*(exp(x) + erf(x)).
                    return self._eval_integral(f, x, **eval_kwargs)

            if h is not None:
                parts.append(coeff * h)
            else:
                return None

        return Add(*parts)

    def _eval_lseries(self, x, logx=None, cdir=0):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        symb = x
        for l in self.limits:
            if x in l[1:]:
                symb = l[0]
                break
        terms, order = self.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *self.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])

    def _eval_simplify(self, **kwargs):
        expr = factor_terms(self)
        if isinstance(expr, Integral):
            from sympy.simplify.simplify import simplify
            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
        return expr.simplify(**kwargs)

    def as_sum(self, n=None, method="midpoint", evaluate=True):
        """
        Approximates a definite integral by a sum.

        Parameters
        ==========

        n :
            The number of subintervals to use, optional.
        method :
            One of: 'left', 'right', 'midpoint', 'trapezoid'.
        evaluate : bool
            If False, returns an unevaluated Sum expression. The default
            is True, evaluate the sum.

        Notes
        =====

        These methods of approximate integration are described in [1].

        Examples
        ========

        >>> from sympy import Integral, sin, sqrt
        >>> from sympy.abc import x, n
        >>> e = Integral(sin(x), (x, 3, 7))
        >>> e
        Integral(sin(x), (x, 3, 7))

        For demonstration purposes, this interval will only be split into 2
        regions, bounded by [3, 5] and [5, 7].

        The left-hand rule uses function evaluations at the left of each
        interval:

        >>> e.as_sum(2, 'left')
        2*sin(5) + 2*sin(3)

        The midpoint rule uses evaluations at the center of each interval:

        >>> e.as_sum(2, 'midpoint')
        2*sin(4) + 2*sin(6)

        The right-hand rule uses function evaluations at the right of each
        interval:

        >>> e.as_sum(2, 'right')
        2*sin(5) + 2*sin(7)

        The trapezoid rule uses function evaluations on both sides of the
        intervals. This is equivalent to taking the average of the left and
        right hand rule results:

        >>> s = e.as_sum(2, 'trapezoid')
        >>> s
        2*sin(5) + sin(3) + sin(7)
        >>> (e.as_sum(2, 'left') + e.as_sum(2, 'right'))/2 == s
        True

        Here, the discontinuity at x = 0 can be avoided by using the
        midpoint or right-hand method:

        >>> e = Integral(1/sqrt(x), (x, 0, 1))
        >>> e.as_sum(5).n(4)
        1.730
        >>> e.as_sum(10).n(4)
        1.809
        >>> e.doit().n(4)  # the actual value is 2
        2.000

        The left- or trapezoid method will encounter the discontinuity and
        return infinity:

        >>> e.as_sum(5, 'left')
        zoo

        The number of intervals can be symbolic. If omitted, a dummy symbol
        will be used for it.

        >>> e = Integral(x**2, (x, 0, 2))
        >>> e.as_sum(n, 'right').expand()
        8/3 + 4/n + 4/(3*n**2)

        This shows that the midpoint rule is more accurate, as its error
        term decays as the square of n:

        >>> e.as_sum(method='midpoint').expand()
        8/3 - 2/(3*_n**2)

        A symbolic sum is returned with evaluate=False:

        >>> e.as_sum(n, 'midpoint', evaluate=False)
        2*Sum((2*_k/n - 1/n)**2, (_k, 1, n))/n

        See Also
        ========

        Integral.doit : Perform the integration using any hints

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Riemann_sum#Riemann_summation_methods
        """

        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError(
                "Multidimensional midpoint rule not implemented yet")
        else:
            limit = limits[0]
            if (len(limit) != 3 or limit[1].is_finite is False or
                limit[2].is_finite is False):
                raise ValueError("Expecting a definite integral over "
                                  "a finite interval.")
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if (n.is_positive is False or n.is_integer is False or
            n.is_finite is False):
            raise ValueError("n must be a positive integer, got %s" % n)
        x, a, b = limit
        dx = (b - a)/n
        k = Dummy('k', integer=True, positive=True)
        f = self.function

        if method == "left":
            result = dx*Sum(f.subs(x, a + (k-1)*dx), (k, 1, n))
        elif method == "right":
            result = dx*Sum(f.subs(x, a + k*dx), (k, 1, n))
        elif method == "midpoint":
            result = dx*Sum(f.subs(x, a + k*dx - dx/2), (k, 1, n))
        elif method == "trapezoid":
            result = dx*((f.subs(x, a) + f.subs(x, b))/2 +
                Sum(f.subs(x, a + k*dx), (k, 1, n - 1)))
        else:
            raise ValueError("Unknown method %s" % method)
        return result.doit() if evaluate else result

    def principal_value(self, **kwargs):
        """
        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval
        on the real axis.

        Explanation
        ===========

        In mathematics, the Cauchy principal value, is a method for assigning values to certain improper
        integrals which would otherwise be undefined.

        Examples
        ========

        >>> from sympy import Integral, oo
        >>> from sympy.abc import x
        >>> Integral(x+1, (x, -oo, oo)).principal_value()
        oo
        >>> f = 1 / (x**3)
        >>> Integral(f, (x, -oo, oo)).principal_value()
        0
        >>> Integral(f, (x, -10, 10)).principal_value()
        0
        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()
        0

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value
        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html
        """
        if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
            raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate "
                             "cauchy's principal value")
        x, a, b = self.limits[0]
        if not (a.is_comparable and b.is_comparable and a <= b):
            raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate "
                             "cauchy's principal value. Also, a and b need to be comparable.")
        if a == b:
            return S.Zero

        from sympy.calculus.singularities import singularities

        r = Dummy('r')
        f = self.function
        singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]
        for i in singularities_list:
            if i in (a, b):
                raise ValueError(
                    'The principal value is not defined in the given interval due to singularity at %d.' % (i))
        F = integrate(f, x, **kwargs)
        if F.has(Integral):
            return self
        if a is -oo and b is oo:
            I = limit(F - F.subs(x, -x), x, oo)
        else:
            I = limit(F, x, b, '-') - limit(F, x, a, '+')
        for s in singularities_list:
            I += limit(((F.subs(x, s - r)) - F.subs(x, s + r)), r, 0, '+')
        return I



def integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs):
    """integrate(f, var, ...)

    .. deprecated:: 1.6

       Using ``integrate()`` with :class:`~.Poly` is deprecated. Use
       :meth:`.Poly.integrate` instead. See :ref:`deprecated-integrate-poly`.

    Explanation
    ===========

    Compute definite or indefinite integral of one or more variables
    using Risch-Norman algorithm and table lookup. This procedure is
    able to handle elementary algebraic and transcendental functions
    and also a huge class of special functions, including Airy,
    Bessel, Whittaker and Lambert.

    var can be:

    - a symbol                   -- indefinite integration
    - a tuple (symbol, a)        -- indefinite integration with result
                                    given with ``a`` replacing ``symbol``
    - a tuple (symbol, a, b)     -- definite integration

    Several variables can be specified, in which case the result is
    multiple integration. (If var is omitted and the integrand is
    univariate, the indefinite integral in that variable will be performed.)

    Indefinite integrals are returned without terms that are independent
    of the integration variables. (see examples)

    Definite improper integrals often entail delicate convergence
    conditions. Pass conds='piecewise', 'separate' or 'none' to have
    these returned, respectively, as a Piecewise function, as a separate
    result (i.e. result will be a tuple), or not at all (default is
    'piecewise').

    **Strategy**

    SymPy uses various approaches to definite integration. One method is to
    find an antiderivative for the integrand, and then use the fundamental
    theorem of calculus. Various functions are implemented to integrate
    polynomial, rational and trigonometric functions, and integrands
    containing DiracDelta terms.

    SymPy also implements the part of the Risch algorithm, which is a decision
    procedure for integrating elementary functions, i.e., the algorithm can
    either find an elementary antiderivative, or prove that one does not
    exist.  There is also a (very successful, albeit somewhat slow) general
    implementation of the heuristic Risch algorithm.  This algorithm will
    eventually be phased out as more of the full Risch algorithm is
    implemented. See the docstring of Integral._eval_integral() for more
    details on computing the antiderivative using algebraic methods.

    The option risch=True can be used to use only the (full) Risch algorithm.
    This is useful if you want to know if an elementary function has an
    elementary antiderivative.  If the indefinite Integral returned by this
    function is an instance of NonElementaryIntegral, that means that the
    Risch algorithm has proven that integral to be non-elementary.  Note that
    by default, additional methods (such as the Meijer G method outlined
    below) are tried on these integrals, as they may be expressible in terms
    of special functions, so if you only care about elementary answers, use
    risch=True.  Also note that an unevaluated Integral returned by this
    function is not necessarily a NonElementaryIntegral, even with risch=True,
    as it may just be an indication that the particular part of the Risch
    algorithm needed to integrate that function is not yet implemented.

    Another family of strategies comes from re-writing the integrand in
    terms of so-called Meijer G-functions. Indefinite integrals of a
    single G-function can always be computed, and the definite integral
    of a product of two G-functions can be computed from zero to
    infinity. Various strategies are implemented to rewrite integrands
    as G-functions, and use this information to compute integrals (see
    the ``meijerint`` module).

    The option manual=True can be used to use only an algorithm that tries
    to mimic integration by hand. This algorithm does not handle as many
    integrands as the other algorithms implemented but may return results in
    a more familiar form. The ``manualintegrate`` module has functions that
    return the steps used (see the module docstring for more information).

    In general, the algebraic methods work best for computing
    antiderivatives of (possibly complicated) combinations of elementary
    functions. The G-function methods work best for computing definite
    integrals from zero to infinity of moderately complicated
    combinations of special functions, or indefinite integrals of very
    simple combinations of special functions.

    The strategy employed by the integration code is as follows:

    - If computing a definite integral, and both limits are real,
      and at least one limit is +- oo, try the G-function method of
      definite integration first.

    - Try to find an antiderivative, using all available methods, ordered
      by performance (that is try fastest method first, slowest last; in
      particular polynomial integration is tried first, Meijer
      G-functions second to last, and heuristic Risch last).

    - If still not successful, try G-functions irrespective of the
      limits.

    The option meijerg=True, False, None can be used to, respectively:
    always use G-function methods and no others, never use G-function
    methods, or use all available methods (in order as described above).
    It defaults to None.

    Examples
    ========

    >>> from sympy import integrate, log, exp, oo
    >>> from sympy.abc import a, x, y

    >>> integrate(x*y, x)
    x**2*y/2

    >>> integrate(log(x), x)
    x*log(x) - x

    >>> integrate(log(x), (x, 1, a))
    a*log(a) - a + 1

    >>> integrate(x)
    x**2/2

    Terms that are independent of x are dropped by indefinite integration:

    >>> from sympy import sqrt
    >>> integrate(sqrt(1 + x), (x, 0, x))
    2*(x + 1)**(3/2)/3 - 2/3
    >>> integrate(sqrt(1 + x), x)
    2*(x + 1)**(3/2)/3

    >>> integrate(x*y)
    Traceback (most recent call last):
    ...
    ValueError: specify integration variables to integrate x*y

    Note that ``integrate(x)`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    >>> integrate(x**a*exp(-x), (x, 0, oo)) # same as conds='piecewise'
    Piecewise((gamma(a + 1), re(a) > -1),
        (Integral(x**a*exp(-x), (x, 0, oo)), True))

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='none')
    gamma(a + 1)

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='separate')
    (gamma(a + 1), re(a) > -1)

    See Also
    ========

    Integral, Integral.doit

    """
    doit_flags = {
        'deep': False,
        'meijerg': meijerg,
        'conds': conds,
        'risch': risch,
        'heurisch': heurisch,
        'manual': manual
        }
    integral = Integral(*args, **kwargs)

    if isinstance(integral, Integral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, Integral) else a
            for a in integral.args]
        return integral.func(*new_args)


def line_integrate(field, curve, vars):
    """line_integrate(field, Curve, variables)

    Compute the line integral.

    Examples
    ========

    >>> from sympy import Curve, line_integrate, E, ln
    >>> from sympy.abc import x, y, t
    >>> C = Curve([E**t + 1, E**t - 1], (t, 0, ln(2)))
    >>> line_integrate(x + y, C, [x, y])
    3*sqrt(2)

    See Also
    ========

    sympy.integrals.integrals.integrate, Integral
    """
    from sympy.geometry import Curve
    F = sympify(field)
    if not F:
        raise ValueError(
            "Expecting function specifying field as first argument.")
    if not isinstance(curve, Curve):
        raise ValueError("Expecting Curve entity as second argument.")
    if not is_sequence(vars):
        raise ValueError("Expecting ordered iterable for variables.")
    if len(curve.functions) != len(vars):
        raise ValueError("Field variable size does not match curve dimension.")

    if curve.parameter in vars:
        raise ValueError("Curve parameter clashes with field parameters.")

    # Calculate derivatives for line parameter functions
    # F(r) -> F(r(t)) and finally F(r(t)*r'(t))
    Ft = F
    dldt = 0
    for i, var in enumerate(vars):
        _f = curve.functions[i]
        _dn = diff(_f, curve.parameter)
        # ...arc length
        dldt = dldt + (_dn * _dn)
        Ft = Ft.subs(var, _f)
    Ft = Ft * sqrt(dldt)

    integral = Integral(Ft, curve.limits).doit(deep=False)
    return integral


### Property function dispatching ###

@shape.register(Integral)
def _(expr):
    return shape(expr.function)

# Delayed imports
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
