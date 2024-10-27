from .add import Add
from .exprtools import gcd_terms
from .function import Function
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .singleton import S


class Mod(Function):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """

    kind = NumberKind

    @classmethod
    def eval(cls, p, q):
        def number_eval(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """

            if q.is_zero:
                raise ZeroDivisionError("Modulo by zero")
            if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
                return S.NaN
            if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
                return S.Zero

            if q.is_Number:
                if p.is_Number:
                    return p%q
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One

            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv

            # by ratio
            r = p/q
            if r.is_integer:
                return S.Zero
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if isinstance(d, int):
                    rv = p - d*q
                    if (rv*q < 0) == True:
                        rv += q
                    return rv

            # by difference
            # -2|q| < p < 2|q|
            d = abs(p)
            for _ in range(2):
                d -= abs(q)
                if d.is_negative:
                    if q.is_positive:
                        if p.is_positive:
                            return d + q
                        elif p.is_negative:
                            return -d
                    elif q.is_negative:
                        if p.is_positive:
                            return d
                        elif p.is_negative:
                            return -d + q
                    break

        rv = number_eval(p, q)
        if rv is not None:
            return rv

        # denest
        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner*(q - qinner)).is_nonnegative:
                # |qinner| < |q| and have same sign
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner*(q + qinner)).is_nonpositive:
                # |qinner| < |q| and have different sign
                return p
        elif isinstance(p, Add):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            # if q same for all
            if mod_l and all(inner.args[1] == q for inner in mod_l):
                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
                return cls(net, q)

        elif isinstance(p, Mul):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)

            if mod_l and all(inner.args[1] == q for inner in mod_l) and all(t.is_integer for t in p.args) and q.is_integer:
                # finding distributive term
                non_mod_l = [cls(x, q) for x in non_mod_l]
                mod = []
                non_mod = []
                for j in non_mod_l:
                    if isinstance(j, cls):
                        mod.append(j.args[0])
                    else:
                        non_mod.append(j)
                prod_mod = Mul(*mod)
                prod_non_mod = Mul(*non_mod)
                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
                net = prod_mod1*prod_mod
                return prod_non_mod*cls(net, q)

            if q.is_Integer and q is not S.One:
                if all(t.is_integer for t in p.args):
                    non_mod_l = [i % q if i.is_Integer else i for i in p.args]
                    if any(iq is S.Zero for iq in non_mod_l):
                        return S.Zero

            p = Mul(*(non_mod_l + mod_l))

        # XXX other possibilities?

        from sympy.polys.polyerrors import PolynomialError
        from sympy.polys.polytools import gcd

        # extract gcd; any further simplification should be done by the user
        try:
            G = gcd(p, q)
            if not equal_valued(G, 1):
                p, q = [gcd_terms(i/G, clear=False, fraction=False)
                        for i in (p, q)]
        except PolynomialError:  # issue 21373
            G = S.One
        pwas, qwas = p, q

        # simplify terms
        # (x + y + 2) % x -> Mod(y + 2, x)
        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)

        else:
            # handle coefficients if they are not Rational
            # since those are not handled by factor_terms
            # e.g. Mod(.6*x, .3*y) -> 0.3*Mod(2*x, y)
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if equal_valued(r, 0):
                    G *= cq
                    p *= int(cp/cq)
                    ok = True
            if not ok:
                p = cp*p
                q = cq*q

        # simple -1 extraction
        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]

        # check again to see if p and q can now be handled as numbers
        rv = number_eval(p, q)
        if rv is not None:
            return rv*G

        # put 1.0 from G on inside
        if G.is_Float and equal_valued(G, 1):
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and equal_valued(G.args[0], 1):
            p = G.args[0]*p
            G = Mul._from_args(G.args[1:])
        return G*cls(p, q, evaluate=(p, q) != (pwas, qwas))

    def _eval_is_integer(self):
        p, q = self.args
        if fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)]):
            return True

    def _eval_is_nonnegative(self):
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        if self.args[1].is_negative:
            return True

    def _eval_rewrite_as_floor(self, a, b, **kwargs):
        from sympy.functions.elementary.integers import floor
        return a - b*floor(a/b)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_nseries(x, n, logx=logx, cdir=cdir)
