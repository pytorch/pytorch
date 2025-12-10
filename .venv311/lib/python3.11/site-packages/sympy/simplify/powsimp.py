from collections import defaultdict
from functools import reduce
from math import prod

from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational, equal_valued
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity



def powsimp(expr, deep=False, combine='all', force=False, measure=count_ops):
    """
    Reduce expression by combining powers with similar bases and exponents.

    Explanation
    ===========

    If ``deep`` is ``True`` then powsimp() will also simplify arguments of
    functions. By default ``deep`` is set to ``False``.

    If ``force`` is ``True`` then bases will be combined without checking for
    assumptions, e.g. sqrt(x)*sqrt(y) -> sqrt(x*y) which is not true
    if x and y are both negative.

    You can make powsimp() only combine bases or only combine exponents by
    changing combine='base' or combine='exp'.  By default, combine='all',
    which does both.  combine='base' will only combine::

         a   a          a                          2x      x
        x * y  =>  (x*y)   as well as things like 2   =>  4

    and combine='exp' will only combine
    ::

         a   b      (a + b)
        x * x  =>  x

    combine='exp' will strictly only combine exponents in the way that used
    to be automatic.  Also use deep=True if you need the old behavior.

    When combine='all', 'exp' is evaluated first.  Consider the first
    example below for when there could be an ambiguity relating to this.
    This is done so things like the second example can be completely
    combined.  If you want 'base' combined first, do something like
    powsimp(powsimp(expr, combine='base'), combine='exp').

    Examples
    ========

    >>> from sympy import powsimp, exp, log, symbols
    >>> from sympy.abc import x, y, z, n
    >>> powsimp(x**y*x**z*y**z, combine='all')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='exp')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='base', force=True)
    x**y*(x*y)**z

    >>> powsimp(x**z*x**y*n**z*n**y, combine='all', force=True)
    (n*x)**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='exp')
    n**(y + z)*x**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='base', force=True)
    (n*x)**y*(n*x)**z

    >>> x, y = symbols('x y', positive=True)
    >>> powsimp(log(exp(x)*exp(y)))
    log(exp(x)*exp(y))
    >>> powsimp(log(exp(x)*exp(y)), deep=True)
    x + y

    Radicals with Mul bases will be combined if combine='exp'

    >>> from sympy import sqrt
    >>> x, y = symbols('x y')

    Two radicals are automatically joined through Mul:

    >>> a=sqrt(x*sqrt(y))
    >>> a*a**3 == a**4
    True

    But if an integer power of that radical has been
    autoexpanded then Mul does not join the resulting factors:

    >>> a**4 # auto expands to a Mul, no longer a Pow
    x**2*y
    >>> _*a # so Mul doesn't combine them
    x**2*y*sqrt(x*sqrt(y))
    >>> powsimp(_) # but powsimp will
    (x*sqrt(y))**(5/2)
    >>> powsimp(x*y*a) # but won't when doing so would violate assumptions
    x*y*sqrt(x*sqrt(y))

    """
    def recurse(arg, **kwargs):
        _deep = kwargs.get('deep', deep)
        _combine = kwargs.get('combine', combine)
        _force = kwargs.get('force', force)
        _measure = kwargs.get('measure', measure)
        return powsimp(arg, _deep, _combine, _force, _measure)

    expr = sympify(expr)

    if (not isinstance(expr, Basic) or isinstance(expr, MatrixSymbol) or (
            expr.is_Atom or expr in (exp_polar(0), exp_polar(1)))):
        return expr

    if deep or expr.is_Add or expr.is_Mul and _y not in expr.args:
        expr = expr.func(*[recurse(w) for w in expr.args])

    if expr.is_Pow:
        return recurse(expr*_y, deep=False)/_y

    if not expr.is_Mul:
        return expr

    # handle the Mul
    if combine in ('exp', 'all'):
        # Collect base/exp data, while maintaining order in the
        # non-commutative parts of the product
        c_powers = defaultdict(list)
        nc_part = []
        newexpr = []
        coeff = S.One
        for term in expr.args:
            if term.is_Rational:
                coeff *= term
                continue
            if term.is_Pow:
                term = _denest_pow(term)
            if term.is_commutative:
                b, e = term.as_base_exp()
                if deep:
                    b, e = [recurse(i) for i in [b, e]]
                if b.is_Pow or isinstance(b, exp):
                    # don't let smthg like sqrt(x**a) split into x**a, 1/2
                    # or else it will be joined as x**(a/2) later
                    b, e = b**e, S.One
                c_powers[b].append(e)
            else:
                # This is the logic that combines exponents for equal,
                # but non-commutative bases: A**x*A**y == A**(x+y).
                if nc_part:
                    b1, e1 = nc_part[-1].as_base_exp()
                    b2, e2 = term.as_base_exp()
                    if (b1 == b2 and
                            e1.is_commutative and e2.is_commutative):
                        nc_part[-1] = Pow(b1, Add(e1, e2))
                        continue
                nc_part.append(term)

        # add up exponents of common bases
        for b, e in ordered(iter(c_powers.items())):
            # allow 2**x/4 -> 2**(x - 2); don't do this when b and e are
            # Numbers since autoevaluation will undo it, e.g.
            # 2**(1/3)/4 -> 2**(1/3 - 2) -> 2**(1/3)/4
            if (b and b.is_Rational and not all(ei.is_Number for ei in e) and \
                    coeff is not S.One and
                    b not in (S.One, S.NegativeOne)):
                m = multiplicity(abs(b), abs(coeff))
                if m:
                    e.append(m)
                    coeff /= b**m
            c_powers[b] = Add(*e)
        if coeff is not S.One:
            if coeff in c_powers:
                c_powers[coeff] += S.One
            else:
                c_powers[coeff] = S.One

        # convert to plain dictionary
        c_powers = dict(c_powers)

        # check for base and inverted base pairs
        be = list(c_powers.items())
        skip = set()  # skip if we already saw them
        for b, e in be:
            if b in skip:
                continue
            bpos = b.is_positive or b.is_polar
            if bpos:
                binv = 1/b
                #Special case for float 1
                if b.is_Float and equal_valued(b, 1):
                    c_powers[b] = S.One
                    continue
                if b != binv and binv in c_powers:
                    if b.as_numer_denom()[0] is S.One:
                        c_powers.pop(b)
                        c_powers[binv] -= e
                    else:
                        skip.add(binv)
                        e = c_powers.pop(binv)
                        c_powers[b] -= e

        # check for base and negated base pairs
        be = list(c_powers.items())
        _n = S.NegativeOne
        for b, e in be:
            if (b.is_Symbol or b.is_Add) and -b in c_powers and b in c_powers:
                if (b.is_positive is not None or e.is_integer):
                    if e.is_integer or b.is_negative:
                        c_powers[-b] += c_powers.pop(b)
                    else:  # (-b).is_positive so use its e
                        e = c_powers.pop(-b)
                        c_powers[b] += e
                    if _n in c_powers:
                        c_powers[_n] += e
                    else:
                        c_powers[_n] = e

        # filter c_powers and convert to a list
        c_powers = [(b, e) for b, e in c_powers.items() if e]

        # ==============================================================
        # check for Mul bases of Rational powers that can be combined with
        # separated bases, e.g. x*sqrt(x*y)*sqrt(x*sqrt(x*y)) ->
        # (x*sqrt(x*y))**(3/2)
        # ---------------- helper functions

        def ratq(x):
            '''Return Rational part of x's exponent as it appears in the bkey.
            '''
            return bkey(x)[0][1]

        def bkey(b, e=None):
            '''Return (b**s, c.q), c.p where e -> c*s. If e is not given then
            it will be taken by using as_base_exp() on the input b.
            e.g.
                x**3/2 -> (x, 2), 3
                x**y -> (x**y, 1), 1
                x**(2*y/3) -> (x**y, 3), 2
                exp(x/2) -> (exp(a), 2), 1

            '''
            if e is not None:  # coming from c_powers or from below
                if e.is_Integer:
                    return (b, S.One), e
                elif e.is_Rational:
                    return (b, Integer(e.q)), Integer(e.p)
                else:
                    c, m = e.as_coeff_Mul(rational=True)
                    if c is not S.One:
                        if m.is_integer:
                            return (b, Integer(c.q)), m*Integer(c.p)
                        return (b**m, Integer(c.q)), Integer(c.p)
                    else:
                        return (b**e, S.One), S.One
            else:
                return bkey(*b.as_base_exp())

        def update(b):
            '''Decide what to do with base, b. If its exponent is now an
            integer multiple of the Rational denominator, then remove it
            and put the factors of its base in the common_b dictionary or
            update the existing bases if necessary. If it has been zeroed
            out, simply remove the base.
            '''
            newe, r = divmod(common_b[b], b[1])
            if not r:
                common_b.pop(b)
                if newe:
                    for m in Mul.make_args(b[0]**newe):
                        b, e = bkey(m)
                        if b not in common_b:
                            common_b[b] = 0
                        common_b[b] += e
                        if b[1] != 1:
                            bases.append(b)
        # ---------------- end of helper functions

        # assemble a dictionary of the factors having a Rational power
        common_b = {}
        done = []
        bases = []
        for b, e in c_powers:
            b, e = bkey(b, e)
            if b in common_b:
                common_b[b] = common_b[b] + e
            else:
                common_b[b] = e
            if b[1] != 1 and b[0].is_Mul:
                bases.append(b)
        bases.sort(key=default_sort_key)  # this makes tie-breaking canonical
        bases.sort(key=measure, reverse=True)  # handle longest first
        for base in bases:
            if base not in common_b:  # it may have been removed already
                continue
            b, exponent = base
            last = False  # True when no factor of base is a radical
            qlcm = 1  # the lcm of the radical denominators
            while True:
                bstart = b
                qstart = qlcm

                bb = []  # list of factors
                ee = []  # (factor's expo. and it's current value in common_b)
                for bi in Mul.make_args(b):
                    bib, bie = bkey(bi)
                    if bib not in common_b or common_b[bib] < bie:
                        ee = bb = []  # failed
                        break
                    ee.append([bie, common_b[bib]])
                    bb.append(bib)
                if ee:
                    # find the number of integral extractions possible
                    # e.g. [(1, 2), (2, 2)] -> min(2/1, 2/2) -> 1
                    min1 = ee[0][1]//ee[0][0]
                    for i in range(1, len(ee)):
                        rat = ee[i][1]//ee[i][0]
                        if rat < 1:
                            break
                        min1 = min(min1, rat)
                    else:
                        # update base factor counts
                        # e.g. if ee = [(2, 5), (3, 6)] then min1 = 2
                        # and the new base counts will be 5-2*2 and 6-2*3
                        for i in range(len(bb)):
                            common_b[bb[i]] -= min1*ee[i][0]
                            update(bb[i])
                        # update the count of the base
                        # e.g. x**2*y*sqrt(x*sqrt(y)) the count of x*sqrt(y)
                        # will increase by 4 to give bkey (x*sqrt(y), 2, 5)
                        common_b[base] += min1*qstart*exponent
                if (last  # no more radicals in base
                    or len(common_b) == 1  # nothing left to join with
                    or all(k[1] == 1 for k in common_b)  # no rad's in common_b
                        ):
                    break
                # see what we can exponentiate base by to remove any radicals
                # so we know what to search for
                # e.g. if base were x**(1/2)*y**(1/3) then we should
                # exponentiate by 6 and look for powers of x and y in the ratio
                # of 2 to 3
                qlcm = lcm([ratq(bi) for bi in Mul.make_args(bstart)])
                if qlcm == 1:
                    break  # we are done
                b = bstart**qlcm
                qlcm *= qstart
                if all(ratq(bi) == 1 for bi in Mul.make_args(b)):
                    last = True  # we are going to be done after this next pass
            # this base no longer can find anything to join with and
            # since it was longer than any other we are done with it
            b, q = base
            done.append((b, common_b.pop(base)*Rational(1, q)))

        # update c_powers and get ready to continue with powsimp
        c_powers = done
        # there may be terms still in common_b that were bases that were
        # identified as needing processing, so remove those, too
        for (b, q), e in common_b.items():
            if (b.is_Pow or isinstance(b, exp)) and \
                    q is not S.One and not b.exp.is_Rational:
                b, be = b.as_base_exp()
                b = b**(be/q)
            else:
                b = root(b, q)
            c_powers.append((b, e))
        check = len(c_powers)
        c_powers = dict(c_powers)
        assert len(c_powers) == check  # there should have been no duplicates
        # ==============================================================

        # rebuild the expression
        newexpr = expr.func(*(newexpr + [Pow(b, e) for b, e in c_powers.items()]))
        if combine == 'exp':
            return expr.func(newexpr, expr.func(*nc_part))
        else:
            return recurse(expr.func(*nc_part), combine='base') * \
                recurse(newexpr, combine='base')

    elif combine == 'base':

        # Build c_powers and nc_part.  These must both be lists not
        # dicts because exp's are not combined.
        c_powers = []
        nc_part = []
        for term in expr.args:
            if term.is_commutative:
                c_powers.append(list(term.as_base_exp()))
            else:
                nc_part.append(term)

        # Pull out numerical coefficients from exponent if assumptions allow
        # e.g., 2**(2*x) => 4**x
        for i in range(len(c_powers)):
            b, e = c_powers[i]
            if not (all(x.is_nonnegative for x in b.as_numer_denom()) or e.is_integer or force or b.is_polar):
                continue
            exp_c, exp_t = e.as_coeff_Mul(rational=True)
            if exp_c is not S.One and exp_t is not S.One:
                c_powers[i] = [Pow(b, exp_c), exp_t]

        # Combine bases whenever they have the same exponent and
        # assumptions allow
        # first gather the potential bases under the common exponent
        c_exp = defaultdict(list)
        for b, e in c_powers:
            if deep:
                e = recurse(e)
            if e.is_Add and (b.is_positive or e.is_integer):
                e = factor_terms(e)
                if _coeff_isneg(e):
                    e = -e
                    b = 1/b
            c_exp[e].append(b)
        del c_powers

        # Merge back in the results of the above to form a new product
        c_powers = defaultdict(list)
        for e in c_exp:
            bases = c_exp[e]

            # calculate the new base for e

            if len(bases) == 1:
                new_base = bases[0]
            elif e.is_integer or force:
                new_base = expr.func(*bases)
            else:
                # see which ones can be joined
                unk = []
                nonneg = []
                neg = []
                for bi in bases:
                    if bi.is_negative:
                        neg.append(bi)
                    elif bi.is_nonnegative:
                        nonneg.append(bi)
                    elif bi.is_polar:
                        nonneg.append(
                            bi)  # polar can be treated like non-negative
                    else:
                        unk.append(bi)
                if len(unk) == 1 and not neg or len(neg) == 1 and not unk:
                    # a single neg or a single unk can join the rest
                    nonneg.extend(unk + neg)
                    unk = neg = []
                elif neg:
                    # their negative signs cancel in groups of 2*q if we know
                    # that e = p/q else we have to treat them as unknown
                    israt = False
                    if e.is_Rational:
                        israt = True
                    else:
                        p, d = e.as_numer_denom()
                        if p.is_integer and d.is_integer:
                            israt = True
                    if israt:
                        neg = [-w for w in neg]
                        unk.extend([S.NegativeOne]*len(neg))
                    else:
                        unk.extend(neg)
                        neg = []
                    del israt

                # these shouldn't be joined
                for b in unk:
                    c_powers[b].append(e)
                # here is a new joined base
                new_base = expr.func(*(nonneg + neg))
                # if there are positive parts they will just get separated
                # again unless some change is made

                def _terms(e):
                    # return the number of terms of this expression
                    # when multiplied out -- assuming no joining of terms
                    if e.is_Add:
                        return sum(_terms(ai) for ai in e.args)
                    if e.is_Mul:
                        return prod([_terms(mi) for mi in e.args])
                    return 1
                xnew_base = expand_mul(new_base, deep=False)
                if len(Add.make_args(xnew_base)) < _terms(new_base):
                    new_base = factor_terms(xnew_base)

            c_powers[new_base].append(e)

        # break out the powers from c_powers now
        c_part = [Pow(b, ei) for b, e in c_powers.items() for ei in e]

        # we're done
        return expr.func(*(c_part + nc_part))

    else:
        raise ValueError("combine must be one of ('all', 'exp', 'base').")


def powdenest(eq, force=False, polar=False):
    r"""
    Collect exponents on powers as assumptions allow.

    Explanation
    ===========

    Given ``(bb**be)**e``, this can be simplified as follows:
        * if ``bb`` is positive, or
        * ``e`` is an integer, or
        * ``|be| < 1`` then this simplifies to ``bb**(be*e)``

    Given a product of powers raised to a power, ``(bb1**be1 *
    bb2**be2...)**e``, simplification can be done as follows:

    - if e is positive, the gcd of all bei can be joined with e;
    - all non-negative bb can be separated from those that are negative
      and their gcd can be joined with e; autosimplification already
      handles this separation.
    - integer factors from powers that have integers in the denominator
      of the exponent can be removed from any term and the gcd of such
      integers can be joined with e

    Setting ``force`` to ``True`` will make symbols that are not explicitly
    negative behave as though they are positive, resulting in more
    denesting.

    Setting ``polar`` to ``True`` will do simplifications on the Riemann surface of
    the logarithm, also resulting in more denestings.

    When there are sums of logs in exp() then a product of powers may be
    obtained e.g. ``exp(3*(log(a) + 2*log(b)))`` - > ``a**3*b**6``.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y, z
    >>> from sympy import Symbol, exp, log, sqrt, symbols, powdenest

    >>> powdenest((x**(2*a/3))**(3*x))
    (x**(2*a/3))**(3*x)
    >>> powdenest(exp(3*x*log(2)))
    2**(3*x)

    Assumptions may prevent expansion:

    >>> powdenest(sqrt(x**2))
    sqrt(x**2)

    >>> p = symbols('p', positive=True)
    >>> powdenest(sqrt(p**2))
    p

    No other expansion is done.

    >>> i, j = symbols('i,j', integer=True)
    >>> powdenest((x**x)**(i + j)) # -X-> (x**x)**i*(x**x)**j
    x**(x*(i + j))

    But exp() will be denested by moving all non-log terms outside of
    the function; this may result in the collapsing of the exp to a power
    with a different base:

    >>> powdenest(exp(3*y*log(x)))
    x**(3*y)
    >>> powdenest(exp(y*(log(a) + log(b))))
    (a*b)**y
    >>> powdenest(exp(3*(log(a) + log(b))))
    a**3*b**3

    If assumptions allow, symbols can also be moved to the outermost exponent:

    >>> i = Symbol('i', integer=True)
    >>> powdenest(((x**(2*i))**(3*y))**x)
    ((x**(2*i))**(3*y))**x
    >>> powdenest(((x**(2*i))**(3*y))**x, force=True)
    x**(6*i*x*y)

    >>> powdenest(((x**(2*a/3))**(3*y/i))**x)
    ((x**(2*a/3))**(3*y/i))**x
    >>> powdenest((x**(2*i)*y**(4*i))**z, force=True)
    (x*y**2)**(2*i*z)

    >>> n = Symbol('n', negative=True)

    >>> powdenest((x**i)**y, force=True)
    x**(i*y)
    >>> powdenest((n**i)**x, force=True)
    (n**i)**x

    """
    from sympy.simplify.simplify import posify

    if force:
        def _denest(b, e):
            if not isinstance(b, (Pow, exp)):
                return b.is_positive, Pow(b, e, evaluate=False)
            return _denest(b.base, b.exp*e)
        reps = []
        for p in eq.atoms(Pow, exp):
            if isinstance(p.base, (Pow, exp)):
                ok, dp = _denest(*p.args)
                if ok is not False:
                    reps.append((p, dp))
        if reps:
            eq = eq.subs(reps)
        eq, reps = posify(eq)
        return powdenest(eq, force=False, polar=polar).xreplace(reps)

    if polar:
        eq, rep = polarify(eq)
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)

    new = powsimp(eq)
    return new.xreplace(Transform(
        _denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))

_y = Dummy('y')


def _denest_pow(eq):
    """
    Denest powers.

    This is a helper function for powdenest that performs the actual
    transformation.
    """
    from sympy.simplify.simplify import logcombine

    b, e = eq.as_base_exp()
    if b.is_Pow or isinstance(b, exp) and e != 1:
        new = b._eval_power(e)
        if new is not None:
            eq = new
            b, e = new.as_base_exp()

    # denest exp with log terms in exponent
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any(isinstance(ai, log) for ai in Add.make_args(ei)):
                logs.append(ei)
            else:
                other.append(ei)
        logs = logcombine(Mul(*logs))
        return Pow(exp(logs), Mul(*other))

    _, be = b.as_base_exp()
    if be is S.One and not (b.is_Mul or
                            b.is_Rational and b.q != 1 or
                            b.is_positive):
        return eq

    # denest eq which is either pos**e or Pow**e or Mul**e or
    # Mul(b1**e1, b2**e2)

    # handle polar numbers specially
    polars, nonpolars = [], []
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())
        else:
            nonpolars.append(bb)
    if len(polars) == 1 and not polars[0][0].is_Mul:
        return Pow(polars[0][0], polars[0][1]*e)*powdenest(Mul(*nonpolars)**e)
    elif polars:
        return Mul(*[powdenest(bb**(ee*e)) for (bb, ee) in polars]) \
            *powdenest(Mul(*nonpolars)**e)

    if b.is_Integer:
        # use log to see if there is a power here
        logb = expand_log(log(b))
        if logb.is_Mul:
            c, logb = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)

    # if b is not a Mul or any factor is an atom then there is nothing to do
    if not b.is_Mul or any(s.is_Atom for s in Mul.make_args(b)):
        return eq

    # let log handle the case of the base of the argument being a Mul, e.g.
    # sqrt(x**(2*i)*y**(6*i)) -> x**i*y**(3**i) if x and y are positive; we
    # will take the log, expand it, and then factor out the common powers that
    # now appear as coefficient. We do this manually since terms_gcd pulls out
    # fractions, terms_gcd(x+x*y/2) -> x*(y + 2)/2 and we don't want the 1/2;
    # gcd won't pull out numerators from a fraction: gcd(3*x, 9*x/2) -> x but
    # we want 3*x. Neither work with noncommutatives.

    def nc_gcd(aa, bb):
        a, b = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*(a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0]))
        return _keep_coeff(c, g)

    glogb = expand_log(log(b))
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)
        if g != 1:
            cg, rg = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg*Add(*[a/g for a in args]))

    # now put the log back together again
    if isinstance(glogb, log) or not glogb.is_Mul:
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            glogb = _denest_pow(glogb.args[0])
            if (abs(glogb.exp) < 1) == True:
                return Pow(glogb.base, glogb.exp*e)
        return eq

    # the log(b) was a Mul so join any adds with logcombine
    add = []
    other = []
    for a in glogb.args:
        if a.is_Add:
            add.append(a)
        else:
            other.append(a)
    return Pow(exp(logcombine(Mul(*add))), e*Mul(*other))
