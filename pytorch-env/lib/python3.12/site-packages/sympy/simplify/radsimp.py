from collections import defaultdict

from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift




def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    """
    Collect additive terms of an expression.

    Explanation
    ===========

    This function collects additive terms of an expression with respect
    to a list of expression up to powers with rational exponents. By the
    term symbol here are meant arbitrary expressions, which can contain
    powers, products, sums etc. In other words symbol is a pattern which
    will be searched for in the expression's terms.

    The input expression is not expanded by :func:`collect`, so user is
    expected to provide an expression in an appropriate form. This makes
    :func:`collect` more predictable as there is no magic happening behind the
    scenes. However, it is important to note, that powers of products are
    converted to products of powers using the :func:`~.expand_power_base`
    function.

    There are two possible types of output. First, if ``evaluate`` flag is
    set, this function will return an expression with collected terms or
    else it will return a dictionary with expressions up to rational powers
    as keys and collected coefficients as values.

    Examples
    ========

    >>> from sympy import S, collect, expand, factor, Wild
    >>> from sympy.abc import a, b, c, x, y

    This function can collect symbolic coefficients in polynomials or
    rational expressions. It will manage to find all integer or rational
    powers of collection variable::

        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)
        c + x**2*(a + b) + x*(a - b)

    The same result can be achieved in dictionary form::

        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)
        >>> d[x**2]
        a + b
        >>> d[x]
        a - b
        >>> d[S.One]
        c

    You can also work with multivariate polynomials. However, remember that
    this function is greedy so it will care only about a single symbol at time,
    in specification order::

        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])
        x**2*(y + 1) + x*y + y*(a + 1)

    Also more complicated expressions can be used as patterns::

        >>> from sympy import sin, log
        >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))
        (a + b)*sin(2*x)

        >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))
        x*(a + b)*log(x)

    You can use wildcards in the pattern::

        >>> w = Wild('w1')
        >>> collect(a*x**y - b*x**y, w**y)
        x**y*(a - b)

    It is also possible to work with symbolic powers, although it has more
    complicated behavior, because in this case power's base and symbolic part
    of the exponent are treated as a single symbol::

        >>> collect(a*x**c + b*x**c, x)
        a*x**c + b*x**c
        >>> collect(a*x**c + b*x**c, x**c)
        x**c*(a + b)

    However if you incorporate rationals to the exponents, then you will get
    well known behavior::

        >>> collect(a*x**(2*c) + b*x**(2*c), x**c)
        x**(2*c)*(a + b)

    Note also that all previously stated facts about :func:`collect` function
    apply to the exponential function, so you can get::

        >>> from sympy import exp
        >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))
        (a + b)*exp(2*x)

    If you are interested only in collecting specific powers of some symbols
    then set ``exact`` flag to True::

        >>> collect(a*x**7 + b*x**7, x, exact=True)
        a*x**7 + b*x**7
        >>> collect(a*x**7 + b*x**7, x**7, exact=True)
        x**7*(a + b)

    If you want to collect on any object containing symbols, set
    ``exact`` to None:

        >>> collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None)
        x*exp(x) + 3*x + (y + 2)*sin(x)
        >>> collect(a*x*y + x*y + b*x + x, [x, y], exact=None)
        x*y*(a + 1) + x*(b + 1)

    You can also apply this function to differential equations, where
    derivatives of arbitrary order can be collected. Note that if you
    collect with respect to a function or a derivative of a function, all
    derivatives of that function will also be collected. Use
    ``exact=True`` to prevent this from happening::

        >>> from sympy import Derivative as D, collect, Function
        >>> f = Function('f') (x)

        >>> collect(a*D(f,x) + b*D(f,x), D(f,x))
        (a + b)*Derivative(f(x), x)

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)
        (a + b)*Derivative(f(x), (x, 2))

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)
        a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))

        >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)
        (a + b)*f(x) + (a + b)*Derivative(f(x), x)

    Or you can even match both derivative order and exponent at the same time::

        >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))
        (a + b)*Derivative(f(x), (x, 2))**2

    Finally, you can apply a function to each of the collected coefficients.
    For example you can factorize symbolic coefficients of polynomial::

        >>> f = expand((x + a + 1)**3)

        >>> collect(f, x, factor)
        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3

    .. note:: Arguments are expected to be in expanded form, so you might have
              to call :func:`~.expand` prior to calling this function.

    See Also
    ========

    collect_const, collect_sqrt, rcollect
    """
    expr = sympify(expr)
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]

    # replace syms[i] if it is not x, -x or has Wild symbols
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(
        x.atoms(Wild))
    _, nonsyms = sift(syms, cond, binary=True)
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        rv = collect(expr.subs(reps), syms,
            func=func, evaluate=evaluate, exact=exact,
            distribute_order_term=distribute_order_term)
        urep = {v: k for k, v in reps.items()}
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            return {urep.get(k, k).xreplace(urep): v.xreplace(urep)
                    for k, v in rv.items()}

    # see if other expressions should be considered
    if exact is None:
        _syms = set()
        for i in Add.make_args(expr):
            if not i.has_free(*syms) or i in syms:
                continue
            if not i.is_Mul and i not in syms:
                _syms.add(i)
            else:
                # identify compound generators
                g = i._new_rawargs(*i.as_coeff_mul(*syms)[1])
                if g not in syms:
                    _syms.add(g)
        simple = all(i.is_Pow and i.base in syms for i in _syms)
        syms = syms + list(ordered(_syms))
        if not simple:
            return collect(expr, syms,
            func=func, evaluate=evaluate, exact=False,
            distribute_order_term=distribute_order_term)

    if evaluate is None:
        evaluate = global_parameters.evaluate

    def make_expression(terms):
        product = []

        for term, rat, sym, deriv in terms:
            if deriv is not None:
                var, order = deriv

                while order > 0:
                    term, order = Derivative(term, var), order - 1

            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                product.append(Pow(term, rat*sym))

        return Mul(*product)

    def parse_derivative(deriv):
        # scan derivatives tower in the input expression and return
        # underlying function and maximal differentiation order
        expr, sym, order = deriv.expr, deriv.variables[0], 1

        for s in deriv.variables[1:]:
            if s == sym:
                order += 1
            else:
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

        while isinstance(expr, Derivative):
            s0 = expr.variables[0]

            for s in expr.variables:
                if s != s0:
                    raise NotImplementedError(
                        'Improve MV Derivative support in collect')

            if s0 == sym:
                expr, order = expr.expr, order + len(expr.variables)
            else:
                break

        return expr, (sym, Rational(order))

    def parse_term(expr):
        """Parses expression expr and outputs tuple (sexpr, rat_expo,
        sym_expo, deriv)
        where:
         - sexpr is the base expression
         - rat_expo is the rational exponent that sexpr is raised to
         - sym_expo is the symbolic exponent that sexpr is raised to
         - deriv contains the derivatives of the expression

         For example, the output of x would be (x, 1, None, None)
         the output of 2**x would be (2, 1, x, None).
        """
        rat_expo, sym_expo = S.One, None
        sexpr, deriv = expr, None

        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                sexpr, deriv = parse_derivative(expr.base)
            else:
                sexpr = expr.base

            if expr.base == S.Exp1:
                arg = expr.exp
                if arg.is_Rational:
                    sexpr, rat_expo = S.Exp1, arg
                elif arg.is_Mul:
                    coeff, tail = arg.as_coeff_Mul(rational=True)
                    sexpr, rat_expo = exp(tail), coeff

            elif expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                coeff, tail = expr.exp.as_coeff_Mul()

                if coeff.is_Number:
                    rat_expo, sym_expo = coeff, tail
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.exp
            if arg.is_Rational:
                sexpr, rat_expo = S.Exp1, arg
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(expr, Derivative):
            sexpr, deriv = parse_derivative(expr)

        return sexpr, rat_expo, sym_expo, deriv

    def parse_expression(terms, pattern):
        """Parse terms searching for a pattern.
        Terms is a list of tuples as returned by parse_terms;
        Pattern is an expression treated as a product of factors.
        """
        pattern = Mul.make_args(pattern)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            pattern = [parse_term(elem) for elem in pattern]

            terms = terms[:]  # need a copy
            elems, common_expo, has_deriv = [], None, False

            for elem, e_rat, e_sym, e_ord in pattern:

                if elem.is_Number and e_rat == 1 and e_sym is None:
                    # a constant is a match for everything
                    continue

                for j in range(len(terms)):
                    if terms[j] is None:
                        continue

                    term, t_rat, t_sym, t_ord = terms[j]

                    # keeping track of whether one of the terms had
                    # a derivative or not as this will require rebuilding
                    # the expression later
                    if t_ord is not None:
                        has_deriv = True

                    if (term.match(elem) is not None and
                            (t_sym == e_sym or t_sym is not None and
                            e_sym is not None and
                            t_sym.match(e_sym) is not None)):
                        if exact is False:
                            # we don't have to be exact so find common exponent
                            # for both expression's term and pattern's element
                            expo = t_rat / e_rat

                            if common_expo is None:
                                # first time
                                common_expo = expo
                            else:
                                # common exponent was negotiated before so
                                # there is no chance for a pattern match unless
                                # common and current exponents are equal
                                if common_expo != expo:
                                    common_expo = 1
                        else:
                            # we ought to be exact so all fields of
                            # interest must match in every details
                            if e_rat != t_rat or e_ord != t_ord:
                                continue

                        # found common term so remove it from the expression
                        # and try to match next element in the pattern
                        elems.append(terms[j])
                        terms[j] = None

                        break

                else:
                    # pattern element not found
                    return None

            return [_f for _f in terms if _f], elems, common_expo, has_deriv

    if evaluate:
        if expr.is_Add:
            o = expr.getO() or 0
            expr = expr.func(*[
                    collect(a, syms, func, True, exact, distribute_order_term)
                    for a in expr.args if a != o]) + o
        elif expr.is_Mul:
            return expr.func(*[
                collect(term, syms, func, True, exact, distribute_order_term)
                for term in expr.args])
        elif expr.is_Pow:
            b = collect(
                expr.base, syms, func, True, exact, distribute_order_term)
            return Pow(b, expr.exp)

    syms = [expand_power_base(i, deep=False) for i in syms]

    order_term = None

    if distribute_order_term:
        order_term = expr.getO()

        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                expr = expr.removeO()

    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]

    collected, disliked = defaultdict(list), S.Zero
    for product in summa:
        c, nc = product.args_cnc(split_1=False)
        args = list(ordered(c)) + nc
        terms = [parse_term(i) for i in args]
        small_first = True

        for symbol in syms:
            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = parse_expression(terms, symbol)

            if result is not None:
                if not symbol.is_commutative:
                    raise AttributeError("Can not collect noncommutative symbol")

                terms, elems, common_expo, has_deriv = result

                # when there was derivative in current pattern we
                # will need to rebuild its expression from scratch
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1]*elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected[index].append(terms)
                break
        else:
            # none of the patterns matched
            disliked += product
    # add terms now for each key
    collected = {k: Add(*v) for k, v in collected.items()}

    if disliked is not S.Zero:
        collected[S.One] = disliked

    if order_term is not None:
        for key, val in collected.items():
            collected[key] = val + order_term

    if func is not None:
        collected = {
            key: func(val) for key, val in collected.items()}

    if evaluate:
        return Add(*[key*val for key, val in collected.items()])
    else:
        return collected


def rcollect(expr, *vars):
    """
    Recursively collect sums in an expression.

    Examples
    ========

    >>> from sympy.simplify import rcollect
    >>> from sympy.abc import x, y

    >>> expr = (x**2*y + x*y + x + y)/(x + y)

    >>> rcollect(expr, y)
    (x + y*(x**2 + x + 1))/(x + y)

    See Also
    ========

    collect, collect_const, collect_sqrt
    """
    if expr.is_Atom or not expr.has(*vars):
        return expr
    else:
        expr = expr.__class__(*[rcollect(arg, *vars) for arg in expr.args])

        if expr.is_Add:
            return collect(expr, vars)
        else:
            return expr


def collect_sqrt(expr, evaluate=None):
    """Return expr with terms having common square roots collected together.
    If ``evaluate`` is False a count indicating the number of sqrt-containing
    terms will be returned and, if non-zero, the terms of the Add will be
    returned, else the expression itself will be returned as a single term.
    If ``evaluate`` is True, the expression with any collected terms will be
    returned.

    Note: since I = sqrt(-1), it is collected, too.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b

    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]
    >>> collect_sqrt(a*r2 + b*r2)
    sqrt(2)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)
    sqrt(2)*(a + b) + sqrt(3)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)
    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)

    If evaluate is False then the arguments will be sorted and
    returned as a list and a count of the number of sqrt-containing
    terms will be returned:

    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)
    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)
    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)
    ((b, sqrt(2)*a), 1)
    >>> collect_sqrt(a + b, evaluate=False)
    ((a + b,), 0)

    See Also
    ========

    collect, collect_const, rcollect
    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    # this step will help to standardize any complex arguments
    # of sqrts
    coeff, expr = expr.as_content_primitive()
    vars = set()
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (
                    m.is_Pow and m.exp.is_Rational and m.exp.q == 2 or
                    m is S.ImaginaryUnit):
                vars.add(m)

    # we only want radicals, so exclude Number handling; in this case
    # d will be evaluated
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d

    if not evaluate:
        nrad = 0
        # make the evaluated args canonical
        args = list(ordered(Add.make_args(d)))
        for i, m in enumerate(args):
            c, nc = m.args_cnc()
            for ci in c:
                # XXX should this be restricted to ci.is_number as above?
                if ci.is_Pow and ci.exp.is_Rational and ci.exp.q == 2 or \
                        ci is S.ImaginaryUnit:
                    nrad += 1
                    break
            args[i] *= coeff
        if not (hit or nrad):
            args = [Add(*args)]
        return tuple(args), nrad

    return coeff*d


def collect_abs(expr):
    """Return ``expr`` with arguments of multiple Abs in a term collected
    under a single instance.

    Examples
    ========

    >>> from sympy.simplify.radsimp import collect_abs
    >>> from sympy.abc import x
    >>> collect_abs(abs(x + 1)/abs(x**2 - 1))
    Abs((x + 1)/(x**2 - 1))
    >>> collect_abs(abs(1/x))
    Abs(1/x)
    """
    def _abs(mul):
      c, nc = mul.args_cnc()
      a = []
      o = []
      for i in c:
          if isinstance(i, Abs):
              a.append(i.args[0])
          elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
              a.append(i.base.args[0]**i.exp)
          else:
              o.append(i)
      if len(a) < 2 and not any(i.exp.is_negative for i in a if isinstance(i, Pow)):
          return mul
      absarg = Mul(*a)
      A = Abs(absarg)
      args = [A]
      args.extend(o)
      if not A.has(Abs):
          args.extend(nc)
          return Mul(*args)
      if not isinstance(A, Abs):
          # reevaluate and make it unevaluated
          A = Abs(absarg, evaluate=False)
      args[0] = A
      _mulsort(args)
      args.extend(nc)  # nc always go last
      return Mul._from_args(args, is_commutative=not nc)

    return expr.replace(
        lambda x: isinstance(x, Mul),
        lambda x: _abs(x)).replace(
            lambda x: isinstance(x, Pow),
            lambda x: _abs(x))


def collect_const(expr, *vars, Numbers=True):
    """A non-greedy collection of terms with similar number coefficients in
    an Add expr. If ``vars`` is given then only those constants will be
    targeted. Although any Number can also be targeted, if this is not
    desired set ``Numbers=False`` and no Float or Rational will be collected.

    Parameters
    ==========

    expr : SymPy expression
        This parameter defines the expression the expression from which
        terms with similar coefficients are to be collected. A non-Add
        expression is returned as it is.

    vars : variable length collection of Numbers, optional
        Specifies the constants to target for collection. Can be multiple in
        number.

    Numbers : bool
        Specifies to target all instance of
        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
        no Float or Rational will be collected.

    Returns
    =======

    expr : Expr
        Returns an expression with similar coefficient terms collected.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import s, x, y, z
    >>> from sympy.simplify.radsimp import collect_const
    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))
    sqrt(3)*(sqrt(2) + 2)
    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))
    (sqrt(3) + sqrt(7))*(s + 1)
    >>> s = sqrt(2) + 2
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))
    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))
    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)

    The collection is sign-sensitive, giving higher precedence to the
    unsigned values:

    >>> collect_const(x - y - z)
    x - (y + z)
    >>> collect_const(-y - z)
    -(y + z)
    >>> collect_const(2*x - 2*y - 2*z, 2)
    2*(x - y - z)
    >>> collect_const(2*x - 2*y - 2*z, -2)
    2*x - 2*(y + z)

    See Also
    ========

    collect, collect_sqrt, rcollect
    """
    if not expr.is_Add:
        return expr

    recurse = False

    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]

    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            q, r = f.div(Fv)
            if r.is_one:
                # only accept this as a true factor if
                # it didn't change an exponent from an Integer
                # to a non-Integer, e.g. 2/sqrt(2) -> sqrt(2)
                # -- we aren't looking for this sort of change
                fwas = f.factors.copy()
                fnow = q.factors
                if not any(k in fwas and fwas[k].is_Integer and not
                        fnow[k].is_Integer for k in fnow):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)

        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue

            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]

            # be careful not to let uneval become True unless
            # it must be because it's going to be more expensive
            # to rebuild the expression as an unevaluated one
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k*v)

        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break

    return expr


def radsimp(expr, symbolic=True, max_terms=4):
    r"""
    Rationalize the denominator by removing square roots.

    Explanation
    ===========

    The expression returned from radsimp must be used with caution
    since if the denominator contains symbols, it will be possible to make
    substitutions that violate the assumptions of the simplification process:
    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If
    there are no symbols, this assumptions is made valid by collecting terms
    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If
    you do not want the simplification to occur for symbolic denominators, set
    ``symbolic`` to False.

    If there are more than ``max_terms`` radical terms then the expression is
    returned unchanged.

    Examples
    ========

    >>> from sympy import radsimp, sqrt, Symbol, pprint
    >>> from sympy import factor_terms, fraction, signsimp
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b, c

    >>> radsimp(1/(2 + sqrt(2)))
    (2 - sqrt(2))/2
    >>> x,y = map(Symbol, 'xy')
    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))
    >>> radsimp(e)
    sqrt(2)*(x + y)

    No simplification beyond removal of the gcd is done. One might
    want to polish the result a little, however, by collecting
    square root terms:

    >>> r2 = sqrt(2)
    >>> r5 = sqrt(5)
    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)
        ___       ___       ___       ___
      \/ 5 *a + \/ 5 *b - \/ 2 *x - \/ 2 *y
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    >>> n, d = fraction(ans)
    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))
            ___             ___
          \/ 5 *(a + b) - \/ 2 *(x + y)
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    If radicals in the denominator cannot be removed or there is no denominator,
    the original expression will be returned.

    >>> radsimp(sqrt(2)*x + sqrt(2))
    sqrt(2)*x + sqrt(2)

    Results with symbols will not always be valid for all substitutions:

    >>> eq = 1/(a + b*sqrt(c))
    >>> eq.subs(a, b*sqrt(c))
    1/(2*b*sqrt(c))
    >>> radsimp(eq).subs(a, b*sqrt(c))
    nan

    If ``symbolic=False``, symbolic denominators will not be transformed (but
    numeric denominators will still be processed):

    >>> radsimp(eq, symbolic=False)
    1/(a + b*sqrt(c))

    """
    from sympy.simplify.simplify import signsimp

    syms = symbols("a:d A:D")
    def _num(rterms):
        # return the multiplier that will simplify the expression described
        # by rterms [(sqrt arg, coeff), ... ]
        a, b, c, d, A, B, C, D = syms
        if len(rterms) == 2:
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            return (
            sqrt(A)*a - sqrt(B)*b).xreplace(reps)
        if len(rterms) == 3:
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            return (
            (sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
            B*b**2 + C*c**2)).xreplace(reps)
        elif len(rterms) == 4:
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                D**2*d**4)).xreplace(reps)
        elif len(rterms) == 1:
            return sqrt(rterms[0][0])
        else:
            raise NotImplementedError

    def ispow2(d, log2=False):
        if not d.is_Pow:
            return False
        e = d.exp
        if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
            return True
        if log2:
            q = 1
            if e.is_Rational:
                q = e.q
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            if q != 1 and log(q, 2).is_Integer:
                return True
        return False

    def handle(expr):
        # Handle first reduces to the case
        # expr = 1/d, where d is an add, or d is base**p/2.
        # We do this by recursively calling handle on each piece.
        from sympy.simplify.simplify import nsimplify

        n, d = fraction(expr)

        if expr.is_Atom or (d.is_Atom and n.is_Atom):
            return expr
        elif not n.is_Atom:
            n = n.func(*[handle(a) for a in n.args])
            return _unevaluated_Mul(n, handle(1/d))
        elif n is not S.One:
            return _unevaluated_Mul(n, handle(1/d))
        elif d.is_Mul:
            return _unevaluated_Mul(*[handle(1/d) for d in d.args])

        # By this step, expr is 1/d, and d is not a mul.
        if not symbolic and d.free_symbols:
            return expr

        if ispow2(d):
            d2 = sqrtdenest(sqrt(d.base))**numer(d.exp)
            if d2 != d:
                return handle(1/d2)
        elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
            # (1/d**i) = (1/d)**i
            return handle(1/d.base)**d.exp

        if not (d.is_Add or ispow2(d)):
            return 1/d.func(*[handle(a) for a in d.args])

        # handle 1/d treating d as an Add (though it may not be)

        keep = True  # keep changes that are made

        # flatten it and collect radicals after checking for special
        # conditions
        d = _mexpand(d)

        # did it change?
        if d.is_Atom:
            return 1/d

        # is it a number that might be handled easily?
        if d.is_number:
            _d = nsimplify(d)
            if _d.is_Number and _d.equals(d):
                return 1/_d

        while True:
            # collect similar terms
            collected = defaultdict(list)
            for m in Add.make_args(d):  # d might have become non-Add
                p2 = []
                other = []
                for i in Mul.make_args(m):
                    if ispow2(i, log2=True):
                        p2.append(i.base if i.exp is S.Half else i.base**(2*i.exp))
                    elif i is S.ImaginaryUnit:
                        p2.append(S.NegativeOne)
                    else:
                        other.append(i)
                collected[tuple(ordered(p2))].append(Mul(*other))
            rterms = list(ordered(list(collected.items())))
            rterms = [(Mul(*i), Add(*j)) for i, j in rterms]
            nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
            if nrad < 1:
                break
            elif nrad > max_terms:
                # there may have been invalid operations leading to this point
                # so don't keep changes, e.g. this expression is troublesome
                # in collecting terms so as not to raise the issue of 2834:
                # r = sqrt(sqrt(5) + 5)
                # eq = 1/(sqrt(5)*r + 2*sqrt(5)*sqrt(-sqrt(5) + 5) + 5*r)
                keep = False
                break
            if len(rterms) > 4:
                # in general, only 4 terms can be removed with repeated squaring
                # but other considerations can guide selection of radical terms
                # so that radicals are removed
                if all(x.is_Integer and (y**2).is_Rational for x, y in rterms):
                    nd, d = rad_rationalize(S.One, Add._from_args(
                        [sqrt(x)*y for x, y in rterms]))
                    n *= nd
                else:
                    # is there anything else that might be attempted?
                    keep = False
                break
            from sympy.simplify.powsimp import powsimp, powdenest

            num = powsimp(_num(rterms))
            n *= num
            d *= num
            d = powdenest(_mexpand(d), force=symbolic)
            if d.has(S.Zero, nan, zoo):
                return expr
            if d.is_Atom:
                break

        if not keep:
            return expr
        return _unevaluated_Mul(n, 1/d)

    coeff, expr = expr.as_coeff_Add()
    expr = expr.normal()
    old = fraction(expr)
    n, d = fraction(handle(expr))
    if old != (n, d):
        if not d.is_Atom:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = Factors(_unevaluated_Mul(n, 1/d))
            u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
            n, d = fraction(u)
            if old == (n, d):
                n, d = was
        n = expand_mul(n)
        if d.is_Number or d.is_Add:
            n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
            if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                n, d = [signsimp(i) for i in (n2, d2)]
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)

    return coeff + _unevaluated_Mul(n, 1/d)


def rad_rationalize(num, den):
    """
    Rationalize ``num/den`` by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    if not den.is_Add:
        return num, den
    g, a, b = split_surds(den)
    a = a*sqrt(g)
    num = _mexpand((a - b)*num)
    den = _mexpand(a**2 - b**2)
    return rad_rationalize(num, den)


def fraction(expr, exact=False):
    """Returns a pair with expression's numerator and denominator.
       If the given expression is not a fraction then this function
       will return the tuple (expr, 1).

       This function will not make any attempt to simplify nested
       fractions or to do any term rewriting at all.

       If only one of the numerator/denominator pair is needed then
       use numer(expr) or denom(expr) functions respectively.

       >>> from sympy import fraction, Rational, Symbol
       >>> from sympy.abc import x, y

       >>> fraction(x/y)
       (x, y)
       >>> fraction(x)
       (x, 1)

       >>> fraction(1/y**2)
       (1, y**2)

       >>> fraction(x*y/2)
       (x*y, 2)
       >>> fraction(Rational(1, 2))
       (1, 2)

       This function will also work fine with assumptions:

       >>> k = Symbol('k', negative=True)
       >>> fraction(x * y**k)
       (x, y**(-k))

       If we know nothing about sign of some exponent and ``exact``
       flag is unset, then the exponent's structure will
       be analyzed and pretty fraction will be returned:

       >>> from sympy import exp, Mul
       >>> fraction(2*x**(-y))
       (2, x**y)

       >>> fraction(exp(-x))
       (1, exp(x))

       >>> fraction(exp(-x), exact=True)
       (exp(-x), 1)

       The ``exact`` flag will also keep any unevaluated Muls from
       being evaluated:

       >>> u = Mul(2, x + 1, evaluate=False)
       >>> fraction(u)
       (2*x + 2, 1)
       >>> fraction(u, exact=True)
       (2*(x  + 1), 1)
    """
    expr = sympify(expr)

    numer, denom = [], []

    for term in Mul.make_args(expr):
        if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
            b, ex = term.as_base_exp()
            if ex.is_negative:
                if ex is S.NegativeOne:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    else:
                        numer.append(term)
                else:
                    denom.append(Pow(b, -ex))
            elif ex.is_positive:
                numer.append(term)
            elif not exact and ex.is_Mul:
                n, d = term.as_numer_denom()  # this will cause evaluation
                if n != 1:
                    numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif term.is_Rational and not term.is_Integer:
            if term.p != 1:
                numer.append(term.p)
            denom.append(term.q)
        else:
            numer.append(term)
    return Mul(*numer, evaluate=not exact), Mul(*denom, evaluate=not exact)


def numer(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[0]


def denom(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[1]


def fraction_expand(expr, **hints):
    return expr.expand(frac=True, **hints)


def numer_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a.expand(numer=True, **hints) / b


def denom_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a / b.expand(denom=True, **hints)


expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand


def split_surds(expr):
    """
    Split an expression with terms whose squares are positive rationals
    into a sum of terms whose surds squared have gcd equal to g
    and a sum of terms with surds squared prime with g.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import split_surds
    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))
    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))
    """
    args = sorted(expr.args, key=default_sort_key)
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = [x[1]**2 for x in coeff_muls if x[1].is_Pow]
    surds.sort(key=default_sort_key)
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x/g for x in b1]
        b1n = [x for x in b1n if x != 1]
        # only a common factor has been factored; split again
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = g*g1
    a1v, a2v = [], []
    for c, s in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c*sqrt(s1/g2))
            else:
                a2v.append(c*s)
        else:
            a2v.append(c*s)
    a = Add(*a1v)
    b = Add(*a2v)
    return g2, a, b


def _split_gcd(*a):
    """
    Split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``.

    Examples
    ========

    >>> from sympy.simplify.radsimp import _split_gcd
    >>> _split_gcd(55, 35, 22, 14, 77, 10)
    (5, [55, 35, 10], [22, 14, 77])
    """
    g = a[0]
    b1 = [g]
    b2 = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if g1 == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return g, b1, b2
