from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from collections import defaultdict
from functools import reduce
from operator import attrgetter
from .basic import _args_sortkey
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import ilcm, igcd
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift


if TYPE_CHECKING:
    from sympy.core.numbers import Number
    from sympy.series.order import Order


def _could_extract_minus_sign(expr):
    # assume expr is Add-like
    # We choose the one with less arguments with minus signs
    negative_args = sum(1 for i in expr.args
        if i.could_extract_minus_sign())
    positive_args = len(expr.args) - negative_args
    if positive_args > negative_args:
        return False
    elif positive_args < negative_args:
        return True
    # choose based on .sort_key() to prefer
    # x - 1 instead of 1 - x and
    # 3 - sqrt(2) instead of -3 + sqrt(2)
    return bool(expr.sort_key() < (-expr).sort_key())


def _addsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Add(*args):
    """Return a well-formed unevaluated Add: Numbers are collected and
    put in slot 0 and args are sorted. Use this when args have changed
    but you still want to return an unevaluated Add.

    Examples
    ========

    >>> from sympy.core.add import _unevaluated_Add as uAdd
    >>> from sympy import S, Add
    >>> from sympy.abc import x, y
    >>> a = uAdd(*[S(1.0), x, S(2)])
    >>> a.args[0]
    3.00000000000000
    >>> a.args[1]
    x

    Beyond the Number being in slot 0, there is no other assurance of
    order for the arguments since they are hash sorted. So, for testing
    purposes, output produced by this in some other function can only
    be tested against the output of this function or as one of several
    options:

    >>> opts = (Add(x, y, evaluate=False), Add(y, x, evaluate=False))
    >>> a = uAdd(x, y)
    >>> assert a in opts and a == uAdd(x, y)
    >>> uAdd(x + 1, x + 2)
    x + x + 3
    """
    args = list(args)
    newargs = []
    co = S.Zero
    while args:
        a = args.pop()
        if a.is_Add:
            # this will keep nesting from building up
            # so that x + (x + 1) -> x + x + 1 (3 args)
            args.extend(a.args)
        elif a.is_Number:
            co += a
        else:
            newargs.append(a)
    _addsort(newargs)
    if co:
        newargs.insert(0, co)
    return Add._from_args(newargs)


class Add(Expr, AssocOp):
    """
    Expression representing addition operation for algebraic group.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Add()`` must be ``Expr``. Infix operator ``+``
    on most scalar objects in SymPy calls this class.

    Another use of ``Add()`` is to represent the structure of abstract
    addition so that its arguments can be substituted to return different
    class. Refer to examples section for this.

    ``Add()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Add(x, Add(y, z))`` -> ``Add(x, y, z)``

    2. Identity removing
        ``Add(x, 0, y)`` -> ``Add(x, y)``

    3. Coefficient collecting by ``.as_coeff_Mul()``
        ``Add(x, 2*x)`` -> ``Mul(3, x)``

    4. Term sorting
        ``Add(y, x, 2)`` -> ``Add(2, x, y)``

    If no argument is passed, identity element 0 is returned. If single
    element is passed, that element is returned.

    Note that ``Add(*args)`` is more efficient than ``sum(args)`` because
    it flattens the arguments. ``sum(a, b, c, ...)`` recursively adds the
    arguments as ``a + (b + (c + ...))``, which has quadratic complexity.
    On the other hand, ``Add(a, b, c, d)`` does not assume nested
    structure, making the complexity linear.

    Since addition is group operation, every argument should have the
    same :obj:`sympy.core.kind.Kind()`.

    Examples
    ========

    >>> from sympy import Add, I
    >>> from sympy.abc import x, y
    >>> Add(x, 1)
    x + 1
    >>> Add(x, x)
    2*x
    >>> 2*x**2 + 3*x + I*y + 2*y + 2*x/5 + 1.0*y + 1
    2*x**2 + 17*x/5 + 3.0*y + I*y + 1

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Add(1, 2, evaluate=False)
    1 + 2
    >>> Add(x, x, evaluate=False)
    x + x

    ``Add()`` also represents the general structure of addition operation.

    >>> from sympy import MatrixSymbol
    >>> A,B = MatrixSymbol('A', 2,2), MatrixSymbol('B', 2,2)
    >>> expr = Add(x,y).subs({x:A, y:B})
    >>> expr
    A + B
    >>> type(expr)
    <class 'sympy.matrices.expressions.matadd.MatAdd'>

    Note that the printers do not display in args order.

    >>> Add(x, 1)
    x + 1
    >>> Add(x, 1).args
    (1, x)

    See Also
    ========

    MatAdd

    """

    __slots__ = ()

    is_Add = True

    _args_type = Expr

    identity: ClassVar[Expr]

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    @classmethod
    def flatten(cls, seq: list[Expr]) -> tuple[list[Expr], list[Expr], None]:
        """
        Takes the sequence "seq" of nested Adds and returns a flatten list.

        Returns: (commutative_part, noncommutative_part, order_symbols)

        Applies associativity, all terms are commutable with respect to
        addition.

        NB: the removal of 0 is already handled by AssocOp.__new__

        See Also
        ========

        sympy.core.mul.Mul.flatten

        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        from sympy.tensor.tensor import TensExpr, TensAdd
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
            if a.is_Rational:
                if b.is_Mul:
                    rv = [a, b], [], None
            if rv:
                if all(s.is_commutative for s in rv[0]):
                    return rv
                return [], rv[0], None

        # term -> coeff
        # e.g. x**2 -> 5   for ... + 5*x**2 + ...
        terms: dict[Expr, Number] = {}

        # coefficient (Number or zoo) to always be in slot 0
        # e.g. 3 + ...
        coeff: Expr = S.Zero

        order_factors: list[Order] = []

        extra: list[MatrixExpr] = []

        for o in seq:

            # O(x)
            if o.is_Order:
                if o.expr.is_zero: # type: ignore
                    continue
                if any(o1.contains(o) for o1 in order_factors):
                    continue
                order_factors = [o1 for o1 in order_factors if not o.contains(o1)] # type: ignore
                order_factors = [o] + order_factors # type: ignore
                continue

            # 3 or NaN
            elif o.is_Number:
                if (o is S.NaN or coeff is S.ComplexInfinity and
                        o.is_finite is False) and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                if coeff.is_Number or isinstance(coeff, AccumBounds):
                    coeff += o
                    if coeff is S.NaN and not extra:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__add__(coeff)
                continue

            elif isinstance(o, MatrixExpr):
                # can't add 0 to Matrix so make sure coeff is not 0
                extra.append(o)
                continue

            elif isinstance(o, TensExpr):
                coeff = TensAdd(o, coeff).doit(deep=False)
                continue

            elif o is S.ComplexInfinity:
                if coeff.is_finite is False and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            # Add([...])
            elif o.is_Add:
                # NB: here we assume Add is always commutative
                o_args: tuple[Expr, ...] = o.args # type: ignore
                seq.extend(o_args)  # TODO zerocopy?
                continue

            # Mul([...])
            elif o.is_Mul:
                c, s = o.as_coeff_Mul()

            # check for unevaluated Pow, e.g. 2**3 or 2**(-1/2)
            elif o.is_Pow:
                b, e = o.as_base_exp()
                if b.is_Number and (e.is_Integer or
                                   (e.is_Rational and e.is_negative)):
                    seq.append(b**e)
                    continue
                c, s = S.One, o

            else:
                # everything else
                c = S.One
                s = o

            # now we have:
            # o = c*s, where
            #
            # c is a Number
            # s is an expression with number factor extracted
            # let's collect terms with the same s, so e.g.
            # 2*x**2 + 3*x**2  ->  5*x**2
            if s in terms:
                terms[s] += c
                if terms[s] is S.NaN and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
            else:
                terms[s] = c

        # now let's construct new args:
        # [2*x**2, x**3, 7*x**4, pi, ...]
        newseq = []
        noncommutative = False
        for s, c in terms.items():
            # 0*s
            if c.is_zero:
                continue
            # 1*s
            elif c is S.One:
                newseq.append(s)
            # c*s
            else:
                if s.is_Mul:
                    # Mul, already keeps its arguments in perfect order.
                    # so we can simply put c in slot0 and go the fast way.
                    #
                    # XXX: This breaks VectorMul unless it overrides
                    # _new_rawargs
                    cs = s._new_rawargs(*((c,) + s.args)) # type: ignore
                    newseq.append(cs)
                elif s.is_Add:
                    # we just re-create the unevaluated Mul
                    newseq.append(Mul(c, s, evaluate=False))
                else:
                    # alternatively we have to call all Mul's machinery (slow)
                    newseq.append(Mul(c, s))

            noncommutative = noncommutative or not s.is_commutative

        # oo, -oo
        if coeff is S.Infinity:
            newseq = [f for f in newseq if not (f.is_extended_nonnegative or f.is_real)]

        elif coeff is S.NegativeInfinity:
            newseq = [f for f in newseq if not (f.is_extended_nonpositive or f.is_real)]

        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + finite_im
            #   finite_real + infinite_im
            #   infinite_real + infinite_im
            # addition of a finite real or imaginary number won't be able to
            # change the zoo nature; adding an infinite qualtity would result
            # in a NaN condition if it had sign opposite of the infinite
            # portion of zoo, e.g., infinite_real - infinite_real.
            newseq = [c for c in newseq if not (c.is_finite and
                                                c.is_extended_real is not None)]

        # process O(x)
        if order_factors:
            newseq2 = []
            for t in newseq:
                # x + O(x) -> O(x)
                if not any(o.contains(t) for o in order_factors):
                    newseq2.append(t)
            newseq = newseq2 + order_factors # type: ignore
            # 1 + O(1) -> O(1)
            for o in order_factors:
                if o.contains(coeff):
                    coeff = S.Zero
                    break

        # order args canonically
        _addsort(newseq)

        # current code expects coeff to be first
        if coeff is not S.Zero:
            newseq.insert(0, coeff)

        if extra:
            newseq += extra
            noncommutative = True

        # we are done
        if noncommutative:
            return [], newseq, None
        else:
            return newseq, [], None

    @classmethod
    def class_key(cls):
        return 3, 1, cls.__name__

    @property
    def kind(self):
        k = attrgetter('kind')
        kinds = map(k, self.args)
        kinds = frozenset(kinds)
        if len(kinds) != 1:
            # Since addition is group operator, kind must be same.
            # We know that this is unexpected signature, so return this.
            result = UndefinedKind
        else:
            result, = kinds
        return result

    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    @cacheit
    def as_coeff_add(self, *deps):
        """
        Returns a tuple (coeff, args) where self is treated as an Add and coeff
        is the Number term and args is a tuple of all other terms.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (7 + 3*x).as_coeff_add()
        (7, (3*x,))
        >>> (7*x).as_coeff_add()
        (0, (7*x,))
        """
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has_free(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        coeff, notrat = self.args[0].as_coeff_add()
        if coeff is not S.Zero:
            return coeff, notrat + self.args[1:]
        return S.Zero, self.args

    def as_coeff_Add(self, rational=False, deps=None) -> tuple[Number, Expr]:
        """
        Efficiently extract the coefficient of a summation.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number and not rational or coeff.is_Rational:
            return coeff, self._new_rawargs(*args) # type: ignore
        return S.Zero, self

    # Note, we intentionally do not implement Add.as_coeff_mul().  Rather, we
    # let Expr.as_coeff_mul() just always return (S.One, self) for an Add.  See
    # issue 5524.

    def _eval_power(self, expt):
        from .evalf import pure_complex
        from .relational import is_eq
        if len(self.args) == 2 and any(_.is_infinite for _ in self.args):
            if expt.is_zero is False and is_eq(expt, S.One) is False:
                # looking for literal a + I*b
                a, b = self.args
                if a.coeff(S.ImaginaryUnit):
                    a, b = b, a
                ico = b.coeff(S.ImaginaryUnit)
                if ico and ico.is_extended_real and a.is_extended_real:
                    if expt.is_extended_negative:
                        return S.Zero
                    if expt.is_extended_positive:
                        return S.ComplexInfinity
            return
        if expt.is_Rational and self.is_number:
            ri = pure_complex(self)
            if ri:
                r, i = ri
                if expt.q == 2:
                    from sympy.functions.elementary.miscellaneous import sqrt
                    D = sqrt(r**2 + i**2)
                    if D.is_Rational:
                        from .exprtools import factor_terms
                        from sympy.functions.elementary.complexes import sign
                        from .function import expand_multinomial
                        # (r, i, D) is a Pythagorean triple
                        root = sqrt(factor_terms((D - r)/2))**expt.p
                        return root*expand_multinomial((
                            # principle value
                            (D + r)/abs(i) + sign(i)*S.ImaginaryUnit)**expt.p)
                elif expt == -1:
                    return _unevaluated_Mul(
                        r - i*S.ImaginaryUnit,
                        1/(r**2 + i**2))

    @cacheit
    def _eval_derivative(self, s):
        return self.func(*[a.diff(s) for a in self.args])

    def _eval_nseries(self, x, n, logx, cdir=0):
        terms = [t.nseries(x, n=n, logx=logx, cdir=cdir) for t in self.args]
        return self.func(*terms)

    def _matches_simple(self, expr, repl_dict):
        # handle (w+3).matches('x+5') -> {w: x+2}
        coeff, terms = self.as_coeff_add()
        if len(terms) == 1:
            return terms[0].matches(expr - coeff, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        return self._matches_commutative(expr, repl_dict, old)

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs - rhs, but treats oo like a symbol so oo - oo
        returns 0, instead of a nan.
        """
        from sympy.simplify.simplify import signsimp
        inf = (S.Infinity, S.NegativeInfinity)
        if lhs.has(*inf) or rhs.has(*inf):
            from .symbol import Dummy
            oo = Dummy('oo')
            reps = {
                S.Infinity: oo,
                S.NegativeInfinity: -oo}
            ireps = {v: k for k, v in reps.items()}
            eq = lhs.xreplace(reps) - rhs.xreplace(reps)
            if eq.has(oo):
                eq = eq.replace(
                    lambda x: x.is_Pow and x.base is oo,
                    lambda x: x.base)
            rv = eq.xreplace(ireps)
        else:
            rv = lhs - rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_add() which gives the head and a tuple containing
          the arguments of the tail when treated as an Add.
        - if you want the coefficient when self is treated as a Mul
          then use self.as_coeff_mul()[0]

        >>> from sympy.abc import x, y
        >>> (3*x - 2*y + 5).as_two_terms()
        (5, 3*x - 2*y)
        """
        return self.args[0], self._new_rawargs(*self.args[1:])

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """
        Decomposes an expression to its numerator part and its
        denominator part.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> (x*y/z).as_numer_denom()
        (x*y, z)
        >>> (x*(y + 1)/y**7).as_numer_denom()
        (x*(y + 1), y**7)

        See Also
        ========

        sympy.core.expr.Expr.as_numer_denom
        """
        # clear rational denominator
        content, expr = self.primitive()
        if not isinstance(expr, Add):
            return Mul(content, expr, evaluate=False).as_numer_denom()
        ncon, dcon = content.as_numer_denom()

        # collect numerators and denominators of the terms
        nd = defaultdict(list)
        for f in expr.args:
            ni, di = f.as_numer_denom()
            nd[di].append(ni)

        # check for quick exit
        if len(nd) == 1:
            d, n = nd.popitem()
            return self.func(
                *[_keep_coeff(ncon, ni) for ni in n]), _keep_coeff(dcon, d)

        # sum up the terms having a common denominator
        nd2 = {d: self.func(*n) if len(n) > 1 else n[0] for d, n in nd.items()}

        # assemble single numerator and denominator
        denoms, numers = [list(i) for i in zip(*iter(nd2.items()))]
        n, d = self.func(*[Mul(*(denoms[:i] + [numers[i]] + denoms[i + 1:]))
                   for i in range(len(numers))]), Mul(*denoms)

        return _keep_coeff(ncon, n), _keep_coeff(dcon, d)

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # assumption methods
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    _eval_is_extended_real = lambda self: _fuzzy_group(
        (a.is_extended_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_infinite(self):
        sawinf = False
        for a in self.args:
            ainf = a.is_infinite
            if ainf is None:
                return None
            elif ainf is True:
                # infinite+infinite might not be infinite
                if sawinf is True:
                    return None
                sawinf = True
        return sawinf

    def _eval_is_imaginary(self):
        nz = []
        im_I = []
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    pass
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im_I.append(a*S.ImaginaryUnit)
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_I.append(-coeff)
                else:
                    return
            else:
                return
        b = self.func(*nz)
        if b != self:
            if b.is_zero:
                return fuzzy_not(self.func(*im_I).is_zero)
            elif b.is_zero is False:
                return False

    def _eval_is_zero(self):
        if self.is_commutative is False:
            # issue 10528: there is no way to know if a nc symbol
            # is zero or not
            return
        nz = []
        z = 0
        im_or_z = False
        im = 0
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im += 1
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_or_z = True
                else:
                    return
            else:
                return
        if z == len(self.args):
            return True
        if len(nz) in [0, len(self.args)]:
            return None
        b = self.func(*nz)
        if b.is_zero:
            if not im_or_z:
                if im == 0:
                    return True
                elif im == 1:
                    return False
        if b.is_zero is False:
            return False

    def _eval_is_odd(self):
        l = [f for f in self.args if not (f.is_even is True)]
        if not l:
            return False
        if l[0].is_odd:
            return self._new_rawargs(*l[1:]).is_even

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all(x.is_rational is True for x in others):
                    return True
                return None
            if a is None:
                return
        return False

    def _all_nonneg_or_nonppos(self):
        nn = np = 0
        for a in self.args:
            if a.is_nonnegative:
                if np:
                    return False
                nn = 1
            elif a.is_nonpositive:
                if nn:
                    return False
                np = 1
            else:
                break
        else:
            return True

    def _eval_is_extended_positive(self):
        if self.is_number:
            return super()._eval_is_extended_positive()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_positive and a.is_extended_nonnegative:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_positive:
                        return True
        pos = nonneg = nonpos = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            ispos = a.is_extended_positive
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((ispos, a.is_extended_nonnegative)))
                if True in saw_INF and False in saw_INF:
                    return
            if ispos:
                pos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonpos and not nonneg and pos:
            return True
        elif not nonpos and pos:
            return True
        elif not pos and not nonneg:
            return False

    def _eval_is_extended_nonnegative(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonnegative:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonnegative:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonnegative:
                            return True

    def _eval_is_extended_nonpositive(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonpositive:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonpositive:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonpositive:
                            return True

    def _eval_is_extended_negative(self):
        if self.is_number:
            return super()._eval_is_extended_negative()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_negative and a.is_extended_nonpositive:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_negative:
                        return True
        neg = nonpos = nonneg = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            isneg = a.is_extended_negative
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((isneg, a.is_extended_nonpositive)))
                if True in saw_INF and False in saw_INF:
                    return
            if isneg:
                neg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonneg and not nonpos and neg:
            return True
        elif not nonneg and neg:
            return True
        elif not neg and not nonpos:
            return False

    def _eval_subs(self, old, new):
        if not old.is_Add:
            if old is S.Infinity and -old in self.args:
                # foo - oo is foo + (-oo) internally
                return self.xreplace({-old: -new})
            return None

        coeff_self, terms_self = self.as_coeff_Add()
        coeff_old, terms_old = old.as_coeff_Add()

        if coeff_self.is_Rational and coeff_old.is_Rational:
            if terms_self == terms_old:   # (2 + a).subs( 3 + a, y) -> -1 + y
                return self.func(new, coeff_self, -coeff_old)
            if terms_self == -terms_old:  # (2 + a).subs(-3 - a, y) -> -1 - y
                return self.func(-new, coeff_self, coeff_old)

        if coeff_self.is_Rational and coeff_old.is_Rational \
                or coeff_self == coeff_old:
            args_old, args_self = self.func.make_args(
                terms_old), self.func.make_args(terms_self)
            if len(args_old) < len(args_self):  # (a+b+c).subs(b+c,x) -> a+x
                self_set = set(args_self)
                old_set = set(args_old)

                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(new, coeff_self, -coeff_old,
                               *[s._subs(old, new) for s in ret_set])

                args_old = self.func.make_args(
                    -terms_old)     # (a+b+c+d).subs(-b-c,x) -> a-x+d
                old_set = set(args_old)
                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(-new, coeff_self, coeff_old,
                               *[s._subs(old, new) for s in ret_set])

    def removeO(self):
        args = [a for a in self.args if not a.is_Order]
        return self._new_rawargs(*args)

    def getO(self):
        args = [a for a in self.args if a.is_Order]
        if args:
            return self._new_rawargs(*args)

    @cacheit
    def extract_leading_order(self, symbols, point=None):
        """
        Returns the leading term and its order.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (x + 1 + 1/x**5).extract_leading_order(x)
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order(x)
        ((1, O(1)),)
        >>> (x + x**2).extract_leading_order(x)
        ((x, O(x)),)

        """
        from sympy.series.order import Order
        lst = []
        symbols = list(symbols if is_sequence(symbols) else [symbols])
        if not point:
            point = [0]*len(symbols)
        seq = [(f, Order(f, *zip(symbols, point))) for f in self.args]
        for ef, of in seq:
            for e, o in lst:
                if o.contains(of) and o != of:
                    of = None
                    break
            if of is None:
                continue
            new_lst = [(ef, of)]
            for e, o in lst:
                if of.contains(o) and o != of:
                    continue
                new_lst.append((e, o))
            lst = new_lst
        return tuple(lst)

    def as_real_imag(self, deep=True, **hints):
        """
        Return a tuple representing a complex number.

        Examples
        ========

        >>> from sympy import I
        >>> (7 + 9*I).as_real_imag()
        (7, 9)
        >>> ((1 + I)/(1 - I)).as_real_imag()
        (0, 1)
        >>> ((1 + 2*I)*(1 + 3*I)).as_real_imag()
        (-5, 5)
        """
        sargs = self.args
        re_part, im_part = [], []
        for term in sargs:
            re, im = term.as_real_imag(deep=deep)
            re_part.append(re)
            im_part.append(im)
        return (self.func(*re_part), self.func(*im_part))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.core.symbol import Dummy, Symbol
        from sympy.series.order import Order
        from sympy.functions.elementary.exponential import log
        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        from .function import expand_mul

        o = self.getO()
        if o is None:
            o = Order(0)
        old = self.removeO()

        if old.has(Piecewise):
            old = piecewise_fold(old)

        # This expansion is the last part of expand_log. expand_log also calls
        # expand_mul with factor=True, which would be more expensive
        if any(isinstance(a, log) for a in self.args):
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": False,
                "factor": False}
            old = old.expand(**logflags)
        expr = expand_mul(old)

        if not expr.is_Add:
            return expr.as_leading_term(x, logx=logx, cdir=cdir)

        infinite = [t for t in expr.args if t.is_infinite]

        _logx = Dummy('logx') if logx is None else logx
        leading_terms = [t.as_leading_term(x, logx=_logx, cdir=cdir) for t in expr.args]

        min, new_expr = Order(0), S.Zero

        try:
            for term in leading_terms:
                order = Order(term, x)
                if not min or order not in min:
                    min = order
                    new_expr = term
                elif min in order:
                    new_expr += term

        except TypeError:
            return expr

        if logx is None:
            new_expr = new_expr.subs(_logx, log(x))

        is_zero = new_expr.is_zero
        if is_zero is None:
            new_expr = new_expr.trigsimp().cancel()
            is_zero = new_expr.is_zero
        if is_zero is True:
            # simple leading term analysis gave us cancelled terms but we have to send
            # back a term, so compute the leading term (via series)
            try:
                n0 = min.getn()
            except NotImplementedError:
                n0 = S.One
            if n0.has(Symbol):
                n0 = S.One
            res = Order(1)
            incr = S.One
            while res.is_Order:
                res = old._eval_nseries(x, n=n0+incr, logx=logx, cdir=cdir).cancel().powsimp().trigsimp()
                incr *= 2
            return res.as_leading_term(x, logx=logx, cdir=cdir)

        elif new_expr is S.NaN:
            return old.func._from_args(infinite) + o

        else:
            return new_expr

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args])

    def primitive(self):
        """
        Return ``(R, self/R)`` where ``R``` is the Rational GCD of ``self```.

        ``R`` is collected only from the leading coefficient of each term.

        Examples
        ========

        >>> from sympy.abc import x, y

        >>> (2*x + 4*y).primitive()
        (2, x + 2*y)

        >>> (2*x/3 + 4*y/9).primitive()
        (2/9, 3*x + 2*y)

        >>> (2*x/3 + 4.2*y).primitive()
        (1/3, 2*x + 12.6*y)

        No subprocessing of term factors is performed:

        >>> ((2 + 2*x)*x + 2).primitive()
        (1, x*(2*x + 2) + 2)

        Recursive processing can be done with the ``as_content_primitive()``
        method:

        >>> ((2 + 2*x)*x + 2).as_content_primitive()
        (2, x*(x + 1) + 1)

        See also: primitive() function in polytools.py

        """

        terms = []
        inf = False
        for a in self.args:
            c, m = a.as_coeff_Mul()
            if not c.is_Rational:
                c = S.One
                m = a
            inf = inf or m is S.ComplexInfinity
            terms.append((c.p, c.q, m))

        if not inf:
            ngcd = reduce(igcd, [t[0] for t in terms], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms], 1)
        else:
            ngcd = reduce(igcd, [t[0] for t in terms if t[1]], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms if t[1]], 1)

        if ngcd == dlcm == 1:
            return S.One, self
        if not inf:
            for i, (p, q, term) in enumerate(terms):
                terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
        else:
            for i, (p, q, term) in enumerate(terms):
                if q:
                    terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
                else:
                    terms[i] = _keep_coeff(Rational(p, q), term)

        # we don't need a complete re-flattening since no new terms will join
        # so we just use the same sort as is used in Add.flatten. When the
        # coefficient changes, the ordering of terms may change, e.g.
        #     (3*x, 6*y) -> (2*y, x)
        #
        # We do need to make sure that term[0] stays in position 0, however.
        #
        if terms[0].is_Number or terms[0] is S.ComplexInfinity:
            c = terms.pop(0)
        else:
            c = None
        _addsort(terms)
        if c:
            terms.insert(0, c)
        return Rational(ngcd, dlcm), self._new_rawargs(*terms)

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self. If radical is True (default is False) then
        common radicals will be removed and included as a factor of the
        primitive expression.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (3 + 3*sqrt(2)).as_content_primitive()
        (3, 1 + sqrt(2))

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        See docstring of Expr.as_content_primitive for more examples.
        """
        con, prim = self.func(*[_keep_coeff(*a.as_content_primitive(
            radical=radical, clear=clear)) for a in self.args]).primitive()
        if not clear and not con.is_Integer and prim.is_Add:
            con, d = con.as_numer_denom()
            _p = prim/d
            if any(a.as_coeff_Mul()[0].is_Integer for a in _p.args):
                prim = _p
            else:
                con /= d
        if radical and prim.is_Add:
            # look for common radicals that can be removed
            args = prim.args
            rads = []
            common_q = None
            for m in args:
                term_rads = defaultdict(list)
                for ai in Mul.make_args(m):
                    if ai.is_Pow:
                        b, e = ai.as_base_exp()
                        if e.is_Rational and b.is_Integer:
                            term_rads[e.q].append(abs(int(b))**e.p)
                if not term_rads:
                    break
                if common_q is None:
                    common_q = set(term_rads.keys())
                else:
                    common_q = common_q & set(term_rads.keys())
                    if not common_q:
                        break
                rads.append(term_rads)
            else:
                # process rads
                # keep only those in common_q
                for r in rads:
                    for q in list(r.keys()):
                        if q not in common_q:
                            r.pop(q)
                    for q in r:
                        r[q] = Mul(*r[q])
                # find the gcd of bases for each q
                G = []
                for q in common_q:
                    g = reduce(igcd, [r[q] for r in rads], 0)
                    if g != 1:
                        G.append(g**Rational(1, q))
                if G:
                    G = Mul(*G)
                    args = [ai/G for ai in args]
                    prim = G*prim.func(*args)

        return con, prim

    @property
    def _sorted_args(self):
        from .sorting import default_sort_key
        return tuple(sorted(self.args, key=default_sort_key))

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        return self.func(*[dd(a, n, step) for a in self.args])

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        re_part, rest = self.as_coeff_Add()
        im_part, imag_unit = rest.as_coeff_Mul()
        if not imag_unit == S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Add to mpc. Must be of the form Number + Number*I")

        return (Float(re_part)._mpf_, Float(im_part)._mpf_)

    def __neg__(self):
        if not global_parameters.distribute:
            return super().__neg__()
        return Mul(S.NegativeOne, self)

add = AssocOpDispatcher('add')

from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
