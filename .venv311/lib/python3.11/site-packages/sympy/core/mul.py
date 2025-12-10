from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar

from collections import defaultdict
from functools import reduce
from itertools import product
import operator

from .sympify import sympify
from .basic import Basic, _args_sortkey
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift


# internal marker to indicate:
#   "there are still non-commutative objects -- don't forget to process them"
class NC_Marker:
    is_Order = False
    is_Mul = False
    is_Number = False
    is_Poly = False

    is_commutative = False


def _mulsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Mul(*args):
    """Return a well-formed unevaluated Mul: Numbers are collected and
    put in slot 0, any arguments that are Muls will be flattened, and args
    are sorted. Use this when args have changed but you still want to return
    an unevaluated Mul.

    Examples
    ========

    >>> from sympy.core.mul import _unevaluated_Mul as uMul
    >>> from sympy import S, sqrt, Mul
    >>> from sympy.abc import x
    >>> a = uMul(*[S(3.0), x, S(2)])
    >>> a.args[0]
    6.00000000000000
    >>> a.args[1]
    x

    Two unevaluated Muls with the same arguments will
    always compare as equal during testing:

    >>> m = uMul(sqrt(2), sqrt(3))
    >>> m == uMul(sqrt(3), sqrt(2))
    True
    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)
    >>> m == uMul(u)
    True
    >>> m == Mul(*m.args)
    False

    """
    cargs = []
    ncargs = []
    args = list(args)
    co = S.One
    for a in args:
        if a.is_Mul:
            a_c, a_nc = a.args_cnc()
            args.extend(a_c)  # grow args
            ncargs.extend(a_nc)
        elif a.is_Number:
            co *= a
        elif a.is_commutative:
            cargs.append(a)
        else:
            ncargs.append(a)
    _mulsort(cargs)
    if co is not S.One:
        cargs.insert(0, co)
    return Mul._from_args(cargs+ncargs)


class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()

    is_Mul = True

    _args_type = Expr
    _kind_dispatcher = KindDispatcher("Mul_kind_dispatcher", commutative=True)

    identity: ClassVar[Expr]

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def could_extract_minus_sign(self):
        if self == (-self):
            return False  # e.g. zoo*x == -zoo*x
        c = self.args[0]
        return c.is_Number and c.is_extended_negative

    def __neg__(self):
        c, args = self.as_coeff_mul()
        if args[0] is not S.ComplexInfinity:
            c = -c
        if c is not S.One:
            if args[0].is_Number:
                args = list(args)
                if c is S.NegativeOne:
                    args[0] = -args[0]
                else:
                    args[0] *= c
            else:
                args = (c,) + args
        return self._from_args(args, self.is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return commutative, noncommutative and order arguments by
        combining related terms.

        Notes
        =====
            * In an expression like ``a*b*c``, Python process this through SymPy
              as ``Mul(Mul(a, b), c)``. This can have undesirable consequences.

              -  Sometimes terms are not combined as one would like:
                 {c.f. https://github.com/sympy/sympy/issues/4596}

                >>> from sympy import Mul, sqrt
                >>> from sympy.abc import x, y, z
                >>> 2*(x + 1) # this is the 2-arg Mul behavior
                2*x + 2
                >>> y*(x + 1)*2
                2*y*(x + 1)
                >>> 2*(x + 1)*y # 2-arg result will be obtained first
                y*(2*x + 2)
                >>> Mul(2, x + 1, y) # all 3 args simultaneously processed
                2*y*(x + 1)
                >>> 2*((x + 1)*y) # parentheses can control this behavior
                2*y*(x + 1)

                Powers with compound bases may not find a single base to
                combine with unless all arguments are processed at once.
                Post-processing may be necessary in such cases.
                {c.f. https://github.com/sympy/sympy/issues/5728}

                >>> a = sqrt(x*sqrt(y))
                >>> a**3
                (x*sqrt(y))**(3/2)
                >>> Mul(a,a,a)
                (x*sqrt(y))**(3/2)
                >>> a*a*a
                x*sqrt(y)*sqrt(x*sqrt(y))
                >>> _.subs(a.base, z).subs(z, a.base)
                (x*sqrt(y))**(3/2)

              -  If more than two terms are being multiplied then all the
                 previous terms will be re-processed for each new argument.
                 So if each of ``a``, ``b`` and ``c`` were :class:`Mul`
                 expression, then ``a*b*c`` (or building up the product
                 with ``*=``) will process all the arguments of ``a`` and
                 ``b`` twice: once when ``a*b`` is computed and again when
                 ``c`` is multiplied.

                 Using ``Mul(a, b, c)`` will process all arguments once.

            * The results of Mul are cached according to arguments, so flatten
              will only be called once for ``Mul(a, b, c)``. If you can
              structure a calculation so the arguments are most likely to be
              repeats then this can save time in computing the answer. For
              example, say you had a Mul, M, that you wished to divide by ``d[i]``
              and multiply by ``n[i]`` and you suspect there are many repeats
              in ``n``. It would be better to compute ``M*n[i]/d[i]`` rather
              than ``M/d[i]*n[i]`` since every time n[i] is a repeat, the
              product, ``M*n[i]`` will be returned without flattening -- the
              cached value will be returned. If you divide by the ``d[i]``
              first (and those are more unique than the ``n[i]``) then that will
              create a new Mul, ``M/d[i]`` the args of which will be traversed
              again when it is multiplied by ``n[i]``.

              {c.f. https://github.com/sympy/sympy/issues/5706}

              This consideration is moot if the cache is turned off.

            NB
            --
              The validity of the above notes depends on the implementation
              details of Mul and flatten which may change at any time. Therefore,
              you should only consider them when your code is highly performance
              sensitive.

              Removal of 1 from the sequence is already handled by AssocOp.__new__.
        """

        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
                seq = [a, b]
            assert a is not S.One
            if a.is_Rational and not a.is_zero:
                r, b = b.as_coeff_Mul()
                if b.is_Add:
                    if r is not S.One:  # 2-arg hack
                        # leave the Mul as a Mul?
                        ar = a*r
                        if ar is S.One:
                            arb = b
                        else:
                            arb = cls(a*r, b, evaluate=False)
                        rv = [arb], [], None
                    elif global_parameters.distribute and b.is_commutative:
                        newb = Add(*[_keep_coeff(a, bi) for bi in b.args])
                        rv = [newb], [], None
            if rv:
                return rv

        # apply associativity, separate commutative part of seq
        c_part = []         # out: commutative factors
        nc_part = []        # out: non-commutative factors

        nc_seq = []

        coeff = S.One       # standalone term
                            # e.g. 3 * ...

        c_powers = []       # (base,exp)      n
                            # e.g. (x,n) for x

        num_exp = []        # (num-base, exp)           y
                            # e.g.  (3, y)  for  ... * 3  * ...

        neg1e = S.Zero      # exponent on -1 extracted from Number-based Pow and I

        pnum_rat = {}       # (num-base, Rat-exp)          1/2
                            # e.g.  (3, 1/2)  for  ... * 3     * ...

        order_symbols = None

        # --- PART 1 ---
        #
        # "collect powers and coeff":
        #
        # o coeff
        # o c_powers
        # o num_exp
        # o neg1e
        # o pnum_rat
        #
        # NOTE: this is optimized for all-objects-are-commutative case
        for o in seq:
            # O(x)
            if o.is_Order:
                o, order_symbols = o.as_expr_variables(order_symbols)

            # Mul([...])
            if o.is_Mul:
                if o.is_commutative:
                    seq.extend(o.args)    # XXX zerocopy?

                else:
                    # NCMul can have commutative parts as well
                    for q in o.args:
                        if q.is_commutative:
                            seq.append(q)
                        else:
                            nc_seq.append(q)

                    # append non-commutative marker, so we don't forget to
                    # process scheduled non-commutative objects
                    seq.append(NC_Marker)

                continue

            # 3
            elif o.is_Number:
                if o is S.NaN or coeff is S.ComplexInfinity and o.is_zero:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                elif coeff.is_Number or isinstance(coeff, AccumBounds):  # it could be zoo
                    coeff *= o
                    if coeff is S.NaN:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__mul__(coeff)
                continue

            elif o is S.ComplexInfinity:
                if not coeff:
                    # 0 * zoo = NaN
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            elif not coeff and isinstance(o, Add) and any(
                    _ in (S.NegativeInfinity, S.ComplexInfinity, S.Infinity)
                    for __ in o.args for _ in Mul.make_args(__)):
                # e.g 0 * (x + oo) = NaN but not
                # 0 * (1 + Integral(x, (x, 0, oo))) which is
                # treated like 0 * x -> 0
                return [S.NaN], [], None

            elif o is S.ImaginaryUnit:
                neg1e += S.Half
                continue

            elif o.is_commutative:
                #      e
                # o = b
                b, e = o.as_base_exp()

                #  y
                # 3
                if o.is_Pow:
                    if b.is_Number:

                        # get all the factors with numeric base so they can be
                        # combined below, but don't combine negatives unless
                        # the exponent is an integer
                        if e.is_Rational:
                            if e.is_Integer:
                                coeff *= Pow(b, e)  # it is an unevaluated power
                                continue
                            elif e.is_negative:    # also a sign of an unevaluated power
                                seq.append(Pow(b, e))
                                continue
                            elif b.is_negative:
                                neg1e += e
                                b = -b
                            if b is not S.One:
                                pnum_rat.setdefault(b, []).append(e)
                            continue
                        elif b.is_positive or e.is_integer:
                            num_exp.append((b, e))
                            continue

                c_powers.append((b, e))

            # NON-COMMUTATIVE
            # TODO: Make non-commutative exponents not combine automatically
            else:
                if o is not NC_Marker:
                    nc_seq.append(o)

                # process nc_seq (if any)
                while nc_seq:
                    o = nc_seq.pop(0)
                    if not nc_part:
                        nc_part.append(o)
                        continue

                    #                             b    c       b+c
                    # try to combine last terms: a  * a   ->  a
                    o1 = nc_part.pop()
                    b1, e1 = o1.as_base_exp()
                    b2, e2 = o.as_base_exp()
                    new_exp = e1 + e2
                    # Only allow powers to combine if the new exponent is
                    # not an Add. This allow things like a**2*b**3 == a**5
                    # if a.is_commutative == False, but prohibits
                    # a**x*a**y and x**a*x**b from combining (x,y commute).
                    if b1 == b2 and (not new_exp.is_Add):
                        o12 = b1 ** new_exp

                        # now o12 could be a commutative object
                        if o12.is_commutative:
                            seq.append(o12)
                            continue
                        else:
                            nc_seq.insert(0, o12)

                    else:
                        nc_part.extend([o1, o])

        # We do want a combined exponent if it would not be an Add, such as
        #  y    2y     3y
        # x  * x   -> x
        # We determine if two exponents have the same term by using
        # as_coeff_Mul.
        #
        # Unfortunately, this isn't smart enough to consider combining into
        # exponents that might already be adds, so things like:
        #  z - y    y
        # x      * x  will be left alone.  This is because checking every possible
        # combination can slow things down.

        # gather exponents of common bases...
        def _gather(c_powers):
            common_b = {}  # b:e
            for b, e in c_powers:
                co = e.as_coeff_Mul()
                common_b.setdefault(b, {}).setdefault(
                    co[1], []).append(co[0])
            for b, d in common_b.items():
                for di, li in d.items():
                    d[di] = Add(*li)
            new_c_powers = []
            for b, e in common_b.items():
                new_c_powers.extend([(b, c*t) for t, c in e.items()])
            return new_c_powers

        # in c_powers
        c_powers = _gather(c_powers)

        # and in num_exp
        num_exp = _gather(num_exp)

        # --- PART 2 ---
        #
        # o process collected powers  (x**0 -> 1; x**1 -> x; otherwise Pow)
        # o combine collected powers  (2**x * 3**x -> 6**x)
        #   with numeric base

        # ................................
        # now we have:
        # - coeff:
        # - c_powers:    (b, e)
        # - num_exp:     (2, e)
        # - pnum_rat:    {(1/3, [1/3, 2/3, 1/4])}

        #  0             1
        # x  -> 1       x  -> x

        # this should only need to run twice; if it fails because
        # it needs to be run more times, perhaps this should be
        # changed to a "while True" loop -- the only reason it
        # isn't such now is to allow a less-than-perfect result to
        # be obtained rather than raising an error or entering an
        # infinite loop
        for i in range(2):
            new_c_powers = []
            changed = False
            for b, e in c_powers:
                if e.is_zero:
                    # canceling out infinities yields NaN
                    if (b.is_Add or b.is_Mul) and any(infty in b.args
                        for infty in (S.ComplexInfinity, S.Infinity,
                                      S.NegativeInfinity)):
                        return [S.NaN], [], None
                    continue
                if e is S.One:
                    if b.is_Number:
                        coeff *= b
                        continue
                    p = b
                if e is not S.One:
                    p = Pow(b, e)
                    # check to make sure that the base doesn't change
                    # after exponentiation; to allow for unevaluated
                    # Pow, we only do so if b is not already a Pow
                    if p.is_Pow and not b.is_Pow:
                        bi = b
                        b, e = p.as_base_exp()
                        if b != bi:
                            changed = True
                c_part.append(p)
                new_c_powers.append((b, e))
            # there might have been a change, but unless the base
            # matches some other base, there is nothing to do
            if changed and len({
                    b for b, e in new_c_powers}) != len(new_c_powers):
                # start over again
                c_part = []
                c_powers = _gather(new_c_powers)
            else:
                break

        #  x    x     x
        # 2  * 3  -> 6
        inv_exp_dict = {}   # exp:Mul(num-bases)     x    x
                            # e.g.  x:6  for  ... * 2  * 3  * ...
        for b, e in num_exp:
            inv_exp_dict.setdefault(e, []).append(b)
        for e, b in inv_exp_dict.items():
            inv_exp_dict[e] = cls(*b)
        c_part.extend([Pow(b, e) for e, b in inv_exp_dict.items() if e])

        # b, e -> e' = sum(e), b
        # {(1/5, [1/3]), (1/2, [1/12, 1/4]} -> {(1/3, [1/5, 1/2])}
        comb_e = {}
        for b, e in pnum_rat.items():
            comb_e.setdefault(Add(*e), []).append(b)
        del pnum_rat
        # process them, reducing exponents to values less than 1
        # and updating coeff if necessary else adding them to
        # num_rat for further processing
        num_rat = []
        for e, b in comb_e.items():
            b = cls(*b)
            if e.q == 1:
                coeff *= Pow(b, e)
                continue
            if e.p > e.q:
                e_i, ep = divmod(e.p, e.q)
                coeff *= Pow(b, e_i)
                e = Rational(ep, e.q)
            num_rat.append((b, e))
        del comb_e

        # extract gcd of bases in num_rat
        # 2**(1/3)*6**(1/4) -> 2**(1/3+1/4)*3**(1/4)
        pnew = defaultdict(list)
        i = 0  # steps through num_rat which may grow
        while i < len(num_rat):
            bi, ei = num_rat[i]
            if bi == 1:
                i += 1
                continue
            grow = []
            for j in range(i + 1, len(num_rat)):
                bj, ej = num_rat[j]
                g = bi.gcd(bj)
                if g is not S.One:
                    # 4**r1*6**r2 -> 2**(r1+r2)  *  2**r1 *  3**r2
                    # this might have a gcd with something else
                    e = ei + ej
                    if e.q == 1:
                        coeff *= Pow(g, e)
                    else:
                        if e.p > e.q:
                            e_i, ep = divmod(e.p, e.q)  # change e in place
                            coeff *= Pow(g, e_i)
                            e = Rational(ep, e.q)
                        grow.append((g, e))
                    # update the jth item
                    num_rat[j] = (bj/g, ej)
                    # update bi that we are checking with
                    bi = bi/g
                    if bi is S.One:
                        break
            if bi is not S.One:
                obj = Pow(bi, ei)
                if obj.is_Number:
                    coeff *= obj
                else:
                    # changes like sqrt(12) -> 2*sqrt(3)
                    for obj in Mul.make_args(obj):
                        if obj.is_Number:
                            coeff *= obj
                        else:
                            assert obj.is_Pow
                            bi, ei = obj.args
                            pnew[ei].append(bi)

            num_rat.extend(grow)
            i += 1

        # combine bases of the new powers
        for e, b in pnew.items():
            pnew[e] = cls(*b)

        # handle -1 and I
        if neg1e:
            # treat I as (-1)**(1/2) and compute -1's total exponent
            p, q =  neg1e.as_numer_denom()
            # if the integer part is odd, extract -1
            n, p = divmod(p, q)
            if n % 2:
                coeff = -coeff
            # if it's a multiple of 1/2 extract I
            if q == 2:
                c_part.append(S.ImaginaryUnit)
            elif p:
                # see if there is any positive base this power of
                # -1 can join
                neg1e = Rational(p, q)
                for e, b in pnew.items():
                    if e == neg1e and b.is_positive:
                        pnew[e] = -b
                        break
                else:
                    # keep it separate; we've already evaluated it as
                    # much as possible so evaluate=False
                    c_part.append(Pow(S.NegativeOne, neg1e, evaluate=False))

        # add all the pnew powers
        c_part.extend([Pow(b, e) for e, b in pnew.items()])

        # oo, -oo
        if coeff in (S.Infinity, S.NegativeInfinity):
            def _handle_for_oo(c_part, coeff_sign):
                new_c_part = []
                for t in c_part:
                    if t.is_extended_positive:
                        continue
                    if t.is_extended_negative:
                        coeff_sign *= -1
                        continue
                    new_c_part.append(t)
                return new_c_part, coeff_sign
            c_part, coeff_sign = _handle_for_oo(c_part, 1)
            nc_part, coeff_sign = _handle_for_oo(nc_part, coeff_sign)
            coeff *= coeff_sign

        # zoo
        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + bounded_im
            #   bounded_real + infinite_im
            #   infinite_real + infinite_im
            # and non-zero real or imaginary will not change that status.
            c_part = [c for c in c_part if not (fuzzy_not(c.is_zero) and
                                                c.is_extended_real is not None)]
            nc_part = [c for c in nc_part if not (fuzzy_not(c.is_zero) and
                                                  c.is_extended_real is not None)]

        # 0
        elif coeff.is_zero:
            # we know for sure the result will be 0 except the multiplicand
            # is infinity or a matrix
            if any(isinstance(c, MatrixExpr) for c in nc_part):
                return [coeff], nc_part, order_symbols
            if any(c.is_finite == False for c in c_part):
                return [S.NaN], [], order_symbols
            return [coeff], [], order_symbols

        # check for straggling Numbers that were produced
        _new = []
        for i in c_part:
            if i.is_Number:
                coeff *= i
            else:
                _new.append(i)
        c_part = _new

        # order commutative part canonically
        _mulsort(c_part)

        # current code expects coeff to be always in slot-0
        if coeff is not S.One:
            c_part.insert(0, coeff)

        # we are done
        if (global_parameters.distribute and not nc_part and len(c_part) == 2 and
                c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add):
            # 2*(1+a) -> 2 + 2 * a
            coeff = c_part[0]
            c_part = [Add(*[coeff*f for f in c_part[1].args])]

        return c_part, nc_part, order_symbols

    def _eval_power(self, expt):

        # don't break up NC terms: (A*B)**3 != A**3*B**3, it is A*B*A*B*A*B
        cargs, nc = self.args_cnc(split_1=False)

        if expt.is_Integer:
            return Mul(*[Pow(b, expt, evaluate=False) for b in cargs]) * \
                Pow(Mul._from_args(nc), expt, evaluate=False)
        if expt.is_Rational and expt.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    n, d = abs(a/2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    if t:
                        d, t = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n)/d
                            return _unevaluated_Mul(r**expt.p, (1 + sign(a)*S.ImaginaryUnit)**expt.p)

        p = Pow(self, expt, evaluate=False)

        if expt.is_Rational or expt.is_Float:
            return p._eval_expand_power_base()

        return p

    @classmethod
    def class_key(cls):
        return 3, 0, cls.__name__

    def _eval_evalf(self, prec):
        c, m = self.as_coeff_Mul()
        if c is S.NegativeOne:
            if m.is_Mul:
                rv = -AssocOp._eval_evalf(m, prec)
            else:
                mnew = m._eval_evalf(prec)
                if mnew is not None:
                    m = mnew
                rv = -m
        else:
            rv = AssocOp._eval_evalf(self, prec)
        if rv.is_number:
            return rv.expand()
        return rv

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        im_part, imag_unit = self.as_coeff_Mul()
        if imag_unit is not S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Mul to mpc. Must be of the form Number*I")

        return (Float(0)._mpf_, Float(im_part)._mpf_)

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_mul() which gives the head and a tuple containing
          the arguments of the tail when treated as a Mul.
        - if you want the coefficient when self is treated as an Add
          then use self.as_coeff_add()[0]

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (3*x*y).as_two_terms()
        (3, x*y)
        """
        args = self.args

        if len(args) == 1:
            return S.One, self
        elif len(args) == 2:
            return args

        else:
            return args[0], self._new_rawargs(*args[1:])

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        args = self.args
        if args[0].is_Number:
            if not rational or args[0].is_Rational:
                return args[0], args[1:]
            elif args[0].is_extended_negative:
                return S.NegativeOne, (-args[0],) + args[1:]
        return S.One, args

    def as_coeff_Mul(self, rational=False):
        """
        Efficiently extract the coefficient of a product.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number:
            if not rational or coeff.is_Rational:
                if len(args) == 1:
                    return coeff, args[0]
                else:
                    return coeff, self._new_rawargs(*args)
            elif coeff.is_extended_negative:
                return S.NegativeOne, self._new_rawargs(*((-coeff,) + args))
        return S.One, self

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []
        coeffr = []
        coeffi = []
        addterms = S.One
        for a in self.args:
            r, i = a.as_real_imag()
            if i.is_zero:
                coeffr.append(r)
            elif r.is_zero:
                coeffi.append(i*S.ImaginaryUnit)
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                # search for complex conjugate pairs:
                for i, x in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x)**2)
                        del other[i]
                        break
                else:
                    if a.is_Add:
                        addterms *= a
                    else:
                        other.append(a)
            else:
                other.append(a)
        m = self.func(*other)
        if hints.get('ignore') == m:
            return
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))
            # all other pairs make a real factor; they will be
            # put into reco below
        else:
            imco = S.Zero
        reco = self.func(*(coeffr + coeffi))
        r, i = (reco*re(m), reco*im(m))
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)
                else:
                    return (S.Zero, reco*imco)
            if imco is S.Zero:
                return (r, i)
            return (-imco*i, imco*r)
        from .function import expand_mul
        addre, addim = expand_mul(addterms, deep=False).as_real_imag()
        if imco is S.Zero:
            return (r*addre - i*addim, i*addre + r*addim)
        else:
            r, i = -imco*i, imco*r
            return (r*addre - i*addim, r*addim + i*addre)

    @staticmethod
    def _expandsums(sums):
        """
        Helper function for _eval_expand_mul.

        sums must be a list of instances of Basic.
        """

        L = len(sums)
        if L == 1:
            return sums[0].args
        terms = []
        left = Mul._expandsums(sums[:L//2])
        right = Mul._expandsums(sums[L//2:])

        terms = [Mul(a, b) for a in left for b in right]
        added = Add(*terms)
        return Add.make_args(added)  # it may have collapsed down to one term

    def _eval_expand_mul(self, **hints):
        from sympy.simplify.radsimp import fraction

        # Handle things like 1/(x*(x + 1)), which are automatically converted
        # to 1/x*1/(x + 1)
        expr = self
        # default matches fraction's default
        n, d = fraction(expr, hints.get('exact', False))
        if d.is_Mul:
            n, d = [i._eval_expand_mul(**hints) if i.is_Mul else i
                for i in (n, d)]
        expr = n/d
        if not expr.is_Mul:
            return expr

        plain, sums, rewrite = [], [], False
        for factor in expr.args:
            if factor.is_Add:
                sums.append(factor)
                rewrite = True
            else:
                if factor.is_commutative:
                    plain.append(factor)
                else:
                    sums.append(Basic(factor))  # Wrapper

        if not rewrite:
            return expr
        else:
            plain = self.func(*plain)
            if sums:
                deep = hints.get("deep", False)
                terms = self.func._expandsums(sums)
                args = []
                for term in terms:
                    t = self.func(plain, term)
                    if t.is_Mul and any(a.is_Add for a in t.args) and deep:
                        t = t._eval_expand_mul()
                    args.append(t)
                return Add(*args)
            else:
                return plain

    @cacheit
    def _eval_derivative(self, s):
        args = list(self.args)
        terms = []
        for i in range(len(args)):
            d = args[i].diff(s)
            if d:
                # Note: reduce is used in step of Mul as Mul is unable to
                # handle subtypes and operation priority:
                terms.append(reduce(lambda x, y: x*y, (args[:i] + [d] + args[i + 1:]), S.One))
        return Add.fromiter(terms)

    @cacheit
    def _eval_derivative_n_times(self, s, n):
        from .function import AppliedUndef
        from .symbol import Symbol, symbols, Dummy
        if not isinstance(s, (AppliedUndef, Symbol)):
            # other types of s may not be well behaved, e.g.
            # (cos(x)*sin(y)).diff([[x, y, z]])
            return super()._eval_derivative_n_times(s, n)
        from .numbers import Integer
        args = self.args
        m = len(args)
        if isinstance(n, (int, Integer)):
            # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
            terms = []
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator
            for kvals, c in multinomial_coefficients_iterator(m, n):
                p = Mul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            return Add(*terms)
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import factorial
        from sympy.functions.elementary.miscellaneous import Max
        kvals = symbols("k1:%i" % m, cls=Dummy)
        klast = n - sum(kvals)
        nfact = factorial(n)
        e, l = (# better to use the multinomial?
            nfact/prod(map(factorial, kvals))/factorial(klast)*\
            Mul(*[args[t].diff((s, kvals[t])) for t in range(m-1)])*\
            args[-1].diff((s, Max(0, klast))),
            [(k, 0, n) for k in kvals])
        return Sum(e, *l)

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        arg0 = self.args[0]
        rest = Mul(*self.args[1:])
        return (arg0.subs(n, n + step) * dd(rest, n, step) + dd(arg0, n, step) *
                rest)

    def _matches_simple(self, expr, repl_dict):
        # handle (w*3).matches('x*5') -> {w: x*5/3}
        coeff, terms = self.as_coeff_Mul()
        terms = Mul.make_args(terms)
        if len(terms) == 1:
            newexpr = self.__class__._combine_inverse(expr, coeff)
            return terms[0].matches(newexpr, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)
        if self.is_commutative and expr.is_commutative:
            return self._matches_commutative(expr, repl_dict, old)
        elif self.is_commutative is not expr.is_commutative:
            return None

        # Proceed only if both both expressions are non-commutative
        c1, nc1 = self.args_cnc()
        c2, nc2 = expr.args_cnc()
        c1, c2 = [c or [1] for c in [c1, c2]]

        # TODO: Should these be self.func?
        comm_mul_self = Mul(*c1)
        comm_mul_expr = Mul(*c2)

        repl_dict = comm_mul_self.matches(comm_mul_expr, repl_dict, old)

        # If the commutative arguments didn't match and aren't equal, then
        # then the expression as a whole doesn't match
        if not repl_dict and c1 != c2:
            return None

        # Now match the non-commutative arguments, expanding powers to
        # multiplications
        nc1 = Mul._matches_expand_pows(nc1)
        nc2 = Mul._matches_expand_pows(nc2)

        repl_dict = Mul._matches_noncomm(nc1, nc2, repl_dict)

        return repl_dict or None

    @staticmethod
    def _matches_expand_pows(arg_list):
        new_args = []
        for arg in arg_list:
            if arg.is_Pow and arg.exp > 0:
                new_args.extend([arg.base] * arg.exp)
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _matches_noncomm(nodes, targets, repl_dict=None):
        """Non-commutative multiplication matcher.

        `nodes` is a list of symbols within the matcher multiplication
        expression, while `targets` is a list of arguments in the
        multiplication expression being matched against.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # List of possible future states to be considered
        agenda = []
        # The current matching state, storing index in nodes and targets
        state = (0, 0)
        node_ind, target_ind = state
        # Mapping between wildcard indices and the index ranges they match
        wildcard_dict = {}

        while target_ind < len(targets) and node_ind < len(nodes):
            node = nodes[node_ind]

            if node.is_Wild:
                Mul._matches_add_wildcard(wildcard_dict, state)

            states_matches = Mul._matches_new_states(wildcard_dict, state,
                                                     nodes, targets)
            if states_matches:
                new_states, new_matches = states_matches
                agenda.extend(new_states)
                if new_matches:
                    for match in new_matches:
                        repl_dict[match] = new_matches[match]
            if not agenda:
                return None
            else:
                state = agenda.pop()
                node_ind, target_ind = state

        return repl_dict

    @staticmethod
    def _matches_add_wildcard(dictionary, state):
        node_ind, target_ind = state
        if node_ind in dictionary:
            begin, end = dictionary[node_ind]
            dictionary[node_ind] = (begin, target_ind)
        else:
            dictionary[node_ind] = (target_ind, target_ind)

    @staticmethod
    def _matches_new_states(dictionary, state, nodes, targets):
        node_ind, target_ind = state
        node = nodes[node_ind]
        target = targets[target_ind]

        # Don't advance at all if we've exhausted the targets but not the nodes
        if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
            return None

        if node.is_Wild:
            match_attempt = Mul._matches_match_wilds(dictionary, node_ind,
                                                     nodes, targets)
            if match_attempt:
                # If the same node has been matched before, don't return
                # anything if the current match is diverging from the previous
                # match
                other_node_inds = Mul._matches_get_other_nodes(dictionary,
                                                               nodes, node_ind)
                for ind in other_node_inds:
                    other_begin, other_end = dictionary[ind]
                    curr_begin, curr_end = dictionary[node_ind]

                    other_targets = targets[other_begin:other_end + 1]
                    current_targets = targets[curr_begin:curr_end + 1]

                    for curr, other in zip(current_targets, other_targets):
                        if curr != other:
                            return None

                # A wildcard node can match more than one target, so only the
                # target index is advanced
                new_state = [(node_ind, target_ind + 1)]
                # Only move on to the next node if there is one
                if node_ind < len(nodes) - 1:
                    new_state.append((node_ind + 1, target_ind + 1))
                return new_state, match_attempt
        else:
            # If we're not at a wildcard, then make sure we haven't exhausted
            # nodes but not targets, since in this case one node can only match
            # one target
            if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
                return None

            match_attempt = node.matches(target)

            if match_attempt:
                return [(node_ind + 1, target_ind + 1)], match_attempt
            elif node == target:
                return [(node_ind + 1, target_ind + 1)], None
            else:
                return None

    @staticmethod
    def _matches_match_wilds(dictionary, wildcard_ind, nodes, targets):
        """Determine matches of a wildcard with sub-expression in `target`."""
        wildcard = nodes[wildcard_ind]
        begin, end = dictionary[wildcard_ind]
        terms = targets[begin:end + 1]
        # TODO: Should this be self.func?
        mult = Mul(*terms) if len(terms) > 1 else terms[0]
        return wildcard.matches(mult)

    @staticmethod
    def _matches_get_other_nodes(dictionary, nodes, node_ind):
        """Find other wildcards that may have already been matched."""
        ind_node = nodes[node_ind]
        return [ind for ind in dictionary if nodes[ind] == ind_node]

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs/rhs, but treats arguments like symbols, so things
        like oo/oo return 1 (instead of a nan) and ``I`` behaves like
        a symbol instead of sqrt(-1).
        """
        from sympy.simplify.simplify import signsimp
        from .symbol import Dummy
        if lhs == rhs:
            return S.One

        def check(l, r):
            if l.is_Float and r.is_comparable:
                # if both objects are added to 0 they will share the same "normalization"
                # and are more likely to compare the same. Since Add(foo, 0) will not allow
                # the 0 to pass, we use __add__ directly.
                return l.__add__(0) == r.evalf().__add__(0)
            return False
        if check(lhs, rhs) or check(rhs, lhs):
            return S.One
        if any(i.is_Pow or i.is_Mul for i in (lhs, rhs)):
            # gruntz and limit wants a literal I to not combine
            # with a power of -1
            d = Dummy('I')
            _i = {S.ImaginaryUnit: d}
            i_ = {d: S.ImaginaryUnit}
            a = lhs.xreplace(_i).as_powers_dict()
            b = rhs.xreplace(_i).as_powers_dict()
            blen = len(b)
            for bi in tuple(b.keys()):
                if bi in a:
                    a[bi] -= b.pop(bi)
                    if not a[bi]:
                        a.pop(bi)
            if len(b) != blen:
                lhs = Mul(*[k**v for k, v in a.items()]).xreplace(i_)
                rhs = Mul(*[k**v for k, v in b.items()]).xreplace(i_)
        rv = lhs/rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    def as_powers_dict(self):
        d = defaultdict(int)
        for term in self.args:
            for b, e in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        # don't use _from_args to rebuild the numerators and denominators
        # as the order is not guaranteed to be the same once they have
        # been separated from each other
        numers, denoms = list(zip(*[f.as_numer_denom() for f in self.args]))
        return self.func(*numers), self.func(*denoms)

    def as_base_exp(self):
        e1 = None
        bases = []
        nc = 0
        for m in self.args:
            b, e = m.as_base_exp()
            if not b.is_commutative:
                nc += 1
            if e1 is None:
                e1 = e
            elif e != e1 or nc > 1 or not e.is_Integer:
                return self, S.One
            bases.append(b)
        return self.func(*bases), e1

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_complex(self):
        comp = _fuzzy_group(a.is_complex for a in self.args)
        if comp is False:
            if any(a.is_infinite for a in self.args):
                if any(a.is_zero is not False for a in self.args):
                    return None
                return False
        return comp

    def _eval_is_zero_infinite_helper(self):
        #
        # Helper used by _eval_is_zero and _eval_is_infinite.
        #
        # Three-valued logic is tricky so let us reason this carefully. It
        # would be nice to say that we just check is_zero/is_infinite in all
        # args but we need to be careful about the case that one arg is zero
        # and another is infinite like Mul(0, oo) or more importantly a case
        # where it is not known if the arguments are zero or infinite like
        # Mul(y, 1/x). If either y or x could be zero then there is a
        # *possibility* that we have Mul(0, oo) which should give None for both
        # is_zero and is_infinite.
        #
        # We keep track of whether we have seen a zero or infinity but we also
        # need to keep track of whether we have *possibly* seen one which
        # would be indicated by None.
        #
        # For each argument there is the possibility that is_zero might give
        # True, False or None and likewise that is_infinite might give True,
        # False or None, giving 9 combinations. The True cases for is_zero and
        # is_infinite are mutually exclusive though so there are 3 main cases:
        #
        # - is_zero = True
        # - is_infinite = True
        # - is_zero and is_infinite are both either False or None
        #
        # At the end seen_zero and seen_infinite can be any of 9 combinations
        # of True/False/None. Unless one is False though we cannot return
        # anything except None:
        #
        # - is_zero=True needs seen_zero=True and seen_infinite=False
        # - is_zero=False needs seen_zero=False
        # - is_infinite=True needs seen_infinite=True and seen_zero=False
        # - is_infinite=False needs seen_infinite=False
        # - anything else gives both is_zero=None and is_infinite=None
        #
        # The loop only sets the flags to True or None and never back to False.
        # Hence as soon as neither flag is False we exit early returning None.
        # In particular as soon as we encounter a single arg that has
        # is_zero=is_infinite=None we exit. This is a common case since it is
        # the default assumptions for a Symbol and also the case for most
        # expressions containing such a symbol. The early exit gives a big
        # speedup for something like Mul(*symbols('x:1000')).is_zero.
        #
        seen_zero = seen_infinite = False

        for a in self.args:
            if a.is_zero:
                if seen_infinite is not False:
                    return None, None
                seen_zero = True
            elif a.is_infinite:
                if seen_zero is not False:
                    return None, None
                seen_infinite = True
            else:
                if seen_zero is False and a.is_zero is None:
                    if seen_infinite is not False:
                        return None, None
                    seen_zero = None
                if seen_infinite is False and a.is_infinite is None:
                    if seen_zero is not False:
                        return None, None
                    seen_infinite = None

        return seen_zero, seen_infinite

    def _eval_is_zero(self):
        # True iff any arg is zero and no arg is infinite but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_zero is False:
            return False
        elif seen_zero is True and seen_infinite is False:
            return True
        else:
            return None

    def _eval_is_infinite(self):
        # True iff any arg is infinite and no arg is zero but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_infinite is True and seen_zero is False:
            return True
        elif seen_infinite is False:
            return False
        else:
            return None

    # We do not need to implement _eval_is_finite because the assumptions
    # system can infer it from finite = not infinite.

    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are rational
            if all(a.is_zero is False for a in self.args):
                return False

    def _eval_is_algebraic(self):
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are algebraic
            if all(a.is_zero is False for a in self.args):
                return False

    # without involving odd/even checks this code would suffice:
    #_eval_is_integer = lambda self: _fuzzy_group(
    #    (a.is_integer for a in self.args), quick_exit=True)
    def _eval_is_integer(self):
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        unknown = False
        for a in self.args:
            hit = False
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer:
                    hit = unknown = True
                if e.is_negative:
                    denominators.append(2 if a is S.Half else
                        Pow(a, S.NegativeOne))
                elif not hit:
                    # int b and pos int e: a = b**e is integer
                    assert not e.is_positive
                    # for rational self and e equal to zero: a = b**e is 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer unknown
                else:
                    # x**2, 2**x, or x**y with x and y int-unknown -> unknown
                    return
            else:
                return

        if not denominators and not unknown:
            return True

        allodd = lambda x: all(i.is_odd for i in x)
        alleven = lambda x: all(i.is_even for i in x)
        anyeven = lambda x: any(i.is_even for i in x)

        from .relational import is_gt
        if not numerators and denominators and all(
                is_gt(_, S.One) for _ in denominators):
            return False
        elif unknown:
            return
        elif allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators
                ) and (Mul(*denominators, evaluate=False) - 1
                ).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                # if minimal power of 2 in num vs den is not
                # negative then we have an integer
                if (Add(*[i.as_base_exp()[1] for i in
                        numerators if i.is_even]) - trailing(d.p)
                        ).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                # if minimal power of 2 in den vs num is positive
                # then we have have a non-integer
                if (Add(*[i.as_base_exp()[1] for i in
                        denominators if i.is_even]) - trailing(n.p)
                        ).is_positive:
                    return False

    def _eval_is_polar(self):
        has_polar = any(arg.is_polar for arg in self.args)
        return has_polar and \
            all(arg.is_polar or arg.is_positive for arg in self.args)

    def _eval_is_extended_real(self):
        return self._eval_real_imag(True)

    def _eval_real_imag(self, real):
        zero = False
        t_not_re_im = None

        for t in self.args:
            if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
                return False
            elif t.is_imaginary:  # I
                real = not real
            elif t.is_extended_real:  # 2
                if not zero:
                    z = t.is_zero
                    if not z and zero is False:
                        zero = z
                    elif z:
                        if all(a.is_finite for a in self.args):
                            return True
                        return
            elif t.is_extended_real is False:
                # symbolic or literal like `2 + I` or symbolic imaginary
                if t_not_re_im:
                    return  # complex terms might cancel
                t_not_re_im = t
            elif t.is_imaginary is False:  # symbolic like `2` or `2 + I`
                if t_not_re_im:
                    return  # complex terms might cancel
                t_not_re_im = t
            else:
                return

        if t_not_re_im:
            if t_not_re_im.is_extended_real is False:
                if real:  # like 3
                    return zero  # 3*(smthng like 2 + I or i) is not real
            if t_not_re_im.is_imaginary is False:  # symbolic 2 or 2 + I
                if not real:  # like I
                    return zero  # I*(smthng like 2 or 2 + I) is not real
        elif zero is False:
            return real  # can't be trumped by 0
        elif real:
            return real  # doesn't matter what zero is

    def _eval_is_imaginary(self):
        if all(a.is_zero is False and a.is_finite for a in self.args):
            return self._eval_real_imag(False)

    def _eval_is_hermitian(self):
        return self._eval_herm_antiherm(True)

    def _eval_is_antihermitian(self):
        return self._eval_herm_antiherm(False)

    def _eval_herm_antiherm(self, herm):
        for t in self.args:
            if t.is_hermitian is None or t.is_antihermitian is None:
                return
            if t.is_hermitian:
                continue
            elif t.is_antihermitian:
                herm = not herm
            else:
                return

        if herm is not False:
            return herm

        is_zero = self._eval_is_zero()
        if is_zero:
            return True
        elif is_zero is False:
            return herm

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all((x.is_rational and fuzzy_not(x.is_zero)) is True for x in others):
                    return True
                return
            if a is None:
                return
        if all(x.is_real for x in self.args):
            return False

    def _eval_is_extended_positive(self):
        """Return True if self is positive, False if not, and None if it
        cannot be determined.

        Explanation
        ===========

        This algorithm is non-recursive and works by keeping track of the
        sign which changes when a negative or nonpositive is encountered.
        Whether a nonpositive or nonnegative is seen is also tracked since
        the presence of these makes it impossible to return True, but
        possible to return False if the end result is nonpositive. e.g.

            pos * neg * nonpositive -> pos or zero -> None is returned
            pos * neg * nonnegative -> neg or zero -> False is returned
        """
        return self._eval_pos_neg(1)

    def _eval_pos_neg(self, sign):
        saw_NON = saw_NOT = False
        for t in self.args:
            if t.is_extended_positive:
                continue
            elif t.is_extended_negative:
                sign = -sign
            elif t.is_zero:
                if all(a.is_finite for a in self.args):
                    return False
                return
            elif t.is_extended_nonpositive:
                sign = -sign
                saw_NON = True
            elif t.is_extended_nonnegative:
                saw_NON = True
            # FIXME: is_positive/is_negative is False doesn't take account of
            # Symbol('x', infinite=True, extended_real=True) which has
            # e.g. is_positive is False but has uncertain sign.
            elif t.is_positive is False:
                sign = -sign
                if saw_NOT:
                    return
                saw_NOT = True
            elif t.is_negative is False:
                if saw_NOT:
                    return
                saw_NOT = True
            else:
                return
        if sign == 1 and saw_NON is False and saw_NOT is False:
            return True
        if sign < 0:
            return False

    def _eval_is_extended_negative(self):
        return self._eval_pos_neg(-1)

    def _eval_is_odd(self):
        is_integer = self._eval_is_integer()
        if is_integer is not True:
            return is_integer

        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if d.is_Integer and d.is_even:
            # if minimal power of 2 in num vs den is
            # positive then we have an even number
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(n) if i.is_even]) - trailing(d.p)
                    ).is_positive:
                return False
            return
        r, acc = True, 1
        for t in self.args:
            if abs(t) is S.One:
                continue
            if t.is_even:
                return False
            if r is False:
                pass
            elif acc != 1 and (acc + t).is_odd:
                r = False
            elif t.is_even is None:
                r = None
            acc = t
        return r

    def _eval_is_even(self):
        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if n.is_Integer and n.is_even:
            # if minimal power of 2 in den vs num is not
            # negative then this is not an integer and
            # can't be even
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
                    ).is_nonnegative:
                return False

    def _eval_is_composite(self):
        """
        Here we count the number of arguments that have a minimum value
        greater than two.
        If there are more than one of such a symbol then the result is composite.
        Else, the result cannot be determined.
        """
        number_of_args = 0 # count of symbols with minimum value greater than one
        for arg in self.args:
            if not (arg.is_integer and arg.is_positive):
                return None
            if (arg-1).is_positive:
                number_of_args += 1

        if number_of_args > 1:
            return True

    def _eval_subs(self, old, new):
        from sympy.functions.elementary.complexes import sign
        from sympy.ntheory.factor_ import multiplicity
        from sympy.simplify.powsimp import powdenest
        from sympy.simplify.radsimp import fraction

        if not old.is_Mul:
            return None

        # try keep replacement literal so -2*x doesn't replace 4*x
        if old.args[0].is_Number and old.args[0] < 0:
            if self.args[0].is_Number:
                if self.args[0] < 0:
                    return self._subs(-old, -new)
                return None

        def base_exp(a):
            # if I and -1 are in a Mul, they get both end up with
            # a -1 base (see issue 6421); all we want here are the
            # true Pow or exp separated into base and exponent
            from sympy.functions.elementary.exponential import exp
            if a.is_Pow or isinstance(a, exp):
                return a.as_base_exp()
            return a, S.One

        def breakup(eq):
            """break up powers of eq when treated as a Mul:
                   b**(Rational*e) -> b**e, Rational
                commutatives come back as a dictionary {b**e: Rational}
                noncommutatives come back as a list [(b**e, Rational)]
            """

            (c, nc) = (defaultdict(int), [])
            for a in Mul.make_args(eq):
                a = powdenest(a)
                (b, e) = base_exp(a)
                if e is not S.One:
                    (co, _) = e.as_coeff_mul()
                    b = Pow(b, e/co)
                    e = co
                if a.is_commutative:
                    c[b] += e
                else:
                    nc.append([b, e])
            return (c, nc)

        def rejoin(b, co):
            """
            Put rational back with exponent; in general this is not ok, but
            since we took it from the exponent for analysis, it's ok to put
            it back.
            """

            (b, e) = base_exp(b)
            return Pow(b, e*co)

        def ndiv(a, b):
            """if b divides a in an extractive way (like 1/4 divides 1/2
            but not vice versa, and 2/5 does not divide 1/3) then return
            the integer number of times it divides, else return 0.
            """
            if not b.q % a.q or not a.q % b.q:
                return int(a/b)
            return 0

        # give Muls in the denominator a chance to be changed (see issue 5651)
        # rv will be the default return value
        rv = None
        n, d = fraction(self)
        self2 = self
        if d is not S.One:
            self2 = n._subs(old, new)/d._subs(old, new)
            if not self2.is_Mul:
                return self2._subs(old, new)
            if self2 != self:
                rv = self2

        # Now continue with regular substitution.

        # handle the leading coefficient and use it to decide if anything
        # should even be started; we always know where to find the Rational
        # so it's a quick test

        co_self = self2.args[0]
        co_old = old.args[0]
        co_xmul = None
        if co_old.is_Rational and co_self.is_Rational:
            # if coeffs are the same there will be no updating to do
            # below after breakup() step; so skip (and keep co_xmul=None)
            if co_old != co_self:
                co_xmul = co_self.extract_multiplicatively(co_old)
        elif co_old.is_Rational:
            return rv

        # break self and old into factors

        (c, nc) = breakup(self2)
        (old_c, old_nc) = breakup(old)

        # update the coefficients if we had an extraction
        # e.g. if co_self were 2*(3/35*x)**2 and co_old = 3/5
        # then co_self in c is replaced by (3/5)**2 and co_residual
        # is 2*(1/7)**2

        if co_xmul and co_xmul.is_Rational and abs(co_old) != 1:
            mult = S(multiplicity(abs(co_old), co_self))
            c.pop(co_self)
            if co_old in c:
                c[co_old] += mult
            else:
                c[co_old] = mult
            co_residual = co_self/co_old**mult
        else:
            co_residual = 1

        # do quick tests to see if we can't succeed

        ok = True
        if len(old_nc) > len(nc):
            # more non-commutative terms
            ok = False
        elif len(old_c) > len(c):
            # more commutative terms
            ok = False
        elif {i[0] for i in old_nc}.difference({i[0] for i in nc}):
            # unmatched non-commutative bases
            ok = False
        elif set(old_c).difference(set(c)):
            # unmatched commutative terms
            ok = False
        elif any(sign(c[b]) != sign(old_c[b]) for b in old_c):
            # differences in sign
            ok = False
        if not ok:
            return rv

        if not old_c:
            cdid = None
        else:
            rat = []
            for (b, old_e) in old_c.items():
                c_e = c[b]
                rat.append(ndiv(c_e, old_e))
                if not rat[-1]:
                    return rv
            cdid = min(rat)

        if not old_nc:
            ncdid = None
            for i in range(len(nc)):
                nc[i] = rejoin(*nc[i])
        else:
            ncdid = 0  # number of nc replacements we did
            take = len(old_nc)  # how much to look at each time
            limit = cdid or S.Infinity  # max number that we can take
            failed = []  # failed terms will need subs if other terms pass
            i = 0
            while limit and i + take <= len(nc):
                hit = False

                # the bases must be equivalent in succession, and
                # the powers must be extractively compatible on the
                # first and last factor but equal in between.

                rat = []
                for j in range(take):
                    if nc[i + j][0] != old_nc[j][0]:
                        break
                    elif j == 0:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif j == take - 1:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif nc[i + j][1] != old_nc[j][1]:
                        break
                    else:
                        rat.append(1)
                    j += 1
                else:
                    ndo = min(rat)
                    if ndo:
                        if take == 1:
                            if cdid:
                                ndo = min(cdid, ndo)
                            nc[i] = Pow(new, ndo)*rejoin(nc[i][0],
                                    nc[i][1] - ndo*old_nc[0][1])
                        else:
                            ndo = 1

                            # the left residual

                            l = rejoin(nc[i][0], nc[i][1] - ndo*
                                    old_nc[0][1])

                            # eliminate all middle terms

                            mid = new

                            # the right residual (which may be the same as the middle if take == 2)

                            ir = i + take - 1
                            r = (nc[ir][0], nc[ir][1] - ndo*
                                 old_nc[-1][1])
                            if r[1]:
                                if i + take < len(nc):
                                    nc[i:i + take] = [l*mid, r]
                                else:
                                    r = rejoin(*r)
                                    nc[i:i + take] = [l*mid*r]
                            else:

                                # there was nothing left on the right

                                nc[i:i + take] = [l*mid]

                        limit -= ndo
                        ncdid += ndo
                        hit = True
                if not hit:

                    # do the subs on this failing factor

                    failed.append(i)
                i += 1
            else:

                if not ncdid:
                    return rv

                # although we didn't fail, certain nc terms may have
                # failed so we rebuild them after attempting a partial
                # subs on them

                failed.extend(range(i, len(nc)))
                for i in failed:
                    nc[i] = rejoin(*nc[i]).subs(old, new)

        # rebuild the expression

        if cdid is None:
            do = ncdid
        elif ncdid is None:
            do = cdid
        else:
            do = min(ncdid, cdid)

        margs = []
        for b in c:
            if b in old_c:

                # calculate the new exponent

                e = c[b] - old_c[b]*do
                margs.append(rejoin(b, e))
            else:
                margs.append(rejoin(b.subs(old, new), c[b]))
        if cdid and not ncdid:

            # in case we are replacing commutative with non-commutative,
            # we want the new term to come at the front just like the
            # rest of this routine

            margs = [Pow(new, cdid)] + margs
        return co_residual*self2.func(*margs)*self2.func(*nc)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from .function import PoleError
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.order import Order

        def coeff_exp(term, x):
            lt = term.as_coeff_exponent(x)
            if lt[0].has(x):
                try:
                    lt = term.leadterm(x)
                except ValueError:
                    return term, S.Zero
            return lt

        ords = []

        try:
            for t in self.args:
                coeff, exp = t.leadterm(x)
                if not coeff.has(x):
                    ords.append((t, exp))
                else:
                    raise ValueError

            n0 = sum(t[1] for t in ords if t[1].is_number)
            facs = []
            for t, m in ords:
                n1 = ceiling(n - n0 + (m if m.is_number else 0))
                s = t.nseries(x, n=n1, logx=logx, cdir=cdir)
                ns = s.getn()
                if ns is not None:
                    if ns < n1:  # less than expected
                        n -= n1 - ns    # reduce n
                facs.append(s)

        except (ValueError, NotImplementedError, TypeError, PoleError):
            # XXX: Catching so many generic exceptions around a large block of
            # code will mask bugs. Whatever purpose catching these exceptions
            # serves should be handled in a different way.
            n0 = sympify(sum(t[1] for t in ords if t[1].is_number))
            if n0.is_nonnegative:
                n0 = S.Zero
            facs = [t.nseries(x, n=ceiling(n-n0), logx=logx, cdir=cdir) for t in self.args]
            from sympy.simplify.powsimp import powsimp
            res = powsimp(self.func(*facs).expand(), combine='exp', deep=True)
            if res.has(Order):
                res += Order(x**n, x)
            return res

        res = S.Zero
        ords2 = [Add.make_args(factor) for factor in facs]

        for fac in product(*ords2):
            ords3 = [coeff_exp(term, x) for term in fac]
            coeffs, powers = zip(*ords3)
            power = sum(powers)
            if (power - n).is_negative:
                res += Mul(*coeffs)*(x**power)

        def max_degree(e, x):
            if e is x:
                return S.One
            if e.is_Atom:
                return S.Zero
            if e.is_Add:
                return max(max_degree(a, x) for a in e.args)
            if e.is_Mul:
                return Add(*[max_degree(a, x) for a in e.args])
            if e.is_Pow:
                return max_degree(e.base, x)*e.exp
            return S.Zero

        if self.is_polynomial(x):
            from sympy.polys.polyerrors import PolynomialError
            from sympy.polys.polytools import degree
            try:
                if max_degree(self, x) >= n or degree(self, x) != degree(res, x):
                    res += Order(x**n, x)
            except PolynomialError:
                pass
            else:
                return res

        if res != self:
            if (self - res).subs(x, 0) == S.Zero and n > 0:
                lt = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
                if lt == S.Zero:
                    return res
            res += Order(x**n, x)
        return res

    def _eval_as_leading_term(self, x, logx, cdir):
        return self.func(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args[::-1]])

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (-3*sqrt(2)*(2 - 2*sqrt(2))).as_content_primitive()
        (6, -sqrt(2)*(1 - sqrt(2)))

        See docstring of Expr.as_content_primitive for more examples.
        """

        coef = S.One
        args = []
        for a in self.args:
            c, p = a.as_content_primitive(radical=radical, clear=clear)
            coef *= c
            if p is not S.One:
                args.append(p)
        # don't use self._from_args here to reconstruct args
        # since there may be identical args now that should be combined
        # e.g. (2+2*x)*(3+3*x) should be (6, (1 + x)**2) not (6, (1+x)*(1+x))
        return coef, self.func(*args)

    def as_ordered_factors(self, order=None):
        """Transform an expression into an ordered list of factors.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x, y

        >>> (2*x*y*sin(x)*cos(x)).as_ordered_factors()
        [2, x, y, sin(x), cos(x)]

        """
        cpart, ncpart = self.args_cnc()
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        return cpart + ncpart

    @property
    def _sorted_args(self):
        return tuple(self.as_ordered_factors())

mul = AssocOpDispatcher('mul')


def prod(a, start=1):
    """Return product of elements of a. Start with int 1 so if only
       ints are included then an int result is returned.

    Examples
    ========

    >>> from sympy import prod, S
    >>> prod(range(3))
    0
    >>> type(_) is int
    True
    >>> prod([S(2), 3])
    6
    >>> _.is_Integer
    True

    You can start the product at something other than 1:

    >>> prod([1, 2], 3)
    6

    """
    return reduce(operator.mul, a, start)


def _keep_coeff(coeff, factors, clear=True, sign=False):
    """Return ``coeff*factors`` unevaluated if necessary.

    If ``clear`` is False, do not keep the coefficient as a factor
    if it can be distributed on a single factor such that one or
    more terms will still have integer coefficients.

    If ``sign`` is True, allow a coefficient of -1 to remain factored out.

    Examples
    ========

    >>> from sympy.core.mul import _keep_coeff
    >>> from sympy.abc import x, y
    >>> from sympy import S

    >>> _keep_coeff(S.Half, x + 2)
    (x + 2)/2
    >>> _keep_coeff(S.Half, x + 2, clear=False)
    x/2 + 1
    >>> _keep_coeff(S.Half, (x + 2)*y, clear=False)
    y*(x + 2)/2
    >>> _keep_coeff(S(-1), x + y)
    -x - y
    >>> _keep_coeff(S(-1), x + y, sign=True)
    -(x + y)
    """
    if not coeff.is_Number:
        if factors.is_Number:
            factors, coeff = coeff, factors
        else:
            return coeff*factors
    if factors is S.One:
        return coeff
    if coeff is S.One:
        return factors
    elif coeff is S.NegativeOne and not sign:
        return -factors
    elif factors.is_Add:
        if not clear and coeff.is_Rational and coeff.q != 1:
            args = [i.as_coeff_Mul() for i in factors.args]
            args = [(_keep_coeff(c, coeff), m) for c, m in args]
            if any(c.is_Integer for c, _ in args):
                return Add._from_args([Mul._from_args(
                    i[1:] if i[0] == 1 else i) for i in args])
        return Mul(coeff, factors, evaluate=False)
    elif factors.is_Mul:
        margs = list(factors.args)
        if margs[0].is_Number:
            margs[0] *= coeff
            if margs[0] == 1:
                margs.pop(0)
        else:
            margs.insert(0, coeff)
        return Mul._from_args(margs)
    else:
        m = coeff*factors
        if m.is_Number and not factors.is_Number:
            m = Mul._from_args((coeff, factors))
        return m

def expand_2arg(e):
    def do(e):
        if e.is_Mul:
            c, r = e.as_coeff_Mul()
            if c.is_Number and r.is_Add:
                return _unevaluated_Add(*[c*ri for ri in r.args])
        return e
    return bottom_up(e, do)


from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
