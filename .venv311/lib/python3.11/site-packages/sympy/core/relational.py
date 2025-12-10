from __future__ import annotations

from .basic import Atom, Basic
from .coreerrors import LazyExceptionMessage
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .numbers import int_valued
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning


__all__ = (
    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',
    'Relational', 'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
    'StrictGreaterThan', 'GreaterThan',
)

from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol


def _nontrivBool(side):
    return isinstance(side, Boolean) and \
           not isinstance(side, Atom)


# Note, see issue 4986.  Ideally, we wouldn't want to subclass both Boolean
# and Expr.
# from .. import Expr


def _canonical(cond):
    # return a condition in which all relationals are canonical
    reps = {r: r.canonical for r in cond.atoms(Relational)}
    return cond.xreplace(reps)
    # XXX: AttributeError was being caught here but it wasn't triggered by any of
    # the tests so I've removed it...


def _canonical_coeff(rel):
    # return -2*x + 1 < 0 as x > 1/2
    # XXX make this part of Relational.canonical?
    rel = rel.canonical
    if not rel.is_Relational or rel.rhs.is_Boolean:
        return rel  # Eq(x, True)
    if not isinstance(rel.lhs, Expr):
        return rel.reversed  # e.g.: Eq(True, x) -> Eq(x, True)
    b, l = rel.lhs.as_coeff_Add(rational=True)
    m, lhs = l.as_coeff_Mul(rational=True)
    rhs = (rel.rhs - b)/m
    if m < 0:
        return rel.reversed.func(lhs, rhs)
    return rel.func(lhs, rhs)


class Relational(Boolean, EvalfMixin):
    """Base class for all relation types.

    Explanation
    ===========

    Subclasses of Relational should generally be instantiated directly, but
    Relational can be instantiated with a valid ``rop`` value to dispatch to
    the appropriate subclass.

    Parameters
    ==========

    rop : str or None
        Indicates what subclass to instantiate.  Valid values can be found
        in the keys of Relational.ValidRelationOperator.

    Examples
    ========

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
    Eq(y, x**2 + x)

    A relation's type can be defined upon creation using ``rop``.
    The relation type of an existing expression can be obtained
    using its ``rel_op`` property.
    Here is a table of all the relation types, along with their
    ``rop`` and ``rel_op`` values:

    +---------------------+----------------------------+------------+
    |Relation             |``rop``                     |``rel_op``  |
    +=====================+============================+============+
    |``Equality``         |``==`` or ``eq`` or ``None``|``==``      |
    +---------------------+----------------------------+------------+
    |``Unequality``       |``!=`` or ``ne``            |``!=``      |
    +---------------------+----------------------------+------------+
    |``GreaterThan``      |``>=`` or ``ge``            |``>=``      |
    +---------------------+----------------------------+------------+
    |``LessThan``         |``<=`` or ``le``            |``<=``      |
    +---------------------+----------------------------+------------+
    |``StrictGreaterThan``|``>`` or ``gt``             |``>``       |
    +---------------------+----------------------------+------------+
    |``StrictLessThan``   |``<`` or ``lt``             |``<``       |
    +---------------------+----------------------------+------------+

    For example, setting ``rop`` to ``==`` produces an
    ``Equality`` relation, ``Eq()``.
    So does setting ``rop`` to ``eq``, or leaving ``rop`` unspecified.
    That is, the first three ``Rel()`` below all produce the same result.
    Using a ``rop`` from a different row in the table produces a
    different relation type.
    For example, the fourth ``Rel()`` below using ``lt`` for ``rop``
    produces a ``StrictLessThan`` inequality:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'eq')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2)
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'lt')
        y < x**2 + x

    To obtain the relation type of an existing expression,
    get its ``rel_op`` property.
    For example, ``rel_op`` is ``==`` for the ``Equality`` relation above,
    and ``<`` for the strict less than inequality above:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> my_equality = Rel(y, x + x**2, '==')
    >>> my_equality.rel_op
        '=='
    >>> my_inequality = Rel(y, x + x**2, 'lt')
    >>> my_inequality.rel_op
        '<'

    """
    __slots__ = ()

    ValidRelationOperator: dict[str | None, type[Relational]] = {}

    is_Relational = True

    # ValidRelationOperator - Defined below, because the necessary classes
    #   have not yet been defined

    def __new__(cls, lhs, rhs, rop=None, **assumptions):
        # If called by a subclass, do nothing special and pass on to Basic.
        if cls is not Relational:
            return Basic.__new__(cls, lhs, rhs, **assumptions)

        # XXX: Why do this? There should be a separate function to make a
        # particular subclass of Relational from a string.
        #
        # If called directly with an operator, look up the subclass
        # corresponding to that operator and delegate to it
        cls = cls.ValidRelationOperator.get(rop, None)
        if cls is None:
            raise ValueError("Invalid relational operator symbol: %r" % rop)

        if not issubclass(cls, (Eq, Ne)):
            # validate that Booleans are not being used in a relational
            # other than Eq/Ne;
            # Note: Symbol is a subclass of Boolean but is considered
            # acceptable here.
            if any(map(_nontrivBool, (lhs, rhs))):
                raise TypeError(filldedent('''
                    A Boolean argument can only be used in
                    Eq and Ne; all other relationals expect
                    real expressions.
                '''))

        return cls(lhs, rhs, **assumptions)

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self._args[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self._args[1]

    @property
    def reversed(self):
        """Return the relationship with sides reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversed
        Eq(1, x)
        >>> x < 1
        x < 1
        >>> _.reversed
        1 > x
        """
        ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
        a, b = self.args
        return Relational.__new__(ops.get(self.func, self.func), b, a)

    @property
    def reversedsign(self):
        """Return the relationship with signs reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversedsign
        Eq(-x, -1)
        >>> x < 1
        x < 1
        >>> _.reversedsign
        -x > -1
        """
        a, b = self.args
        if not (isinstance(a, BooleanAtom) or isinstance(b, BooleanAtom)):
            ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
            return Relational.__new__(ops.get(self.func, self.func), -a, -b)
        else:
            return self

    @property
    def negated(self):
        """Return the negated relationship.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.negated
        Ne(x, 1)
        >>> x < 1
        x < 1
        >>> _.negated
        x >= 1

        Notes
        =====

        This works more or less identical to ``~``/``Not``. The difference is
        that ``negated`` returns the relationship even if ``evaluate=False``.
        Hence, this is useful in code when checking for e.g. negated relations
        to existing ones as it will not be affected by the `evaluate` flag.

        """
        ops = {Eq: Ne, Ge: Lt, Gt: Le, Le: Gt, Lt: Ge, Ne: Eq}
        # If there ever will be new Relational subclasses, the following line
        # will work until it is properly sorted out
        # return ops.get(self.func, lambda a, b, evaluate=False: ~(self.func(a,
        #      b, evaluate=evaluate)))(*self.args, evaluate=False)
        return Relational.__new__(ops.get(self.func), *self.args)

    @property
    def weak(self):
        """return the non-strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x < 1).weak
        x <= 1
        >>> _.weak
        x <= 1
        """
        return self

    @property
    def strict(self):
        """return the strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x <= 1).strict
        x < 1
        >>> _.strict
        x < 1
        """
        return self

    def _eval_evalf(self, prec):
        return self.func(*[s._evalf(prec) for s in self.args])

    @property
    def canonical(self):
        """Return a canonical form of the relational by putting a
        number on the rhs, canonically removing a sign or else
        ordering the args canonically. No other simplification is
        attempted.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x < 2
        x < 2
        >>> _.reversed.canonical
        x < 2
        >>> (-y < x).canonical
        x > -y
        >>> (-y > x).canonical
        x < -y
        >>> (-y < -x).canonical
        x < y

        The canonicalization is recursively applied:

        >>> from sympy import Eq
        >>> Eq(x < y, y > x).canonical
        True
        """
        args = tuple([i.canonical if isinstance(i, Relational) else i for i in self.args])
        if args != self.args:
            r = self.func(*args)
            if not isinstance(r, Relational):
                return r
        else:
            r = self
        if r.rhs.is_number:
            if r.rhs.is_Number and r.lhs.is_Number and r.lhs > r.rhs:
                r = r.reversed
        elif r.lhs.is_number:
            r = r.reversed
        elif tuple(ordered(args)) != args:
            r = r.reversed

        LHS_CEMS = getattr(r.lhs, 'could_extract_minus_sign', None)
        RHS_CEMS = getattr(r.rhs, 'could_extract_minus_sign', None)

        if isinstance(r.lhs, BooleanAtom) or isinstance(r.rhs, BooleanAtom):
            return r

        # Check if first value has negative sign
        if LHS_CEMS and LHS_CEMS():
            return r.reversedsign
        elif not r.rhs.is_number and RHS_CEMS and RHS_CEMS():
            # Right hand side has a minus, but not lhs.
            # How does the expression with reversed signs behave?
            # This is so that expressions of the type
            # Eq(x, -y) and Eq(-x, y)
            # have the same canonical representation
            expr1, _ = ordered([r.lhs, -r.rhs])
            if expr1 != r.lhs:
                return r.reversed.reversedsign

        return r

    def equals(self, other, failing_expression=False):
        """Return True if the sides of the relationship are mathematically
        identical and the type of relationship is the same.
        If failing_expression is True, return the expression whose truth value
        was unknown."""
        if isinstance(other, Relational):
            if other in (self, self.reversed):
                return True
            a, b = self, other
            if a.func in (Eq, Ne) or b.func in (Eq, Ne):
                if a.func != b.func:
                    return False
                left, right = [i.equals(j,
                                        failing_expression=failing_expression)
                               for i, j in zip(a.args, b.args)]
                if left is True:
                    return right
                if right is True:
                    return left
                lr, rl = [i.equals(j, failing_expression=failing_expression)
                          for i, j in zip(a.args, b.reversed.args)]
                if lr is True:
                    return rl
                if rl is True:
                    return lr
                e = (left, right, lr, rl)
                if all(i is False for i in e):
                    return False
                for i in e:
                    if i not in (True, False):
                        return i
            else:
                if b.func != a.func:
                    b = b.reversed
                if a.func != b.func:
                    return False
                left = a.lhs.equals(b.lhs,
                                    failing_expression=failing_expression)
                if left is False:
                    return False
                right = a.rhs.equals(b.rhs,
                                     failing_expression=failing_expression)
                if right is False:
                    return False
                if left is True:
                    return right
                return left

    def _eval_simplify(self, **kwargs):
        from .add import Add
        from .expr import Expr
        r = self
        r = r.func(*[i.simplify(**kwargs) for i in r.args])
        if r.is_Relational:
            if not isinstance(r.lhs, Expr) or not isinstance(r.rhs, Expr):
                return r
            dif = r.lhs - r.rhs
            # replace dif with a valid Number that will
            # allow a definitive comparison with 0
            v = None
            if dif.is_comparable:
                v = dif.n(2)
                if any(i._prec == 1 for i in v.as_real_imag()):
                    rv, iv = [i.n(2) for i in dif.as_real_imag()]
                    v = rv + S.ImaginaryUnit*iv
            elif dif.equals(0):  # XXX this is expensive
                v = S.Zero
            if v is not None:
                r = r.func._eval_relation(v, S.Zero)
            r = r.canonical
            # If there is only one symbol in the expression,
            # try to write it on a simplified form
            free = list(filter(lambda x: x.is_real is not False, r.free_symbols))
            if len(free) == 1:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    x = free.pop()
                    dif = r.lhs - r.rhs
                    m, b = linear_coeffs(dif, x)
                    if m.is_zero is False:
                        if m.is_negative:
                            # Dividing with a negative number, so change order of arguments
                            # canonical will put the symbol back on the lhs later
                            r = r.func(-b / m, x)
                        else:
                            r = r.func(x, -b / m)
                    else:
                        r = r.func(b, S.Zero)
                except ValueError:
                    # maybe not a linear function, try polynomial
                    from sympy.polys.polyerrors import PolynomialError
                    from sympy.polys.polytools import gcd, Poly, poly
                    try:
                        p = poly(dif, x)
                        c = p.all_coeffs()
                        constant = c[-1]
                        c[-1] = 0
                        scale = gcd(c)
                        c = [ctmp / scale for ctmp in c]
                        r = r.func(Poly.from_list(c, x).as_expr(), -constant / scale)
                    except PolynomialError:
                        pass
            elif len(free) >= 2:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    from sympy.polys.polytools import gcd
                    free = list(ordered(free))
                    dif = r.lhs - r.rhs
                    m = linear_coeffs(dif, *free)
                    constant = m[-1]
                    del m[-1]
                    scale = gcd(m)
                    m = [mtmp / scale for mtmp in m]
                    nzm = list(filter(lambda f: f[0] != 0, list(zip(m, free))))
                    if scale.is_zero is False:
                        if constant != 0:
                            # lhs: expression, rhs: constant
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(newexpr, -constant / scale)
                        else:
                            # keep first term on lhs
                            lhsterm = nzm[0][0] * nzm[0][1]
                            del nzm[0]
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(lhsterm, -newexpr)

                    else:
                        r = r.func(constant, S.Zero)
                except ValueError:
                    pass
        # Did we get a simplified result?
        r = r.canonical
        measure = kwargs['measure']
        if measure(r) < kwargs['ratio'] * measure(self):
            return r
        else:
            return self

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.func(trigsimp(self.lhs, **opts), trigsimp(self.rhs, **opts))

    def expand(self, **kwargs):
        args = (arg.expand(**kwargs) for arg in self.args)
        return self.func(*args)

    def __bool__(self) -> bool:
        raise TypeError(
            LazyExceptionMessage(
                lambda: f"cannot determine truth value of Relational: {self}"
            )
        )

    def _eval_as_set(self):
        # self is univariate and periodicity(self, x) in (0, None)
        from sympy.solvers.inequalities import solve_univariate_inequality
        from sympy.sets.conditionset import ConditionSet
        syms = self.free_symbols
        assert len(syms) == 1
        x = syms.pop()
        try:
            xset = solve_univariate_inequality(self, x, relational=False)
        except NotImplementedError:
            # solve_univariate_inequality raises NotImplementedError for
            # unsolvable equations/inequalities.
            xset = ConditionSet(x, self, S.Reals)
        return xset

    @property
    def binary_symbols(self):
        # override where necessary
        return set()


Rel = Relational


class Equality(Relational):
    """
    An equal relation between two objects.

    Explanation
    ===========

    Represents that two objects are equal.  If they can be easily shown
    to be definitively equal (or unequal), this will reduce to True (or
    False).  Otherwise, the relation is maintained as an unevaluated
    Equality object.  Use the ``simplify`` function on this object for
    more nontrivial evaluation of the equality relation.

    As usual, the keyword argument ``evaluate=False`` can be used to
    prevent any evaluation.

    Examples
    ========

    >>> from sympy import Eq, simplify, exp, cos
    >>> from sympy.abc import x, y
    >>> Eq(y, x + x**2)
    Eq(y, x**2 + x)
    >>> Eq(2, 5)
    False
    >>> Eq(2, 5, evaluate=False)
    Eq(2, 5)
    >>> _.doit()
    False
    >>> Eq(exp(x), exp(x).rewrite(cos))
    Eq(exp(x), sinh(x) + cosh(x))
    >>> simplify(_)
    True

    See Also
    ========

    sympy.logic.boolalg.Equivalent : for representing equality between two
        boolean expressions

    Notes
    =====

    Python treats 1 and True (and 0 and False) as being equal; SymPy
    does not. And integer will always compare as unequal to a Boolean:

    >>> Eq(True, 1), True == 1
    (False, True)

    This class is not the same as the == operator.  The == operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    If either object defines an ``_eval_Eq`` method, it can be used in place of
    the default algorithm.  If ``lhs._eval_Eq(rhs)`` or ``rhs._eval_Eq(lhs)``
    returns anything other than None, that return value will be substituted for
    the Equality.  If None is returned by ``_eval_Eq``, an Equality object will
    be created as usual.

    Since this object is already an expression, it does not respond to
    the method ``as_expr`` if one tries to create `x - y` from ``Eq(x, y)``.
    If ``eq = Eq(x, y)`` then write `eq.lhs - eq.rhs` to get ``x - y``.

    .. deprecated:: 1.5

       ``Eq(expr)`` with a single argument is a shorthand for ``Eq(expr, 0)``,
       but this behavior is deprecated and will be removed in a future version
       of SymPy.

    """
    rel_op = '=='

    __slots__ = ()

    is_Equality = True

    def __new__(cls, lhs, rhs, **options):
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        if evaluate:
            val = is_eq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs == rhs)

    def _eval_rewrite_as_Add(self, L, R, evaluate=True, **kwargs):
        """
        return Eq(L, R) as L - R. To control the evaluation of
        the result set pass `evaluate=True` to give L - R;
        if `evaluate=None` then terms in L and R will not cancel
        but they will be listed in canonical order; otherwise
        non-canonical args will be returned. If one side is 0, the
        non-zero side will be returned.

        .. deprecated:: 1.13

           The method ``Eq.rewrite(Add)`` is deprecated.
           See :ref:`eq-rewrite-Add` for details.

        Examples
        ========

        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)  #doctest: +SKIP
        2*b
        >>> eq.rewrite(Add, evaluate=None).args  #doctest: +SKIP
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args  #doctest: +SKIP
        (b, x, b, -x)
        """
        sympy_deprecation_warning("""
        Eq.rewrite(Add) is deprecated.

        For ``eq = Eq(a, b)`` use ``eq.lhs - eq.rhs`` to obtain
        ``a - b``.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="eq-rewrite-Add",
            stacklevel=5,
        )
        from .add import _unevaluated_Add, Add
        if L == 0:
            return R
        if R == 0:
            return L
        if evaluate:
            # allow cancellation of args
            return L - R
        args = Add.make_args(L) + Add.make_args(-R)
        if evaluate is None:
            # no cancellation, but canonical
            return _unevaluated_Add(*args)
        # no cancellation, not canonical
        return Add._from_args(args)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # standard simplify
        e = super()._eval_simplify(**kwargs)
        if not isinstance(e, Equality):
            return e
        from .expr import Expr
        if not isinstance(e.lhs, Expr) or not isinstance(e.rhs, Expr):
            return e
        free = self.free_symbols
        if len(free) == 1:
            try:
                from .add import Add
                from sympy.solvers.solveset import linear_coeffs
                x = free.pop()
                m, b = linear_coeffs(
                    Add(e.lhs, -e.rhs, evaluate=False), x)
                if m.is_zero is False:
                    enew = e.func(x, -b / m)
                else:
                    enew = e.func(m * x, -b)
                measure = kwargs['measure']
                if measure(enew) <= kwargs['ratio'] * measure(e):
                    e = enew
            except ValueError:
                pass
        return e.canonical

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def as_poly(self, *gens, **kwargs):
        '''Returns lhs-rhs as a Poly

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x**2, 1).as_poly(x)
        Poly(x**2 - 1, x, domain='ZZ')
        '''
        return (self.lhs - self.rhs).as_poly(*gens, **kwargs)


Eq = Equality


class Unequality(Relational):
    """An unequal relation between two objects.

    Explanation
    ===========

    Represents that two objects are not equal.  If they can be shown to be
    definitively equal, this will reduce to False; if definitively unequal,
    this will reduce to True.  Otherwise, the relation is maintained as an
    Unequality object.

    Examples
    ========

    >>> from sympy import Ne
    >>> from sympy.abc import x, y
    >>> Ne(y, x+x**2)
    Ne(y, x**2 + x)

    See Also
    ========
    Equality

    Notes
    =====
    This class is not the same as the != operator.  The != operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    This class is effectively the inverse of Equality.  As such, it uses the
    same algorithms, including any available `_eval_Eq` methods.

    """
    rel_op = '!='

    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            val = is_neq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs != rhs)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # simplify as an equality
        eq = Equality(*self.args)._eval_simplify(**kwargs)
        if isinstance(eq, Equality):
            # send back Ne with the new args
            return self.func(*eq.args)
        return eq.negated  # result of Ne is the negated Eq


Ne = Unequality


class _Inequality(Relational):
    """Internal base class for all *Than types.

    Each subclass must implement _eval_relation to provide the method for
    comparing two real numbers.

    """
    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):

        try:
            lhs = _sympify(lhs)
            rhs = _sympify(rhs)
        except SympifyError:
            return NotImplemented

        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            for me in (lhs, rhs):
                if me.is_extended_real is False:
                    raise TypeError("Invalid comparison of non-real %s" % me)
                if me is S.NaN:
                    raise TypeError("Invalid NaN comparison")
            # First we invoke the appropriate inequality method of `lhs`
            # (e.g., `lhs.__lt__`).  That method will try to reduce to
            # boolean or raise an exception.  It may keep calling
            # superclasses until it reaches `Expr` (e.g., `Expr.__lt__`).
            # In some cases, `Expr` will just invoke us again (if neither it
            # nor a subclass was able to reduce to boolean or raise an
            # exception).  In that case, it must call us with
            # `evaluate=False` to prevent infinite recursion.
            return cls._eval_relation(lhs, rhs, **options)

        # make a "non-evaluated" Expr for the inequality
        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs, **options):
        val = cls._eval_fuzzy_relation(lhs, rhs)
        if val is None:
            return cls(lhs, rhs, evaluate=False)
        else:
            return _sympify(val)


class _Greater(_Inequality):
    """Not intended for general use

    _Greater is only used so that GreaterThan and StrictGreaterThan may
    subclass it for the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[0]

    @property
    def lts(self):
        return self._args[1]


class _Less(_Inequality):
    """Not intended for general use.

    _Less is only used so that LessThan and StrictLessThan may subclass it for
    the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[1]

    @property
    def lts(self):
        return self._args[0]


class GreaterThan(_Greater):
    r"""Class representations of inequalities.

    Explanation
    ===========

    The ``*Than`` classes represent inequal relationships, where the left-hand
    side is generally bigger or smaller than the right-hand side.  For example,
    the GreaterThan class represents an inequal relationship where the
    left-hand side is at least as big as the right side, if not bigger.  In
    mathematical notation:

    lhs $\ge$ rhs

    In total, there are four ``*Than`` classes, to represent the four
    inequalities:

    +-----------------+--------+
    |Class Name       | Symbol |
    +=================+========+
    |GreaterThan      | ``>=`` |
    +-----------------+--------+
    |LessThan         | ``<=`` |
    +-----------------+--------+
    |StrictGreaterThan| ``>``  |
    +-----------------+--------+
    |StrictLessThan   | ``<``  |
    +-----------------+--------+

    All classes take two arguments, lhs and rhs.

    +----------------------------+-----------------+
    |Signature Example           | Math Equivalent |
    +============================+=================+
    |GreaterThan(lhs, rhs)       |   lhs $\ge$ rhs |
    +----------------------------+-----------------+
    |LessThan(lhs, rhs)          |   lhs $\le$ rhs |
    +----------------------------+-----------------+
    |StrictGreaterThan(lhs, rhs) |   lhs $>$ rhs   |
    +----------------------------+-----------------+
    |StrictLessThan(lhs, rhs)    |   lhs $<$ rhs   |
    +----------------------------+-----------------+

    In addition to the normal .lhs and .rhs of Relations, ``*Than`` inequality
    objects also have the .lts and .gts properties, which represent the "less
    than side" and "greater than side" of the operator.  Use of .lts and .gts
    in an algorithm rather than .lhs and .rhs as an assumption of inequality
    direction will make more explicit the intent of a certain section of code,
    and will make it similarly more robust to client code changes:

    >>> from sympy import GreaterThan, StrictGreaterThan
    >>> from sympy import LessThan, StrictLessThan
    >>> from sympy import And, Ge, Gt, Le, Lt, Rel, S
    >>> from sympy.abc import x, y, z
    >>> from sympy.core.relational import Relational

    >>> e = GreaterThan(x, 1)
    >>> e
    x >= 1
    >>> '%s >= %s is the same as %s <= %s' % (e.gts, e.lts, e.lts, e.gts)
    'x >= 1 is the same as 1 <= x'

    Examples
    ========

    One generally does not instantiate these classes directly, but uses various
    convenience methods:

    >>> for f in [Ge, Gt, Le, Lt]:  # convenience wrappers
    ...     print(f(x, 2))
    x >= 2
    x > 2
    x <= 2
    x < 2

    Another option is to use the Python inequality operators (``>=``, ``>``,
    ``<=``, ``<``) directly.  Their main advantage over the ``Ge``, ``Gt``,
    ``Le``, and ``Lt`` counterparts, is that one can write a more
    "mathematical looking" statement rather than littering the math with
    oddball function calls.  However there are certain (minor) caveats of
    which to be aware (search for 'gotcha', below).

    >>> x >= 2
    x >= 2
    >>> _ == Ge(x, 2)
    True

    However, it is also perfectly valid to instantiate a ``*Than`` class less
    succinctly and less conveniently:

    >>> Rel(x, 1, ">")
    x > 1
    >>> Relational(x, 1, ">")
    x > 1

    >>> StrictGreaterThan(x, 1)
    x > 1
    >>> GreaterThan(x, 1)
    x >= 1
    >>> LessThan(x, 1)
    x <= 1
    >>> StrictLessThan(x, 1)
    x < 1

    Notes
    =====

    There are a couple of "gotchas" to be aware of when using Python's
    operators.

    The first is that what your write is not always what you get:

        >>> 1 < x
        x > 1

        Due to the order that Python parses a statement, it may
        not immediately find two objects comparable.  When ``1 < x``
        is evaluated, Python recognizes that the number 1 is a native
        number and that x is *not*.  Because a native Python number does
        not know how to compare itself with a SymPy object
        Python will try the reflective operation, ``x > 1`` and that is the
        form that gets evaluated, hence returned.

        If the order of the statement is important (for visual output to
        the console, perhaps), one can work around this annoyance in a
        couple ways:

        (1) "sympify" the literal before comparison

        >>> S(1) < x
        1 < x

        (2) use one of the wrappers or less succinct methods described
        above

        >>> Lt(1, x)
        1 < x
        >>> Relational(1, x, "<")
        1 < x

    The second gotcha involves writing equality tests between relationals
    when one or both sides of the test involve a literal relational:

        >>> e = x < 1; e
        x < 1
        >>> e == e  # neither side is a literal
        True
        >>> e == x < 1  # expecting True, too
        False
        >>> e != x < 1  # expecting False
        x < 1
        >>> x < 1 != x < 1  # expecting False or the same thing as before
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

        The solution for this case is to wrap literal relationals in
        parentheses:

        >>> e == (x < 1)
        True
        >>> e != (x < 1)
        False
        >>> (x < 1) != (x < 1)
        False

    The third gotcha involves chained inequalities not involving
    ``==`` or ``!=``. Occasionally, one may be tempted to write:

        >>> e = x < y < z
        Traceback (most recent call last):
        ...
        TypeError: symbolic boolean expression has no truth value.

        Due to an implementation detail or decision of Python [1]_,
        there is no way for SymPy to create a chained inequality with
        that syntax so one must use And:

        >>> e = And(x < y, y < z)
        >>> type( e )
        And
        >>> e
        (x < y) & (y < z)

        Although this can also be done with the '&' operator, it cannot
        be done with the 'and' operarator:

        >>> (x < y) & (y < z)
        (x < y) & (y < z)
        >>> (x < y) and (y < z)
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

    .. [1] This implementation detail is that Python provides no reliable
       method to determine that a chained inequality is being built.
       Chained comparison operators are evaluated pairwise, using "and"
       logic (see
       https://docs.python.org/3/reference/expressions.html#not-in). This
       is done in an efficient way, so that each object being compared
       is only evaluated once and the comparison can short-circuit. For
       example, ``1 > 2 > 3`` is evaluated by Python as ``(1 > 2) and (2
       > 3)``. The ``and`` operator coerces each side into a bool,
       returning the object itself when it short-circuits. The bool of
       the --Than operators will raise TypeError on purpose, because
       SymPy cannot determine the mathematical ordering of symbolic
       expressions. Thus, if we were to compute ``x > y > z``, with
       ``x``, ``y``, and ``z`` being Symbols, Python converts the
       statement (roughly) into these steps:

        (1) x > y > z
        (2) (x > y) and (y > z)
        (3) (GreaterThanObject) and (y > z)
        (4) (GreaterThanObject.__bool__()) and (y > z)
        (5) TypeError

       Because of the ``and`` added at step 2, the statement gets turned into a
       weak ternary statement, and the first object's ``__bool__`` method will
       raise TypeError.  Thus, creating a chained inequality is not possible.

           In Python, there is no way to override the ``and`` operator, or to
           control how it short circuits, so it is impossible to make something
           like ``x > y > z`` work.  There was a PEP to change this,
           :pep:`335`, but it was officially closed in March, 2012.

    """
    __slots__ = ()

    rel_op = '>='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_ge(lhs, rhs)

    @property
    def strict(self):
        return Gt(*self.args)

Ge = GreaterThan


class LessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_le(lhs, rhs)

    @property
    def strict(self):
        return Lt(*self.args)

Le = LessThan


class StrictGreaterThan(_Greater):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '>'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_gt(lhs, rhs)

    @property
    def weak(self):
        return Ge(*self.args)


Gt = StrictGreaterThan


class StrictLessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_lt(lhs, rhs)

    @property
    def weak(self):
        return Le(*self.args)

Lt = StrictLessThan

# A class-specific (not object-specific) data item used for a minor speedup.
# It is defined here, rather than directly in the class, because the classes
# that it references have not been defined until now (e.g. StrictLessThan).
Relational.ValidRelationOperator = {
    None: Equality,
    '==': Equality,
    'eq': Equality,
    '!=': Unequality,
    '<>': Unequality,
    'ne': Unequality,
    '>=': GreaterThan,
    'ge': GreaterThan,
    '<=': LessThan,
    'le': LessThan,
    '>': StrictGreaterThan,
    'gt': StrictGreaterThan,
    '<': StrictLessThan,
    'lt': StrictLessThan,
}


def _n2(a, b):
    """Return (a - b).evalf(2) if a and b are comparable, else None.
    This should only be used when a and b are already sympified.
    """
    # /!\ it is very important (see issue 8245) not to
    # use a re-evaluated number in the calculation of dif
    if a.is_comparable and b.is_comparable:
        dif = (a - b).evalf(2)
        if dif.is_comparable:
            return dif


@dispatch(Expr, Expr)
def _eval_is_ge(lhs, rhs):
    return None


@dispatch(Basic, Basic)
def _eval_is_eq(lhs, rhs):
    return None


@dispatch(Tuple, Expr) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return False


@dispatch(Tuple, AppliedUndef) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Symbol) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Tuple) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    if len(lhs) != len(rhs):
        return False

    return fuzzy_and(fuzzy_bool(is_eq(s, o)) for s, o in zip(lhs, rhs))


def is_lt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly less than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_ge(lhs, rhs, assumptions))


def is_gt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly greater than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_le(lhs, rhs, assumptions))


def is_le(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is less than or equal to rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return is_ge(rhs, lhs, assumptions)


def is_ge(lhs, rhs, assumptions=None):
    """
    Fuzzy bool for *lhs* is greater than or equal to *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified,
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    rhs : Expr
        The right-hand side of the expression, must be sympified
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the inequality.

    Returns
    =======

    ``True`` if *lhs* is greater than or equal to *rhs*, ``False`` if *lhs*
    is less than *rhs*, and ``None`` if the comparison between *lhs* and
    *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    The four comparison functions ``is_le``, ``is_lt``, ``is_ge``, and ``is_gt`` are
    each implemented in terms of ``is_ge`` in the following way:

    is_ge(x, y) := is_ge(x, y)
    is_le(x, y) := is_ge(y, x)
    is_lt(x, y) := fuzzy_not(is_ge(x, y))
    is_gt(x, y) := fuzzy_not(is_ge(y, x))

    Therefore, supporting new type with this function will ensure behavior for
    other three functions as well.

    To maintain these equivalences in fuzzy logic it is important that in cases where
    either x or y is non-real all comparisons will give None.

    Examples
    ========

    >>> from sympy import S, Q
    >>> from sympy.core.relational import is_ge, is_le, is_gt, is_lt
    >>> from sympy.abc import x
    >>> is_ge(S(2), S(0))
    True
    >>> is_ge(S(0), S(2))
    False
    >>> is_le(S(0), S(2))
    True
    >>> is_gt(S(0), S(2))
    False
    >>> is_lt(S(2), S(0))
    False

    Assumptions can be passed to evaluate the quality which is otherwise
    indeterminate.

    >>> print(is_ge(x, S(0)))
    None
    >>> is_ge(x, S(0), assumptions=Q.positive(x))
    True

    New types can be supported by dispatching to ``_eval_is_ge``.

    >>> from sympy import Expr, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyExpr(Expr):
    ...     def __new__(cls, arg):
    ...         return super().__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    >>> @dispatch(MyExpr, MyExpr)
    ... def _eval_is_ge(a, b):
    ...     return is_ge(a.value, b.value)
    >>> a = MyExpr(1)
    >>> b = MyExpr(2)
    >>> is_ge(b, a)
    True
    >>> is_le(a, b)
    True
    """
    from sympy.assumptions.wrapper import AssumptionsWrapper, is_extended_nonnegative

    if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
        raise TypeError("Can only compare inequalities with Expr")

    retval = _eval_is_ge(lhs, rhs)

    if retval is not None:
        return retval
    else:
        n2 = _n2(lhs, rhs)
        if n2 is not None:
            # use float comparison for infinity.
            # otherwise get stuck in infinite recursion
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            return n2 >= 0

        _lhs = AssumptionsWrapper(lhs, assumptions)
        _rhs = AssumptionsWrapper(rhs, assumptions)
        if _lhs.is_extended_real and _rhs.is_extended_real:
            if (_lhs.is_infinite and _lhs.is_extended_positive) or (_rhs.is_infinite and _rhs.is_extended_negative):
                return True
            diff = lhs - rhs
            if diff is not S.NaN:
                rv = is_extended_nonnegative(diff, assumptions)
                if rv is not None:
                    return rv


def is_neq(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs does not equal rhs.

    See the docstring for :func:`~.is_eq` for more.
    """
    return fuzzy_not(is_eq(lhs, rhs, assumptions))


def is_eq(lhs, rhs, assumptions=None):
    """
    Fuzzy bool representing mathematical equality between *lhs* and *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified.

    rhs : Expr
        The right-hand side of the expression, must be sympified.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the equality.

    Returns
    =======

    ``True`` if *lhs* is equal to *rhs*, ``False`` is *lhs* is not equal to *rhs*,
    and ``None`` if the comparison between *lhs* and *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    :func:`~.is_neq` calls this function to return its value, so supporting
    new type with this function will ensure correct behavior for ``is_neq``
    as well.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.core.relational import is_eq, is_neq
    >>> from sympy.abc import x
    >>> is_eq(S(0), S(0))
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> is_neq(S(0), S(2))
    True

    Assumptions can be passed to evaluate the equality which is otherwise
    indeterminate.

    >>> print(is_eq(x, S(0)))
    None
    >>> is_eq(x, S(0), assumptions=Q.zero(x))
    True

    New types can be supported by dispatching to ``_eval_is_eq``.

    >>> from sympy import Basic, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> is_eq(a, b)
    True
    >>> is_neq(a, b)
    False

    """
    # here, _eval_Eq is only called for backwards compatibility
    # new code should use is_eq with multiple dispatch as
    # outlined in the docstring
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval

    # retval is still None, so go through the equality logic
    # If expressions have the same structure, they must be equal.
    if lhs == rhs:
        return True  # e.g. True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # only Booleans can equal Booleans

    from sympy.assumptions.wrapper import (AssumptionsWrapper,
        is_infinite, is_extended_real)
    from .add import Add

    _lhs = AssumptionsWrapper(lhs, assumptions)
    _rhs = AssumptionsWrapper(rhs, assumptions)

    if _lhs.is_infinite or _rhs.is_infinite:
        if fuzzy_xor([_lhs.is_infinite, _rhs.is_infinite]):
            return False
        if fuzzy_xor([_lhs.is_extended_real, _rhs.is_extended_real]):
            return False
        if fuzzy_and([_lhs.is_extended_real, _rhs.is_extended_real]):
            return fuzzy_xor([_lhs.is_extended_positive, fuzzy_not(_rhs.is_extended_positive)])

        # Try to split real/imaginary parts and equate them
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            real_imag = lambda t: (
                'real' if is_extended_real(t, assumptions) else
                'imag' if is_extended_real(I*t, assumptions) else None)
            return sift(Add.make_args(expr), real_imag)

        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                eq_real = is_eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']), assumptions)
                eq_imag = is_eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']), assumptions)
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        from sympy.functions.elementary.complexes import arg
        # Compare e.g. zoo with 1+I*oo by comparing args
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # Guard against Eq(nan, nan) -> False
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(is_eq(arglhs, argrhs, assumptions))

    if all(isinstance(i, Expr) for i in (lhs, rhs)):
        # see if the difference evaluates
        dif = lhs - rhs
        _dif = AssumptionsWrapper(dif, assumptions)
        z = _dif.is_zero
        if z is not None:
            if z is False and _dif.is_commutative:  # issue 10728
                return False
            if z:
                return True

        # is_zero cannot help decide integer/rational with Float
        c, t = dif.as_coeff_Add()
        if c.is_Float:
            if int_valued(c):
                if t.is_integer is False:
                    return False
            elif t.is_rational is False:
                return False

        n2 = _n2(lhs, rhs)
        if n2 is not None:
            return _sympify(n2 == 0)

        # see if the ratio evaluates
        n, d = dif.as_numer_denom()
        rv = None
        _n = AssumptionsWrapper(n, assumptions)
        _d = AssumptionsWrapper(d, assumptions)
        if _n.is_zero:
            rv = _d.is_nonzero
        elif _n.is_finite:
            if _d.is_infinite:
                rv = True
            elif _n.is_zero is False:
                rv = _d.is_infinite
                if rv is None:
                    # if the condition that makes the denominator
                    # infinite does not make the original expression
                    # True then False can be returned
                    from sympy.simplify.simplify import clear_coefficients
                    l, r = clear_coefficients(d, S.Infinity)
                    args = [_.subs(l, r) for _ in (lhs, rhs)]
                    if args != [lhs, rhs]:
                        rv = fuzzy_bool(is_eq(*args, assumptions))
                        if rv is True:
                            rv = None
        elif any(is_infinite(a, assumptions) for a in Add.make_args(n)):
            # (inf or nan)/x != 0
            rv = False
        if rv is not None:
            return rv
