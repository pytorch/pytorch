"""Base class for all the objects in SymPy"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Iterable
from itertools import zip_longest
from functools import cmp_to_key
from typing import TYPE_CHECKING, overload

from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable

from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name


if TYPE_CHECKING:
    from typing import ClassVar, TypeVar, Any
    from typing_extensions import Self
    from .assumptions import StdFactKB
    from .symbol import Symbol

    Tbasic = TypeVar("Tbasic", bound='Basic')


def as_Basic(expr):
    """Return expr as a Basic instance using strict sympify
    or raise a TypeError; this is just a wrapper to _sympify,
    raising a TypeError instead of a SympifyError."""
    try:
        return _sympify(expr)
    except SympifyError:
        raise TypeError(
            'Argument must be a Basic object, not `%s`' % func_name(
            expr))


# Key for sorting commutative args in canonical order
# by name. This is used for canonical ordering of the
# args for Add and Mul *if* the names of both classes
# being compared appear here. Some things in this list
# are not spelled the same as their name so they do not,
# in effect, appear here. See Basic.compare.
ordering_of_classes = [
    # singleton numbers
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity',
    # numbers
    'Integer', 'Rational', 'Float',
    # singleton symbols
    'Exp1', 'Pi', 'ImaginaryUnit',
    # symbols
    'Symbol', 'Wild',
    # arithmetic operations
    'Pow', 'Mul', 'Add',
    # function values
    'Derivative', 'Integral',
    # defined singleton functions
    'Abs', 'Sign', 'Sqrt',
    'Floor', 'Ceiling',
    'Re', 'Im', 'Arg',
    'Conjugate',
    'Exp', 'Log',
    'Sin', 'Cos', 'Tan', 'Cot', 'ASin', 'ACos', 'ATan', 'ACot',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'ASinh', 'ACosh', 'ATanh', 'ACoth',
    'RisingFactorial', 'FallingFactorial',
    'factorial', 'binomial',
    'Gamma', 'LowerGamma', 'UpperGamma', 'PolyGamma',
    'Erf',
    # special polynomials
    'Chebyshev', 'Chebyshev2',
    # undefined functions
    'Function', 'WildFunction',
    # anonymous functions
    'Lambda',
    # Landau O symbol
    'Order',
    # relational operations
    'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan',
]

def _cmp_name(x: type, y: type) -> int:
    """return -1, 0, 1 if the name of x is before that of y.
    A string comparison is done if either name does not appear
    in `ordering_of_classes`. This is the helper for
    ``Basic.compare``

    Examples
    ========

    >>> from sympy import cos, tan, sin
    >>> from sympy.core import basic
    >>> save = basic.ordering_of_classes
    >>> basic.ordering_of_classes = ()
    >>> basic._cmp_name(cos, tan)
    -1
    >>> basic.ordering_of_classes = ["tan", "sin", "cos"]
    >>> basic._cmp_name(cos, tan)
    1
    >>> basic._cmp_name(sin, cos)
    -1
    >>> basic.ordering_of_classes = save

    """
    n1 = x.__name__
    n2 = y.__name__
    if n1 == n2:
        return 0

    # If the other object is not a Basic subclass, then we are not equal to it.
    if not issubclass(y, Basic):
        return -1

    UNKNOWN = len(ordering_of_classes) + 1
    try:
        i1 = ordering_of_classes.index(n1)
    except ValueError:
        i1 = UNKNOWN
    try:
        i2 = ordering_of_classes.index(n2)
    except ValueError:
        i2 = UNKNOWN
    if i1 == UNKNOWN and i2 == UNKNOWN:
        return (n1 > n2) - (n1 < n2)
    return (i1 > i2) - (i1 < i2)



@cacheit
def _get_postprocessors(clsname, arg_type):
    # Since only Add, Mul, Pow can be clsname, this cache
    # is not quadratic.
    postprocessors = set()
    mappings = _get_postprocessors_for_type(arg_type)
    for mapping in mappings:
        f = mapping.get(clsname, None)
        if f is not None:
            postprocessors.update(f)
    return postprocessors

@cacheit
def _get_postprocessors_for_type(arg_type):
    return tuple(
        Basic._constructor_postprocessor_mapping[cls]
        for cls in arg_type.mro()
        if cls in Basic._constructor_postprocessor_mapping
    )


class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """
    __slots__ = ('_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'
                )

    _args: tuple[Basic, ...]
    _mhash: int | None

    @property
    def __sympy__(self):
        return True

    def __init_subclass__(cls):
        # Initialize the default_assumptions FactKB and also any assumptions
        # property methods. This method will only be called for subclasses of
        # Basic but not for Basic itself so we call
        # _prepare_class_assumptions(Basic) below the class definition.
        super().__init_subclass__()
        _prepare_class_assumptions(cls)

    # To be overridden with True in the appropriate subclasses
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False

    default_assumptions: ClassVar[StdFactKB]

    is_composite: bool | None
    is_noninteger: bool | None
    is_extended_positive: bool | None
    is_negative: bool | None
    is_complex: bool | None
    is_extended_nonpositive: bool | None
    is_integer: bool | None
    is_positive: bool | None
    is_rational: bool | None
    is_extended_nonnegative: bool | None
    is_infinite: bool | None
    is_antihermitian: bool | None
    is_extended_negative: bool | None
    is_extended_real: bool | None
    is_finite: bool | None
    is_polar: bool | None
    is_imaginary: bool | None
    is_transcendental: bool | None
    is_extended_nonzero: bool | None
    is_nonzero: bool | None
    is_odd: bool | None
    is_algebraic: bool | None
    is_prime: bool | None
    is_commutative: bool | None
    is_nonnegative: bool | None
    is_nonpositive: bool | None
    is_hermitian: bool | None
    is_irrational: bool | None
    is_real: bool | None
    is_zero: bool | None
    is_even: bool | None

    kind: Kind = UndefinedKind

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._assumptions = cls.default_assumptions
        obj._mhash = None  # will be set by __hash__ method.

        obj._args = args  # all items in args must be Basic objects
        return obj

    def copy(self):
        return self.func(*self.args)

    def __getnewargs__(self):
        return self.args

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            msg = "Only pickle protocol 2 or higher is supported by SymPy"
            raise NotImplementedError(msg)
        return super().__reduce_ex__(protocol)

    def __hash__(self) -> int:
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    @property
    def assumptions0(self):
        """
        Return object `type` assumptions.

        For example:

          Symbol('x', real=True)
          Symbol('x', integer=True)

        are different objects. In other words, besides Python type (Symbol in
        this case), the initial assumptions are also forming their typeinfo.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.abc import x
        >>> x.assumptions0
        {'commutative': True}
        >>> x = Symbol("x", positive=True)
        >>> x.assumptions0
        {'commutative': True, 'complex': True, 'extended_negative': False,
         'extended_nonnegative': True, 'extended_nonpositive': False,
         'extended_nonzero': True, 'extended_positive': True, 'extended_real':
         True, 'finite': True, 'hermitian': True, 'imaginary': False,
         'infinite': False, 'negative': False, 'nonnegative': True,
         'nonpositive': False, 'nonzero': True, 'positive': True, 'real':
         True, 'zero': False}
        """
        return {}

    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        # all redefinitions of __cmp__ method should start with the
        # following lines:
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        c = _cmp_name(n1, n2)
        if c:
            return c
        #
        st = self._hashable_content()
        ot = other._hashable_content()
        len_st = len(st)
        len_ot = len(ot)
        c = (len_st > len_ot) - (len_st < len_ot)
        if c:
            return c
        for l, r in zip(st, ot):
            if isinstance(l, Basic):
                c = l.compare(r)
            elif isinstance(l, frozenset):
                l = Basic(*l) if isinstance(l, frozenset) else l
                r = Basic(*r) if isinstance(r, frozenset) else r
                c = l.compare(r)
            else:
                c = (l > r) - (l < r)
            if c:
                return c
        return 0

    @classmethod
    def fromiter(cls, args, **assumptions):
        """
        Create a new object from an iterable.

        This is a convenience function that allows one to create objects from
        any iterable, without having to convert to a list or tuple first.

        Examples
        ========

        >>> from sympy import Tuple
        >>> Tuple.fromiter(i for i in range(5))
        (0, 1, 2, 3, 4)

        """
        return cls(*tuple(args), **assumptions)

    @classmethod
    def class_key(cls) -> tuple[int, int, str]:
        """Nice order of classes."""
        return 5, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        """
        Return a sort key.

        Examples
        ========

        >>> from sympy import S, I

        >>> sorted([S(1)/2, I, -I], key=lambda x: x.sort_key())
        [1/2, -I, I]

        >>> S("[x, 1/x, 1/x**2, x**2, x**(1/2), x**(1/4), x**(3/2)]")
        [x, 1/x, x**(-2), x**2, sqrt(x), x**(1/4), x**(3/2)]
        >>> sorted(_, key=lambda x: x.sort_key())
        [x**(-2), 1/x, x**(1/4), sqrt(x), x, x**(3/2), x**2]

        """

        # XXX: remove this when issue 5169 is fixed
        def inner_key(arg):
            if isinstance(arg, Basic):
                return arg.sort_key(order)
            else:
                return arg

        args = self._sorted_args
        args = len(args), tuple([inner_key(arg) for arg in args])
        return self.class_key(), args, S.One.sort_key(), S.One

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True

        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)

        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True

    def __ne__(self, other):
        """``a != b``  -> Compare two symbolic trees and see whether they are different

        this is the same as:

        ``a.compare(b) != 0``

        but faster
        """
        return not self == other

    def dummy_eq(self, other, symbol=None):
        """
        Compare two expressions and handle dummy symbols.

        Examples
        ========

        >>> from sympy import Dummy
        >>> from sympy.abc import x, y

        >>> u = Dummy('u')

        >>> (u**2 + 1).dummy_eq(x**2 + 1)
        True
        >>> (u**2 + 1) == (x**2 + 1)
        False

        >>> (u**2 + y).dummy_eq(x**2 + y, x)
        True
        >>> (u**2 + y).dummy_eq(x**2 + y, y)
        False

        """
        s = self.as_dummy()
        o = _sympify(other)
        o = o.as_dummy()

        dummy_symbols = [i for i in s.free_symbols if i.is_Dummy]

        if len(dummy_symbols) == 1:
            dummy = dummy_symbols.pop()
        else:
            return s == o

        if symbol is None:
            symbols = o.free_symbols

            if len(symbols) == 1:
                symbol = symbols.pop()
            else:
                return s == o

        tmp = dummy.__class__()

        return s.xreplace({dummy: tmp}) == o.xreplace({symbol: tmp})

    @overload
    def atoms(self) -> set[Basic]: ...
    @overload
    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Tbasic]: ...

    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Basic] | set[Tbasic]:
        """Returns the atoms that form the current object.

        By default, only objects that are truly atomic and cannot
        be divided into smaller pieces are returned: symbols, numbers,
        and number symbols like I and pi. It is possible to request
        atoms of any type, however, as demonstrated below.

        Examples
        ========

        >>> from sympy import I, pi, sin
        >>> from sympy.abc import x, y
        >>> (1 + x + 2*sin(y + I*pi)).atoms()
        {1, 2, I, pi, x, y}

        If one or more types are given, the results will contain only
        those types of atoms.

        >>> from sympy import Number, NumberSymbol, Symbol
        >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
        {x, y}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
        {1, 2}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
        {1, 2, pi}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
        {1, 2, I, pi}

        Note that I (imaginary unit) and zoo (complex infinity) are special
        types of number symbols and are not part of the NumberSymbol class.

        The type can be given implicitly, too:

        >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
        {x, y}

        Be careful to check your assumptions when using the implicit option
        since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
        of SymPy atom, while ``type(S(2))`` is type ``Integer`` and will find all
        integers in an expression:

        >>> from sympy import S
        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
        {1}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
        {1, 2}

        Finally, arguments to atoms() can select more than atomic atoms: any
        SymPy type (loaded in core/__init__.py) can be listed as an argument
        and those types of "atoms" as found in scanning the arguments of the
        expression recursively:

        >>> from sympy import Function, Mul
        >>> from sympy.core.function import AppliedUndef
        >>> f = Function('f')
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
        {f(x), sin(y + I*pi)}
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
        {f(x)}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
        {I*pi, 2*sin(y + I*pi)}

        """
        nodes = _preorder_traversal(self)
        if types:
            types2 = tuple([t if isinstance(t, type) else type(t) for t in types])
            return {node for node in nodes if isinstance(node, types2)}
        else:
            return {node for node in nodes if not node.args}

    @property
    def free_symbols(self) -> set[Basic]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol`` (see examples)

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method.

        Examples
        ========

        >>> from sympy import Derivative, Integral, IndexedBase
        >>> from sympy.abc import x, y, n
        >>> (x + 1).free_symbols
        {x}
        >>> Integral(x, y).free_symbols
        {x, y}

        Not all free symbols are actually symbols:

        >>> IndexedBase('F')[0].free_symbols
        {F, F[0]}

        The symbols of differentiation are not included unless they
        appear in the expression being differentiated.

        >>> Derivative(x + y, y).free_symbols
        {x, y}
        >>> Derivative(x, y).free_symbols
        {x}
        >>> Derivative(x, (y, n)).free_symbols
        {n, x}

        If you want to know if a symbol is in the variables of the
        Derivative you can do so as follows:

        >>> Derivative(x, y).has_free(y)
        True
        """
        empty: set[Basic] = set()
        return empty.union(*(a.free_symbols for a in self.args))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return set()

    def as_dummy(self) -> "Self":
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True. When applied to a
        symbol a new symbol having only the same commutativity will be
        returned.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True
        >>> r.as_dummy()
        _r

        Notes
        =====

        Any object that has structurally bound variables should have
        a property, ``bound_symbols`` that returns those symbols
        appearing in the object.
        """
        from .symbol import Dummy, Symbol
        def can(x):
            # mask free that shadow bound
            free = x.free_symbols
            bound = set(x.bound_symbols)
            d = {i: Dummy() for i in bound & free}
            x = x.subs(d)
            # replace bound with canonical names
            x = x.xreplace(x.canonical_variables)
            # return after undoing masking
            return x.xreplace({v: k for k, v in d.items()})
        if not self.has(Symbol):
            return self
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            can,
            simultaneous=False) # type:ignore

    @property
    def canonical_variables(self) -> dict[Basic, Symbol]:
        """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
        bound: list[Basic] | None = getattr(self, 'bound_symbols', None)
        if bound is None:
            return {}
        dums = numbered_symbols('_')
        reps = {}
        # watch out for free symbol that are not in bound symbols;
        # those that are in bound symbols are about to get changed

        # XXX: free_symbols only returns particular kinds of expressions that
        # generally have a .name attribute. There is not a proper class/type
        # that represents this.
        names = {i.name for i in self.free_symbols - set(bound)} # type: ignore
        for b in bound:
            d = next(dums)
            if b.is_Symbol:
                while d.name in names:
                    d = next(dums)
            reps[b] = d
        return reps

    def rcall(self, *args):
        """Apply on the argument recursively through the expression tree.

        This method is used to simulate a common abuse of notation for
        operators. For instance, in SymPy the following will not work:

        ``(x+Lambda(y, 2*y))(z) == x+2*z``,

        however, you can use:

        >>> from sympy import Lambda
        >>> from sympy.abc import x, y, z
        >>> (x + Lambda(y, 2*y)).rcall(z)
        x + 2*z
        """
        if callable(self):
            return self(*args)
        elif self.args:
            newargs = [sub.rcall(*args) for sub in self.args]
            return self.func(*newargs)
        else:
            return self

    def is_hypergeometric(self, k):
        from sympy.simplify.simplify import hypersimp
        from sympy.functions.elementary.piecewise import Piecewise
        if self.has(Piecewise):
            return None
        return hypersimp(self, k) is not None

    @property
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        return self._eval_is_comparable()

    def _eval_is_comparable(self) -> bool:
        # Expr.is_comparable overrides this
        return False

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args

    @property
    def _sorted_args(self):
        """
        The same as ``args``.  Derived classes which do not fix an
        order on their arguments should override this method to
        produce the sorted representation.
        """
        return self.args

    def as_content_primitive(self, radical=False, clear=True):
        """A stub to allow Basic args (like Tuple) to be skipped when computing
        the content and primitive components of an expression.

        See Also
        ========

        sympy.core.expr.Expr.as_content_primitive
        """
        return S.One, self

    @overload
    def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

    def subs(self, arg1: Mapping[Basic | complex, Basic | complex]
            | Iterable[tuple[Basic | complex, Basic | complex]] | Basic | complex,
             arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
        """
        Substitutes old for new in an expression after sympifying args.

        `args` is either:
          - two arguments, e.g. foo.subs(old, new)
          - one iterable argument, e.g. foo.subs(iterable). The iterable may be
             o an iterable container with (old, new) pairs. In this case the
               replacements are processed in the order given with successive
               patterns possibly affecting replacements already made.
             o a dict or set whose key/value items correspond to old/new pairs.
               In this case the old/new pairs will be sorted by op count and in
               case of a tie, by number of args and the default_sort_key. The
               resulting sorted list is then processed as an iterable container
               (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Examples
        ========

        >>> from sympy import pi, exp, limit, oo
        >>> from sympy.abc import x, y
        >>> (1 + x*y).subs(x, pi)
        pi*y + 1
        >>> (1 + x*y).subs({x:pi, y:2})
        1 + 2*pi
        >>> (1 + x*y).subs([(x, pi), (y, 2)])
        1 + 2*pi
        >>> reps = [(y, x**2), (x, 2)]
        >>> (x + y).subs(reps)
        6
        >>> (x + y).subs(reversed(reps))
        x**2 + 2

        >>> (x**2 + x**4).subs(x**2, y)
        y**2 + y

        To replace only the x**2 but not the x**4, use xreplace:

        >>> (x**2 + x**4).xreplace({x**2: y})
        x**4 + y

        To delay evaluation until all substitutions have been made,
        set the keyword ``simultaneous`` to True:

        >>> (x/y).subs([(x, 0), (y, 0)])
        0
        >>> (x/y).subs([(x, 0), (y, 0)], simultaneous=True)
        nan

        This has the added feature of not allowing subsequent substitutions
        to affect those already made:

        >>> ((x + y)/y).subs({x + y: y, y: x + y})
        1
        >>> ((x + y)/y).subs({x + y: y, y: x + y}, simultaneous=True)
        y/(x + y)

        In order to obtain a canonical result, unordered iterables are
        sorted by count_op length, number of arguments and by the
        default_sort_key to break any ties. All other iterables are left
        unsorted.

        >>> from sympy import sqrt, sin, cos
        >>> from sympy.abc import a, b, c, d, e

        >>> A = (sqrt(sin(2*x)), a)
        >>> B = (sin(2*x), b)
        >>> C = (cos(2*x), c)
        >>> D = (x, d)
        >>> E = (exp(x), e)

        >>> expr = sqrt(sin(2*x))*sin(exp(x)*x)*cos(2*x) + sin(2*x)

        >>> expr.subs(dict([A, B, C, D, E]))
        a*c*sin(d*e) + b

        The resulting expression represents a literal replacement of the
        old arguments with the new arguments. This may not reflect the
        limiting behavior of the expression:

        >>> (x**3 - 3*x).subs({x: oo})
        nan

        >>> limit(x**3 - 3*x, x, oo)
        oo

        If the substitution will be followed by numerical
        evaluation, it is better to pass the substitution to
        evalf as

        >>> (1/x).evalf(subs={x: 3.0}, n=21)
        0.333333333333333333333

        rather than

        >>> (1/x).subs({x: 3.0}).evalf(21)
        0.333333333333333314830

        as the former will ensure that the desired level of precision is
        obtained.

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules
        sympy.core.evalf.EvalfMixin.evalf: calculates the given formula to a desired level of precision

        """
        from .containers import Dict
        from .symbol import Dummy, Symbol
        from .numbers import _illegal

        items: Iterable[tuple[Basic | complex, Basic | complex]]

        unordered = False
        if arg2 is None:

            if isinstance(arg1, set):
                items = arg1
                unordered = True
            elif isinstance(arg1, (Dict, Mapping)):
                unordered = True
                items = arg1.items() # type: ignore
            elif not iterable(arg1):
                raise ValueError(filldedent("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
            else:
                items = arg1 # type: ignore
        else:
            items = [(arg1, arg2)] # type: ignore

        def sympify_old(old) -> Basic:
            if isinstance(old, str):
                # Use Symbol rather than parse_expr for old
                return Symbol(old)
            elif isinstance(old, type):
                # Allow a type e.g. Function('f') or sin
                return sympify(old, strict=False)
            else:
                return sympify(old, strict=True)

        def sympify_new(new) -> Basic:
            if isinstance(new, (str, type)):
                # Allow a type or parse a string input
                return sympify(new, strict=False)
            else:
                return sympify(new, strict=True)

        sequence = [(sympify_old(s1), sympify_new(s2)) for s1, s2 in items]

        # skip if there is no change
        sequence = [(s1, s2) for s1, s2 in sequence if not _aresame(s1, s2)]

        simultaneous = kwargs.pop('simultaneous', False)

        if unordered:
            from .sorting import _nodes, default_sort_key
            sequence_dict = dict(sequence)
            # order so more complex items are first and items
            # of identical complexity are ordered so
            # f(x) < f(y) < x < y
            # \___ 2 __/    \_1_/  <- number of nodes
            #
            # For more complex ordering use an unordered sequence.
            k = list(ordered(sequence_dict, default=False, keys=(
                lambda x: -_nodes(x),
                default_sort_key,
                )))
            sequence = [(k, sequence_dict[k]) for k in k]
            # do infinities first
            if not simultaneous:
                redo = [i for i, seq in enumerate(sequence) if seq[1] in _illegal]
                for i in reversed(redo):
                    sequence.insert(0, sequence.pop(i))

        if simultaneous:  # XXX should this be the default for dict subs?
            reps = {}
            rv = self
            kwargs['hack2'] = True
            m = Dummy('subs_m')
            for old, new in sequence:
                com = new.is_commutative
                if com is None:
                    com = True
                d = Dummy('subs_d', commutative=com)
                # using d*m so Subs will be used on dummy variables
                # in things like Derivative(f(x, y), x) in which x
                # is both free and bound
                rv = rv._subs(old, d*m, **kwargs)
                if not isinstance(rv, Basic):
                    break
                reps[d] = new
            reps[m] = S.One  # get rid of m
            return rv.xreplace(reps)
        else:
            rv = self
            for old, new in sequence:
                rv = rv._subs(old, new, **kwargs)
                if not isinstance(rv, Basic):
                    break
            return rv

    @cacheit
    def _subs(self, old, new, **hints):
        """Substitutes an expression old -> new.

        If self is not equal to old then _eval_subs is called.
        If _eval_subs does not want to make any special replacement
        then a None is received which indicates that the fallback
        should be applied wherein a search for replacements is made
        amongst the arguments of self.

        >>> from sympy import Add
        >>> from sympy.abc import x, y, z

        Examples
        ========

        Add's _eval_subs knows how to target x + y in the following
        so it makes the change:

        >>> (x + y + z).subs(x + y, 1)
        z + 1

        Add's _eval_subs does not need to know how to find x + y in
        the following:

        >>> Add._eval_subs(z*(x + y) + 3, x + y, 1) is None
        True

        The returned None will cause the fallback routine to traverse the args and
        pass the z*(x + y) arg to Mul where the change will take place and the
        substitution will succeed:

        >>> (z*(x + y) + 3).subs(x + y, 1)
        z + 3

        ** Developers Notes **

        An _eval_subs routine for a class should be written if:

            1) any arguments are not instances of Basic (e.g. bool, tuple);

            2) some arguments should not be targeted (as in integration
               variables);

            3) if there is something other than a literal replacement
               that should be attempted (as in Piecewise where the condition
               may be updated without doing a replacement).

        If it is overridden, here are some special cases that might arise:

            1) If it turns out that no special change was made and all
               the original sub-arguments should be checked for
               replacements then None should be returned.

            2) If it is necessary to do substitutions on a portion of
               the expression then _subs should be called. _subs will
               handle the case of any sub-expression being equal to old
               (which usually would not be the case) while its fallback
               will handle the recursion into the sub-arguments. For
               example, after Add's _eval_subs removes some matching terms
               it must process the remaining terms so it calls _subs
               on each of the un-matched terms and then adds them
               onto the terms previously obtained.

           3) If the initial expression should remain unchanged then
              the original expression should be returned. (Whenever an
              expression is returned, modified or not, no further
              substitution of old -> new is attempted.) Sum's _eval_subs
              routine uses this strategy when a substitution is attempted
              on any of its summation variables.
        """

        def fallback(self, old, new):
            """
            Try to replace old with new in any of self's arguments.
            """
            hit = False
            args = list(self.args)
            for i, arg in enumerate(args):
                if not hasattr(arg, '_eval_subs'):
                    continue
                arg = arg._subs(old, new, **hints)
                if not _aresame(arg, args[i]):
                    hit = True
                    args[i] = arg
            if hit:
                rv = self.func(*args)
                hack2 = hints.get('hack2', False)
                if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
                    coeff = S.One
                    nonnumber = []
                    for i in args:
                        if i.is_Number:
                            coeff *= i
                        else:
                            nonnumber.append(i)
                    nonnumber = self.func(*nonnumber)
                    if coeff is S.One:
                        return nonnumber
                    else:
                        return self.func(coeff, nonnumber, evaluate=False)
                return rv
            return self

        if _aresame(self, old):
            return new

        rv = self._eval_subs(old, new)
        if rv is None:
            rv = fallback(self, old, new)
        return rv

    def _eval_subs(self, old, new) -> Basic | None:
        """Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        return None

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the expression.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule

        Returns
        =======

        xreplace : the result of the replacement

        Examples
        ========

        >>> from sympy import symbols, pi, exp
        >>> x, y, z = symbols('x y z')
        >>> (1 + x*y).xreplace({x: pi})
        pi*y + 1
        >>> (1 + x*y).xreplace({x: pi, y: 2})
        1 + 2*pi

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> (x*y + z).xreplace({x*y: pi})
        z + pi
        >>> (x*y*z).xreplace({x*y: pi})
        x*y*z
        >>> (2*x).xreplace({2*x: y, x: z})
        y
        >>> (2*2*x).xreplace({2*x: y, x: z})
        4*z
        >>> (x + y + 2).xreplace({x + y: 2})
        x + y + 2
        >>> (x + 2 + exp(x + 2)).xreplace({x + 2: y})
        x + exp(y) + 2

        xreplace does not differentiate between free and bound symbols. In the
        following, subs(x, y) would not change x since it is a bound symbol,
        but xreplace does:

        >>> from sympy import Integral
        >>> Integral(x, (x, 1, 2*x)).xreplace({x: y})
        Integral(y, (y, 1, 2*y))

        Trying to replace x with an expression raises an error:

        >>> Integral(x, (x, 1, 2*x)).xreplace({x: 2*y}) # doctest: +SKIP
        ValueError: Invalid limits given: ((2*y, 1, 4*y),)

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        subs: substitution of subexpressions as defined by the objects
              themselves.

        """
        value, _ = self._xreplace(rule)
        return value

    def _xreplace(self, rule):
        """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
        if self in rule:
            return rule[self], True
        elif rule:
            args = []
            changed = False
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            if changed:
                return self.func(*args), True
        return self, False

    @cacheit
    def has(self, *patterns):
        """
        Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y, z
        >>> (x**2 + sin(x*y)).has(z)
        False
        >>> (x**2 + sin(x*y)).has(x, y, z)
        True
        >>> x.has(x)
        True

        Note ``has`` is a structural algorithm with no knowledge of
        mathematics. Consider the following half-open interval:

        >>> from sympy import Interval
        >>> i = Interval.Lopen(0, 5); i
        Interval.Lopen(0, 5)
        >>> i.args
        (0, 5, True, False)
        >>> i.has(4)  # there is no "4" in the arguments
        False
        >>> i.has(0)  # there *is* a "0" in the arguments
        True

        Instead, use ``contains`` to determine whether a number is in the
        interval or not:

        >>> i.contains(4)
        True
        >>> i.contains(0)
        False


        Note that ``expr.has(*patterns)`` is exactly equivalent to
        ``any(expr.has(p) for p in patterns)``. In particular, ``False`` is
        returned when the list of patterns is empty.

        >>> x.has()
        False

        """
        return self._has(iterargs, *patterns)

    def has_xfree(self, s: set[Basic]):
        """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
        # protect O(1) containment check by requiring:
        if type(s) is not set:
            raise TypeError('expecting set argument')
        return any(a in s for a in iterfreeargs(self))

    @cacheit
    def has_free(self, *patterns):
        """Return True if self has object(s) ``x`` as a free expression
        else False.

        Examples
        ========

        >>> from sympy import Integral, Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> g = Function('g')
        >>> expr = Integral(f(x), (f(x), 1, g(y)))
        >>> expr.free_symbols
        {y}
        >>> expr.has_free(g(y))
        True
        >>> expr.has_free(*(x, f(x)))
        False

        This works for subexpressions and types, too:

        >>> expr.has_free(g)
        True
        >>> (x + y + 1).has_free(y + 1)
        True
        """
        if not patterns:
            return False
        p0 = patterns[0]
        if len(patterns) == 1 and iterable(p0) and not isinstance(p0, Basic):
            # Basic can contain iterables (though not non-Basic, ideally)
            # but don't encourage mixed passing patterns
            raise TypeError(filldedent('''
                Expecting 1 or more Basic args, not a single
                non-Basic iterable. Don't forget to unpack
                iterables: `eq.has_free(*patterns)`'''))
        # try quick test first
        s = set(patterns)
        rv = self.has_xfree(s)
        if rv:
            return rv
        # now try matching through slower _has
        return self._has(iterfreeargs, *patterns)

    def _has(self, iterargs, *patterns):
        # separate out types and unhashable objects
        type_set = set()  # only types
        p_set = set()  # hashable non-types
        for p in patterns:
            if isinstance(p, type) and issubclass(p, Basic):
                type_set.add(p)
                continue
            if not isinstance(p, Basic):
                try:
                    p = _sympify(p)
                except SympifyError:
                    continue  # Basic won't have this in it
            p_set.add(p)  # fails if object defines __eq__ but
                          # doesn't define __hash__
        types = tuple(type_set)   #
        for i in iterargs(self):  #
            if i in p_set:        # <--- here, too
                return True
            if isinstance(i, types):
                return True

        # use matcher if defined, e.g. operations defines
        # matcher that checks for exact subset containment,
        # (x + y + 1).has(x + 1) -> True
        for i in p_set - type_set:  # types don't have matchers
            if not hasattr(i, '_has_matcher'):
                continue
            match = i._has_matcher()
            if any(match(arg) for arg in iterargs(self)):
                return True

        # no success
        return False

    def replace(self, query, value, map=False, simultaneous=True, exact=None) -> Basic:
        """
        Replace matching subexpressions of ``self`` with ``value``.

        If ``map = True`` then also return the mapping {old: new} where ``old``
        was a sub-expression found with query and ``new`` is the replacement
        value for it. If the expression itself does not match the query, then
        the returned value will be ``self.xreplace(map)`` otherwise it should
        be ``self.subs(ordered(map.items()))``.

        Traverses an expression tree and performs replacement of matching
        subexpressions from the bottom to the top of the tree. The default
        approach is to do the replacement in a simultaneous fashion so
        changes made are targeted only once. If this is not desired or causes
        problems, ``simultaneous`` can be set to False.

        In addition, if an expression containing more than one Wild symbol
        is being used to match subexpressions and the ``exact`` flag is None
        it will be set to True so the match will only succeed if all non-zero
        values are received for each Wild that appears in the match pattern.
        Setting this to False accepts a match of 0; while setting it True
        accepts all matches that have a 0 in them. See example below for
        cautions.

        The list of possible combinations of queries and replacement values
        is listed below:

        Examples
        ========

        Initial setup

        >>> from sympy import log, sin, cos, tan, Wild, Mul, Add
        >>> from sympy.abc import x, y
        >>> f = log(sin(x)) + tan(sin(x**2))

        1.1. type -> type
            obj.replace(type, newtype)

            When object of type ``type`` is found, replace it with the
            result of passing its argument(s) to ``newtype``.

            >>> f.replace(sin, cos)
            log(cos(x)) + tan(cos(x**2))
            >>> sin(x).replace(sin, cos, map=True)
            (cos(x), {sin(x): cos(x)})
            >>> (x*y).replace(Mul, Add)
            x + y

        1.2. type -> func
            obj.replace(type, func)

            When object of type ``type`` is found, apply ``func`` to its
            argument(s). ``func`` must be written to handle the number
            of arguments of ``type``.

            >>> f.replace(sin, lambda arg: sin(2*arg))
            log(sin(2*x)) + tan(sin(2*x**2))
            >>> (x*y).replace(Mul, lambda *args: sin(2*Mul(*args)))
            sin(2*x*y)

        2.1. pattern -> expr
            obj.replace(pattern(wild), expr(wild))

            Replace subexpressions matching ``pattern`` with the expression
            written in terms of the Wild symbols in ``pattern``.

            >>> a, b = map(Wild, 'ab')
            >>> f.replace(sin(a), tan(a))
            log(tan(x)) + tan(tan(x**2))
            >>> f.replace(sin(a), tan(a/2))
            log(tan(x/2)) + tan(tan(x**2/2))
            >>> f.replace(sin(a), a)
            log(x) + tan(x**2)
            >>> (x*y).replace(a*x, a)
            y

            Matching is exact by default when more than one Wild symbol
            is used: matching fails unless the match gives non-zero
            values for all Wild symbols:

            >>> (2*x + y).replace(a*x + b, b - a)
            y - 2
            >>> (2*x).replace(a*x + b, b - a)
            2*x

            When set to False, the results may be non-intuitive:

            >>> (2*x).replace(a*x + b, b - a, exact=False)
            2/x

        2.2. pattern -> func
            obj.replace(pattern(wild), lambda wild: expr(wild))

            All behavior is the same as in 2.1 but now a function in terms of
            pattern variables is used rather than an expression:

            >>> f.replace(sin(a), lambda a: sin(2*a))
            log(sin(2*x)) + tan(sin(2*x**2))

        3.1. func -> func
            obj.replace(filter, func)

            Replace subexpression ``e`` with ``func(e)`` if ``filter(e)``
            is True.

            >>> g = 2*sin(x**3)
            >>> g.replace(lambda expr: expr.is_Number, lambda expr: expr**2)
            4*sin(x**9)

        The expression itself is also targeted by the query but is done in
        such a fashion that changes are not made twice.

            >>> e = x*(x*y + 1)
            >>> e.replace(lambda x: x.is_Mul, lambda x: 2*x)
            2*x*(2*x*y + 1)

        When matching a single symbol, `exact` will default to True, but
        this may or may not be the behavior that is desired:

        Here, we want `exact=False`:

        >>> from sympy import Function
        >>> f = Function('f')
        >>> e = f(1) + f(0)
        >>> q = f(a), lambda a: f(a + 1)
        >>> e.replace(*q, exact=False)
        f(1) + f(2)
        >>> e.replace(*q, exact=True)
        f(0) + f(2)

        But here, the nature of matching makes selecting
        the right setting tricky:

        >>> e = x**(1 + y)
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(-x - y + 1)
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(1 - y)

        It is probably better to use a different form of the query
        that describes the target expression more precisely:

        >>> (1 + x**(1 + y)).replace(
        ... lambda x: x.is_Pow and x.exp.is_Add and x.exp.args[0] == 1,
        ... lambda x: x.base**(1 - (x.exp - 1)))
        ...
        x**(1 - y) + 1

        See Also
        ========

        subs: substitution of subexpressions as defined by the objects
              themselves.
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules

        """

        try:
            query = _sympify(query)
        except SympifyError:
            pass
        try:
            value = _sympify(value)
        except SympifyError:
            pass
        if isinstance(query, type):
            _query = lambda expr: isinstance(expr, query)

            if isinstance(value, type):
                _value = lambda expr, result: value(*expr.args)
            elif callable(value):
                _value = lambda expr, result: value(*expr.args)
            else:
                raise TypeError(
                    "given a type, replace() expects another "
                    "type or a callable")
        elif isinstance(query, Basic):
            _query = lambda expr: expr.match(query)
            if exact is None:
                from .symbol import Wild
                exact = (len(query.atoms(Wild)) > 1)

            if isinstance(value, Basic):
                if exact:
                    _value = lambda expr, result: (value.subs(result)
                        if all(result.values()) else expr)
                else:
                    _value = lambda expr, result: value.subs(result)
            elif callable(value):
                # match dictionary keys get the trailing underscore stripped
                # from them and are then passed as keywords to the callable;
                # if ``exact`` is True, only accept match if there are no null
                # values amongst those matched.
                if exact:
                    _value = lambda expr, result: (value(**
                        {str(k)[:-1]: v for k, v in result.items()})
                        if all(val for val in result.values()) else expr)
                else:
                    _value = lambda expr, result: value(**
                        {str(k)[:-1]: v for k, v in result.items()})
            else:
                raise TypeError(
                    "given an expression, replace() expects "
                    "another expression or a callable")
        elif callable(query):
            _query = query

            if callable(value):
                _value = lambda expr, result: value(expr)
            else:
                raise TypeError(
                    "given a callable, replace() expects "
                    "another callable")
        else:
            raise TypeError(
                "first argument to replace() must be a "
                "type, an expression or a callable")

        def walk(rv, F):
            """Apply ``F`` to args and then to result.
            """
            args = getattr(rv, 'args', None)
            if args is not None:
                if args:
                    newargs = tuple([walk(a, F) for a in args])
                    if args != newargs:
                        rv = rv.func(*newargs)
                        if simultaneous:
                            # if rv is something that was already
                            # matched (that was changed) then skip
                            # applying F again
                            for i, e in enumerate(args):
                                if rv == e and e != newargs[i]:
                                    return rv
                rv = F(rv)
            return rv

        mapping = {}  # changes that took place

        def rec_replace(expr):
            result = _query(expr)
            if result or result == {}:
                v = _value(expr, result)
                if v is not None and v != expr:
                    if map:
                        mapping[expr] = v
                    expr = v
            return expr

        rv = walk(self, rec_replace)
        return (rv, mapping) if map else rv # type: ignore

    def find(self, query, group=False):
        """Find all subexpressions matching a query."""
        query = _make_find_query(query)
        results = list(filter(query, _preorder_traversal(self)))

        if not group:
            return set(results)
        return dict(Counter(results))

    def count(self, query):
        """Count the number of matching subexpressions."""
        query = _make_find_query(query)
        return sum(bool(query(sub)) for sub in _preorder_traversal(self))

    def matches(self, expr, repl_dict=None, old=False):
        """
        Helper method for match() that looks for a match between Wild symbols
        in self and expressions in expr.

        Examples
        ========

        >>> from sympy import symbols, Wild, Basic
        >>> a, b, c = symbols('a b c')
        >>> x = Wild('x')
        >>> Basic(a + x, x).matches(Basic(a + b, c)) is None
        True
        >>> Basic(a + x, x).matches(Basic(a + b + c, b + c))
        {x_: b + c}
        """
        expr = sympify(expr)
        if not isinstance(expr, self.__class__):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if self == expr:
            return repl_dict

        if len(self.args) != len(expr.args):
            return None

        d = repl_dict  # already a copy
        for arg, other_arg in zip(self.args, expr.args):
            if arg == other_arg:
                continue
            if arg.is_Relational:
                try:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
                except TypeError: # Should be InvalidComparisonError when introduced
                    d = None
            else:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
            if d is None:
                return None
        return d

    def match(self, pattern, old=False):
        """
        Pattern matching.

        Wild symbols match all.

        Return ``None`` when expression (self) does not match with pattern.
        Otherwise return a dictionary such that::

          pattern.xreplace(self.match(pattern)) == self

        Examples
        ========

        >>> from sympy import Wild, Sum
        >>> from sympy.abc import x, y
        >>> p = Wild("p")
        >>> q = Wild("q")
        >>> r = Wild("r")
        >>> e = (x+y)**(x+y)
        >>> e.match(p**p)
        {p_: x + y}
        >>> e.match(p**q)
        {p_: x + y, q_: x + y}
        >>> e = (2*x)**2
        >>> e.match(p*q**r)
        {p_: 4, q_: x, r_: 2}
        >>> (p*q**r).xreplace(e.match(p*q**r))
        4*x**2

        Since match is purely structural expressions that are equivalent up to
        bound symbols will not match:

        >>> print(Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, p))))
        None

        An expression with bound symbols can be matched if the pattern uses
        a distinct ``Wild`` for each bound symbol:

        >>> Sum(x, (x, 1, 2)).match(Sum(q, (q, 1, p)))
        {p_: 2, q_: x}

        The ``old`` flag will give the old-style pattern matching where
        expressions and patterns are essentially solved to give the match. Both
        of the following give None unless ``old=True``:

        >>> (x - 2).match(p - x, old=True)
        {p_: 2*x - 2}
        >>> (2/x).match(p*x, old=True)
        {p_: 2/x**2}

        See Also
        ========

        matches: pattern.matches(expr) is the same as expr.match(pattern)
        xreplace: exact structural replacement
        replace: structural replacement with pattern matching
        Wild: symbolic placeholders for expressions in pattern matching
        """
        pattern = sympify(pattern)
        return pattern.matches(self, old=old)

    def count_ops(self, visual=False):
        """Wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)

    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
        integrals, sums and products. All objects of this kind will be
        evaluated recursively, unless some species were excluded via 'hints'
        or unless the 'deep' hint was set to 'False'.

        >>> from sympy import Integral
        >>> from sympy.abc import x

        >>> 2*Integral(x, x)
        2*Integral(x, x)

        >>> (2*Integral(x, x)).doit()
        x**2

        >>> (2*Integral(x, x)).doit(deep=False)
        2*Integral(x, x)

        """
        if hints.get('deep', True):
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            return self.func(*terms)
        else:
            return self

    def simplify(self, **kwargs) -> Basic:
        """See the simplify function in sympy.simplify"""
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions.refine import refine
        return refine(self, assumption)

    def _eval_derivative_n_times(self, s, n):
        # This is the default evaluator for derivatives (as called by `diff`
        # and `Derivative`), it will attempt a loop to derive the expression
        # `n` times by calling the corresponding `_eval_derivative` method,
        # while leaving the derivative unevaluated if `n` is symbolic.  This
        # method should be overridden if the object has a closed form for its
        # symbolic n-th derivative.
        from .numbers import Integer
        if isinstance(n, (int, Integer)):
            obj = self
            for i in range(n):
                prev = obj
                obj = obj._eval_derivative(s)
                if obj is None:
                    return None
                elif obj == prev:
                    break
            return obj
        else:
            return None

    def rewrite(self, *args, deep=True, **hints):
        """
        Rewrite *self* using a defined rule.

        Rewriting transforms an expression to another, which is mathematically
        equivalent but structurally different. For example you can rewrite
        trigonometric functions as complex exponentials or combinatorial
        functions as gamma function.

        This method takes a *pattern* and a *rule* as positional arguments.
        *pattern* is optional parameter which defines the types of expressions
        that will be transformed. If it is not passed, all possible expressions
        will be rewritten. *rule* defines how the expression will be rewritten.

        Parameters
        ==========

        args : Expr
            A *rule*, or *pattern* and *rule*.
            - *pattern* is a type or an iterable of types.
            - *rule* can be any object.

        deep : bool, optional
            If ``True``, subexpressions are recursively transformed. Default is
            ``True``.

        Examples
        ========

        If *pattern* is unspecified, all possible expressions are transformed.

        >>> from sympy import cos, sin, exp, I
        >>> from sympy.abc import x
        >>> expr = cos(x) + I*sin(x)
        >>> expr.rewrite(exp)
        exp(I*x)

        Pattern can be a type or an iterable of types.

        >>> expr.rewrite(sin, exp)
        exp(I*x)/2 + cos(x) - exp(-I*x)/2
        >>> expr.rewrite([cos,], exp)
        exp(I*x)/2 + I*sin(x) + exp(-I*x)/2
        >>> expr.rewrite([cos, sin], exp)
        exp(I*x)

        Rewriting behavior can be implemented by defining ``_eval_rewrite()``
        method.

        >>> from sympy import Expr, sqrt, pi
        >>> class MySin(Expr):
        ...     def _eval_rewrite(self, rule, args, **hints):
        ...         x, = args
        ...         if rule == cos:
        ...             return cos(pi/2 - x, evaluate=False)
        ...         if rule == sqrt:
        ...             return sqrt(1 - cos(x)**2)
        >>> MySin(MySin(x)).rewrite(cos)
        cos(-cos(-x + pi/2) + pi/2)
        >>> MySin(x).rewrite(sqrt)
        sqrt(1 - cos(x)**2)

        Defining ``_eval_rewrite_as_[...]()`` method is supported for backwards
        compatibility reason. This may be removed in the future and using it is
        discouraged.

        >>> class MySin(Expr):
        ...     def _eval_rewrite_as_cos(self, *args, **hints):
        ...         x, = args
        ...         return cos(pi/2 - x, evaluate=False)
        >>> MySin(x).rewrite(cos)
        cos(-x + pi/2)

        """
        if not args:
            return self

        hints.update(deep=deep)

        pattern = args[:-1]
        rule = args[-1]

        # Special case: map `abs` to `Abs`
        if rule is abs:
            from sympy.functions.elementary.complexes import Abs
            rule = Abs

        # support old design by _eval_rewrite_as_[...] method
        if isinstance(rule, str):
            method = "_eval_rewrite_as_%s" % rule
        elif hasattr(rule, "__name__"):
            # rule is class or function
            clsname = rule.__name__
            method = "_eval_rewrite_as_%s" % clsname
        else:
            # rule is instance
            clsname = rule.__class__.__name__
            method = "_eval_rewrite_as_%s" % clsname

        if pattern:
            if iterable(pattern[0]):
                pattern = pattern[0]
            pattern = tuple(p for p in pattern if self.has(p))
            if not pattern:
                return self
        # hereafter, empty pattern is interpreted as all pattern.

        return self._rewrite(pattern, rule, method, **hints)

    def _rewrite(self, pattern, rule, method, **hints):
        deep = hints.pop('deep', True)
        if deep:
            args = [a._rewrite(pattern, rule, method, **hints)
                    for a in self.args]
        else:
            args = self.args
        if not pattern or any(isinstance(self, p) for p in pattern):
            meth = getattr(self, method, None)
            if meth is not None:
                rewritten = meth(*args, **hints)
            else:
                rewritten = self._eval_rewrite(rule, args, **hints)
            if rewritten is not None:
                return rewritten
        if not args:
            return self
        return self.func(*args)

    def _eval_rewrite(self, rule, args, **hints):
        return None

    _constructor_postprocessor_mapping = {}  # type: ignore

    @classmethod
    def _exec_constructor_postprocessors(cls, obj):
        # WARNING: This API is experimental.

        # This is an experimental API that introduces constructor
        # postprosessors for SymPy Core elements. If an argument of a SymPy
        # expression has a `_constructor_postprocessor_mapping` attribute, it will
        # be interpreted as a dictionary containing lists of postprocessing
        # functions for matching expression node names.

        clsname = obj.__class__.__name__
        postprocessors = {f for i in obj.args
                            for f in _get_postprocessors(clsname, type(i))}
        for f in postprocessors:
            obj = f(obj)

        return obj

    def _sage_(self):
        """
        Convert *self* to a symbolic expression of SageMath.

        This version of the method is merely a placeholder.
        """
        old_method = self._sage_
        from sage.interfaces.sympy import sympy_init # type: ignore
        sympy_init()  # may monkey-patch _sage_ method into self's class or superclasses
        if old_method == self._sage_:
            raise NotImplementedError('conversion to SageMath is not implemented')
        else:
            # call the freshly monkey-patched method
            return self._sage_()

    def could_extract_minus_sign(self) -> bool:
        return False  # see Expr.could_extract_minus_sign

    def is_same(a, b, approx=None):
        """Return True if a and b are structurally the same, else False.
        If `approx` is supplied, it will be used to test whether two
        numbers are the same or not. By default, only numbers of the
        same type will compare equal, so S.Half != Float(0.5).

        Examples
        ========

        In SymPy (unlike Python) two numbers do not compare the same if they are
        not of the same type:

        >>> from sympy import S
        >>> 2.0 == S(2)
        False
        >>> 0.5 == S.Half
        False

        By supplying a function with which to compare two numbers, such
        differences can be ignored. e.g. `equal_valued` will return True
        for decimal numbers having a denominator that is a power of 2,
        regardless of precision.

        >>> from sympy import Float
        >>> from sympy.core.numbers import equal_valued
        >>> (S.Half/4).is_same(Float(0.125, 1), equal_valued)
        True
        >>> Float(1, 2).is_same(Float(1, 10), equal_valued)
        True

        But decimals without a power of 2 denominator will compare
        as not being the same.

        >>> Float(0.1, 9).is_same(Float(0.1, 10), equal_valued)
        False

        But arbitrary differences can be ignored by supplying a function
        to test the equivalence of two numbers:

        >>> import math
        >>> Float(0.1, 9).is_same(Float(0.1, 10), math.isclose)
        True

        Other objects might compare the same even though types are not the
        same. This routine will only return True if two expressions are
        identical in terms of class types.

        >>> from sympy import eye, Basic
        >>> eye(1) == S(eye(1))  # mutable vs immutable
        True
        >>> Basic.is_same(eye(1), S(eye(1)))
        False

        """
        from .numbers import Number
        from .traversal import postorder_traversal as pot
        for t in zip_longest(pot(a), pot(b)):
            if None in t:
                return False
            a, b = t
            if isinstance(a, Number):
                if not isinstance(b, Number):
                    return False
                if approx:
                    return approx(a, b)
            if not (a == b and a.__class__ == b.__class__):
                return False
        return True

_aresame = Basic.is_same  # for sake of others importing this

# key used by Mul and Add to make canonical args
_args_sortkey = cmp_to_key(Basic.compare)

# For all Basic subclasses _prepare_class_assumptions is called by
# Basic.__init_subclass__ but that method is not called for Basic itself so we
# call the function here instead.
_prepare_class_assumptions(Basic)


class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    def matches(self, expr, repl_dict=None, old=False):
        if self == expr:
            if repl_dict is None:
                return {}
            return repl_dict.copy()

    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    def doit(self, **hints):
        return self

    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def _eval_simplify(self, **kwargs):
        return self

    @property
    def _sorted_args(self):
        # this is here as a safeguard against accidentally using _sorted_args
        # on Atoms -- they cannot be rebuilt as atom.func(*atom._sorted_args)
        # since there are no args. So the calling routine should be checking
        # to see that this property is not called for Atoms.
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')


def _atomic(e, recursive=False):
    """Return atom-like quantities as far as substitution is
    concerned: Derivatives, Functions and Symbols. Do not
    return any 'atoms' that are inside such quantities unless
    they also appear outside, too, unless `recursive` is True.

    Examples
    ========

    >>> from sympy import Derivative, Function, cos
    >>> from sympy.abc import x, y
    >>> from sympy.core.basic import _atomic
    >>> f = Function('f')
    >>> _atomic(x + y)
    {x, y}
    >>> _atomic(x + f(y))
    {x, f(y)}
    >>> _atomic(Derivative(f(x), x) + cos(x) + y)
    {y, cos(x), Derivative(f(x), x)}

    """
    pot = _preorder_traversal(e)
    seen = set()
    if isinstance(e, Basic):
        free = getattr(e, "free_symbols", None)
        if free is None:
            return {e}
    else:
        return set()
    from .symbol import Symbol
    from .function import Derivative, Function
    atoms = set()
    for p in pot:
        if p in seen:
            pot.skip()
            continue
        seen.add(p)
        if isinstance(p, Symbol) and p in free:
            atoms.add(p)
        elif isinstance(p, (Derivative, Function)):
            if not recursive:
                pot.skip()
            atoms.add(p)
    return atoms


def _make_find_query(query):
    """Convert the argument of Basic.find() into a callable"""
    try:
        query = _sympify(query)
    except SympifyError:
        pass
    if isinstance(query, type):
        return lambda expr: isinstance(expr, query)
    elif isinstance(query, Basic):
        return lambda expr: expr.match(query) is not None
    return query

# Delayed to avoid cyclic import
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
   iterargs, iterfreeargs)

preorder_traversal = deprecated(
    """
    Using preorder_traversal from the sympy.core.basic submodule is
    deprecated.

    Instead, use preorder_traversal from the top-level sympy namespace, like

        sympy.preorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_preorder_traversal)
