r"""Module that defines indexed objects.

The classes ``IndexedBase``, ``Indexed``, and ``Idx`` represent a
matrix element ``M[i, j]`` as in the following diagram::

       1) The Indexed class represents the entire indexed object.
                  |
               ___|___
              '       '
               M[i, j]
              /   \__\______
              |             |
              |             |
              |     2) The Idx class represents indices; each Idx can
              |        optionally contain information about its range.
              |
        3) IndexedBase represents the 'stem' of an indexed object, here `M`.
           The stem used by itself is usually taken to represent the entire
           array.

There can be any number of indices on an Indexed object.  No
transformation properties are implemented in these Base objects, but
implicit contraction of repeated indices is supported.

Note that the support for complicated (i.e. non-atomic) integer
expressions as indices is limited.  (This should be improved in
future releases.)

Examples
========

To express the above matrix element example you would write:

>>> from sympy import symbols, IndexedBase, Idx
>>> M = IndexedBase('M')
>>> i, j = symbols('i j', cls=Idx)
>>> M[i, j]
M[i, j]

Repeated indices in a product implies a summation, so to express a
matrix-vector product in terms of Indexed objects:

>>> x = IndexedBase('x')
>>> M[i, j]*x[j]
M[i, j]*x[j]

If the indexed objects will be converted to component based arrays, e.g.
with the code printers or the autowrap framework, you also need to provide
(symbolic or numerical) dimensions.  This can be done by passing an
optional shape parameter to IndexedBase upon construction:

>>> dim1, dim2 = symbols('dim1 dim2', integer=True)
>>> A = IndexedBase('A', shape=(dim1, 2*dim1, dim2))
>>> A.shape
(dim1, 2*dim1, dim2)
>>> A[i, j, 3].shape
(dim1, 2*dim1, dim2)

If an IndexedBase object has no shape information, it is assumed that the
array is as large as the ranges of its indices:

>>> n, m = symbols('n m', integer=True)
>>> i = Idx('i', m)
>>> j = Idx('j', n)
>>> M[i, j].shape
(m, n)
>>> M[i, j].ranges
[(0, m - 1), (0, n - 1)]

The above can be compared with the following:

>>> A[i, 2, j].shape
(dim1, 2*dim1, dim2)
>>> A[i, 2, j].ranges
[(0, m - 1), None, (0, n - 1)]

To analyze the structure of indexed expressions, you can use the methods
get_indices() and get_contraction_structure():

>>> from sympy.tensor import get_indices, get_contraction_structure
>>> get_indices(A[i, j, j])
({i}, {})
>>> get_contraction_structure(A[i, j, j])
{(j,): {A[i, j, j]}}

See the appropriate docstrings for a detailed explanation of the output.
"""

#   TODO:  (some ideas for improvement)
#
#   o test and guarantee numpy compatibility
#      - implement full support for broadcasting
#      - strided arrays
#
#   o more functions to analyze indexed expressions
#      - identify standard constructs, e.g matrix-vector product in a subexpression
#
#   o functions to generate component based arrays (numpy and sympy.Matrix)
#      - generate a single array directly from Indexed
#      - convert simple sub-expressions
#
#   o sophisticated indexing (possibly in subclasses to preserve simplicity)
#      - Idx with range smaller than dimension of Indexed
#      - Idx with stepsize != 1
#      - Idx with step determined by function call
from collections.abc import Iterable

from sympy.core.numbers import Number
from sympy.core.assumptions import StdFactKB
from sympy.core import Expr, Tuple, sympify, S
from sympy.core.symbol import _filter_assumptions, Symbol
from sympy.core.logic import fuzzy_bool, fuzzy_not
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.multipledispatch import dispatch
from sympy.utilities.iterables import is_sequence, NotIterable
from sympy.utilities.misc import filldedent


class IndexException(Exception):
    pass


class Indexed(Expr):
    """Represents a mathematical object with indices.

    >>> from sympy import Indexed, IndexedBase, Idx, symbols
    >>> i, j = symbols('i j', cls=Idx)
    >>> Indexed('A', i, j)
    A[i, j]

    It is recommended that ``Indexed`` objects be created by indexing ``IndexedBase``:
    ``IndexedBase('A')[i, j]`` instead of ``Indexed(IndexedBase('A'), i, j)``.

    >>> A = IndexedBase('A')
    >>> a_ij = A[i, j]           # Prefer this,
    >>> b_ij = Indexed(A, i, j)  # over this.
    >>> a_ij == b_ij
    True

    """
    is_Indexed = True
    is_symbol = True
    is_Atom = True

    def __new__(cls, base, *args, **kw_args):
        from sympy.tensor.array.ndim_array import NDimArray
        from sympy.matrices.matrixbase import MatrixBase

        if not args:
            raise IndexException("Indexed needs at least one index.")
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base, IndexedBase):
            raise TypeError(filldedent("""
                The base can only be replaced with a string, Symbol,
                IndexedBase or an object with a method for getting
                items (i.e. an object with a `__getitem__` method).
                """))
        args = list(map(sympify, args))
        if isinstance(base, (NDimArray, Iterable, Tuple, MatrixBase)) and all(i.is_number for i in args):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]

        base = _sympify(base)

        obj = Expr.__new__(cls, base, *args, **kw_args)

        IndexedBase._set_assumptions(obj, base.assumptions0)

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def name(self):
        return str(self)

    @property
    def _diff_wrt(self):
        """Allow derivatives with respect to an ``Indexed`` object."""
        return True

    def _eval_derivative(self, wrt):
        from sympy.tensor.array.ndim_array import NDimArray

        if isinstance(wrt, Indexed) and wrt.base == self.base:
            if len(self.indices) != len(wrt.indices):
                msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                       wrt)
                raise IndexException(msg)
            result = S.One
            for index1, index2 in zip(self.indices, wrt.indices):
                result *= KroneckerDelta(index1, index2)
            return result
        elif isinstance(self.base, NDimArray):
            from sympy.tensor.array import derive_by_array
            return Indexed(derive_by_array(self.base, wrt), *self.args[1:])
        else:
            if Tuple(self.indices).has(wrt):
                return S.NaN
            return S.Zero

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    @property
    def base(self):
        """Returns the ``IndexedBase`` of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, IndexedBase, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).base
        A
        >>> B = IndexedBase('B')
        >>> B == B[i, j].base
        True

        """
        return self.args[0]

    @property
    def indices(self):
        """
        Returns the indices of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).indices
        (i, j)

        """
        return self.args[1:]

    @property
    def rank(self):
        """
        Returns the rank of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j, k, l, m = symbols('i:m', cls=Idx)
        >>> Indexed('A', i, j).rank
        2
        >>> q = Indexed('A', i, j, k, l, m)
        >>> q.rank
        5
        >>> q.rank == len(q.indices)
        True

        """
        return len(self.args) - 1

    @property
    def shape(self):
        """Returns a list with dimensions of each index.

        Dimensions is a property of the array, not of the indices.  Still, if
        the ``IndexedBase`` does not define a shape attribute, it is assumed
        that the ranges of the indices correspond to the shape of the array.

        >>> from sympy import IndexedBase, Idx, symbols
        >>> n, m = symbols('n m', integer=True)
        >>> i = Idx('i', m)
        >>> j = Idx('j', m)
        >>> A = IndexedBase('A', shape=(n, n))
        >>> B = IndexedBase('B')
        >>> A[i, j].shape
        (n, n)
        >>> B[i, j].shape
        (m, m)
        """

        if self.base.shape:
            return self.base.shape
        sizes = []
        for i in self.indices:
            upper = getattr(i, 'upper', None)
            lower = getattr(i, 'lower', None)
            if None in (upper, lower):
                raise IndexException(filldedent("""
                    Range is not defined for all indices in: %s""" % self))
            try:
                size = upper - lower + 1
            except TypeError:
                raise IndexException(filldedent("""
                    Shape cannot be inferred from Idx with
                    undefined range: %s""" % self))
            sizes.append(size)
        return Tuple(*sizes)

    @property
    def ranges(self):
        """Returns a list of tuples with lower and upper range of each index.

        If an index does not define the data members upper and lower, the
        corresponding slot in the list contains ``None`` instead of a tuple.

        Examples
        ========

        >>> from sympy import Indexed,Idx, symbols
        >>> Indexed('A', Idx('i', 2), Idx('j', 4), Idx('k', 8)).ranges
        [(0, 1), (0, 3), (0, 7)]
        >>> Indexed('A', Idx('i', 3), Idx('j', 3), Idx('k', 3)).ranges
        [(0, 2), (0, 2), (0, 2)]
        >>> x, y, z = symbols('x y z', integer=True)
        >>> Indexed('A', x, y, z).ranges
        [None, None, None]

        """
        ranges = []
        sentinel = object()
        for i in self.indices:
            upper = getattr(i, 'upper', sentinel)
            lower = getattr(i, 'lower', sentinel)
            if sentinel not in (upper, lower):
                ranges.append((lower, upper))
            else:
                ranges.append(None)
        return ranges

    def _sympystr(self, p):
        indices = list(map(p.doprint, self.indices))
        return "%s[%s]" % (p.doprint(self.base), ", ".join(indices))

    @property
    def free_symbols(self):
        base_free_symbols = self.base.free_symbols
        indices_free_symbols = {
            fs for i in self.indices for fs in i.free_symbols}
        if base_free_symbols:
            return {self} | base_free_symbols | indices_free_symbols
        else:
            return indices_free_symbols

    @property
    def expr_free_symbols(self):
        from sympy.utilities.exceptions import sympy_deprecation_warning
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")

        return {self}


class IndexedBase(Expr, NotIterable):
    """Represent the base or stem of an indexed object

    The IndexedBase class represent an array that contains elements. The main purpose
    of this class is to allow the convenient creation of objects of the Indexed
    class.  The __getitem__ method of IndexedBase returns an instance of
    Indexed.  Alone, without indices, the IndexedBase class can be used as a
    notation for e.g. matrix equations, resembling what you could do with the
    Symbol class.  But, the IndexedBase class adds functionality that is not
    available for Symbol instances:

      -  An IndexedBase object can optionally store shape information.  This can
         be used in to check array conformance and conditions for numpy
         broadcasting.  (TODO)
      -  An IndexedBase object implements syntactic sugar that allows easy symbolic
         representation of array operations, using implicit summation of
         repeated indices.
      -  The IndexedBase object symbolizes a mathematical structure equivalent
         to arrays, and is recognized as such for code generation and automatic
         compilation and wrapping.

    >>> from sympy.tensor import IndexedBase, Idx
    >>> from sympy import symbols
    >>> A = IndexedBase('A'); A
    A
    >>> type(A)
    <class 'sympy.tensor.indexed.IndexedBase'>

    When an IndexedBase object receives indices, it returns an array with named
    axes, represented by an Indexed object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'sympy.tensor.indexed.Indexed'>

    The IndexedBase constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedBase('B', shape=(o, p))
    >>> B[i, j].shape
    (o, p)

    Assumptions can be specified with keyword arguments the same way as for Symbol:

    >>> A_real = IndexedBase('A', real=True)
    >>> A_real.is_real
    True
    >>> A != A_real
    True

    Assumptions can also be inherited if a Symbol is used to initialize the IndexedBase:

    >>> I = symbols('I', integer=True)
    >>> C_inherit = IndexedBase(I)
    >>> C_explicit = IndexedBase('I', integer=True)
    >>> C_inherit == C_explicit
    True
    """
    is_symbol = True
    is_Atom = True

    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        tmp_asm_copy = assumptions.copy()
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873

    def __new__(cls, label, shape=None, *, offset=S.Zero, strides=None, **kw_args):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array.ndim_array import NDimArray

        assumptions, kw_args = _filter_assumptions(kw_args)
        if isinstance(label, str):
            label = Symbol(label, **assumptions)
        elif isinstance(label, Symbol):
            assumptions = label._merge(assumptions)
        elif isinstance(label, (MatrixBase, NDimArray)):
            return label
        elif isinstance(label, Iterable):
            return _sympify(label)
        else:
            label = _sympify(label)

        if is_sequence(shape):
            shape = Tuple(*shape)
        elif shape is not None:
            shape = Tuple(shape)

        if shape is not None:
            obj = Expr.__new__(cls, label, shape)
        else:
            obj = Expr.__new__(cls, label)
        obj._shape = shape
        obj._offset = offset
        obj._strides = strides
        obj._name = str(label)

        IndexedBase._set_assumptions(obj, assumptions)
        return obj

    @property
    def name(self):
        return self._name

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return Indexed(self, *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return Indexed(self, indices, **kw_args)

    @property
    def shape(self):
        """Returns the shape of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase, Idx
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).shape
        (x, y)

        Note: If the shape of the ``IndexedBase`` is specified, it will override
        any shape information given by the indices.

        >>> A = IndexedBase('A', shape=(x, y))
        >>> B = IndexedBase('B')
        >>> i = Idx('i', 2)
        >>> j = Idx('j', 1)
        >>> A[i, j].shape
        (x, y)
        >>> B[i, j].shape
        (2, 1)

        """
        return self._shape

    @property
    def strides(self):
        """Returns the strided scheme for the ``IndexedBase`` object.

        Normally this is a tuple denoting the number of
        steps to take in the respective dimension when traversing
        an array. For code generation purposes strides='C' and
        strides='F' can also be used.

        strides='C' would mean that code printer would unroll
        in row-major order and 'F' means unroll in column major
        order.

        """

        return self._strides

    @property
    def offset(self):
        """Returns the offset for the ``IndexedBase`` object.

        This is the value added to the resulting index when the
        2D Indexed object is unrolled to a 1D form. Used in code
        generation.

        Examples
        ==========
        >>> from sympy.printing import ccode
        >>> from sympy.tensor import IndexedBase, Idx
        >>> from sympy import symbols
        >>> l, m, n, o = symbols('l m n o', integer=True)
        >>> A = IndexedBase('A', strides=(l, m, n), offset=o)
        >>> i, j, k = map(Idx, 'ijk')
        >>> ccode(A[i, j, k])
        'A[l*i + m*j + n*k + o]'

        """
        return self._offset

    @property
    def label(self):
        """Returns the label of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).label
        A

        """
        return self.args[0]

    def _sympystr(self, p):
        return p.doprint(self.label)


class Idx(Expr):
    """Represents an integer index as an ``Integer`` or integer expression.

    There are a number of ways to create an ``Idx`` object.  The constructor
    takes two arguments:

    ``label``
        An integer or a symbol that labels the index.
    ``range``
        Optionally you can specify a range as either

        * ``Symbol`` or integer: This is interpreted as a dimension. Lower and
          upper bounds are set to ``0`` and ``range - 1``, respectively.
        * ``tuple``: The two elements are interpreted as the lower and upper
          bounds of the range, respectively.

    Note: bounds of the range are assumed to be either integer or infinite (oo
    and -oo are allowed to specify an unbounded range). If ``n`` is given as a
    bound, then ``n.is_integer`` must not return false.

    For convenience, if the label is given as a string it is automatically
    converted to an integer symbol.  (Note: this conversion is not done for
    range or dimension arguments.)

    Examples
    ========

    >>> from sympy import Idx, symbols, oo
    >>> n, i, L, U = symbols('n i L U', integer=True)

    If a string is given for the label an integer ``Symbol`` is created and the
    bounds are both ``None``:

    >>> idx = Idx('qwerty'); idx
    qwerty
    >>> idx.lower, idx.upper
    (None, None)

    Both upper and lower bounds can be specified:

    >>> idx = Idx(i, (L, U)); idx
    i
    >>> idx.lower, idx.upper
    (L, U)

    When only a single bound is given it is interpreted as the dimension
    and the lower bound defaults to 0:

    >>> idx = Idx(i, n); idx.lower, idx.upper
    (0, n - 1)
    >>> idx = Idx(i, 4); idx.lower, idx.upper
    (0, 3)
    >>> idx = Idx(i, oo); idx.lower, idx.upper
    (0, oo)

    """

    is_integer = True
    is_finite = True
    is_real = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True

    def __new__(cls, label, range=None, **kw_args):

        if isinstance(label, str):
            label = Symbol(label, integer=True)
        label, range = list(map(sympify, (label, range)))

        if label.is_Number:
            if not label.is_integer:
                raise TypeError("Index is not an integer number.")
            return label

        if not label.is_integer:
            raise TypeError("Idx object requires an integer label.")

        elif is_sequence(range):
            if len(range) != 2:
                raise ValueError(filldedent("""
                    Idx range tuple must have length 2, but got %s""" % len(range)))
            for bound in range:
                if (bound.is_integer is False and bound is not S.Infinity
                        and bound is not S.NegativeInfinity):
                    raise TypeError("Idx object requires integer bounds.")
            args = label, Tuple(*range)
        elif isinstance(range, Expr):
            if range is not S.Infinity and fuzzy_not(range.is_integer):
                raise TypeError("Idx object requires an integer dimension.")
            args = label, Tuple(0, range - 1)
        elif range:
            raise TypeError(filldedent("""
                The range must be an ordered iterable or
                integer SymPy expression."""))
        else:
            args = label,

        obj = Expr.__new__(cls, *args, **kw_args)
        obj._assumptions["finite"] = True
        obj._assumptions["real"] = True
        return obj

    @property
    def label(self):
        """Returns the label (Integer or integer expression) of the Idx object.

        Examples
        ========

        >>> from sympy import Idx, Symbol
        >>> x = Symbol('x', integer=True)
        >>> Idx(x).label
        x
        >>> j = Symbol('j', integer=True)
        >>> Idx(j).label
        j
        >>> Idx(j + 1).label
        j + 1

        """
        return self.args[0]

    @property
    def lower(self):
        """Returns the lower bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).lower
        0
        >>> Idx('j', 5).lower
        0
        >>> Idx('j').lower is None
        True

        """
        try:
            return self.args[1][0]
        except IndexError:
            return

    @property
    def upper(self):
        """Returns the upper bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).upper
        1
        >>> Idx('j', 5).upper
        4
        >>> Idx('j').upper is None
        True

        """
        try:
            return self.args[1][1]
        except IndexError:
            return

    def _sympystr(self, p):
        return p.doprint(self.label)

    @property
    def name(self):
        return self.label.name if self.label.is_Symbol else str(self.label)

    @property
    def free_symbols(self):
        return {self}


@dispatch(Idx, Idx)
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs if rhs.upper is None else rhs.upper
    other_lower = rhs if rhs.lower is None else rhs.lower

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Idx, Number)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs
    other_lower = rhs

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Number, Idx)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = lhs
    other_lower = lhs

    if rhs.upper is not None and (rhs.upper <= other_lower) == True:
        return True
    if rhs.lower is not None and (rhs.lower > other_upper) == True:
        return False
    return None
