"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------
::

  *Basic*
       |
       |
   CodegenAST
       |
       |--->AssignmentBase
       |             |--->Assignment
       |             |--->AugmentedAssignment
       |                                    |--->AddAugmentedAssignment
       |                                    |--->SubAugmentedAssignment
       |                                    |--->MulAugmentedAssignment
       |                                    |--->DivAugmentedAssignment
       |                                    |--->ModAugmentedAssignment
       |
       |--->CodeBlock
       |
       |
       |--->Token
                |--->Attribute
                |--->For
                |--->String
                |       |--->QuotedString
                |       |--->Comment
                |--->Type
                |       |--->IntBaseType
                |       |              |--->_SizedIntType
                |       |                               |--->SignedIntType
                |       |                               |--->UnsignedIntType
                |       |--->FloatBaseType
                |                        |--->FloatType
                |                        |--->ComplexBaseType
                |                                           |--->ComplexType
                |--->Node
                |       |--->Variable
                |       |           |---> Pointer
                |       |--->FunctionPrototype
                |                            |--->FunctionDefinition
                |--->Element
                |--->Declaration
                |--->While
                |--->Scope
                |--->Stream
                |--->Print
                |--->FunctionCall
                |--->BreakToken
                |--->ContinueToken
                |--->NoneToken
                |--->Return


Predefined types
----------------

A number of ``Type`` instances are provided in the ``sympy.codegen.ast`` module
for convenience. Perhaps the two most common ones for code-generation (of numeric
codes) are ``float32`` and ``float64`` (known as single and double precision respectively).
There are also precision generic versions of Types (for which the codeprinters selects the
underlying data type at time of printing): ``real``, ``integer``, ``complex_``, ``bool_``.

The other ``Type`` instances defined are:

- ``intc``: Integer type used by C's "int".
- ``intp``: Integer type used by C's "unsigned".
- ``int8``, ``int16``, ``int32``, ``int64``: n-bit integers.
- ``uint8``, ``uint16``, ``uint32``, ``uint64``: n-bit unsigned integers.
- ``float80``: known as "extended precision" on modern x86/amd64 hardware.
- ``complex64``: Complex number represented by two ``float32`` numbers
- ``complex128``: Complex number represented by two ``float64`` numbers

Using the nodes
---------------

It is possible to construct simple algorithms using the AST nodes. Let's construct a loop applying
Newton's method::

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import While, Assignment, aug_assign, Print, QuotedString
    >>> t, dx, x = symbols('tol delta val')
    >>> expr = cos(x) - x**3
    >>> whl = While(abs(dx) > t, [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx),
    ...     Print([x])
    ... ])
    >>> from sympy import pycode
    >>> py_str = pycode(whl)
    >>> print(py_str)
    while (abs(delta) > tol):
        delta = (val**3 - math.cos(val))/(-3*val**2 - math.sin(val))
        val += delta
        print(val)
    >>> import math
    >>> tol, val, delta = 1e-5, 0.5, float('inf')
    >>> exec(py_str)
    1.1121416371
    0.909672693737
    0.867263818209
    0.865477135298
    0.865474033111
    >>> print('%3.1g' % (math.cos(val) - val**3))
    -3e-11

If we want to generate Fortran code for the same while loop we simple call ``fcode``::

    >>> from sympy import fcode
    >>> print(fcode(whl, standard=2003, source_format='free'))
    do while (abs(delta) > tol)
       delta = (val**3 - cos(val))/(-3*val**2 - sin(val))
       val = val + delta
       print *, val
    end do

There is a function constructing a loop (or a complete function) like this in
:mod:`sympy.codegen.algorithms`.

"""

from __future__ import annotations
from typing import Any

from collections import defaultdict

from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
                                       numbered_symbols, filter_symbols)


def _mk_Tuple(args):
    """
    Create a SymPy Tuple object from an iterable, converting Python strings to
    AST strings.

    Parameters
    ==========

    args: iterable
        Arguments to :class:`sympy.Tuple`.

    Returns
    =======

    sympy.Tuple
    """
    args = [String(arg) if isinstance(arg, str) else arg for arg in args]
    return Tuple(*args)


class CodegenAST(Basic):
    __slots__ = ()


class Token(CodegenAST):
    """ Base class for the AST types.

    Explanation
    ===========

    Defining fields are set in ``_fields``. Attributes (defined in _fields)
    are only allowed to contain instances of Basic (unless atomic, see
    ``String``). The arguments to ``__new__()`` correspond to the attributes in
    the order defined in ``_fields`. The ``defaults`` class attribute is a
    dictionary mapping attribute names to their default values.

    Subclasses should not need to override the ``__new__()`` method. They may
    define a class or static method named ``_construct_<attr>`` for each
    attribute to process the value passed to ``__new__()``. Attributes listed
    in the class attribute ``not_in_args`` are not passed to :class:`~.Basic`.
    """

    __slots__: tuple[str, ...] = ()
    _fields = __slots__
    defaults: dict[str, Any] = {}
    not_in_args: list[str] = []
    indented_args = ['body']

    @property
    def is_Atom(self):
        return len(self._fields) == 0

    @classmethod
    def _get_constructor(cls, attr):
        """ Get the constructor function for an attribute by name. """
        return getattr(cls, '_construct_%s' % attr, lambda x: x)

    @classmethod
    def _construct(cls, attr, arg):
        """ Construct an attribute value from argument passed to ``__new__()``. """
        # arg may be ``NoneToken()``, so comparison is done using == instead of ``is`` operator
        if arg == None:
            return cls.defaults.get(attr, none)
        else:
            if isinstance(arg, Dummy):  # SymPy's replace uses Dummy instances
                return arg
            else:
                return cls._get_constructor(attr)(arg)

    def __new__(cls, *args, **kwargs):
        # Pass through existing instances when given as sole argument
        if len(args) == 1 and not kwargs and isinstance(args[0], cls):
            return args[0]

        if len(args) > len(cls._fields):
            raise ValueError("Too many arguments (%d), expected at most %d" % (len(args), len(cls._fields)))

        attrvals = []

        # Process positional arguments
        for attrname, argval in zip(cls._fields, args):
            if attrname in kwargs:
                raise TypeError('Got multiple values for attribute %r' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        # Process keyword arguments
        for attrname in cls._fields[len(args):]:
            if attrname in kwargs:
                argval = kwargs.pop(attrname)

            elif attrname in cls.defaults:
                argval = cls.defaults[attrname]

            else:
                raise TypeError('No value for %r given and attribute has no default' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        if kwargs:
            raise ValueError("Unknown keyword arguments: %s" % ' '.join(kwargs))

        # Parent constructor
        basic_args = [
            val for attr, val in zip(cls._fields, attrvals)
            if attr not in cls.not_in_args
        ]
        obj = CodegenAST.__new__(cls, *basic_args)

        # Set attributes
        for attr, arg in zip(cls._fields, attrvals):
            setattr(obj, attr, arg)

        return obj

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for attr in self._fields:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _hashable_content(self):
        return tuple([getattr(self, attr) for attr in self._fields])

    def __hash__(self):
        return super().__hash__()

    def _joiner(self, k, indent_level):
        return (',\n' + ' '*indent_level) if k in self.indented_args else ', '

    def _indented(self, printer, k, v, *args, **kwargs):
        il = printer._context['indent_level']
        def _print(arg):
            if isinstance(arg, Token):
                return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
            else:
                return printer._print(arg, *args, **kwargs)

        if isinstance(v, Tuple):
            joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
            if k in self.indented_args:
                return '(\n' + ' '*il + joined + ',\n' + ' '*(il - 4) + ')'
            else:
                return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
        else:
            return _print(v)

    def _sympyrepr(self, printer, *args, joiner=', ', **kwargs):
        from sympy.printing.printer import printer_context
        exclude = kwargs.get('exclude', ())
        values = [getattr(self, k) for k in self._fields]
        indent_level = printer._context.get('indent_level', 0)

        arg_reprs = []

        for i, (attr, value) in enumerate(zip(self._fields, values)):
            if attr in exclude:
                continue

            # Skip attributes which have the default value
            if attr in self.defaults and value == self.defaults[attr]:
                continue

            ilvl = indent_level + 4 if attr in self.indented_args else 0
            with printer_context(printer, indent_level=ilvl):
                indented = self._indented(printer, attr, value, *args, **kwargs)
            arg_reprs.append(('{1}' if i == 0 else '{0}={1}').format(attr, indented.lstrip()))

        return "{}({})".format(self.__class__.__name__, joiner.join(arg_reprs))

    _sympystr = _sympyrepr

    def __repr__(self):  # sympy.core.Basic.__repr__ uses sstr
        from sympy.printing import srepr
        return srepr(self)

    def kwargs(self, exclude=(), apply=None):
        """ Get instance's attributes as dict of keyword arguments.

        Parameters
        ==========

        exclude : collection of str
            Collection of keywords to exclude.

        apply : callable, optional
            Function to apply to all values.
        """
        kwargs = {k: getattr(self, k) for k in self._fields if k not in exclude}
        if apply is not None:
            return {k: apply(v) for k, v in kwargs.items()}
        else:
            return kwargs

class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """

break_ = BreakToken()


class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """

continue_ = ContinueToken()

class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """
    def __eq__(self, other):
        return other is None or isinstance(other, NoneToken)

    def _hashable_content(self):
        return ()

    def __hash__(self):
        return super().__hash__()


none = NoneToken()


class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)

        cls._check_args(lhs, rhs)

        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @classmethod
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable,
                ArrayElement)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        # Indexed types implement shape, but don't define it until later. This
        # causes issues in assignment validation. For now, matrices are defined
        # as anything with a shape that is not an Indexed
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

        # If lhs and rhs have same structure, then this assignment is ok
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs do not align.")
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")


class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation.

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.codegen.ast import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """

    op = ':='


class AugmentedAssignment(AssignmentBase):
    """
    Base class for augmented assignments.

    Attributes:
    ===========

    binop : str
       Symbol for binary operation being applied in the assignment, such as "+",
       "*", etc.
    """
    binop: str | None

    @property
    def op(self):
        return self.binop + '='


class AddAugmentedAssignment(AugmentedAssignment):
    binop = '+'


class SubAugmentedAssignment(AugmentedAssignment):
    binop = '-'


class MulAugmentedAssignment(AugmentedAssignment):
    binop = '*'


class DivAugmentedAssignment(AugmentedAssignment):
    binop = '/'


class ModAugmentedAssignment(AugmentedAssignment):
    binop = '%'


# Mapping from binary op strings to AugmentedAssignment subclasses
augassign_classes = {
    cls.binop: cls for cls in [
        AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
        DivAugmentedAssignment, ModAugmentedAssignment
    ]
}


def aug_assign(lhs, op, rhs):
    """
    Create 'lhs op= rhs'.

    Explanation
    ===========

    Represents augmented variable assignment for code generation. This is a
    convenience function. You can also use the AugmentedAssignment classes
    directly, like AddAugmentedAssignment(x, y).

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : str
        Operator (+, -, /, \\*, %).

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.codegen.ast import aug_assign
    >>> x, y = symbols('x, y')
    >>> aug_assign(x, '+', y)
    AddAugmentedAssignment(x, y)
    """
    if op not in augassign_classes:
        raise ValueError("Unrecognized operator %s" % op)
    return augassign_classes[op](lhs, rhs)


class CodeBlock(CodegenAST):
    """
    Represents a block of code.

    Explanation
    ===========

    For now only assignments are supported. This restriction will be lifted in
    the future.

    Useful attributes on this object are:

    ``left_hand_sides``:
        Tuple of left-hand sides of assignments, in order.
    ``left_hand_sides``:
        Tuple of right-hand sides of assignments, in order.
    ``free_symbols``: Free symbols of the expressions in the right-hand sides
        which do not appear in the left-hand side of an assignment.

    Useful methods on this object are:

    ``topological_sort``:
        Class method. Return a CodeBlock with assignments
        sorted so that variables are assigned before they
        are used.
    ``cse``:
        Return a new CodeBlock with common subexpressions eliminated and
        pulled out as assignments.

    Examples
    ========

    >>> from sympy import symbols, ccode
    >>> from sympy.codegen.ast import CodeBlock, Assignment
    >>> x, y = symbols('x y')
    >>> c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    >>> print(ccode(c))
    x = 1;
    y = x + 1;

    """
    def __new__(cls, *args):
        left_hand_sides = []
        right_hand_sides = []
        for i in args:
            if isinstance(i, Assignment):
                lhs, rhs = i.args
                left_hand_sides.append(lhs)
                right_hand_sides.append(rhs)

        obj = CodegenAST.__new__(cls, *args)

        obj.left_hand_sides = Tuple(*left_hand_sides)
        obj.right_hand_sides = Tuple(*right_hand_sides)
        return obj

    def __iter__(self):
        return iter(self.args)

    def _sympyrepr(self, printer, *args, **kwargs):
        il = printer._context.get('indent_level', 0)
        joiner = ',\n' + ' '*il
        joined = joiner.join(map(printer._print, self.args))
        return ('{}(\n'.format(' '*(il-4) + self.__class__.__name__,) +
                ' '*il + joined + '\n' + ' '*(il - 4) + ')')

    _sympystr = _sympyrepr

    @property
    def free_symbols(self):
        return super().free_symbols - set(self.left_hand_sides)

    @classmethod
    def topological_sort(cls, assignments):
        """
        Return a CodeBlock with topologically sorted assignments so that
        variables are assigned before they are used.

        Examples
        ========

        The existing order of assignments is preserved as much as possible.

        This function assumes that variables are assigned to only once.

        This is a class constructor so that the default constructor for
        CodeBlock can error when variables are used before they are assigned.

        >>> from sympy import symbols
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> assignments = [
        ...     Assignment(x, y + z),
        ...     Assignment(y, z + 1),
        ...     Assignment(z, 2),
        ... ]
        >>> CodeBlock.topological_sort(assignments)
        CodeBlock(
            Assignment(z, 2),
            Assignment(y, z + 1),
            Assignment(x, y + z)
        )

        """

        if not all(isinstance(i, Assignment) for i in assignments):
            # Will support more things later
            raise NotImplementedError("CodeBlock.topological_sort only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in assignments):
            raise NotImplementedError("CodeBlock.topological_sort does not yet work with AugmentedAssignments")

        # Create a graph where the nodes are assignments and there is a directed edge
        # between nodes that use a variable and nodes that assign that
        # variable, like

        # [(x := 1, y := x + 1), (x := 1, z := y + z), (y := x + 1, z := y + z)]

        # If we then topologically sort these nodes, they will be in
        # assignment order, like

        # x := 1
        # y := x + 1
        # z := y + z

        # A = The nodes
        #
        # enumerate keeps nodes in the same order they are already in if
        # possible. It will also allow us to handle duplicate assignments to
        # the same variable when those are implemented.
        A = list(enumerate(assignments))

        # var_map = {variable: [nodes for which this variable is assigned to]}
        # like {x: [(1, x := y + z), (4, x := 2 * w)], ...}
        var_map = defaultdict(list)
        for node in A:
            i, a = node
            var_map[a.lhs].append(node)

        # E = Edges in the graph
        E = []
        for dst_node in A:
            i, a = dst_node
            for s in a.rhs.free_symbols:
                for src_node in var_map[s]:
                    E.append((src_node, dst_node))

        ordered_assignments = topological_sort([A, E])

        # De-enumerate the result
        return cls(*[a for i, a in ordered_assignments])

    def cse(self, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
        """
        Return a new code block with common subexpressions eliminated.

        Explanation
        ===========

        See the docstring of :func:`sympy.simplify.cse_main.cse` for more
        information.

        Examples
        ========

        >>> from sympy import symbols, sin
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> c = CodeBlock(
        ...     Assignment(x, 1),
        ...     Assignment(y, sin(x) + 1),
        ...     Assignment(z, sin(x) - 1),
        ... )
        ...
        >>> c.cse()
        CodeBlock(
            Assignment(x, 1),
            Assignment(x0, sin(x)),
            Assignment(y, x0 + 1),
            Assignment(z, x0 - 1)
        )

        """
        from sympy.simplify.cse_main import cse

        # Check that the CodeBlock only contains assignments to unique variables
        if not all(isinstance(i, Assignment) for i in self.args):
            # Will support more things later
            raise NotImplementedError("CodeBlock.cse only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in self.args):
            raise NotImplementedError("CodeBlock.cse does not yet work with AugmentedAssignments")

        for i, lhs in enumerate(self.left_hand_sides):
            if lhs in self.left_hand_sides[:i]:
                raise NotImplementedError("Duplicate assignments to the same "
                    "variable are not yet supported (%s)" % lhs)

        # Ensure new symbols for subexpressions do not conflict with existing
        existing_symbols = self.atoms(Symbol)
        if symbols is None:
            symbols = numbered_symbols()
        symbols = filter_symbols(symbols, existing_symbols)

        replacements, reduced_exprs = cse(list(self.right_hand_sides),
            symbols=symbols, optimizations=optimizations, postprocess=postprocess,
            order=order)

        new_block = [Assignment(var, expr) for var, expr in
            zip(self.left_hand_sides, reduced_exprs)]
        new_assignments = [Assignment(var, expr) for var, expr in replacements]
        return self.topological_sort(new_assignments + new_block)


class For(Token):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ==========

    target : symbol
    iter : iterable
    body : CodeBlock or iterable
!        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Range
    >>> from sympy.codegen.ast import aug_assign, For
    >>> x, i, j, k = symbols('x i j k')
    >>> for_i = For(i, Range(10), [aug_assign(x, '+', i*j*k)])
    >>> for_i  # doctest: -NORMALIZE_WHITESPACE
    For(i, iterable=Range(0, 10, 1), body=CodeBlock(
        AddAugmentedAssignment(x, i*j*k)
    ))
    >>> for_ji = For(j, Range(7), [for_i])
    >>> for_ji  # doctest: -NORMALIZE_WHITESPACE
    For(j, iterable=Range(0, 7, 1), body=CodeBlock(
        For(i, iterable=Range(0, 10, 1), body=CodeBlock(
            AddAugmentedAssignment(x, i*j*k)
        ))
    ))
    >>> for_kji =For(k, Range(5), [for_ji])
    >>> for_kji  # doctest: -NORMALIZE_WHITESPACE
    For(k, iterable=Range(0, 5, 1), body=CodeBlock(
        For(j, iterable=Range(0, 7, 1), body=CodeBlock(
            For(i, iterable=Range(0, 10, 1), body=CodeBlock(
                AddAugmentedAssignment(x, i*j*k)
            ))
        ))
    ))
    """
    __slots__ = _fields = ('target', 'iterable', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_iterable(cls, itr):
        if not iterable(itr):
            raise TypeError("iterable must be an iterable")
        if isinstance(itr, list):  # _sympify errors on lists because they are mutable
            itr = tuple(itr)
        return _sympify(itr)


class String(Atom, Token):
    """ SymPy object representing a string.

    Atomic object which is not an expression (as opposed to Symbol).

    Parameters
    ==========

    text : str

    Examples
    ========

    >>> from sympy.codegen.ast import String
    >>> f = String('foo')
    >>> f
    foo
    >>> str(f)
    'foo'
    >>> f.text
    'foo'
    >>> print(repr(f))
    String('foo')

    """
    __slots__ = _fields = ('text',)
    not_in_args = ['text']
    is_Atom = True

    @classmethod
    def _construct_text(cls, text):
        if not isinstance(text, str):
            raise TypeError("Argument text is not a string type.")
        return text

    def _sympystr(self, printer, *args, **kwargs):
        return self.text

    def kwargs(self, exclude = (), apply = None):
        return {}

    #to be removed when Atom is given a suitable func
    @property
    def func(self):
        return lambda: self

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        return r'\texttt{{"{}"}}'.format(latex_escape(self.text))

class QuotedString(String):
    """ Represents a string which should be printed with quotes. """

class Comment(String):
    """ Represents a comment. """

class Node(Token):
    """ Subclass of Token, carrying the attribute 'attrs' (Tuple)

    Examples
    ========

    >>> from sympy.codegen.ast import Node, value_const, pointer_const
    >>> n1 = Node([value_const])
    >>> n1.attr_params('value_const')  # get the parameters of attribute (by name)
    ()
    >>> from sympy.codegen.fnodes import dimension
    >>> n2 = Node([value_const, dimension(5, 3)])
    >>> n2.attr_params(value_const)  # get the parameters of attribute (by Attribute instance)
    ()
    >>> n2.attr_params('dimension')  # get the parameters of attribute (by name)
    (5, 3)
    >>> n2.attr_params(pointer_const) is None
    True

    """

    __slots__: tuple[str, ...] = ('attrs',)
    _fields = __slots__

    defaults: dict[str, Any] = {'attrs': Tuple()}

    _construct_attrs = staticmethod(_mk_Tuple)

    def attr_params(self, looking_for):
        """ Returns the parameters of the Attribute with name ``looking_for`` in self.attrs """
        for attr in self.attrs:
            if str(attr.name) == str(looking_for):
                return attr.parameters


class Type(Token):
    """ Represents a type.

    Explanation
    ===========

    The naming is a super-set of NumPy naming. Type has a classmethod
    ``from_expr`` which offer type deduction. It also has a method
    ``cast_check`` which casts the argument to its type, possibly raising an
    exception if rounding error is not within tolerances, or if the value is not
    representable by the underlying data type (e.g. unsigned integers).

    Parameters
    ==========

    name : str
        Name of the type, e.g. ``object``, ``int16``, ``float16`` (where the latter two
        would use the ``Type`` sub-classes ``IntType`` and ``FloatType`` respectively).
        If a ``Type`` instance is given, the said instance is returned.

    Examples
    ========

    >>> from sympy.codegen.ast import Type
    >>> t = Type.from_expr(42)
    >>> t
    integer
    >>> print(repr(t))
    IntBaseType(String('integer'))
    >>> from sympy.codegen.ast import uint8
    >>> uint8.cast_check(-1)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Minimum value for data type bigger than new value.
    >>> from sympy.codegen.ast import float32
    >>> v6 = 0.123456
    >>> float32.cast_check(v6)
    0.123456
    >>> v10 = 12345.67894
    >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Casting gives a significantly different value.
    >>> boost_mp50 = Type('boost::multiprecision::cpp_dec_float_50')
    >>> from sympy import cxxcode
    >>> from sympy.codegen.ast import Declaration, Variable
    >>> cxxcode(Declaration(Variable('x', type=boost_mp50)))
    'boost::multiprecision::cpp_dec_float_50 x'

    References
    ==========

    .. [1] https://numpy.org/doc/stable/user/basics.types.html

    """
    __slots__: tuple[str, ...] = ('name',)
    _fields = __slots__

    _construct_name = String

    def _sympystr(self, printer, *args, **kwargs):
        return str(self.name)

    @classmethod
    def from_expr(cls, expr):
        """ Deduces type from an expression or a ``Symbol``.

        Parameters
        ==========

        expr : number or SymPy object
            The type will be deduced from type or properties.

        Examples
        ========

        >>> from sympy.codegen.ast import Type, integer, complex_
        >>> Type.from_expr(2) == integer
        True
        >>> from sympy import Symbol
        >>> Type.from_expr(Symbol('z', complex=True)) == complex_
        True
        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Could not deduce type from expr.

        Raises
        ======

        ValueError when type deduction fails.

        """
        if isinstance(expr, (float, Float)):
            return real
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return integer
        if getattr(expr, 'is_real', False):
            return real
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return complex_
        if isinstance(expr, bool) or getattr(expr, 'is_Relational', False):
            return bool_
        else:
            raise ValueError("Could not deduce type from expr.")

    def _check(self, value):
        pass

    def cast_check(self, value, rtol=None, atol=0, precision_targets=None):
        """ Casts a value to the data type of the instance.

        Parameters
        ==========

        value : number
        rtol : floating point number
            Relative tolerance. (will be deduced if not given).
        atol : floating point number
            Absolute tolerance (in addition to ``rtol``).
        type_aliases : dict
            Maps substitutions for Type, e.g. {integer: int64, real: float32}

        Examples
        ========

        >>> from sympy.codegen.ast import integer, float32, int8
        >>> integer.cast_check(3.0) == 3
        True
        >>> float32.cast_check(1e-40)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Minimum value for data type bigger than new value.
        >>> int8.cast_check(256)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Maximum value for data type smaller than new value.
        >>> v10 = 12345.67894
        >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float64
        >>> float64.cast_check(v10)
        12345.67894
        >>> from sympy import Float
        >>> v18 = Float('0.123456789012345646')
        >>> float64.cast_check(v18)
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float80
        >>> float80.cast_check(v18)
        0.123456789012345649

        """
        val = sympify(value)

        ten = Integer(10)
        exp10 = getattr(self, 'decimal_dig', None)

        if rtol is None:
            rtol = 1e-15 if exp10 is None else 2.0*ten**(-exp10)

        def tol(num):
            return atol + rtol*abs(num)

        new_val = self.cast_nocheck(value)
        self._check(new_val)

        delta = new_val - val
        if abs(delta) > tol(val):  # rounding, e.g. int(3.5) != 3.5
            raise ValueError("Casting gives a significantly different value.")

        return new_val

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        type_name = latex_escape(self.__class__.__name__)
        name = latex_escape(self.name.text)
        return r"\text{{{}}}\left(\texttt{{{}}}\right)".format(type_name, name)


class IntBaseType(Type):
    """ Integer base type, contains no size information. """
    __slots__ = ()
    cast_nocheck = lambda self, i: Integer(int(i))


class _SizedIntType(IntBaseType):
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__

    _construct_nbits = Integer

    def _check(self, value):
        if value < self.min:
            raise ValueError("Value is too small: %d < %d" % (value, self.min))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))


class SignedIntType(_SizedIntType):
    """ Represents a signed integer type. """
    __slots__ = ()
    @property
    def min(self):
        return -2**(self.nbits-1)

    @property
    def max(self):
        return 2**(self.nbits-1) - 1


class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()
    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 2**self.nbits - 1

two = Integer(2)

class FloatBaseType(Type):
    """ Represents a floating point number type. """
    __slots__ = ()
    cast_nocheck = Float

class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """

    __slots__ = ('nbits', 'nmant', 'nexp',)
    _fields = Type._fields + __slots__

    _construct_nbits = _construct_nmant = _construct_nexp = Integer


    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        # cf. C++'s ``std::numeric_limits::max_exponent``
        return two**(self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        # cf. C++'s ``std::numeric_limits::min_exponent``
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        return (1 - two**-(self.nmant+1))*two**self.max_exponent

    @property
    def tiny(self):
        """ The minimum positive normalized value. """
        # See C macros: FLT_MIN, DBL_MIN, LDBL_MIN
        # or C++'s ``std::numeric_limits::min``
        # or numpy.finfo(dtype).tiny
        return two**(self.min_exponent - 1)


    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        return two**(-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        from sympy.functions import floor, log
        return floor(self.nmant * log(2)/log(10))

    @property
    def decimal_dig(self):
        """ Number of digits needed to store & load without loss.

        Explanation
        ===========

        Number of decimal digits needed to guarantee that two consecutive conversions
        (float -> text -> float) to be idempotent. This is useful when one do not want
        to loose precision due to rounding errors when storing a floating point value
        as text.
        """
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2)/log(10) + 1)

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        if value == oo:  # float(oo) or oo
            return float(oo)
        elif value == -oo:  # float(-oo) or -oo
            return float(-oo)
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        if value < -self.max:
            raise ValueError("Value is too small: %d < %d" % (value, -self.max))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))
        if abs(value) < self.tiny:
            raise ValueError("Smallest (absolute) value for data type bigger than new value.")

class ComplexBaseType(FloatBaseType):

    __slots__ = ()

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        from sympy.functions import re, im
        return (
            super().cast_nocheck(re(value)) +
            super().cast_nocheck(im(value))*1j
        )

    def _check(self, value):
        from sympy.functions import re, im
        super()._check(re(value))
        super()._check(im(value))


class ComplexType(ComplexBaseType, FloatType):
    """ Represents a complex floating point number. """
    __slots__ = ()


# NumPy types:
intc = IntBaseType('intc')
intp = IntBaseType('intp')
int8 = SignedIntType('int8', 8)
int16 = SignedIntType('int16', 16)
int32 = SignedIntType('int32', 32)
int64 = SignedIntType('int64', 64)
uint8 = UnsignedIntType('uint8', 8)
uint16 = UnsignedIntType('uint16', 16)
uint32 = UnsignedIntType('uint32', 32)
uint64 = UnsignedIntType('uint64', 64)
float16 = FloatType('float16', 16, nexp=5, nmant=10)  # IEEE 754 binary16, Half precision
float32 = FloatType('float32', 32, nexp=8, nmant=23)  # IEEE 754 binary32, Single precision
float64 = FloatType('float64', 64, nexp=11, nmant=52)  # IEEE 754 binary64, Double precision
float80 = FloatType('float80', 80, nexp=15, nmant=63)  # x86 extended precision (1 integer part bit), "long double"
float128 = FloatType('float128', 128, nexp=15, nmant=112)  # IEEE 754 binary128, Quadruple precision
float256 = FloatType('float256', 256, nexp=19, nmant=236)  # IEEE 754 binary256, Octuple precision

complex64 = ComplexType('complex64', nbits=64, **float32.kwargs(exclude=('name', 'nbits')))
complex128 = ComplexType('complex128', nbits=128, **float64.kwargs(exclude=('name', 'nbits')))

# Generic types (precision may be chosen by code printers):
untyped = Type('untyped')
real = FloatBaseType('real')
integer = IntBaseType('integer')
complex_ = ComplexBaseType('complex')
bool_ = Type('bool')


class Attribute(Token):
    """ Attribute (possibly parametrized)

    For use with :class:`sympy.codegen.ast.Node` (which takes instances of
    ``Attribute`` as ``attrs``).

    Parameters
    ==========

    name : str
    parameters : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import Attribute
    >>> volatile = Attribute('volatile')
    >>> volatile
    volatile
    >>> print(repr(volatile))
    Attribute(String('volatile'))
    >>> a = Attribute('foo', [1, 2, 3])
    >>> a
    foo(1, 2, 3)
    >>> a.parameters == (1, 2, 3)
    True
    """
    __slots__ = _fields = ('name', 'parameters')
    defaults = {'parameters': Tuple()}

    _construct_name = String
    _construct_parameters = staticmethod(_mk_Tuple)

    def _sympystr(self, printer, *args, **kwargs):
        result = str(self.name)
        if self.parameters:
            result += '(%s)' % ', '.join((printer._print(
                arg, *args, **kwargs) for arg in self.parameters))
        return result

value_const = Attribute('value_const')
pointer_const = Attribute('pointer_const')


class Variable(Node):
    """ Represents a variable.

    Parameters
    ==========

    symbol : Symbol
    type : Type (optional)
        Type of the variable.
    attrs : iterable of Attribute instances
        Will be stored as a Tuple.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, float32, integer
    >>> x = Symbol('x')
    >>> v = Variable(x, type=float32)
    >>> v.attrs
    ()
    >>> v == Variable('x')
    False
    >>> v == Variable('x', type=float32)
    True
    >>> v
    Variable(x, type=float32)

    One may also construct a ``Variable`` instance with the type deduced from
    assumptions about the symbol using the ``deduced`` classmethod:

    >>> i = Symbol('i', integer=True)
    >>> v = Variable.deduced(i)
    >>> v.type == integer
    True
    >>> v == Variable('i')
    False
    >>> from sympy.codegen.ast import value_const
    >>> value_const in v.attrs
    False
    >>> w = Variable('w', attrs=[value_const])
    >>> w
    Variable(w, attrs=(value_const,))
    >>> value_const in w.attrs
    True
    >>> w.as_Declaration(value=42)
    Declaration(Variable(w, value=42, attrs=(value_const,)))

    """

    __slots__ = ('symbol', 'type', 'value')
    _fields = __slots__ + Node._fields

    defaults = Node.defaults.copy()
    defaults.update({'type': untyped, 'value': none})

    _construct_symbol = staticmethod(sympify)
    _construct_value = staticmethod(sympify)

    @classmethod
    def deduced(cls, symbol, value=None, attrs=Tuple(), cast_check=True):
        """ Alt. constructor with type deduction from ``Type.from_expr``.

        Deduces type primarily from ``symbol``, secondarily from ``value``.

        Parameters
        ==========

        symbol : Symbol
        value : expr
            (optional) value of the variable.
        attrs : iterable of Attribute instances
        cast_check : bool
            Whether to apply ``Type.cast_check`` on ``value``.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.codegen.ast import Variable, complex_
        >>> n = Symbol('n', integer=True)
        >>> str(Variable.deduced(n).type)
        'integer'
        >>> x = Symbol('x', real=True)
        >>> v = Variable.deduced(x)
        >>> v.type
        real
        >>> z = Symbol('z', complex=True)
        >>> Variable.deduced(z).type == complex_
        True

        """
        if isinstance(symbol, Variable):
            return symbol

        try:
            type_ = Type.from_expr(symbol)
        except ValueError:
            type_ = Type.from_expr(value)

        if value is not None and cast_check:
            value = type_.cast_check(value)
        return cls(symbol, type=type_, value=value, attrs=attrs)

    def as_Declaration(self, **kwargs):
        """ Convenience method for creating a Declaration instance.

        Explanation
        ===========

        If the variable of the Declaration need to wrap a modified
        variable keyword arguments may be passed (overriding e.g.
        the ``value`` of the Variable instance).

        Examples
        ========

        >>> from sympy.codegen.ast import Variable, NoneToken
        >>> x = Variable('x')
        >>> decl1 = x.as_Declaration()
        >>> # value is special NoneToken() which must be tested with == operator
        >>> decl1.variable.value is None  # won't work
        False
        >>> decl1.variable.value == None  # not PEP-8 compliant
        True
        >>> decl1.variable.value == NoneToken()  # OK
        True
        >>> decl2 = x.as_Declaration(value=42.0)
        >>> decl2.variable.value == 42.0
        True

        """
        kw = self.kwargs()
        kw.update(kwargs)
        return Declaration(self.func(**kw))

    def _relation(self, rhs, op):
        try:
            rhs = _sympify(rhs)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, rhs))
        return op(self, rhs, evaluate=False)

    __lt__ = lambda self, other: self._relation(other, Lt)
    __le__ = lambda self, other: self._relation(other, Le)
    __ge__ = lambda self, other: self._relation(other, Ge)
    __gt__ = lambda self, other: self._relation(other, Gt)

class Pointer(Variable):
    """ Represents a pointer. See ``Variable``.

    Examples
    ========

    Can create instances of ``Element``:

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Pointer
    >>> i = Symbol('i', integer=True)
    >>> p = Pointer('x')
    >>> p[i+1]
    Element(x, indices=(i + 1,))

    """
    __slots__ = ()

    def __getitem__(self, key):
        try:
            return Element(self.symbol, key)
        except TypeError:
            return Element(self.symbol, (key,))


class Element(Token):
    """ Element in (a possibly N-dimensional) array.

    Examples
    ========

    >>> from sympy.codegen.ast import Element
    >>> elem = Element('x', 'ijk')
    >>> elem.symbol.name == 'x'
    True
    >>> elem.indices
    (i, j, k)
    >>> from sympy import ccode
    >>> ccode(elem)
    'x[i][j][k]'
    >>> ccode(Element('x', 'ijk', strides='lmn', offset='o'))
    'x[i*l + j*m + k*n + o]'

    """
    __slots__ = _fields = ('symbol', 'indices', 'strides', 'offset')
    defaults = {'strides': none, 'offset': none}
    _construct_symbol = staticmethod(sympify)
    _construct_indices = staticmethod(lambda arg: Tuple(*arg))
    _construct_strides = staticmethod(lambda arg: Tuple(*arg))
    _construct_offset = staticmethod(sympify)


class Declaration(Token):
    """ Represents a variable declaration

    Parameters
    ==========

    variable : Variable

    Examples
    ========

    >>> from sympy.codegen.ast import Declaration, NoneToken, untyped
    >>> z = Declaration('z')
    >>> z.variable.type == untyped
    True
    >>> # value is special NoneToken() which must be tested with == operator
    >>> z.variable.value is None  # won't work
    False
    >>> z.variable.value == None  # not PEP-8 compliant
    True
    >>> z.variable.value == NoneToken()  # OK
    True
    """
    __slots__ = _fields = ('variable',)
    _construct_variable = Variable


class While(Token):
    """ Represents a 'for-loop' in the code.

    Expressions are of the form:
        "while condition:
             body..."

    Parameters
    ==========

    condition : expression convertible to Boolean
    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Gt, Abs
    >>> from sympy.codegen import aug_assign, Assignment, While
    >>> x, dx = symbols('x dx')
    >>> expr = 1 - x**2
    >>> whl = While(Gt(Abs(dx), 1e-9), [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx)
    ... ])

    """
    __slots__ = _fields = ('condition', 'body')
    _construct_condition = staticmethod(lambda cond: _sympify(cond))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Scope(Token):
    """ Represents a scope in the code.

    Parameters
    ==========

    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    """
    __slots__ = _fields = ('body',)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Stream(Token):
    """ Represents a stream.

    There are two predefined Stream instances ``stdout`` & ``stderr``.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import pycode, Symbol
    >>> from sympy.codegen.ast import Print, stderr, QuotedString
    >>> print(pycode(Print(['x'], file=stderr)))
    print(x, file=sys.stderr)
    >>> x = Symbol('x')
    >>> print(pycode(Print([QuotedString('x')], file=stderr)))  # print literally "x"
    print("x", file=sys.stderr)

    """
    __slots__ = _fields = ('name',)
    _construct_name = String

stdout = Stream('stdout')
stderr = Stream('stderr')


class Print(Token):
    r""" Represents print command in the code.

    Parameters
    ==========

    formatstring : str
    *args : Basic instances (or convertible to such through sympify)

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy import pycode
    >>> print(pycode(Print('x y'.split(), "coordinate: %12.5g %12.5g\\n")))
    print("coordinate: %12.5g %12.5g\n" % (x, y), end="")

    """

    __slots__ = _fields = ('print_args', 'format_string', 'file')
    defaults = {'format_string': none, 'file': none}

    _construct_print_args = staticmethod(_mk_Tuple)
    _construct_format_string = QuotedString
    _construct_file = Stream


class FunctionPrototype(Node):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """

    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields

    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError("func_def is not an instance of FunctionDefinition")
        return cls(**func_def.kwargs(exclude=('body',)))


class FunctionDefinition(FunctionPrototype):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    body : CodeBlock or iterable
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'
    >>> from sympy.codegen.ast import FunctionDefinition, Return
    >>> body = [Return(x*y)]
    >>> fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    >>> print(ccode(fd))
    double foo(double x, double y){
        return x*y;
    }
    """

    __slots__ = ('body', )
    _fields = FunctionPrototype._fields[:-1] + __slots__ + Node._fields

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def from_FunctionPrototype(cls, func_proto, body):
        if not isinstance(func_proto, FunctionPrototype):
            raise TypeError("func_proto is not an instance of FunctionPrototype")
        return cls(body=body, **func_proto.kwargs())


class Return(Token):
    """ Represents a return command in the code.

    Parameters
    ==========

    return : Basic

    Examples
    ========

    >>> from sympy.codegen.ast import Return
    >>> from sympy.printing.pycode import pycode
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> print(pycode(Return(x)))
    return x

    """
    __slots__ = _fields = ('return',)
    _construct_return=staticmethod(_sympify)


class FunctionCall(Token, Expr):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str
    function_args : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)

    """
    __slots__ = _fields = ('name', 'function_args')

    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))


class Raise(Token):
    """ Prints as 'raise ...' in Python, 'throw ...' in C++"""
    __slots__ = _fields = ('exception',)


class RuntimeError_(Token):
    """ Represents 'std::runtime_error' in C++ and 'RuntimeError' in Python.

    Note that the latter is uncommon, and you might want to use e.g. ValueError.
    """
    __slots__ = _fields = ('message',)
    _construct_message = String
