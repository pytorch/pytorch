"""
AST nodes specific to Fortran.

The functions defined in this module allows the user to express functions such as ``dsign``
as a SymPy function for symbolic manipulation.
"""

from __future__ import annotations
from sympy.codegen.ast import (
    Attribute, CodeBlock, FunctionCall, Node, none, String,
    Token, _mk_Tuple, Variable
)
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable



pure = Attribute('pure')
elemental = Attribute('elemental')  # (all elemental procedures are also pure)

intent_in = Attribute('intent_in')
intent_out = Attribute('intent_out')
intent_inout = Attribute('intent_inout')

allocatable = Attribute('allocatable')

class Program(Token):
    """ Represents a 'program' block in Fortran.

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Program
    >>> prog = Program('myprogram', [Print([42])])
    >>> from sympy import fcode
    >>> print(fcode(prog, source_format='free'))
    program myprogram
        print *, 42
    end program

    """
    __slots__ = _fields = ('name', 'body')
    _construct_name = String
    _construct_body = staticmethod(lambda body: CodeBlock(*body))


class use_rename(Token):
    """ Represents a renaming in a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use_rename, use
    >>> from sympy import fcode
    >>> ren = use_rename("thingy", "convolution2d")
    >>> print(fcode(ren, source_format='free'))
    thingy => convolution2d
    >>> full = use('signallib', only=['snr', ren])
    >>> print(fcode(full, source_format='free'))
    use signallib, only: snr, thingy => convolution2d

    """
    __slots__ = _fields = ('local', 'original')
    _construct_local = String
    _construct_original = String

def _name(arg):
    if hasattr(arg, 'name'):
        return arg.name
    else:
        return String(arg)

class use(Token):
    """ Represents a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use
    >>> from sympy import fcode
    >>> fcode(use('signallib'), source_format='free')
    'use signallib'
    >>> fcode(use('signallib', [('metric', 'snr')]), source_format='free')
    'use signallib, metric => snr'
    >>> fcode(use('signallib', only=['snr', 'convolution2d']), source_format='free')
    'use signallib, only: snr, convolution2d'

    """
    __slots__ = _fields = ('namespace', 'rename', 'only')
    defaults = {'rename': none, 'only': none}
    _construct_namespace = staticmethod(_name)
    _construct_rename = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else use_rename(*arg) for arg in args]))
    _construct_only = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else _name(arg) for arg in args]))


class Module(Token):
    """ Represents a module in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Module
    >>> from sympy import fcode
    >>> print(fcode(Module('signallib', ['implicit none'], []), source_format='free'))
    module signallib
    implicit none
    <BLANKLINE>
    contains
    <BLANKLINE>
    <BLANKLINE>
    end module

    """
    __slots__ = _fields = ('name', 'declarations', 'definitions')
    defaults = {'declarations': Tuple()}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        args = [Str(arg) if isinstance(arg, str) else arg for arg in args]
        return CodeBlock(*args)

    _construct_definitions = staticmethod(lambda arg: CodeBlock(*arg))


class Subroutine(Node):
    """ Represents a subroutine in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Subroutine
    >>> x, y = symbols('x y', real=True)
    >>> sub = Subroutine('mysub', [x, y], [Print([x**2 + y**2, x*y])])
    >>> print(fcode(sub, source_format='free', standard=2003))
    subroutine mysub(x, y)
    real*8 :: x
    real*8 :: y
    print *, x**2 + y**2, x*y
    end subroutine

    """
    __slots__ = ('name', 'parameters', 'body')
    _fields = __slots__ + Node._fields
    _construct_name = String
    _construct_parameters = staticmethod(lambda params: Tuple(*map(Variable.deduced, params)))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

class SubroutineCall(Token):
    """ Represents a call to a subroutine in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import SubroutineCall
    >>> from sympy import fcode
    >>> fcode(SubroutineCall('mysub', 'x y'.split()))
    '       call mysub(x, y)'

    """
    __slots__ = _fields = ('name', 'subroutine_args')
    _construct_name = staticmethod(_name)
    _construct_subroutine_args = staticmethod(_mk_Tuple)


class Do(Token):
    """ Represents a Do loop in in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import aug_assign, Print
    >>> from sympy.codegen.fnodes import Do
    >>> i, n = symbols('i n', integer=True)
    >>> r = symbols('r', real=True)
    >>> body = [aug_assign(r, '+', 1/i), Print([i, r])]
    >>> do1 = Do(body, i, 1, n)
    >>> print(fcode(do1, source_format='free'))
    do i = 1, n
        r = r + 1d0/i
        print *, i, r
    end do
    >>> do2 = Do(body, i, 1, n, 2)
    >>> print(fcode(do2, source_format='free'))
    do i = 1, n, 2
        r = r + 1d0/i
        print *, i, r
    end do

    """

    __slots__ = _fields = ('body', 'counter', 'first', 'last', 'step', 'concurrent')
    defaults = {'step': Integer(1), 'concurrent': false}
    _construct_body = staticmethod(lambda body: CodeBlock(*body))
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)
    _construct_concurrent = staticmethod(lambda arg: true if arg else false)


class ArrayConstructor(Token):
    """ Represents an array constructor.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ArrayConstructor
    >>> ac = ArrayConstructor([1, 2, 3])
    >>> fcode(ac, standard=95, source_format='free')
    '(/1, 2, 3/)'
    >>> fcode(ac, standard=2003, source_format='free')
    '[1, 2, 3]'

    """
    __slots__ = _fields = ('elements',)
    _construct_elements = staticmethod(_mk_Tuple)


class ImpliedDoLoop(Token):
    """ Represents an implied do loop in Fortran.

    Examples
    ========

    >>> from sympy import Symbol, fcode
    >>> from sympy.codegen.fnodes import ImpliedDoLoop, ArrayConstructor
    >>> i = Symbol('i', integer=True)
    >>> idl = ImpliedDoLoop(i**3, i, -3, 3, 2)  # -27, -1, 1, 27
    >>> ac = ArrayConstructor([-28, idl, 28]) # -28, -27, -1, 1, 27, 28
    >>> fcode(ac, standard=2003, source_format='free')
    '[-28, (i**3, i = -3, 3, 2), 28]'

    """
    __slots__ = _fields = ('expr', 'counter', 'first', 'last', 'step')
    defaults = {'step': Integer(1)}
    _construct_expr = staticmethod(sympify)
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)


class Extent(Basic):
    """ Represents a dimension extent.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Extent
    >>> e = Extent(-3, 3)  # -3, -2, -1, 0, 1, 2, 3
    >>> from sympy import fcode
    >>> fcode(e, source_format='free')
    '-3:3'
    >>> from sympy.codegen.ast import Variable, real
    >>> from sympy.codegen.fnodes import dimension, intent_out
    >>> dim = dimension(e, e)
    >>> arr = Variable('x', real, attrs=[dim, intent_out])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'real*8, dimension(-3:3, -3:3), intent(out) :: x'

    """
    def __new__(cls, *args):
        if len(args) == 2:
            low, high = args
            return Basic.__new__(cls, sympify(low), sympify(high))
        elif len(args) == 0 or (len(args) == 1 and args[0] in (':', None)):
            return Basic.__new__(cls)  # assumed shape
        else:
            raise ValueError("Expected 0 or 2 args (or one argument == None or ':')")

    def _sympystr(self, printer):
        if len(self.args) == 0:
            return ':'
        return ":".join(str(arg) for arg in self.args)

assumed_extent = Extent() # or Extent(':'), Extent(None)


def dimension(*args):
    """ Creates a 'dimension' Attribute with (up to 7) extents.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import dimension, intent_in
    >>> dim = dimension('2', ':')  # 2 rows, runtime determined number of columns
    >>> from sympy.codegen.ast import Variable, integer
    >>> arr = Variable('a', integer, attrs=[dim, intent_in])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'integer*4, dimension(2, :), intent(in) :: a'

    """
    if len(args) > 7:
        raise ValueError("Fortran only supports up to 7 dimensional arrays")
    parameters = []
    for arg in args:
        if isinstance(arg, Extent):
            parameters.append(arg)
        elif isinstance(arg, str):
            if arg == ':':
                parameters.append(Extent())
            else:
                parameters.append(String(arg))
        elif iterable(arg):
            parameters.append(Extent(*arg))
        else:
            parameters.append(sympify(arg))
    if len(args) == 0:
        raise ValueError("Need at least one dimension")
    return Attribute('dimension', parameters)


assumed_size = dimension('*')

def array(symbol, dim, intent=None, *, attrs=(), value=None, type=None):
    """ Convenience function for creating a Variable instance for a Fortran array.

    Parameters
    ==========

    symbol : symbol
    dim : Attribute or iterable
        If dim is an ``Attribute`` it need to have the name 'dimension'. If it is
        not an ``Attribute``, then it is passed to :func:`dimension` as ``*dim``
    intent : str
        One of: 'in', 'out', 'inout' or None
    \\*\\*kwargs:
        Keyword arguments for ``Variable`` ('type' & 'value')

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.ast import integer, real
    >>> from sympy.codegen.fnodes import array
    >>> arr = array('a', '*', 'in', type=integer)
    >>> print(fcode(arr.as_Declaration(), source_format='free', standard=2003))
    integer*4, dimension(*), intent(in) :: a
    >>> x = array('x', [3, ':', ':'], intent='out', type=real)
    >>> print(fcode(x.as_Declaration(value=1), source_format='free', standard=2003))
    real*8, dimension(3, :, :), intent(out) :: x = 1

    """
    if isinstance(dim, Attribute):
        if str(dim.name) != 'dimension':
            raise ValueError("Got an unexpected Attribute argument as dim: %s" % str(dim))
    else:
        dim = dimension(*dim)

    attrs = list(attrs) + [dim]
    if intent is not None:
        if intent not in (intent_in, intent_out, intent_inout):
            intent = {'in': intent_in, 'out': intent_out, 'inout': intent_inout}[intent]
        attrs.append(intent)
    if type is None:
        return Variable.deduced(symbol, value=value, attrs=attrs)
    else:
        return Variable(symbol, type, value=value, attrs=attrs)

def _printable(arg):
    return String(arg) if isinstance(arg, str) else sympify(arg)


def allocated(array):
    """ Creates an AST node for a function call to Fortran's "allocated(...)"

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import allocated
    >>> alloc = allocated('x')
    >>> fcode(alloc, source_format='free')
    'allocated(x)'

    """
    return FunctionCall('allocated', [_printable(array)])


def lbound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "lbound(...)"

    Parameters
    ==========

    array : Symbol or String
    dim : expr
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import lbound
    >>> lb = lbound('arr', dim=2)
    >>> fcode(lb, source_format='free')
    'lbound(arr, 2)'

    """
    return FunctionCall(
        'lbound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def ubound(array, dim=None, kind=None):
    return FunctionCall(
        'ubound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def shape(source, kind=None):
    """ Creates an AST node for a function call to Fortran's "shape(...)"

    Parameters
    ==========

    source : Symbol or String
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import shape
    >>> shp = shape('x')
    >>> fcode(shp, source_format='free')
    'shape(x)'

    """
    return FunctionCall(
        'shape',
        [_printable(source)] +
        ([_printable(kind)] if kind else [])
    )


def size(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "size(...)"

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, size
    >>> a = Symbol('a', real=True)
    >>> body = [Return((sum_(a**2)/size(a))**.5)]
    >>> arr = array(a, dim=[':'], intent='in')
    >>> fd = FunctionDefinition(real, 'rms', [arr], body)
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a)
    real*8, dimension(:), intent(in) :: a
    rms = sqrt(sum(a**2)*1d0/size(a))
    end function

    """
    return FunctionCall(
        'size',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def reshape(source, shape, pad=None, order=None):
    """ Creates an AST node for a function call to Fortran's "reshape(...)"

    Parameters
    ==========

    source : Symbol or String
    shape : ArrayExpr

    """
    return FunctionCall(
        'reshape',
        [_printable(source), _printable(shape)] +
        ([_printable(pad)] if pad else []) +
        ([_printable(order)] if pad else [])
    )


def bind_C(name=None):
    """ Creates an Attribute ``bind_C`` with a name.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, bind_C
    >>> a = Symbol('a', real=True)
    >>> s = Symbol('s', integer=True)
    >>> arr = array(a, dim=[s], intent='in')
    >>> body = [Return((sum_(a**2)/s)**.5)]
    >>> fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a, s) bind(C, name="rms")
    real*8, dimension(s), intent(in) :: a
    integer*4 :: s
    rms = sqrt(sum(a**2)/s)
    end function

    """
    return Attribute('bind_C', [String(name)] if name else [])

class GoTo(Token):
    """ Represents a goto statement in Fortran

    Examples
    ========

    >>> from sympy.codegen.fnodes import GoTo
    >>> go = GoTo([10, 20, 30], 'i')
    >>> from sympy import fcode
    >>> fcode(go, source_format='free')
    'go to (10, 20, 30), i'

    """
    __slots__ = _fields = ('labels', 'expr')
    defaults = {'expr': none}
    _construct_labels = staticmethod(_mk_Tuple)
    _construct_expr = staticmethod(sympify)


class FortranReturn(Token):
    """ AST node explicitly mapped to a fortran "return".

    Explanation
    ===========

    Because a return statement in fortran is different from C, and
    in order to aid reuse of our codegen ASTs the ordinary
    ``.codegen.ast.Return`` is interpreted as assignment to
    the result variable of the function. If one for some reason needs
    to generate a fortran RETURN statement, this node should be used.

    Examples
    ========

    >>> from sympy.codegen.fnodes import FortranReturn
    >>> from sympy import fcode
    >>> fcode(FortranReturn('x'))
    '       return x'

    """
    __slots__ = _fields = ('return_value',)
    defaults = {'return_value': none}
    _construct_return_value = staticmethod(sympify)


class FFunction(Function):
    _required_standard = 77

    def _fcode(self, printer):
        name = self.__class__.__name__
        if printer._settings['standard'] < self._required_standard:
            raise NotImplementedError("%s requires Fortran %d or newer" %
                                      (name, self._required_standard))
        return '{}({})'.format(name, ', '.join(map(printer._print, self.args)))


class F95Function(FFunction):
    _required_standard = 95


class isign(FFunction):
    """ Fortran sign intrinsic for integer arguments. """
    nargs = 2


class dsign(FFunction):
    """ Fortran sign intrinsic for double precision arguments. """
    nargs = 2


class cmplx(FFunction):
    """ Fortran complex conversion function. """
    nargs = 2  # may be extended to (2, 3) at a later point


class kind(FFunction):
    """ Fortran kind function. """
    nargs = 1


class merge(F95Function):
    """ Fortran merge function """
    nargs = 3


class _literal(Float):
    _token: str
    _decimals: int

    def _fcode(self, printer, *args, **kwargs):
        mantissa, sgnd_ex = ('%.{}e'.format(self._decimals) % self).split('e')
        mantissa = mantissa.strip('0').rstrip('.')
        ex_sgn, ex_num = sgnd_ex[0], sgnd_ex[1:].lstrip('0')
        ex_sgn = '' if ex_sgn == '+' else ex_sgn
        return (mantissa or '0') + self._token + ex_sgn + (ex_num or '0')


class literal_sp(_literal):
    """ Fortran single precision real literal """
    _token = 'e'
    _decimals = 9


class literal_dp(_literal):
    """ Fortran double precision real literal """
    _token = 'd'
    _decimals = 17


class sum_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)


class product_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)
