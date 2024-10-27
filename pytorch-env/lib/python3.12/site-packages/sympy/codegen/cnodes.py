"""
AST nodes specific to the C family of languages
"""

from sympy.codegen.ast import (
    Attribute, Declaration, Node, String, Token, Type, none,
    FunctionCall, CodeBlock
    )
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify

void = Type('void')

restrict = Attribute('restrict')  # guarantees no pointer aliasing
volatile = Attribute('volatile')
static = Attribute('static')


def alignof(arg):
    """ Generate of FunctionCall instance for calling 'alignof' """
    return FunctionCall('alignof', [String(arg) if isinstance(arg, str) else arg])


def sizeof(arg):
    """ Generate of FunctionCall instance for calling 'sizeof'

    Examples
    ========

    >>> from sympy.codegen.ast import real
    >>> from sympy.codegen.cnodes import sizeof
    >>> from sympy import ccode
    >>> ccode(sizeof(real))
    'sizeof(double)'
    """
    return FunctionCall('sizeof', [String(arg) if isinstance(arg, str) else arg])


class CommaOperator(Basic):
    """ Represents the comma operator in C """
    def __new__(cls, *args):
        return Basic.__new__(cls, *[sympify(arg) for arg in args])


class Label(Node):
    """ Label for use with e.g. goto statement.

    Examples
    ========

    >>> from sympy import ccode, Symbol
    >>> from sympy.codegen.cnodes import Label, PreIncrement
    >>> print(ccode(Label('foo')))
    foo:
    >>> print(ccode(Label('bar', [PreIncrement(Symbol('a'))])))
    bar:
    ++(a);

    """
    __slots__ = _fields = ('name', 'body')
    defaults = {'body': none}
    _construct_name = String

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class goto(Token):
    """ Represents goto in C """
    __slots__ = _fields = ('label',)
    _construct_label = Label


class PreDecrement(Basic):
    """ Represents the pre-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreDecrement
    >>> from sympy import ccode
    >>> ccode(PreDecrement(x))
    '--(x)'

    """
    nargs = 1


class PostDecrement(Basic):
    """ Represents the post-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostDecrement
    >>> from sympy import ccode
    >>> ccode(PostDecrement(x))
    '(x)--'

    """
    nargs = 1


class PreIncrement(Basic):
    """ Represents the pre-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreIncrement
    >>> from sympy import ccode
    >>> ccode(PreIncrement(x))
    '++(x)'

    """
    nargs = 1


class PostIncrement(Basic):
    """ Represents the post-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostIncrement
    >>> from sympy import ccode
    >>> ccode(PostIncrement(x))
    '(x)++'

    """
    nargs = 1


class struct(Node):
    """ Represents a struct in C """
    __slots__ = _fields = ('name', 'declarations')
    defaults = {'name': none}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        return Tuple(*[Declaration(arg) for arg in args])


class union(struct):
    """ Represents a union in C """
    __slots__ = ()
