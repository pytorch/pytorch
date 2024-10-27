"""
AST nodes specific to C++.
"""

from sympy.codegen.ast import Attribute, String, Token, Type, none

class using(Token):
    """ Represents a 'using' statement in C++ """
    __slots__ = _fields = ('type', 'alias')
    defaults = {'alias': none}
    _construct_type = Type
    _construct_alias = String

constexpr = Attribute('constexpr')
