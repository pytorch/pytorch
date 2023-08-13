import typing

import sympy as real_sympy

from .cached_sympy_impl import _setup, _unwrap, _wrap  # noqa: F401


Abs = _setup(real_sympy.Abs)
Add = _setup(real_sympy.Add)
And = _setup(real_sympy.And)
Basic = _setup(real_sympy.Basic)
BooleanAtom = _setup(real_sympy.logic.boolalg.BooleanAtom)
Boolean = _setup(real_sympy.logic.boolalg.Boolean)
Eq = _setup(real_sympy.Eq)
Expr = _setup(real_sympy.Expr)
Float = _setup(real_sympy.Float)
Ge = _setup(real_sympy.Ge)
Gt = _setup(real_sympy.Gt)
Integer = _setup(real_sympy.Integer)
Le = _setup(real_sympy.Le)
Lt = _setup(real_sympy.Lt)
Max = _setup(real_sympy.Max)
Min = _setup(real_sympy.Min)
Mod = _setup(real_sympy.Mod)
Mul = _setup(real_sympy.Mul)
Ne = _setup(real_sympy.Ne)
Not = _setup(real_sympy.Not)
Number = _setup(real_sympy.Number)
Or = _setup(real_sympy.Or)
Pow = _setup(real_sympy.Pow)
Rational = _setup(real_sympy.Rational)
Rel = _setup(real_sympy.Rel)
Symbol = _setup(real_sympy.Symbol)
Wild = _setup(real_sympy.Wild)

# Don't add this one, handle custom functions like functions.py does
# Function = _setup(real_sympy.Function)

oo = _setup(real_sympy.oo)
zoo = _setup(real_sympy.zoo)
true = _setup(real_sympy.true)
false = _setup(real_sympy.false)
nan = _setup(real_sympy.nan)

ceiling = _setup(real_sympy.ceiling)
divisors = _setup(real_sympy.divisors)
expand = _setup(real_sympy.expand)
exp = _setup(real_sympy.exp)
floor = _setup(real_sympy.floor)
gcd = _setup(real_sympy.gcd)
log = _setup(real_sympy.log)
simplify = _setup(real_sympy.simplify)
sqrt = _setup(real_sympy.sqrt)
symbols = _setup(real_sympy.symbols)
sympify = _setup(real_sympy.sympify)


class logic:
    class boolalg:
        BooleanAtom = _setup(real_sympy.logic.boolalg.BooleanAtom)
        Boolean = _setup(real_sympy.logic.boolalg.Boolean)


class core:
    class numbers:
        Half = _setup(real_sympy.core.numbers.Half)
        Integer = _setup(real_sympy.core.numbers.Integer)


class solvers:
    class inequalities:
        reduce_inequalities = _setup(
            real_sympy.solvers.inequalities.reduce_inequalities
        )


def preorder_traversal(expr):
    """`real_sympy.preorder_traversal` a stateful class, so we can't cache it"""
    for item in real_sympy.preorder_traversal(_unwrap(expr)):
        yield _wrap(item)


if typing.TYPE_CHECKING:  # mypy hackery
    Basic = real_sympy.Basic
    Expr = real_sympy.Expr
    Symbol = real_sympy.Symbol
    Boolean = real_sympy.logic.boolalg.Boolean
