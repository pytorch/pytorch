"""Implementation of :class:`ExpressionRawDomain` class. """


from sympy.core import Expr, S, sympify, Add
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public


@public
class ExpressionRawDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions but without automatic simplification. """

    is_SymbolicRawDomain = is_EXRAW = True

    dtype = Expr

    zero = S.Zero
    one = S.One

    rep = 'EXRAW'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    @classmethod
    def new(self, a):
        return sympify(a)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        if not isinstance(a, Expr):
            raise CoercionFailed(f"Expecting an Expr instance but found: {type(a).__name__}")
        return a

    def convert_from(self, a, K):
        """Convert a domain element from another domain to EXRAW"""
        return K.to_sympy(a)

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def sum(self, items):
        return Add(*items)


EXRAW = ExpressionRawDomain()
