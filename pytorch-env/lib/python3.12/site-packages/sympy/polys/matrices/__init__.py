"""

sympy.polys.matrices package.

The main export from this package is the DomainMatrix class which is a
lower-level implementation of matrices based on the polys Domains. This
implementation is typically a lot faster than SymPy's standard Matrix class
but is a work in progress and is still experimental.

"""
from .domainmatrix import DomainMatrix, DM

__all__ = [
    'DomainMatrix', 'DM',
]
