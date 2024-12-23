"""
Stub module to expose DomainMatrix which has now moved to
sympy.polys.matrices package. It should now be imported as:

    >>> from sympy.polys.matrices import DomainMatrix

This module might be removed in future.
"""

from sympy.polys.matrices.domainmatrix import DomainMatrix

__all__ = ['DomainMatrix']
