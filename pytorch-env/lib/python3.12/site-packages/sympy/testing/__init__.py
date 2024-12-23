"""This module contains code for running the tests in SymPy."""


from .runtests import doctest
from .runtests_pytest import test


__all__ = [
    'test', 'doctest',
]
