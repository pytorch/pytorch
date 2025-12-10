"""Definitions of common exceptions for :mod:`sympy.core` module. """

from typing import Callable


class BaseCoreError(Exception):
    """Base class for core related exceptions. """


class NonCommutativeExpression(BaseCoreError):
    """Raised when expression didn't have commutative property. """


class LazyExceptionMessage:
    """Wrapper class that lets you specify an expensive to compute
    error message that is only evaluated if the error is rendered."""
    callback: Callable[[], str]

    def __init__(self, callback: Callable[[], str]):
        self.callback = callback

    def __str__(self):
        return self.callback()
