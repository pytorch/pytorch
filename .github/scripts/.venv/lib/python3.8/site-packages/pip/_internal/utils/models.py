"""Utilities for defining models
"""
# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

import operator


class KeyBasedCompareMixin(object):
    """Provides comparison capabilities that is based on a key
    """

    __slots__ = ['_compare_key', '_defining_class']

    def __init__(self, key, defining_class):
        self._compare_key = key
        self._defining_class = defining_class

    def __hash__(self):
        return hash(self._compare_key)

    def __lt__(self, other):
        return self._compare(other, operator.__lt__)

    def __le__(self, other):
        return self._compare(other, operator.__le__)

    def __gt__(self, other):
        return self._compare(other, operator.__gt__)

    def __ge__(self, other):
        return self._compare(other, operator.__ge__)

    def __eq__(self, other):
        return self._compare(other, operator.__eq__)

    def __ne__(self, other):
        return self._compare(other, operator.__ne__)

    def _compare(self, other, method):
        if not isinstance(other, self._defining_class):
            return NotImplemented

        return method(self._compare_key, other._compare_key)
