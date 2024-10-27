"""This module provides containers for python objects that are valid
printing targets but are not a subclass of SymPy's Printable.
"""


from sympy.core.containers import Tuple


class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""
    def __eq__(self, other):
        if isinstance(other, list):
            return self == List(*other)
        else:
            return self.args == other

    def __hash__(self):
        return super().__hash__()
