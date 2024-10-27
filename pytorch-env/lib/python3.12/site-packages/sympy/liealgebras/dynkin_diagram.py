from .cartan_type import CartanType


def DynkinDiagram(t):
    """Display the Dynkin diagram of a given Lie algebra

    Works by generating the CartanType for the input, t, and then returning the
    Dynkin diagram method from the individual classes.

    Examples
    ========

    >>> from sympy.liealgebras.dynkin_diagram import DynkinDiagram
    >>> print(DynkinDiagram("A3"))
    0---0---0
    1   2   3

    >>> print(DynkinDiagram("B4"))
    0---0---0=>=0
    1   2   3   4

    """

    return CartanType(t).dynkin_diagram()
