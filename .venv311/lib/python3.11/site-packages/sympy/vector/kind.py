#sympy.vector.kind

from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul

class VectorKind(Kind):
    """
    Kind for all vector objects in SymPy.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the vector contains only numbers.

    Examples
    ========

    Any instance of Vector class has kind ``VectorKind``:

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> Sys = CoordSys3D('Sys')
    >>> Sys.i.kind
    VectorKind(NumberKind)

    Operations between instances of Vector keep also have the kind ``VectorKind``:

    >>> from sympy.core.add import Add
    >>> v1 = Sys.i * 2 + Sys.j * 3 + Sys.k * 4
    >>> v2 = Sys.i * Sys.x + Sys.j * Sys.y + Sys.k * Sys.z
    >>> v1.kind
    VectorKind(NumberKind)
    >>> v2.kind
    VectorKind(NumberKind)
    >>> Add(v1, v2).kind
    VectorKind(NumberKind)

    Subclasses of Vector also have the kind ``VectorKind``, such as
    Cross, VectorAdd, VectorMul or VectorZero.

    See Also
    ========

    sympy.core.kind.Kind
    sympy.matrices.kind.MatrixKind

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "VectorKind(%s)" % self.element_kind

@Mul._kind_dispatcher.register(_NumberKind, VectorKind)
def num_vec_mul(k1, k2):
    """
    The result of a multiplication between a number and a Vector should be of VectorKind.
    The element kind is selected by recursive dispatching.
    """
    if not isinstance(k2, VectorKind):
        k1, k2 = k2, k1
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return VectorKind(elemk)
