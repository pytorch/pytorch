from sympy.core import Basic
from sympy.vector.operators import gradient, divergence, curl


class Del(Basic):
    """
    Represents the vector differential operator, usually represented in
    mathematical expressions as the 'nabla' symbol.
    """

    def __new__(cls):
        obj = super().__new__(cls)
        obj._name = "delop"
        return obj

    def gradient(self, scalar_field, doit=False):
        """
        Returns the gradient of the given scalar field, as a
        Vector instance.

        Parameters
        ==========

        scalar_field : SymPy expression
            The scalar field to calculate the gradient of.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> delop.gradient(9)
        0
        >>> delop(C.x*C.y*C.z).doit()
        C.y*C.z*C.i + C.x*C.z*C.j + C.x*C.y*C.k

        """

        return gradient(scalar_field, doit=doit)

    __call__ = gradient
    __call__.__doc__ = gradient.__doc__

    def dot(self, vect, doit=False):
        """
        Represents the dot product between this operator and a given
        vector - equal to the divergence of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose divergence is to be calculated.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> delop = Del()
        >>> C = CoordSys3D('C')
        >>> delop.dot(C.x*C.i)
        Derivative(C.x, C.x)
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> (delop & v).doit()
        C.x*C.y + C.x*C.z + C.y*C.z

        """
        return divergence(vect, doit=doit)

    __and__ = dot
    __and__.__doc__ = dot.__doc__

    def cross(self, vect, doit=False):
        """
        Represents the cross product between this operator and a given
        vector - equal to the curl of the vector field.

        Parameters
        ==========

        vect : Vector
            The vector whose curl is to be calculated.

        doit : bool
            If True, the result is returned after calling .doit() on
            each component. Else, the returned expression contains
            Derivative instances

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> v = C.x*C.y*C.z * (C.i + C.j + C.k)
        >>> delop.cross(v, doit = True)
        (-C.x*C.y + C.x*C.z)*C.i + (C.x*C.y - C.y*C.z)*C.j +
            (-C.x*C.z + C.y*C.z)*C.k
        >>> (delop ^ C.i).doit()
        0

        """

        return curl(vect, doit=doit)

    __xor__ = cross
    __xor__.__doc__ = cross.__doc__

    def _sympystr(self, printer):
        return self._name
