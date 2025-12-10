from __future__ import annotations
from itertools import product

from sympy.core import Add, Basic
from sympy.core.assumptions import StdFactKB
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.basisdependent import (BasisDependentZero,
    BasisDependent, BasisDependentMul, BasisDependentAdd)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd
from sympy.vector.kind import VectorKind


class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """

    is_scalar = False
    is_Vector = True
    _op_priority = 12.0

    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero

    kind: VectorKind = VectorKind()

    @property
    def components(self):
        """
        Returns the components of this vector in the form of a
        Python dictionary mapping BaseVector instances to the
        corresponding measure numbers.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.components
        {C.i: 3, C.j: 4, C.k: 5}

        """
        # The '_components' attribute is defined according to the
        # subclass of Vector the instance belongs to.
        return self._components

    def magnitude(self):
        """
        Returns the magnitude of this vector.
        """
        return sqrt(self & self)

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self / self.magnitude()

    def equals(self, other):
        """
        Check if ``self`` and ``other`` are identically equal vectors.

        Explanation
        ===========

        Checks if two vector expressions are equal for all possible values of
        the symbols present in the expressions.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy.abc import x, y
        >>> from sympy import pi
        >>> C = CoordSys3D('C')

        Compare vectors that are equal or not:

        >>> C.i.equals(C.j)
        False
        >>> C.i.equals(C.i)
        True

        These two vectors are equal if `x = y` but are not identically equal
        as expressions since for some values of `x` and `y` they are unequal:

        >>> v1 = x*C.i + C.j
        >>> v2 = y*C.i + C.j
        >>> v1.equals(v1)
        True
        >>> v1.equals(v2)
        False

        Vectors from different coordinate systems can be compared:

        >>> D = C.orient_new_axis('D', pi/2, C.i)
        >>> D.j.equals(C.j)
        False
        >>> D.j.equals(C.k)
        True

        Parameters
        ==========

        other: Vector
            The other vector expression to compare with.

        Returns
        =======

        ``True``, ``False`` or ``None``. A return value of ``True`` indicates
        that the two vectors are identically equal. A return value of ``False``
        indicates that they are not. In some cases it is not possible to
        determine if the two vectors are identically equal and ``None`` is
        returned.

        See Also
        ========

        sympy.core.expr.Expr.equals
        """
        diff = self - other
        diff_mag2 = diff.dot(diff)
        return diff_mag2.equals(0)

    def dot(self, other):
        """
        Returns the dot product of this Vector, either with another
        Vector, or a Dyadic, or a Del operator.
        If 'other' is a Vector, returns the dot product scalar (SymPy
        expression).
        If 'other' is a Dyadic, the dot product is returned as a Vector.
        If 'other' is an instance of Del, returns the directional
        derivative operator as a Python function. If this function is
        applied to a scalar expression, it returns the directional
        derivative of the scalar field wrt this Vector.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator .

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> C.i.dot(C.j)
        0
        >>> C.i & C.i
        1
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.dot(C.k)
        5
        >>> (C.i & delop)(C.x*C.y*C.z)
        C.y*C.z
        >>> d = C.i.outer(C.i)
        >>> C.i.dot(d)
        C.i

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Vector.zero
            outvec = Vector.zero
            for k, v in other.components.items():
                vect_dot = k.args[0].dot(self)
                outvec += vect_dot * v * k.args[1]
            return outvec
        from sympy.vector.deloperator import Del
        if not isinstance(other, (Del, Vector)):
            raise TypeError(str(other) + " is not a vector, dyadic or " +
                            "del operator")

        # Check if the other is a del operator
        if isinstance(other, Del):
            def directional_derivative(field):
                from sympy.vector.functions import directional_derivative
                return directional_derivative(field, self)
            return directional_derivative

        return dot(self, other)

    def __and__(self, other):
        return self.dot(other)

    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        """
        Returns the cross product of this Vector with another Vector or
        Dyadic instance.
        The cross product is a Vector, if 'other' is a Vector. If 'other'
        is a Dyadic, this returns a Dyadic instance.

        Parameters
        ==========

        other: Vector/Dyadic
            The Vector or Dyadic we are crossing with.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> C.i.cross(C.j)
        C.k
        >>> C.i ^ C.i
        0
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v ^ C.i
        5*C.j + (-4)*C.k
        >>> d = C.i.outer(C.i)
        >>> C.j.cross(d)
        (-1)*(C.k|C.i)

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Dyadic.zero
            outdyad = Dyadic.zero
            for k, v in other.components.items():
                cross_product = self.cross(k.args[0])
                outer = cross_product.outer(k.args[1])
                outdyad += v * outer
            return outdyad

        return cross(self, other)

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def outer(self, other):
        """
        Returns the outer product of this vector with another, in the
        form of a Dyadic instance.

        Parameters
        ==========

        other : Vector
            The Vector with respect to which the outer product is to
            be computed.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N.i.outer(N.j)
        (N.i|N.j)

        """

        # Handle the special cases
        if not isinstance(other, Vector):
            raise TypeError("Invalid operand for outer product")
        elif (isinstance(self, VectorZero) or
                isinstance(other, VectorZero)):
            return Dyadic.zero

        # Iterate over components of both the vectors to generate
        # the required Dyadic instance
        args = [(v1 * v2) * BaseDyadic(k1, k2) for (k1, v1), (k2, v2)
                in product(self.components.items(), other.components.items())]

        return DyadicAdd(*args)

    def projection(self, other, scalar=False):
        """
        Returns the vector or scalar projection of the 'other' on 'self'.

        Examples
        ========

        >>> from sympy.vector.coordsysrect import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> i, j, k = C.base_vectors()
        >>> v1 = i + j + k
        >>> v2 = 3*i + 4*j
        >>> v1.projection(v2)
        7/3*C.i + 7/3*C.j + 7/3*C.k
        >>> v1.projection(v2, scalar=True)
        7/3

        """
        if self.equals(Vector.zero):
            return S.Zero if scalar else Vector.zero

        if scalar:
            return self.dot(other) / self.dot(self)
        else:
            return self.dot(other) / self.dot(self) * self

    @property
    def _projections(self):
        """
        Returns the components of this vector but the output includes
        also zero values components.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Vector
        >>> C = CoordSys3D('C')
        >>> v1 = 3*C.i + 4*C.j + 5*C.k
        >>> v1._projections
        (3, 4, 5)
        >>> v2 = C.x*C.y*C.z*C.i
        >>> v2._projections
        (C.x*C.y*C.z, 0, 0)
        >>> v3 = Vector.zero
        >>> v3._projections
        (0, 0, 0)
        """

        from sympy.vector.operators import _get_coord_systems
        if isinstance(self, VectorZero):
            return (S.Zero, S.Zero, S.Zero)
        base_vec = next(iter(_get_coord_systems(self))).base_vectors()
        return tuple([self.dot(i) for i in base_vec])

    def __or__(self, other):
        return self.outer(other)

    __or__.__doc__ = outer.__doc__

    def to_matrix(self, system):
        """
        Returns the matrix form of this vector with respect to the
        specified coordinate system.

        Parameters
        ==========

        system : CoordSys3D
            The system wrt which the matrix form is to be computed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> from sympy.abc import a, b, c
        >>> v = a*C.i + b*C.j + c*C.k
        >>> v.to_matrix(C)
        Matrix([
        [a],
        [b],
        [c]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       system.base_vectors()])

    def separate(self):
        """
        The constituents of this vector in different coordinate systems,
        as per its definition.

        Returns a dict mapping each CoordSys3D to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> R1 = CoordSys3D('R1')
        >>> R2 = CoordSys3D('R2')
        >>> v = R1.i + R2.i
        >>> v.separate() == {R1: R1.i, R2: R2.i}
        True

        """

        parts = {}
        for vect, measure in self.components.items():
            parts[vect.system] = (parts.get(vect.system, Vector.zero) +
                                  vect * measure)
        return parts

    def _div_helper(one, other):
        """ Helper for division involving vectors. """
        if isinstance(one, Vector) and isinstance(other, Vector):
            raise TypeError("Cannot divide two vectors")
        elif isinstance(one, Vector):
            if other == S.Zero:
                raise ValueError("Cannot divide a vector by zero")
            return VectorMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("Invalid division involving a vector")

# The following is adapted from the matrices.expressions.matexpr file

def get_postprocessor(cls):
    def _postprocessor(expr):
        vec_class = {Add: VectorAdd}[cls]
        vectors = []
        for term in expr.args:
            if isinstance(term.kind, VectorKind):
                vectors.append(term)

        if vec_class == VectorAdd:
            return VectorAdd(*vectors).doit(deep=False)
    return _postprocessor


Basic._constructor_postprocessor_mapping[Vector] = {
    "Add": [get_postprocessor(Add)],
}

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        if latex_str is None:
            latex_str = "x_{}".format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        # Verify arguments
        if index not in range(0, 3):
            raise ValueError("index must be 0, 1 or 2")
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        name = system._vector_names[index]
        # Initialize an object
        obj = super().__new__(cls, S(index), system)
        # Assign important attributes
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        # The _id is used for printing purposes
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)

        # This attr is used for re-expression to one of the systems
        # involved in the definition of the Vector. Applies to
        # VectorMul and VectorAdd too.
        obj._sys = system

        return obj

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name

    def _sympyrepr(self, printer):
        index, system = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        return {self}

    def _eval_conjugate(self):
        return self


class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        ret_str = ''
        items = list(self.separate().items())
        items.sort(key=lambda x: x[0].__str__())
        for system, vect in items:
            base_vects = system.base_vectors()
            for x in base_vects:
                if x in vect.components:
                    temp_vect = self.components[x] * x
                    ret_str += printer._print(temp_vect) + " + "
        return ret_str[:-3]


class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_vector(self):
        """ The BaseVector involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this VectorMul.
        """
        return self._measure_number


class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    """

    _op_priority = 12.1
    _pretty_form = '0'
    _latex_form = r'\mathbf{\hat{0}}'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


class Cross(Vector):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        if default_sort_key(expr1) > default_sort_key(expr2):
            return -Cross(expr2, expr1)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return cross(self._expr1, self._expr2)


class Dot(Expr):
    """
    Represents unevaluated Dot product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        expr1, expr2 = sorted([expr1, expr2], key=default_sort_key)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return dot(self._expr1, self._expr2)


def cross(vect1, vect2):
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> cross(v1, v2)
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
    if isinstance(vect1, Add):
        return VectorAdd.fromiter(cross(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return VectorAdd.fromiter(cross(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            if n1 == n2:
                return Vector.zero
            n3 = ({0,1,2}.difference({n1, n2})).pop()
            sign = 1 if ((n1 + 1) % 3 == n2) else -1
            return sign*vect1._sys.base_vectors()[n3]
        from .functions import express
        try:
            v = express(vect1, vect2._sys)
        except ValueError:
            return Cross(vect1, vect2)
        else:
            return cross(v, vect2)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*cross(vect1, v2)

    return Cross(vect1, vect2)


def dot(vect1, vect2):
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import dot
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> dot(v1, v2)
    R.x + R.y + R.z

    """
    if isinstance(vect1, Add):
        return Add.fromiter(dot(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return Add.fromiter(dot(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            return S.One if vect1 == vect2 else S.Zero
        from .functions import express
        try:
            v = express(vect2, vect1._sys)
        except ValueError:
            return Dot(vect1, vect2)
        else:
            return dot(vect1, v)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return S.Zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*dot(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*dot(vect1, v2)

    return Dot(vect1, vect2)


Vector._expr_type = Vector
Vector._mul_func = VectorMul
Vector._add_func = VectorAdd
Vector._zero_func = VectorZero
Vector._base_func = BaseVector
Vector.zero = VectorZero()
