from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit


class Point(Basic):
    """
    Represents a point in 3-D space.
    """

    def __new__(cls, name, position=Vector.zero, parent_point=None):
        name = str(name)
        # Check the args first
        if not isinstance(position, Vector):
            raise TypeError(
                "position should be an instance of Vector, not %s" % type(
                    position))
        if (not isinstance(parent_point, Point) and
                parent_point is not None):
            raise TypeError(
                "parent_point should be an instance of Point, not %s" % type(
                    parent_point))
        # Super class construction
        if parent_point is None:
            obj = super().__new__(cls, Str(name), position)
        else:
            obj = super().__new__(cls, Str(name), position, parent_point)
        # Decide the object parameters
        obj._name = name
        obj._pos = position
        if parent_point is None:
            obj._parent = None
            obj._root = obj
        else:
            obj._parent = parent_point
            obj._root = parent_point._root
        # Return object
        return obj

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of this Point with respect to
        another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this Point wrt it is
            returned. If its an instance of CoordSyRect, the position
            wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> N.origin.position_wrt(p1)
        (-10)*N.i

        """

        if (not isinstance(other, Point) and
                not isinstance(other, CoordSys3D)):
            raise TypeError(str(other) +
                            "is not a Point or CoordSys3D")
        if isinstance(other, CoordSys3D):
            other = other.origin
        # Handle special cases
        if other == self:
            return Vector.zero
        elif other == self._parent:
            return self._pos
        elif other._parent == self:
            return -1 * other._pos
        # Else, use point tree to calculate position
        rootindex, path = _path(self, other)
        result = Vector.zero
        i = -1
        for i in range(rootindex):
            result += path[i]._pos
        i += 2
        while i < len(path):
            result -= path[i]._pos
            i += 1
        return result

    def locate_new(self, name, position):
        """
        Returns a new Point located at the given position wrt this
        Point.
        Thus, the position vector of the new Point wrt this one will
        be equal to the given 'position' parameter.

        Parameters
        ==========

        name : str
            Name of the new point

        position : Vector
            The position vector of the new Point wrt this one

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p1.position_wrt(N.origin)
        10*N.i

        """
        return Point(name, position, self)

    def express_coordinates(self, coordinate_system):
        """
        Returns the Cartesian/rectangular coordinates of this point
        wrt the origin of the given CoordSys3D instance.

        Parameters
        ==========

        coordinate_system : CoordSys3D
            The coordinate system to express the coordinates of this
            Point in.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p2 = p1.locate_new('p2', 5 * N.j)
        >>> p2.express_coordinates(N)
        (10, 5, 0)

        """

        # Determine the position vector
        pos_vect = self.position_wrt(coordinate_system.origin)
        # Express it in the given coordinate system
        return tuple(pos_vect.to_matrix(coordinate_system))

    def _sympystr(self, printer):
        return self._name
