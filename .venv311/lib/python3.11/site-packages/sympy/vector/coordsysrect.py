from collections.abc import Callable

from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)


class CoordSys3D(Basic):
    """
    Represents a coordinate system in 3-D space.
    """

    def __new__(cls, name, transformation=None, parent=None, location=None,
                rotation_matrix=None, vector_names=None, variable_names=None):
        """
        The orientation/location parameters are necessary if this system
        is being defined at a certain orientation or location wrt another.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        location : Vector
            The position vector of the new system's origin wrt the parent
            instance.

        rotation_matrix : SymPy ImmutableMatrix
            The rotation matrix of the new coordinate system with respect
            to the parent. In other words, the output of
            new_system.rotation_matrix(parent).

        parent : CoordSys3D
            The coordinate system wrt which the orientation/location
            (or both) is being defined.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        """

        name = str(name)
        Vector = sympy.vector.Vector
        Point = sympy.vector.Point

        if not isinstance(name, str):
            raise TypeError("name should be a string")

        if transformation is not None:
            if (location is not None) or (rotation_matrix is not None):
                raise ValueError("specify either `transformation` or "
                                 "`location`/`rotation_matrix`")
            if isinstance(transformation, (Tuple, tuple, list)):
                if isinstance(transformation[0], MatrixBase):
                    rotation_matrix = transformation[0]
                    location = transformation[1]
                else:
                    transformation = Lambda(transformation[0],
                                            transformation[1])
            elif isinstance(transformation, Callable):
                x1, x2, x3 = symbols('x1 x2 x3', cls=Dummy)
                transformation = Lambda((x1, x2, x3),
                                        transformation(x1, x2, x3))
            elif isinstance(transformation, str):
                transformation = Str(transformation)
            elif isinstance(transformation, (Str, Lambda)):
                pass
            else:
                raise TypeError("transformation: "
                                "wrong type {}".format(type(transformation)))

        # If orientation information has been provided, store
        # the rotation matrix accordingly
        if rotation_matrix is None:
            rotation_matrix = ImmutableDenseMatrix(eye(3))
        else:
            if not isinstance(rotation_matrix, MatrixBase):
                raise TypeError("rotation_matrix should be an Immutable" +
                                "Matrix instance")
            rotation_matrix = rotation_matrix.as_immutable()

        # If location information is not given, adjust the default
        # location as Vector.zero
        if parent is not None:
            if not isinstance(parent, CoordSys3D):
                raise TypeError("parent should be a " +
                                "CoordSys3D/None")
            if location is None:
                location = Vector.zero
            else:
                if not isinstance(location, Vector):
                    raise TypeError("location should be a Vector")
                # Check that location does not contain base
                # scalars
                for x in location.free_symbols:
                    if isinstance(x, BaseScalar):
                        raise ValueError("location should not contain" +
                                         " BaseScalars")
            origin = parent.origin.locate_new(name + '.origin',
                                              location)
        else:
            location = Vector.zero
            origin = Point(name + '.origin')

        if transformation is None:
            transformation = Tuple(rotation_matrix, location)

        if isinstance(transformation, Tuple):
            lambda_transformation = CoordSys3D._compose_rotation_and_translation(
                transformation[0],
                transformation[1],
                parent
            )
            r, l = transformation
            l = l._projections
            lambda_lame = CoordSys3D._get_lame_coeff('cartesian')
            lambda_inverse = lambda x, y, z: r.inv()*Matrix(
                [x-l[0], y-l[1], z-l[2]])
        elif isinstance(transformation, Str):
            trname = transformation.name
            lambda_transformation = CoordSys3D._get_transformation_lambdas(trname)
            if parent is not None:
                if parent.lame_coefficients() != (S.One, S.One, S.One):
                    raise ValueError('Parent for pre-defined coordinate '
                                 'system should be Cartesian.')
            lambda_lame = CoordSys3D._get_lame_coeff(trname)
            lambda_inverse = CoordSys3D._set_inv_trans_equations(trname)
        elif isinstance(transformation, Lambda):
            if not CoordSys3D._check_orthogonality(transformation):
                raise ValueError("The transformation equation does not "
                                 "create orthogonal coordinate system")
            lambda_transformation = transformation
            lambda_lame = CoordSys3D._calculate_lame_coeff(lambda_transformation)
            lambda_inverse = None
        else:
            lambda_transformation = lambda x, y, z: transformation(x, y, z)
            lambda_lame = CoordSys3D._get_lame_coeff(transformation)
            lambda_inverse = None

        if variable_names is None:
            if isinstance(transformation, Lambda):
                variable_names = ["x1", "x2", "x3"]
            elif isinstance(transformation, Str):
                if transformation.name == 'spherical':
                    variable_names = ["r", "theta", "phi"]
                elif transformation.name == 'cylindrical':
                    variable_names = ["r", "theta", "z"]
                else:
                    variable_names = ["x", "y", "z"]
            else:
                variable_names = ["x", "y", "z"]
        if vector_names is None:
            vector_names = ["i", "j", "k"]

        # All systems that are defined as 'roots' are unequal, unless
        # they have the same name.
        # Systems defined at same orientation/position wrt the same
        # 'parent' are equal, irrespective of the name.
        # This is true even if the same orientation is provided via
        # different methods like Axis/Body/Space/Quaternion.
        # However, coincident systems may be seen as unequal if
        # positioned/oriented wrt different parents, even though
        # they may actually be 'coincident' wrt the root system.
        if parent is not None:
            obj = super().__new__(
                cls, Str(name), transformation, parent)
        else:
            obj = super().__new__(
                cls, Str(name), transformation)
        obj._name = name
        # Initialize the base vectors

        _check_strings('vector_names', vector_names)
        vector_names = list(vector_names)
        latex_vects = [(r'\mathbf{\hat{%s}_{%s}}' % (x, name)) for
                           x in vector_names]
        pretty_vects = ['%s_%s' % (x, name) for x in vector_names]

        obj._vector_names = vector_names

        v1 = BaseVector(0, obj, pretty_vects[0], latex_vects[0])
        v2 = BaseVector(1, obj, pretty_vects[1], latex_vects[1])
        v3 = BaseVector(2, obj, pretty_vects[2], latex_vects[2])

        obj._base_vectors = (v1, v2, v3)

        # Initialize the base scalars

        _check_strings('variable_names', vector_names)
        variable_names = list(variable_names)
        latex_scalars = [(r"\mathbf{{%s}_{%s}}" % (x, name)) for
                         x in variable_names]
        pretty_scalars = ['%s_%s' % (x, name) for x in variable_names]

        obj._variable_names = variable_names
        obj._vector_names = vector_names

        x1 = BaseScalar(0, obj, pretty_scalars[0], latex_scalars[0])
        x2 = BaseScalar(1, obj, pretty_scalars[1], latex_scalars[1])
        x3 = BaseScalar(2, obj, pretty_scalars[2], latex_scalars[2])

        obj._base_scalars = (x1, x2, x3)

        obj._transformation = transformation
        obj._transformation_lambda = lambda_transformation
        obj._lame_coefficients = lambda_lame(x1, x2, x3)
        obj._transformation_from_parent_lambda = lambda_inverse

        setattr(obj, variable_names[0], x1)
        setattr(obj, variable_names[1], x2)
        setattr(obj, variable_names[2], x3)

        setattr(obj, vector_names[0], v1)
        setattr(obj, vector_names[1], v2)
        setattr(obj, vector_names[2], v3)

        # Assign params
        obj._parent = parent
        if obj._parent is not None:
            obj._root = obj._parent._root
        else:
            obj._root = obj

        obj._parent_rotation_matrix = rotation_matrix
        obj._origin = origin

        # Return the instance
        return obj

    def _sympystr(self, printer):
        return self._name

    def __iter__(self):
        return iter(self.base_vectors())

    @staticmethod
    def _check_orthogonality(equations):
        """
        Helper method for _connect_to_cartesian. It checks if
        set of transformation equations create orthogonal curvilinear
        coordinate system

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """

        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy)
        equations = equations(x1, x2, x3)
        v1 = Matrix([diff(equations[0], x1),
                     diff(equations[1], x1), diff(equations[2], x1)])

        v2 = Matrix([diff(equations[0], x2),
                     diff(equations[1], x2), diff(equations[2], x2)])

        v3 = Matrix([diff(equations[0], x3),
                     diff(equations[1], x3), diff(equations[2], x3)])

        if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
            return False
        else:
            if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
                and simplify(v3.dot(v1)) == 0:
                return True
            else:
                return False

    @staticmethod
    def _set_inv_trans_equations(curv_coord_name):
        """
        Store information about inverse transformation equations for
        pre-defined coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if curv_coord_name == 'cartesian':
            return lambda x, y, z: (x, y, z)

        if curv_coord_name == 'spherical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2 + z**2),
                acos(z/sqrt(x**2 + y**2 + z**2)),
                atan2(y, x)
            )
        if curv_coord_name == 'cylindrical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2),
                atan2(y, x),
                z
            )
        raise ValueError('Wrong set of parameters.'
                         'Type of coordinate system is defined')

    def _calculate_inv_trans_equations(self):
        """
        Helper method for set_coordinate_type. It calculates inverse
        transformation equations for given transformations equations.

        """
        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy, reals=True)
        x, y, z = symbols("x, y, z", cls=Dummy)

        equations = self._transformation(x1, x2, x3)

        solved = solve([equations[0] - x,
                        equations[1] - y,
                        equations[2] - z], (x1, x2, x3), dict=True)[0]
        solved = solved[x1], solved[x2], solved[x3]
        self._transformation_from_parent_lambda = \
            lambda x1, x2, x3: tuple(i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved)

    @staticmethod
    def _get_lame_coeff(curv_coord_name):
        """
        Store information about Lame coefficients for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (S.One, S.One, S.One)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (S.One, r, r*sin(theta))
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (S.One, r, S.One)
            raise ValueError('Wrong set of parameters.'
                             ' Type of coordinate system is not defined')
        return CoordSys3D._calculate_lame_coefficients(curv_coord_name)

    @staticmethod
    def _calculate_lame_coeff(equations):
        """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        return lambda x1, x2, x3: (
                          sqrt(diff(equations(x1, x2, x3)[0], x1)**2 +
                               diff(equations(x1, x2, x3)[1], x1)**2 +
                               diff(equations(x1, x2, x3)[2], x1)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x2)**2 +
                               diff(equations(x1, x2, x3)[1], x2)**2 +
                               diff(equations(x1, x2, x3)[2], x2)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x3)**2 +
                               diff(equations(x1, x2, x3)[1], x3)**2 +
                               diff(equations(x1, x2, x3)[2], x3)**2)
                      )

    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    @staticmethod
    def _get_transformation_lambdas(curv_coord_name):
        """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (x, y, z)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (
                    r*sin(theta)*cos(phi),
                    r*sin(theta)*sin(phi),
                    r*cos(theta)
                )
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (
                    r*cos(theta),
                    r*sin(theta),
                    h
                )
            raise ValueError('Wrong set of parameters.'
                             'Type of coordinate system is defined')

    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        return tuple(matrix * Matrix(equations))

    @property
    def origin(self):
        return self._origin

    def base_vectors(self):
        return self._base_vectors

    def base_scalars(self):
        return self._base_scalars

    def lame_coefficients(self):
        return self._lame_coefficients

    def transformation_to_parent(self):
        return self._transformation_lambda(*self.base_scalars())

    def transformation_from_parent(self):
        if self._parent is None:
            raise ValueError("no parent coordinate system, use "
                             "`transformation_from_parent_function()`")
        return self._transformation_from_parent_lambda(
                            *self._parent.base_scalars())

    def transformation_from_parent_function(self):
        return self._transformation_from_parent_lambda

    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
        from sympy.vector.functions import _path
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) +
                            " is not a CoordSys3D")
        # Handle special cases
        if other == self:
            return eye(3)
        elif other == self._parent:
            return self._parent_rotation_matrix
        elif other._parent == self:
            return other._parent_rotation_matrix.T
        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(3)
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._parent_rotation_matrix.T
        return result

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)

    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        otherframe.

        Parameters
        ==========

        otherframe : CoordSys3D
            The other system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import Symbol
        >>> A = CoordSys3D('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        origin_coords = tuple(self.position_wrt(other).to_matrix(other))
        relocated_scalars = [x - origin_coords[i]
                             for i, x in enumerate(other.base_scalars())]

        vars_matrix = (self.rotation_matrix(other) *
                       Matrix(relocated_scalars))
        return {x: trigsimp(vars_matrix[i])
                for i, x in enumerate(self.base_scalars())}

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSys3D with its origin located at the given
        position wrt this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        position : Vector
            The position vector of the new system's origin wrt this
            one.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> A = CoordSys3D('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        return CoordSys3D(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)

    def orient_new(self, name, orienters, location=None,
                   vector_names=None, variable_names=None):
        """
        Creates a new CoordSys3D oriented in the user-specified way
        with respect to this system.

        Please refer to the documentation of the orienter classes
        for more information about the orientation procedure.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        orienters : iterable/Orienter
            An Orienter or an iterable of Orienters for orienting the
            new coordinate system.
            If an Orienter is provided, it is applied to get the new
            system.
            If an iterable is provided, the orienters will be applied
            in the order in which they appear in the iterable.

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')

        Using an AxisOrienter

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> A = N.orient_new('A', (axis_orienter, ))

        Using a BodyOrienter

        >>> from sympy.vector import BodyOrienter
        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> B = N.orient_new('B', (body_orienter, ))

        Using a SpaceOrienter

        >>> from sympy.vector import SpaceOrienter
        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> C = N.orient_new('C', (space_orienter, ))

        Using a QuaternionOrienter

        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> D = N.orient_new('D', (q_orienter, ))
        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        if isinstance(orienters, Orienter):
            if isinstance(orienters, AxisOrienter):
                final_matrix = orienters.rotation_matrix(self)
            else:
                final_matrix = orienters.rotation_matrix()
            # TODO: trigsimp is needed here so that the matrix becomes
            # canonical (scalar_map also calls trigsimp; without this, you can
            # end up with the same CoordinateSystem that compares differently
            # due to a differently formatted matrix). However, this is
            # probably not so good for performance.
            final_matrix = trigsimp(final_matrix)
        else:
            final_matrix = Matrix(eye(3))
            for orienter in orienters:
                if isinstance(orienter, AxisOrienter):
                    final_matrix *= orienter.rotation_matrix(self)
                else:
                    final_matrix *= orienter.rotation_matrix()

        return CoordSys3D(name, rotation_matrix=final_matrix,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          location=location,
                          parent=self)

    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        orienter = AxisOrienter(angle, axis)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSys3D.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        orienter = QuaternionOrienter(q0, q1, q2, q3)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSys3D which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> a = CoordSys3D('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        return CoordSys3D(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        r = lambda x, y, z: CoordSys3D._rotation_trans_equations(rot, (x, y, z))
        if parent is None:
            return r

        dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        t = lambda x, y, z: (
            x + dx,
            y + dy,
            z + dz,
        )
        return lambda x, y, z: t(*r(x, y, z))


def _check_strings(arg_name, arg):
    errorstr = arg_name + " must be an iterable of 3 string-types"
    if len(arg) != 3:
        raise ValueError(errorstr)
    for s in arg:
        if not isinstance(s, str):
            raise TypeError(errorstr)


# Delayed import to avoid cyclic import problems:
from sympy.vector.vector import BaseVector
