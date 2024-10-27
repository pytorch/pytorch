from functools import reduce

from sympy import (sympify, diff, sin, cos, Matrix, symbols,
                                Function, S, Symbol, linear_eq_to_matrix)
from sympy.integrals.integrals import integrate
from sympy.simplify.trigsimp import trigsimp
from .vector import Vector, _check_vector
from .frame import CoordinateSym, _check_frame
from .dyadic import Dyadic
from .printing import vprint, vsprint, vpprint, vlatex, init_vprinting
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import translate

__all__ = ['cross', 'dot', 'express', 'time_derivative', 'outer',
           'kinematic_equations', 'get_motion_params', 'partial_velocity',
           'dynamicsymbols', 'vprint', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


def cross(vec1, vec2):
    """Cross product convenience wrapper for Vector.cross(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Cross product is between two vectors')
    return vec1 ^ vec2


cross.__doc__ += Vector.cross.__doc__  # type: ignore


def dot(vec1, vec2):
    """Dot product convenience wrapper for Vector.dot(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Dot product is between two vectors')
    return vec1 & vec2


dot.__doc__ += Vector.dot.__doc__  # type: ignore


def express(expr, frame, frame2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, scalar(sympyfiable) or Dyadic in given frame.

    Refer to the local methods of Vector and Dyadic for details.
    If 'variables' is True, then the coordinate variables (CoordinateSym
    instances) of other frames present in the vector/scalar field or
    dyadic expression are also substituted in terms of the base scalars of
    this frame.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in ReferenceFrame 'frame'

    frame: ReferenceFrame
        The reference frame to express expr in

    frame2 : ReferenceFrame
        The other frame required for re-expression(only for Dyadic expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of frame

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> B = N.orientnew('B', 'Axis', [q, N.z])
    >>> d = outer(N.x, N.x)
    >>> from sympy.physics.vector import express
    >>> express(d, B, N)
    cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)
    >>> express(B.x, N)
    cos(q)*N.x + sin(q)*N.y
    >>> express(N[0], B, variables=True)
    B_x*cos(q) - B_y*sin(q)

    """

    _check_frame(frame)

    if expr == 0:
        return expr

    if isinstance(expr, Vector):
        # Given expr is a Vector
        if variables:
            # If variables attribute is True, substitute the coordinate
            # variables in the Vector
            frame_list = [x[-1] for x in expr.args]
            subs_dict = {}
            for f in frame_list:
                subs_dict.update(f.variable_map(frame))
            expr = expr.subs(subs_dict)
        # Re-express in this frame
        outvec = Vector([])
        for v in expr.args:
            if v[1] != frame:
                temp = frame.dcm(v[1]) * v[0]
                if Vector.simp:
                    temp = temp.applyfunc(lambda x:
                                          trigsimp(x, method='fu'))
                outvec += Vector([(temp, frame)])
            else:
                outvec += Vector([v])
        return outvec

    if isinstance(expr, Dyadic):
        if frame2 is None:
            frame2 = frame
        _check_frame(frame2)
        ol = Dyadic(0)
        for v in expr.args:
            ol += express(v[0], frame, variables=variables) * \
                  (express(v[1], frame, variables=variables) |
                   express(v[2], frame2, variables=variables))
        return ol

    else:
        if variables:
            # Given expr is a scalar field
            frame_set = set()
            expr = sympify(expr)
            # Substitute all the coordinate variables
            for x in expr.free_symbols:
                if isinstance(x, CoordinateSym) and x.frame != frame:
                    frame_set.add(x.frame)
            subs_dict = {}
            for f in frame_set:
                subs_dict.update(f.variable_map(frame))
            return expr.subs(subs_dict)
        return expr


def time_derivative(expr, frame, order=1):
    """
    Calculate the time derivative of a vector/scalar field function
    or dyadic expression in given frame.

    References
    ==========

    https://en.wikipedia.org/wiki/Rotating_reference_frame#Time_derivatives_in_the_two_frames

    Parameters
    ==========

    expr : Vector/Dyadic/sympifyable
        The expression whose time derivative is to be calculated

    frame : ReferenceFrame
        The reference frame to calculate the time derivative in

    order : integer
        The order of the derivative to be calculated

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import Symbol
    >>> q1 = Symbol('q1')
    >>> u1 = dynamicsymbols('u1')
    >>> N = ReferenceFrame('N')
    >>> A = N.orientnew('A', 'Axis', [q1, N.x])
    >>> v = u1 * N.x
    >>> A.set_ang_vel(N, 10*A.x)
    >>> from sympy.physics.vector import time_derivative
    >>> time_derivative(v, N)
    u1'*N.x
    >>> time_derivative(u1*A[0], N)
    N_x*u1'
    >>> B = N.orientnew('B', 'Axis', [u1, N.z])
    >>> from sympy.physics.vector import outer
    >>> d = outer(N.x, N.x)
    >>> time_derivative(d, B)
    - u1'*(N.y|N.x) - u1'*(N.x|N.y)

    """

    t = dynamicsymbols._t
    _check_frame(frame)

    if order == 0:
        return expr
    if order % 1 != 0 or order < 0:
        raise ValueError("Unsupported value of order entered")

    if isinstance(expr, Vector):
        outlist = []
        for v in expr.args:
            if v[1] == frame:
                outlist += [(express(v[0], frame, variables=True).diff(t),
                             frame)]
            else:
                outlist += (time_derivative(Vector([v]), v[1]) +
                            (v[1].ang_vel_in(frame) ^ Vector([v]))).args
        outvec = Vector(outlist)
        return time_derivative(outvec, frame, order - 1)

    if isinstance(expr, Dyadic):
        ol = Dyadic(0)
        for v in expr.args:
            ol += (v[0].diff(t) * (v[1] | v[2]))
            ol += (v[0] * (time_derivative(v[1], frame) | v[2]))
            ol += (v[0] * (v[1] | time_derivative(v[2], frame)))
        return time_derivative(ol, frame, order - 1)

    else:
        return diff(express(expr, frame, variables=True), t, order)


def outer(vec1, vec2):
    """Outer product convenience wrapper for Vector.outer():\n"""
    if not isinstance(vec1, Vector):
        raise TypeError('Outer product is between two Vectors')
    return vec1.outer(vec2)


outer.__doc__ += Vector.outer.__doc__  # type: ignore


def kinematic_equations(speeds, coords, rot_type, rot_order=''):
    """Gives equations relating the qdot's to u's for a rotation type.

    Supply rotation type and order as in orient. Speeds are assumed to be
    body-fixed; if we are defining the orientation of B in A using by rot_type,
    the angular velocity of B in A is assumed to be in the form: speed[0]*B.x +
    speed[1]*B.y + speed[2]*B.z

    Parameters
    ==========

    speeds : list of length 3
        The body fixed angular velocity measure numbers.
    coords : list of length 3 or 4
        The coordinates used to define the orientation of the two frames.
    rot_type : str
        The type of rotation used to create the equations. Body, Space, or
        Quaternion only
    rot_order : str or int
        If applicable, the order of a series of rotations.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import kinematic_equations, vprint
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    >>> vprint(kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313'),
    ...     order=None)
    [-(u1*sin(q3) + u2*cos(q3))/sin(q2) + q1', -u1*cos(q3) + u2*sin(q3) + q2', (u1*sin(q3) + u2*cos(q3))*cos(q2)/sin(q2) - u3 + q3']

    """

    # Code below is checking and sanitizing input
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131',
                       '212', '232', '313', '323', '1', '2', '3', '')
    # make sure XYZ => 123 and rot_type is in lower case
    rot_order = translate(str(rot_order), 'XYZxyz', '123123')
    rot_type = rot_type.lower()

    if not isinstance(speeds, (list, tuple)):
        raise TypeError('Need to supply speeds in a list')
    if len(speeds) != 3:
        raise TypeError('Need to supply 3 body-fixed speeds')
    if not isinstance(coords, (list, tuple)):
        raise TypeError('Need to supply coordinates in a list')
    if rot_type in ['body', 'space']:
        if rot_order not in approved_orders:
            raise ValueError('Not an acceptable rotation order')
        if len(coords) != 3:
            raise ValueError('Need 3 coordinates for body or space')
        # Actual hard-coded kinematic differential equations
        w1, w2, w3 = speeds
        if w1 == w2 == w3 == 0:
            return [S.Zero]*3
        q1, q2, q3 = coords
        q1d, q2d, q3d = [diff(i, dynamicsymbols._t) for i in coords]
        s1, s2, s3 = [sin(q1), sin(q2), sin(q3)]
        c1, c2, c3 = [cos(q1), cos(q2), cos(q3)]
        if rot_type == 'body':
            if rot_order == '123':
                return [q1d - (w1 * c3 - w2 * s3) / c2, q2d - w1 * s3 - w2 *
                        c3, q3d - (-w1 * c3 + w2 * s3) * s2 / c2 - w3]
            if rot_order == '231':
                return [q1d - (w2 * c3 - w3 * s3) / c2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (- w2 * c3 + w3 * s3) * s2 / c2]
            if rot_order == '312':
                return [q1d - (-w1 * s3 + w3 * c3) / c2, q2d - w1 * c3 - w3 *
                        s3, q3d - (w1 * s3 - w3 * c3) * s2 / c2 - w2]
            if rot_order == '132':
                return [q1d - (w1 * c3 + w3 * s3) / c2, q2d + w1 * s3 - w3 *
                        c3, q3d - (w1 * c3 + w3 * s3) * s2 / c2 - w2]
            if rot_order == '213':
                return [q1d - (w1 * s3 + w2 * c3) / c2, q2d - w1 * c3 + w2 *
                        s3, q3d - (w1 * s3 + w2 * c3) * s2 / c2 - w3]
            if rot_order == '321':
                return [q1d - (w2 * s3 + w3 * c3) / c2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 - (w2 * s3 + w3 * c3) * s2 / c2]
            if rot_order == '121':
                return [q1d - (w2 * s3 + w3 * c3) / s2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 + (w2 * s3 + w3 * c3) * c2 / s2]
            if rot_order == '131':
                return [q1d - (-w2 * c3 + w3 * s3) / s2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (w2 * c3 - w3 * s3) * c2 / s2]
            if rot_order == '212':
                return [q1d - (w1 * s3 - w3 * c3) / s2, q2d - w1 * c3 - w3 *
                        s3, q3d - (-w1 * s3 + w3 * c3) * c2 / s2 - w2]
            if rot_order == '232':
                return [q1d - (w1 * c3 + w3 * s3) / s2, q2d + w1 * s3 - w3 *
                        c3, q3d + (w1 * c3 + w3 * s3) * c2 / s2 - w2]
            if rot_order == '313':
                return [q1d - (w1 * s3 + w2 * c3) / s2, q2d - w1 * c3 + w2 *
                        s3, q3d + (w1 * s3 + w2 * c3) * c2 / s2 - w3]
            if rot_order == '323':
                return [q1d - (-w1 * c3 + w2 * s3) / s2, q2d - w1 * s3 - w2 *
                        c3, q3d - (w1 * c3 - w2 * s3) * c2 / s2 - w3]
        if rot_type == 'space':
            if rot_order == '123':
                return [q1d - w1 - (w2 * s1 + w3 * c1) * s2 / c2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / c2]
            if rot_order == '231':
                return [q1d - (w1 * c1 + w3 * s1) * s2 / c2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / c2]
            if rot_order == '312':
                return [q1d - (w1 * s1 + w2 * c1) * s2 / c2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / c2]
            if rot_order == '132':
                return [q1d - w1 - (-w2 * c1 + w3 * s1) * s2 / c2, q2d - w2 *
                        s1 - w3 * c1, q3d - (w2 * c1 - w3 * s1) / c2]
            if rot_order == '213':
                return [q1d - (w1 * s1 - w3 * c1) * s2 / c2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (-w1 * s1 + w3 * c1) / c2]
            if rot_order == '321':
                return [q1d - (-w1 * c1 + w2 * s1) * s2 / c2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (w1 * c1 - w2 * s1) / c2]
            if rot_order == '121':
                return [q1d - w1 + (w2 * s1 + w3 * c1) * c2 / s2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / s2]
            if rot_order == '131':
                return [q1d - w1 - (w2 * c1 - w3 * s1) * c2 / s2, q2d - w2 *
                        s1 - w3 * c1, q3d - (-w2 * c1 + w3 * s1) / s2]
            if rot_order == '212':
                return [q1d - (-w1 * s1 + w3 * c1) * c2 / s2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (w1 * s1 - w3 * c1) / s2]
            if rot_order == '232':
                return [q1d + (w1 * c1 + w3 * s1) * c2 / s2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / s2]
            if rot_order == '313':
                return [q1d + (w1 * s1 + w2 * c1) * c2 / s2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / s2]
            if rot_order == '323':
                return [q1d - (w1 * c1 - w2 * s1) * c2 / s2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (-w1 * c1 + w2 * s1) / s2]
    elif rot_type == 'quaternion':
        if rot_order != '':
            raise ValueError('Cannot have rotation order for quaternion')
        if len(coords) != 4:
            raise ValueError('Need 4 coordinates for quaternion')
        # Actual hard-coded kinematic differential equations
        e0, e1, e2, e3 = coords
        w = Matrix(speeds + [0])
        E = Matrix([[e0, -e3, e2, e1],
                    [e3, e0, -e1, e2],
                    [-e2, e1, e0, e3],
                    [-e1, -e2, -e3, e0]])
        edots = Matrix([diff(i, dynamicsymbols._t) for i in [e1, e2, e3, e0]])
        return list(edots.T - 0.5 * w.T * E.T)
    else:
        raise ValueError('Not an approved rotation type for this function')


def get_motion_params(frame, **kwargs):
    """
    Returns the three motion parameters - (acceleration, velocity, and
    position) as vectorial functions of time in the given frame.

    If a higher order differential function is provided, the lower order
    functions are used as boundary conditions. For example, given the
    acceleration, the velocity and position parameters are taken as
    boundary conditions.

    The values of time at which the boundary conditions are specified
    are taken from timevalue1(for position boundary condition) and
    timevalue2(for velocity boundary condition).

    If any of the boundary conditions are not provided, they are taken
    to be zero by default (zero vectors, in case of vectorial inputs). If
    the boundary conditions are also functions of time, they are converted
    to constants by substituting the time values in the dynamicsymbols._t
    time Symbol.

    This function can also be used for calculating rotational motion
    parameters. Have a look at the Parameters and Examples for more clarity.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame to express the motion parameters in

    acceleration : Vector
        Acceleration of the object/frame as a function of time

    velocity : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    position : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    timevalue1 : sympyfiable
        Value of time for position boundary condition

    timevalue2 : sympyfiable
        Value of time for velocity boundary condition

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, get_motion_params, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> R = ReferenceFrame('R')
    >>> v1, v2, v3 = dynamicsymbols('v1 v2 v3')
    >>> v = v1*R.x + v2*R.y + v3*R.z
    >>> get_motion_params(R, position = v)
    (v1''*R.x + v2''*R.y + v3''*R.z, v1'*R.x + v2'*R.y + v3'*R.z, v1*R.x + v2*R.y + v3*R.z)
    >>> a, b, c = symbols('a b c')
    >>> v = a*R.x + b*R.y + c*R.z
    >>> get_motion_params(R, velocity = v)
    (0, a*R.x + b*R.y + c*R.z, a*t*R.x + b*t*R.y + c*t*R.z)
    >>> parameters = get_motion_params(R, acceleration = v)
    >>> parameters[1]
    a*t*R.x + b*t*R.y + c*t*R.z
    >>> parameters[2]
    a*t**2/2*R.x + b*t**2/2*R.y + c*t**2/2*R.z

    """

    def _process_vector_differential(vectdiff, condition, variable, ordinate,
                                     frame):
        """
        Helper function for get_motion methods. Finds derivative of vectdiff
        wrt variable, and its integral using the specified boundary condition
        at value of variable = ordinate.
        Returns a tuple of - (derivative, function and integral) wrt vectdiff

        """

        # Make sure boundary condition is independent of 'variable'
        if condition != 0:
            condition = express(condition, frame, variables=True)
        # Special case of vectdiff == 0
        if vectdiff == Vector(0):
            return (0, 0, condition)
        # Express vectdiff completely in condition's frame to give vectdiff1
        vectdiff1 = express(vectdiff, frame)
        # Find derivative of vectdiff
        vectdiff2 = time_derivative(vectdiff, frame)
        # Integrate and use boundary condition
        vectdiff0 = Vector(0)
        lims = (variable, ordinate, variable)
        for dim in frame:
            function1 = vectdiff1.dot(dim)
            abscissa = dim.dot(condition).subs({variable: ordinate})
            # Indefinite integral of 'function1' wrt 'variable', using
            # the given initial condition (ordinate, abscissa).
            vectdiff0 += (integrate(function1, lims) + abscissa) * dim
        # Return tuple
        return (vectdiff2, vectdiff, vectdiff0)

    _check_frame(frame)
    # Decide mode of operation based on user's input
    if 'acceleration' in kwargs:
        mode = 2
    elif 'velocity' in kwargs:
        mode = 1
    else:
        mode = 0
    # All the possible parameters in kwargs
    # Not all are required for every case
    # If not specified, set to default values(may or may not be used in
    # calculations)
    conditions = ['acceleration', 'velocity', 'position',
                  'timevalue', 'timevalue1', 'timevalue2']
    for i, x in enumerate(conditions):
        if x not in kwargs:
            if i < 3:
                kwargs[x] = Vector(0)
            else:
                kwargs[x] = S.Zero
        elif i < 3:
            _check_vector(kwargs[x])
        else:
            kwargs[x] = sympify(kwargs[x])
    if mode == 2:
        vel = _process_vector_differential(kwargs['acceleration'],
                                           kwargs['velocity'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue2'], frame)[2]
        pos = _process_vector_differential(vel, kwargs['position'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue1'], frame)[2]
        return (kwargs['acceleration'], vel, pos)
    elif mode == 1:
        return _process_vector_differential(kwargs['velocity'],
                                            kwargs['position'],
                                            dynamicsymbols._t,
                                            kwargs['timevalue1'], frame)
    else:
        vel = time_derivative(kwargs['position'], frame)
        acc = time_derivative(vel, frame)
        return (acc, vel, kwargs['position'])


def partial_velocity(vel_vecs, gen_speeds, frame):
    """Returns a list of partial velocities with respect to the provided
    generalized speeds in the given reference frame for each of the supplied
    velocity vectors.

    The output is a list of lists. The outer list has a number of elements
    equal to the number of supplied velocity vectors. The inner lists are, for
    each velocity vector, the partial derivatives of that velocity vector with
    respect to the generalized speeds supplied.

    Parameters
    ==========

    vel_vecs : iterable
        An iterable of velocity vectors (angular or linear).
    gen_speeds : iterable
        An iterable of generalized speeds.
    frame : ReferenceFrame
        The reference frame that the partial derivatives are going to be taken
        in.

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import partial_velocity
    >>> u = dynamicsymbols('u')
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, u * N.x)
    >>> vel_vecs = [P.vel(N)]
    >>> gen_speeds = [u]
    >>> partial_velocity(vel_vecs, gen_speeds, N)
    [[N.x]]

    """

    if not iterable(vel_vecs):
        raise TypeError('Velocity vectors must be contained in an iterable.')

    if not iterable(gen_speeds):
        raise TypeError('Generalized speeds must be contained in an iterable')

    vec_partials = []
    gen_speeds = list(gen_speeds)
    for vel in vel_vecs:
        partials = [Vector(0) for _ in gen_speeds]
        for components, ref in vel.args:
            mat, _ = linear_eq_to_matrix(components, gen_speeds)
            for i in range(len(gen_speeds)):
                for dim, direction in enumerate(ref):
                    if mat[dim, i] != 0:
                        partials[i] += direction * mat[dim, i]

        vec_partials.append(partials)

    return vec_partials


def dynamicsymbols(names, level=0, **assumptions):
    """Uses symbols and Function for functions of time.

    Creates a SymPy UndefinedFunction, which is then initialized as a function
    of a variable, the default being Symbol('t').

    Parameters
    ==========

    names : str
        Names of the dynamic symbols you want to create; works the same way as
        inputs to symbols
    level : int
        Level of differentiation of the returned function; d/dt once of t,
        twice of t, etc.
    assumptions :
        - real(bool) : This is used to set the dynamicsymbol as real,
                    by default is False.
        - positive(bool) : This is used to set the dynamicsymbol as positive,
                    by default is False.
        - commutative(bool) : This is used to set the commutative property of
                    a dynamicsymbol, by default is True.
        - integer(bool) : This is used to set the dynamicsymbol as integer,
                    by default is False.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy import diff, Symbol
    >>> q1 = dynamicsymbols('q1')
    >>> q1
    q1(t)
    >>> q2 = dynamicsymbols('q2', real=True)
    >>> q2.is_real
    True
    >>> q3 = dynamicsymbols('q3', positive=True)
    >>> q3.is_positive
    True
    >>> q4, q5 = dynamicsymbols('q4,q5', commutative=False)
    >>> bool(q4*q5 != q5*q4)
    True
    >>> q6 = dynamicsymbols('q6', integer=True)
    >>> q6.is_integer
    True
    >>> diff(q1, Symbol('t'))
    Derivative(q1(t), t)

    """
    esses = symbols(names, cls=Function, **assumptions)
    t = dynamicsymbols._t
    if iterable(esses):
        esses = [reduce(diff, [t] * level, e(t)) for e in esses]
        return esses
    else:
        return reduce(diff, [t] * level, esses(t))


dynamicsymbols._t = Symbol('t')  # type: ignore
dynamicsymbols._str = '\''  # type: ignore
