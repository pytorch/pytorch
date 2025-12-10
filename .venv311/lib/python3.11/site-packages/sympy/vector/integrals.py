from sympy.core import Basic, diff
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.matrices import Matrix
from sympy.integrals import Integral, integrate
from sympy.geometry.entity import GeometryEntity
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import topological_sort
from sympy.vector import (CoordSys3D, Vector, ParametricRegion,
                        parametric_region_list, ImplicitRegion)
from sympy.vector.operators import _get_coord_systems


class ParametricIntegral(Basic):
    """
    Represents integral of a scalar or vector field
    over a Parametric Region

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.vector import CoordSys3D, ParametricRegion, ParametricIntegral
    >>> from sympy.abc import r, t, theta, phi

    >>> C = CoordSys3D('C')
    >>> curve = ParametricRegion((3*t - 2, t + 1), (t, 1, 2))
    >>> ParametricIntegral(C.x, curve)
    5*sqrt(10)/2
    >>> length = ParametricIntegral(1, curve)
    >>> length
    sqrt(10)
    >>> semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    >>> ParametricIntegral(C.z, semisphere)
    8*pi

    >>> ParametricIntegral(C.j + C.k, ParametricRegion((r*cos(theta), r*sin(theta)), r, theta))
    0

    """

    def __new__(cls, field, parametricregion):

        coord_set = _get_coord_systems(field)

        if len(coord_set) == 0:
            coord_sys = CoordSys3D('C')
        elif len(coord_set) > 1:
            raise ValueError
        else:
            coord_sys = next(iter(coord_set))

        if parametricregion.dimensions == 0:
            return S.Zero

        base_vectors = coord_sys.base_vectors()
        base_scalars = coord_sys.base_scalars()

        parametricfield = field

        r = Vector.zero
        for i in range(len(parametricregion.definition)):
            r += base_vectors[i]*parametricregion.definition[i]

        if len(coord_set) != 0:
            for i in range(len(parametricregion.definition)):
                parametricfield = parametricfield.subs(base_scalars[i], parametricregion.definition[i])

        if parametricregion.dimensions == 1:
            parameter = parametricregion.parameters[0]

            r_diff = diff(r, parameter)
            lower, upper = parametricregion.limits[parameter][0], parametricregion.limits[parameter][1]

            if isinstance(parametricfield, Vector):
                integrand = simplify(r_diff.dot(parametricfield))
            else:
                integrand = simplify(r_diff.magnitude()*parametricfield)

            result = integrate(integrand, (parameter, lower, upper))

        elif parametricregion.dimensions == 2:
            u, v = cls._bounds_case(parametricregion.parameters, parametricregion.limits)

            r_u = diff(r, u)
            r_v = diff(r, v)
            normal_vector = simplify(r_u.cross(r_v))

            if isinstance(parametricfield, Vector):
                integrand = parametricfield.dot(normal_vector)
            else:
                integrand = parametricfield*normal_vector.magnitude()

            integrand = simplify(integrand)

            lower_u, upper_u = parametricregion.limits[u][0], parametricregion.limits[u][1]
            lower_v, upper_v = parametricregion.limits[v][0], parametricregion.limits[v][1]

            result = integrate(integrand, (u, lower_u, upper_u), (v, lower_v, upper_v))

        else:
            variables = cls._bounds_case(parametricregion.parameters, parametricregion.limits)
            coeff = Matrix(parametricregion.definition).jacobian(variables).det()
            integrand = simplify(parametricfield*coeff)

            l = [(var, parametricregion.limits[var][0], parametricregion.limits[var][1]) for var in variables]
            result = integrate(integrand, *l)

        if not isinstance(result, Integral):
            return result
        else:
            return super().__new__(cls, field, parametricregion)

    @classmethod
    def _bounds_case(cls, parameters, limits):

        V = list(limits.keys())
        E = []

        for p in V:
            lower_p = limits[p][0]
            upper_p = limits[p][1]

            lower_p = lower_p.atoms()
            upper_p = upper_p.atoms()
            E.extend((p, q) for q in V if p != q and
                     (lower_p.issuperset({q}) or upper_p.issuperset({q})))

        if not E:
            return parameters
        else:
            return topological_sort((V, E), key=default_sort_key)

    @property
    def field(self):
        return self.args[0]

    @property
    def parametricregion(self):
        return self.args[1]


def vector_integrate(field, *region):
    """
    Compute the integral of a vector/scalar field
    over a a region or a set of parameters.

    Examples
    ========
    >>> from sympy.vector import CoordSys3D, ParametricRegion, vector_integrate
    >>> from sympy.abc import x, y, t
    >>> C = CoordSys3D('C')

    >>> region = ParametricRegion((t, t**2), (t, 1, 5))
    >>> vector_integrate(C.x*C.i, region)
    12

    Integrals over some objects of geometry module can also be calculated.

    >>> from sympy.geometry import Point, Circle, Triangle
    >>> c = Circle(Point(0, 2), 5)
    >>> vector_integrate(C.x**2 + C.y**2, c)
    290*pi
    >>> triangle = Triangle(Point(-2, 3), Point(2, 3), Point(0, 5))
    >>> vector_integrate(3*C.x**2*C.y*C.i + C.j, triangle)
    -8

    Integrals over some simple implicit regions can be computed. But in most cases,
    it takes too long to compute over them. This is due to the expressions of parametric
    representation becoming large.

    >>> from sympy.vector import ImplicitRegion
    >>> c2 = ImplicitRegion((x, y), (x - 2)**2 + (y - 1)**2 - 9)
    >>> vector_integrate(1, c2)
    6*pi

    Integral of fields with respect to base scalars:

    >>> vector_integrate(12*C.y**3, (C.y, 1, 3))
    240
    >>> vector_integrate(C.x**2*C.z, C.x)
    C.x**3*C.z/3
    >>> vector_integrate(C.x*C.i - C.y*C.k, C.x)
    (Integral(C.x, C.x))*C.i + (Integral(-C.y, C.x))*C.k
    >>> _.doit()
    C.x**2/2*C.i + (-C.x*C.y)*C.k

    """
    if len(region) == 1:
        if isinstance(region[0], ParametricRegion):
            return ParametricIntegral(field, region[0])

        if isinstance(region[0], ImplicitRegion):
            region = parametric_region_list(region[0])[0]
            return vector_integrate(field, region)

        if isinstance(region[0], GeometryEntity):
            regions_list = parametric_region_list(region[0])

            result = 0
            for reg in regions_list:
                result += vector_integrate(field, reg)
            return result

    return integrate(field, *region)
