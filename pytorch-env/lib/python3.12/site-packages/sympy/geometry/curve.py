"""Curves in 2-dimensional Euclidean space.

Contains
========
Curve

"""

from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import diff
from sympy.core.containers import Tuple
from sympy.core.symbol import _symbol
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point
from sympy.integrals import integrate
from sympy.matrices import Matrix, rot_axis3
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


class Curve(GeometrySet):
    """A curve in space.

    A curve is defined by parametric functions for the coordinates, a
    parameter and the lower and upper bounds for the parameter value.

    Parameters
    ==========

    function : list of functions
    limits : 3-tuple
        Function parameter and lower and upper bounds.

    Attributes
    ==========

    functions
    parameter
    limits

    Raises
    ======

    ValueError
        When `functions` are specified incorrectly.
        When `limits` are specified incorrectly.

    Examples
    ========

    >>> from sympy import Curve, sin, cos, interpolate
    >>> from sympy.abc import t, a
    >>> C = Curve((sin(t), cos(t)), (t, 0, 2))
    >>> C.functions
    (sin(t), cos(t))
    >>> C.limits
    (t, 0, 2)
    >>> C.parameter
    t
    >>> C = Curve((t, interpolate([1, 4, 9, 16], t)), (t, 0, 1)); C
    Curve((t, t**2), (t, 0, 1))
    >>> C.subs(t, 4)
    Point2D(4, 16)
    >>> C.arbitrary_point(a)
    Point2D(a, a**2)

    See Also
    ========

    sympy.core.function.Function
    sympy.polys.polyfuncs.interpolate

    """

    def __new__(cls, function, limits):
        if not is_sequence(function) or len(function) != 2:
            raise ValueError("Function argument should be (x(t), y(t)) "
                "but got %s" % str(function))
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*function), Tuple(*limits))

    def __call__(self, f):
        return self.subs(self.parameter, f)

    def _eval_subs(self, old, new):
        if old == self.parameter:
            return Point(*[f.subs(old, new) for f in self.functions])

    def _eval_evalf(self, prec=15, **options):
        f, (t, a, b) = self.args
        dps = prec_to_dps(prec)
        f = tuple([i.evalf(n=dps, **options) for i in f])
        a, b = [i.evalf(n=dps, **options) for i in (a, b)]
        return self.func(f, (t, a, b))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't'.
            The Curve's parameter is selected with None or self.parameter
            otherwise the provided symbol is used.

        Returns
        =======

        Point :
            Returns a point in parametric form.

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        Examples
        ========

        >>> from sympy import Curve, Symbol
        >>> from sympy.abc import s
        >>> C = Curve([2*s, s**2], (s, 0, 2))
        >>> C.arbitrary_point()
        Point2D(2*t, t**2)
        >>> C.arbitrary_point(C.parameter)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(None)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(Symbol('a'))
        Point2D(2*a, a**2)

        See Also
        ========

        sympy.geometry.point.Point

        """
        if parameter is None:
            return Point(*self.functions)

        tnew = _symbol(parameter, self.parameter, real=True)
        t = self.parameter
        if (tnew.name != t.name and
                tnew.name in (f.name for f in self.free_symbols)):
            raise ValueError('Symbol %s already appears in object '
                'and cannot be used as a parameter.' % tnew.name)
        return Point(*[w.subs(t, tnew) for w in self.functions])

    @property
    def free_symbols(self):
        """Return a set of symbols other than the bound symbols used to
        parametrically define the Curve.

        Returns
        =======

        set :
            Set of all non-parameterized symbols.

        Examples
        ========

        >>> from sympy.abc import t, a
        >>> from sympy import Curve
        >>> Curve((t, t**2), (t, 0, 2)).free_symbols
        set()
        >>> Curve((t, t**2), (t, a, 2)).free_symbols
        {a}

        """
        free = set()
        for a in self.functions + self.limits[1:]:
            free |= a.free_symbols
        free = free.difference({self.parameter})
        return free

    @property
    def ambient_dimension(self):
        """The dimension of the curve.

        Returns
        =======

        int :
            the dimension of curve.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.ambient_dimension
        2

        """

        return len(self.args[0])

    @property
    def functions(self):
        """The functions specifying the curve.

        Returns
        =======

        functions :
            list of parameterized coordinate functions.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.functions
        (t, t**2)

        See Also
        ========

        parameter

        """
        return self.args[0]

    @property
    def limits(self):
        """The limits for the curve.

        Returns
        =======

        limits : tuple
            Contains parameter and lower and upper limits.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**3], (t, -2, 2))
        >>> C.limits
        (t, -2, 2)

        See Also
        ========

        plot_interval

        """
        return self.args[1]

    @property
    def parameter(self):
        """The curve function variable.

        Returns
        =======

        Symbol :
            returns a bound symbol.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**2], (t, 0, 2))
        >>> C.parameter
        t

        See Also
        ========

        functions

        """
        return self.args[1][0]

    @property
    def length(self):
        """The curve length.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import t
        >>> Curve((t, t), (t, 0, 1)).length
        sqrt(2)

        """
        integrand = sqrt(sum(diff(func, self.limits[0])**2 for func in self.functions))
        return integrate(integrand, self.limits)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't';
            otherwise the provided symbol is used.

        Returns
        =======

        List :
            the plot interval as below:
                [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Curve, sin
        >>> from sympy.abc import x, s
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval()
        [t, 1, 2]
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval(s)
        [s, 1, 2]

        See Also
        ========

        limits : Returns limits of the parameter interval

        """
        t = _symbol(parameter, self.parameter, real=True)
        return [t] + list(self.limits[1:])

    def rotate(self, angle=0, pt=None):
        """This function is used to rotate a curve along given point ``pt`` at given angle(in radian).

        Parameters
        ==========

        angle :
            the angle at which the curve will be rotated(in radian) in counterclockwise direction.
            default value of angle is 0.

        pt : Point
            the point along which the curve will be rotated.
            If no point given, the curve will be rotated around origin.

        Returns
        =======

        Curve :
            returns a curve rotated at given angle along given point.

        Examples
        ========

        >>> from sympy import Curve, pi
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).rotate(pi/2)
        Curve((-x, x), (x, 0, 1))

        """
        if pt:
            pt = -Point(pt, dim=2)
        else:
            pt = Point(0,0)
        rv = self.translate(*pt.args)
        f = list(rv.functions)
        f.append(0)
        f = Matrix(1, 3, f)
        f *= rot_axis3(angle)
        rv = self.func(f[0, :2].tolist()[0], self.limits)
        pt = -pt
        return rv.translate(*pt.args)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since Curve is not made up of Points.

        Returns
        =======

        Curve :
            returns scaled curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).scale(2)
        Curve((2*x, x), (x, 0, 1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        fx, fy = self.functions
        return self.func((fx*x, fy*y), self.limits)

    def translate(self, x=0, y=0):
        """Translate the Curve by (x, y).

        Returns
        =======

        Curve :
            returns a translated curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).translate(1, 2)
        Curve((x + 1, x + 2), (x, 0, 1))

        """
        fx, fy = self.functions
        return self.func((fx + x, fy + y), self.limits)
