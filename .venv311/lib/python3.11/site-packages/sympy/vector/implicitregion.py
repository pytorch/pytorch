from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core


class ImplicitRegion(Basic):
    """
    Represents an implicit region in space.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z, t
    >>> from sympy.vector import ImplicitRegion

    >>> ImplicitRegion((x, y), x**2 + y**2 - 4)
    ImplicitRegion((x, y), x**2 + y**2 - 4)
    >>> ImplicitRegion((x, y), Eq(y*x, 1))
    ImplicitRegion((x, y), x*y - 1)

    >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
    >>> parabola.degree
    2
    >>> parabola.equation
    -4*x + y**2
    >>> parabola.rational_parametrization(t)
    (4/t**2, 4/t)

    >>> r = ImplicitRegion((x, y, z), Eq(z, x**2 + y**2))
    >>> r.variables
    (x, y, z)
    >>> r.singular_points()
    EmptySet
    >>> r.regular_point()
    (-10, -10, 200)

    Parameters
    ==========

    variables : tuple to map variables in implicit equation to base scalars.

    equation : An expression or Eq denoting the implicit equation of the region.

    """
    def __new__(cls, variables, equation):
        if not isinstance(variables, Tuple):
            variables = Tuple(*variables)

        if isinstance(equation, Eq):
            equation = equation.lhs - equation.rhs

        return super().__new__(cls, variables, equation)

    @property
    def variables(self):
        return self.args[0]

    @property
    def equation(self):
        return self.args[1]

    @property
    def degree(self):
        return total_degree(self.equation)

    def regular_point(self):
        """
        Returns a point on the implicit region.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> circle = ImplicitRegion((x, y), (x + 2)**2 + (y - 3)**2 - 16)
        >>> circle.regular_point()
        (-2, -1)
        >>> parabola = ImplicitRegion((x, y), x**2 - 4*y)
        >>> parabola.regular_point()
        (0, 0)
        >>> r = ImplicitRegion((x, y, z), (x + y + z)**4)
        >>> r.regular_point()
        (-10, -10, 20)

        References
        ==========

        - Erik Hillgarter, "Rational Points on Conics", Diploma Thesis, RISC-Linz,
          J. Kepler Universitat Linz, 1996. Available:
          https://www3.risc.jku.at/publications/download/risc_1355/Rational%20Points%20on%20Conics.pdf

        """
        equation = self.equation

        if len(self.variables) == 1:
            return (list(solveset(equation, self.variables[0], domain=S.Reals))[0],)
        elif len(self.variables) == 2:

            if self.degree == 2:
                coeffs = a, b, c, d, e, f = conic_coeff(self.variables, equation)

                if b**2 == 4*a*c:
                    x_reg, y_reg = self._regular_point_parabola(*coeffs)
                else:
                    x_reg, y_reg = self._regular_point_ellipse(*coeffs)
                return x_reg, y_reg

        if len(self.variables) == 3:
            x, y, z = self.variables

            for x_reg in range(-10, 10):
                for y_reg in range(-10, 10):
                    if not solveset(equation.subs({x: x_reg, y: y_reg}), self.variables[2], domain=S.Reals).is_empty:
                        return (x_reg, y_reg, list(solveset(equation.subs({x: x_reg, y: y_reg})))[0])

        if len(self.singular_points()) != 0:
            return list[self.singular_points()][0]

        raise NotImplementedError()

    def _regular_point_parabola(self, a, b, c, d, e, f):
            ok = (a, d) != (0, 0) and (c, e) != (0, 0) and b**2 == 4*a*c and (a, c) != (0, 0)

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a != 0:
                d_dash, f_dash = (4*a*e - 2*b*d, 4*a*f - d**2)
                if d_dash != 0:
                    y_reg = -f_dash/d_dash
                    x_reg = -(d + b*y_reg)/(2*a)
                else:
                    ok = False
            elif c != 0:
                d_dash, f_dash = (4*c*d - 2*b*e, 4*c*f - e**2)
                if d_dash != 0:
                    x_reg = -f_dash/d_dash
                    y_reg = -(e + b*x_reg)/(2*c)
                else:
                    ok = False

            if ok:
                return x_reg, y_reg
            else:
                raise ValueError("Rational Point on the conic does not exist")

    def _regular_point_ellipse(self, a, b, c, d, e, f):
            D = 4*a*c - b**2
            ok = D

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a == 0 and c == 0:
                K = -1
                L = 4*(d*e - b*f)
            elif c != 0:
                K = D
                L = 4*c**2*d**2 - 4*b*c*d*e + 4*a*c*e**2 + 4*b**2*c*f - 16*a*c**2*f
            else:
                K = D
                L = 4*a**2*e**2 - 4*b*a*d*e + 4*b**2*a*f

            ok = L != 0 and not(K > 0 and L < 0)
            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            K = Rational(K).limit_denominator(10**12)
            L = Rational(L).limit_denominator(10**12)

            k1, k2 = K.p, K.q
            l1, l2 = L.p, L.q
            g = gcd(k2, l2)

            a1 = (l2*k2)/g
            b1 = (k1*l2)/g
            c1 = -(l1*k2)/g
            a2 = sign(a1)*core(abs(a1), 2)
            r1 = sqrt(a1/a2)
            b2 = sign(b1)*core(abs(b1), 2)
            r2 = sqrt(b1/b2)
            c2 = sign(c1)*core(abs(c1), 2)
            r3 = sqrt(c1/c2)

            g = gcd(gcd(a2, b2), c2)
            a2 = a2/g
            b2 = b2/g
            c2 = c2/g

            g1 = gcd(a2, b2)
            a2 = a2/g1
            b2 = b2/g1
            c2 = c2*g1

            g2 = gcd(a2,c2)
            a2 = a2/g2
            b2 = b2*g2
            c2 = c2/g2

            g3 = gcd(b2, c2)
            a2 = a2*g3
            b2 = b2/g3
            c2 = c2/g3

            x, y, z = symbols("x y z")
            eq = a2*x**2 + b2*y**2 + c2*z**2

            solutions = diophantine(eq)

            if len(solutions) == 0:
                raise ValueError("Rational Point on the conic does not exist")

            flag = False
            for sol in solutions:
                syms = Tuple(*sol).free_symbols
                rep = dict.fromkeys(syms, 3)
                sol_z = sol[2]

                if sol_z == 0:
                    flag = True
                    continue

                if not isinstance(sol_z, (int, Integer)):
                    syms_z = sol_z.free_symbols

                    if len(syms_z) == 1:
                        p = next(iter(syms_z))
                        p_values = Complement(S.Integers, solveset(Eq(sol_z, 0), p, S.Integers))
                        rep[p] = next(iter(p_values))

                    if len(syms_z) == 2:
                        p, q = list(ordered(syms_z))

                        for i in S.Integers:
                            subs_sol_z = sol_z.subs(p, i)
                            q_values = Complement(S.Integers, solveset(Eq(subs_sol_z, 0), q, S.Integers))

                            if not q_values.is_empty:
                                rep[p] = i
                                rep[q] = next(iter(q_values))
                                break

                    if len(syms) != 0:
                        x, y, z = tuple(s.subs(rep) for s in sol)
                    else:
                        x, y, z =   sol
                    flag = False
                    break

            if flag:
                raise ValueError("Rational Point on the conic does not exist")

            x = (x*g3)/r1
            y = (y*g2)/r2
            z = (z*g1)/r3
            x = x/z
            y = y/z

            if a == 0 and c == 0:
                x_reg = (x + y - 2*e)/(2*b)
                y_reg = (x - y - 2*d)/(2*b)
            elif c != 0:
                x_reg = (x - 2*d*c + b*e)/K
                y_reg = (y - b*x_reg - e)/(2*c)
            else:
                y_reg = (x - 2*e*a + b*d)/K
                x_reg = (y - b*y_reg - d)/(2*a)

            return x_reg, y_reg

    def singular_points(self):
        """
        Returns a set of singular points of the region.

        The singular points are those points on the region
        where all partial derivatives vanish.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
        >>> I.singular_points()
        {(1, 1)}

        """
        eq_list = [self.equation]
        for var in self.variables:
            eq_list += [diff(self.equation, var)]

        return nonlinsolve(eq_list, list(self.variables))

    def multiplicity(self, point):
        """
        Returns the multiplicity of a singular point on the region.

        A singular point (x,y) of region is said to be of multiplicity m
        if all the partial derivatives off to order m - 1 vanish there.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
        >>> I.singular_points()
        {(0, 0, 0)}
        >>> I.multiplicity((0, 0, 0))
        2

        """
        if isinstance(point, Point):
            point = point.args

        modified_eq = self.equation

        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        if len(modified_eq.args) != 0:
            terms = modified_eq.args
            m = min(total_degree(term) for term in terms)
        else:
            terms = modified_eq
            m = total_degree(terms)

        return m

    def rational_parametrization(self, parameters=('t', 's'), reg_point=None):
        """
        Returns the rational parametrization of implicit region.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z, s, t
        >>> from sympy.vector import ImplicitRegion

        >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
        >>> parabola.rational_parametrization()
        (4/t**2, 4/t)

        >>> circle = ImplicitRegion((x, y), Eq(x**2 + y**2, 4))
        >>> circle.rational_parametrization()
        (4*t/(t**2 + 1), 4*t**2/(t**2 + 1) - 2)

        >>> I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> I.rational_parametrization()
        (t**2 - 1, t*(t**2 - 1))

        >>> cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> cubic_curve.rational_parametrization(parameters=(t))
        (t**2 - 1, t*(t**2 - 1))

        >>> sphere = ImplicitRegion((x, y, z), x**2 + y**2 + z**2 - 4)
        >>> sphere.rational_parametrization(parameters=(t, s))
        (-2 + 4/(s**2 + t**2 + 1), 4*s/(s**2 + t**2 + 1), 4*t/(s**2 + t**2 + 1))

        For some conics, regular_points() is unable to find a point on curve.
        To calulcate the parametric representation in such cases, user need
        to determine a point on the region and pass it using reg_point.

        >>> c = ImplicitRegion((x, y), (x  - 1/2)**2 + (y)**2 - (1/4)**2)
        >>> c.rational_parametrization(reg_point=(3/4, 0))
        (0.75 - 0.5/(t**2 + 1), -0.5*t/(t**2 + 1))

        References
        ==========

        - Christoph M. Hoffmann, "Conversion Methods between Parametric and
          Implicit Curves and Surfaces", Purdue e-Pubs, 1990. Available:
          https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1827&context=cstech

        """
        equation = self.equation
        degree = self.degree

        if degree == 1:
            if len(self.variables) == 1:
                return (equation,)
            elif len(self.variables) == 2:
                x, y = self.variables
                y_par = list(solveset(equation, y))[0]
                return x, y_par
            else:
                raise NotImplementedError()

        point = ()

        # Finding the (n - 1) fold point of the monoid of degree
        if degree == 2:
            # For degree 2 curves, either a regular point or a singular point can be used.
            if reg_point is not None:
                # Using point provided by the user as regular point
                point = reg_point
            else:
                if len(self.singular_points()) != 0:
                    point = list(self.singular_points())[0]
                else:
                    point = self.regular_point()

        if len(self.singular_points()) != 0:
            singular_points = self.singular_points()
            for spoint in singular_points:
                syms = Tuple(*spoint).free_symbols
                rep = dict.fromkeys(syms, 2)

                if len(syms) != 0:
                    spoint = tuple(s.subs(rep) for s in spoint)

                if self.multiplicity(spoint) == degree - 1:
                    point = spoint
                    break

        if len(point) == 0:
            # The region in not a monoid
            raise NotImplementedError()

        modified_eq = equation

        # Shifting the region such that fold point moves to origin
        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        hn = hn_1 = 0
        for term in modified_eq.args:
            if total_degree(term) == degree:
                hn += term
            else:
                hn_1 += term

        hn_1 = -1*hn_1

        if not isinstance(parameters, tuple):
            parameters = (parameters,)

        if len(self.variables) == 2:

            parameter1 = parameters[0]
            if parameter1 == 's':
                # To avoid name conflict between parameters
                s = _symbol('s_', real=True)
            else:
                s = _symbol('s', real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: s, self.variables[1]: t})
            hn_1 = hn_1.subs({self.variables[0]: s, self.variables[1]: t})

            x_par = (s*(hn_1/hn)).subs(s, 1) + point[0]
            y_par = (t*(hn_1/hn)).subs(s, 1) + point[1]

            return x_par, y_par

        elif len(self.variables) == 3:

            parameter1, parameter2 = parameters
            if 'r' in parameters:
                # To avoid name conflict between parameters
                r = _symbol('r_', real=True)
            else:
                r = _symbol('r', real=True)
            s = _symbol(parameter2, real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})
            hn_1 = hn_1.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})

            x_par = (r*(hn_1/hn)).subs(r, 1) + point[0]
            y_par = (s*(hn_1/hn)).subs(r, 1) + point[1]
            z_par = (t*(hn_1/hn)).subs(r, 1) + point[2]

            return x_par, y_par, z_par

        raise NotImplementedError()

def conic_coeff(variables, equation):
    if total_degree(equation) != 2:
        raise ValueError()
    x = variables[0]
    y = variables[1]

    equation = expand(equation)
    a = equation.coeff(x**2)
    b = equation.coeff(x*y)
    c = equation.coeff(y**2)
    d = equation.coeff(x, 1).coeff(y, 0)
    e = equation.coeff(y, 1).coeff(x, 0)
    f = equation.coeff(x, 0).coeff(y, 0)
    return a, b, c, d, e, f
