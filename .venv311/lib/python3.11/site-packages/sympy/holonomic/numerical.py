"""Numerical Methods for Holonomic Functions"""

from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs

from mpmath import mp


def _evalf(func, points, derivatives=False, method='RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """

    ann = func.annihilator
    a = ann.order
    R = ann.parent.base
    K = R.get_field()

    if method == 'Euler':
        meth = _euler
    else:
        meth = _rk4

    dmf = [K.new(j.to_list()) for j in ann.listofpoly]
    red = [-dmf[i] / dmf[a] for i in range(a)]

    y0 = func.y0
    if len(y0) < a:
        raise TypeError("Not Enough Initial Conditions")
    x0 = func.x0
    sol = [meth(red, x0, points[0], y0, a)]

    for i, j in enumerate(points[1:]):
        sol.append(meth(red, points[i], j, sol[-1], a))

    if not derivatives:
        return [sympify(i[0]) for i in sol]
    else:
        return sympify(sol)


def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A
    f_0 = y_0[1:]
    f_0_n = 0

    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    return [y_0[i] + h * f_0[i] for i in range(a)]


def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A

    f_0_n = 0
    f_1_n = 0
    f_2_n = 0
    f_3_n = 0

    f_0 = y_0[1:]
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    f_1 = [y_0[i] + f_0[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_1_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_0[i]*h/2)
    f_1.append(f_1_n)

    f_2 = [y_0[i] + f_1[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_2_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_1[i]*h/2)
    f_2.append(f_2_n)

    f_3 = [y_0[i] + f_2[i]*h for i in range(1, a)]
    for i in range(a):
        f_3_n += sympify(DMFsubs(red[i], A + h, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_2[i]*h)
    f_3.append(f_3_n)

    return [y_0[i] + h*(f_0[i]+2*f_1[i]+2*f_2[i]+f_3[i])/6 for i in range(a)]
