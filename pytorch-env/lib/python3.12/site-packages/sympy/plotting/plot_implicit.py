"""Implicit plotting module for SymPy.

Explanation
===========

The module implements a data series called ImplicitSeries which is used by
``Plot`` class to plot implicit plots for different backends. The module,
by default, implements plotting using interval arithmetic. It switches to a
fall back algorithm if the expression cannot be plotted using interval arithmetic.
It is also possible to specify to use the fall back algorithm for all plots.

Boolean combinations of expressions cannot be plotted by the fall back
algorithm.

See Also
========

sympy.plotting.plot

References
==========

.. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
Mathematical Formulae with Two Free Variables.

.. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
Arithmetic. Master's thesis. University of Toronto, 1996

"""


from sympy.core.containers import Tuple
from sympy.core.symbol import (Dummy, Symbol)
from sympy.polys.polyutils import _sort_gens
from sympy.plotting.series import ImplicitSeries, _set_discretization_points
from sympy.plotting.plot import plot_factory
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten


__doctest_requires__ = {'plot_implicit': ['matplotlib']}


@doctest_depends_on(modules=('matplotlib',))
def plot_implicit(expr, x_var=None, y_var=None, adaptive=True, depth=0,
                  n=300, line_color="blue", show=True, **kwargs):
    """A plot function to plot implicit equations / inequalities.

    Arguments
    =========

    - expr : The equation / inequality that is to be plotted.
    - x_var (optional) : symbol to plot on x-axis or tuple giving symbol
      and range as ``(symbol, xmin, xmax)``
    - y_var (optional) : symbol to plot on y-axis or tuple giving symbol
      and range as ``(symbol, ymin, ymax)``

    If neither ``x_var`` nor ``y_var`` are given then the free symbols in the
    expression will be assigned in the order they are sorted.

    The following keyword arguments can also be used:

    - ``adaptive`` Boolean. The default value is set to True. It has to be
        set to False if you want to use a mesh grid.

    - ``depth`` integer. The depth of recursion for adaptive mesh grid.
        Default value is 0. Takes value in the range (0, 4).

    - ``n`` integer. The number of points if adaptive mesh grid is not
        used. Default value is 300. This keyword argument replaces ``points``,
        which should be considered deprecated.

    - ``show`` Boolean. Default value is True. If set to False, the plot will
        not be shown. See ``Plot`` for further information.

    - ``title`` string. The title for the plot.

    - ``xlabel`` string. The label for the x-axis

    - ``ylabel`` string. The label for the y-axis

    Aesthetics options:

    - ``line_color``: float or string. Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Default value is "Blue"

    plot_implicit, by default, uses interval arithmetic to plot functions. If
    the expression cannot be plotted using interval arithmetic, it defaults to
    a generating a contour using a mesh grid of fixed number of points. By
    setting adaptive to False, you can force plot_implicit to use the mesh
    grid. The mesh grid method can be effective when adaptive plotting using
    interval arithmetic, fails to plot with small line width.

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import plot_implicit, symbols, Eq, And
        >>> x, y = symbols('x y')

    Without any ranges for the symbols in the expression:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p1 = plot_implicit(Eq(x**2 + y**2, 5))

    With the range for the symbols:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p2 = plot_implicit(
        ...     Eq(x**2 + y**2, 3), (x, -3, 3), (y, -3, 3))

    With depth of recursion as argument:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p3 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -4, 4), (y, -4, 4), depth = 2)

    Using mesh grid and not using adaptive meshing:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p4 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2),
        ...     adaptive=False)

    Using mesh grid without using adaptive meshing with number of points
    specified:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p5 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2),
        ...     adaptive=False, n=400)

    Plotting regions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p6 = plot_implicit(y > x**2)

    Plotting Using boolean conjunctions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p7 = plot_implicit(And(y > x, y > -x))

    When plotting an expression with a single variable (y - 1, for example),
    specify the x or the y variable explicitly:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p8 = plot_implicit(y - 1, y_var=y)
        >>> p9 = plot_implicit(x - 1, x_var=x)
    """

    xyvar = [i for i in (x_var, y_var) if i is not None]
    free_symbols = expr.free_symbols
    range_symbols = Tuple(*flatten(xyvar)).free_symbols
    undeclared = free_symbols - range_symbols
    if len(free_symbols & range_symbols) > 2:
        raise NotImplementedError("Implicit plotting is not implemented for "
                                  "more than 2 variables")

    #Create default ranges if the range is not provided.
    default_range = Tuple(-5, 5)
    def _range_tuple(s):
        if isinstance(s, Symbol):
            return Tuple(s) + default_range
        if len(s) == 3:
            return Tuple(*s)
        raise ValueError('symbol or `(symbol, min, max)` expected but got %s' % s)

    if len(xyvar) == 0:
        xyvar = list(_sort_gens(free_symbols))
    var_start_end_x = _range_tuple(xyvar[0])
    x = var_start_end_x[0]
    if len(xyvar) != 2:
        if x in undeclared or not undeclared:
            xyvar.append(Dummy('f(%s)' % x.name))
        else:
            xyvar.append(undeclared.pop())
    var_start_end_y = _range_tuple(xyvar[1])

    kwargs = _set_discretization_points(kwargs, ImplicitSeries)
    series_argument = ImplicitSeries(
        expr, var_start_end_x, var_start_end_y,
        adaptive=adaptive, depth=depth,
        n=n, line_color=line_color)

    #set the x and y limits
    kwargs['xlim'] = tuple(float(x) for x in var_start_end_x[1:])
    kwargs['ylim'] = tuple(float(y) for y in var_start_end_y[1:])
    # set the x and y labels
    kwargs.setdefault('xlabel', var_start_end_x[0])
    kwargs.setdefault('ylabel', var_start_end_y[0])
    p = plot_factory(series_argument, **kwargs)
    if show:
        p.show()
    return p
