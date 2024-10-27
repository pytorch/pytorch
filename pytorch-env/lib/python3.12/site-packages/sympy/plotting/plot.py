"""Plotting module for SymPy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from SymPy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Do not use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""

from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.external import import_module
from sympy.functions import sign
from sympy.plotting.backends.base_backend import Plot
from sympy.plotting.backends.matplotlibbackend import MatplotlibBackend
from sympy.plotting.backends.textbackend import TextBackend
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, ContourSeries)
from sympy.plotting.utils import _check_arguments, _plot_sympify
from sympy.tensor.indexed import Indexed
# to maintain back-compatibility
from sympy.plotting.plotgrid import PlotGrid # noqa: F401
from sympy.plotting.series import BaseSeries # noqa: F401
from sympy.plotting.series import Line2DBaseSeries # noqa: F401
from sympy.plotting.series import Line3DBaseSeries # noqa: F401
from sympy.plotting.series import SurfaceBaseSeries # noqa: F401
from sympy.plotting.series import List2DSeries # noqa: F401
from sympy.plotting.series import GenericDataSeries # noqa: F401
from sympy.plotting.series import centers_of_faces # noqa: F401
from sympy.plotting.series import centers_of_segments # noqa: F401
from sympy.plotting.series import flat # noqa: F401
from sympy.plotting.backends.base_backend import unset_show # noqa: F401
from sympy.plotting.backends.matplotlibbackend import _matplotlib_list # noqa: F401
from sympy.plotting.textplot import textplot # noqa: F401


__doctest_requires__ = {
    ('plot3d',
     'plot3d_parametric_line',
     'plot3d_parametric_surface',
     'plot_parametric'): ['matplotlib'],
    # XXX: The plot doctest possibly should not require matplotlib. It fails at
    # plot(x**2, (x, -5, 5)) which should be fine for text backend.
    ('plot',): ['matplotlib'],
}


def _process_summations(sum_bound, *args):
    """Substitute oo (infinity) in the lower/upper bounds of a summation with
    some integer number.

    Parameters
    ==========

    sum_bound : int
        oo will be substituted with this integer number.
    *args : list/tuple
        pre-processed arguments of the form (expr, range, ...)

    Notes
    =====
    Let's consider the following summation: ``Sum(1 / x**2, (x, 1, oo))``.
    The current implementation of lambdify (SymPy 1.12 at the time of
    writing this) will create something of this form:
    ``sum(1 / x**2 for x in range(1, INF))``
    The problem is that ``type(INF)`` is float, while ``range`` requires
    integers: the evaluation fails.
    Instead of modifying ``lambdify`` (which requires a deep knowledge), just
    replace it with some integer number.
    """
    def new_bound(t, bound):
        if (not t.is_number) or t.is_finite:
            return t
        if sign(t) >= 0:
            return bound
        return -bound

    args = list(args)
    expr = args[0]

    # select summations whose lower/upper bound is infinity
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
    ])

    for t in list(expr.find(w)):
        sums_args = list(t.args)
        for i, a in enumerate(sums_args):
            if i > 0:
                sums_args[i] = (a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound))
        s = Sum(*sums_args)
        expr = expr.subs(t, s)
    args[0] = expr
    return args


def _build_line_series(*args, **kwargs):
    """Loop over the provided arguments and create the necessary line series.
    """
    series = []
    sum_bound = int(kwargs.get("sum_bound", 1000))
    for arg in args:
        expr, r, label, rendering_kw = arg
        kw = kwargs.copy()
        if rendering_kw is not None:
            kw["rendering_kw"] = rendering_kw
        # TODO: _process_piecewise check goes here
        if not callable(expr):
            arg = _process_summations(sum_bound, *arg)
        series.append(LineOver1DRangeSeries(*arg[:-1], **kw))
    return series


def _create_series(series_type, plot_expr, **kwargs):
    """Extract the rendering_kw dictionary from the provided arguments and
    create an appropriate data series.
    """
    series = []
    for args in plot_expr:
        kw = kwargs.copy()
        if args[-1] is not None:
            kw["rendering_kw"] = args[-1]
        series.append(series_type(*args[:-1], **kw))
    return series


def _set_labels(series, labels, rendering_kw):
    """Apply the `label` and `rendering_kw` keyword arguments to the series.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    if len(labels) > 0:
        if len(labels) == 1 and len(series) > 1:
            # if one label is provided and multiple series are being plotted,
            # set the same label to all data series. It maintains
            # back-compatibility
            labels *= len(series)
        if len(series) != len(labels):
            raise ValueError("The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")

        for s, l in zip(series, labels):
            s.label = l

    if rendering_kw:
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        elif len(series) != len(rendering_kw):
            raise ValueError("The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")
        for s, r in zip(series, rendering_kw):
            s.rendering_kw = r


def plot_factory(*args, **kwargs):
    backend = kwargs.pop("backend", "default")
    if isinstance(backend, str):
        if backend == "default":
            matplotlib = import_module('matplotlib',
                min_module_version='1.1.0', catch=(RuntimeError,))
            if matplotlib:
                return MatplotlibBackend(*args, **kwargs)
            return TextBackend(*args, **kwargs)
        return plot_backends[backend](*args, **kwargs)
    elif (type(backend) == type) and issubclass(backend, Plot):
        return backend(*args, **kwargs)
    else:
        raise TypeError("backend must be either a string or a subclass of ``Plot``.")


plot_backends = {
    'matplotlib': MatplotlibBackend,
    'text': TextBackend,
}


####New API for plotting module ####

# TODO: Add color arrays for plots.
# TODO: Add more plotting options for 3d plots.
# TODO: Adaptive sampling for 3D plots.

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The last argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        Typical usage examples are in the following:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str or expression, optional
        Label for the x-axis.

    ylabel : str or expression, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.annotate` method.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the :external:mod:`matplotlib`'s :external:func:`~matplotlib.pyplot.plot()` function
        along with the marker related keyworded arguments.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the :external:mod:`matplotlib`'s
        :external:class:`~matplotlib.patches.Rectangle` class.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.fill_between` method.

    adaptive : bool, optional
        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

        The plotting uses an adaptive algorithm which samples
        recursively to accurately plot. The adaptive algorithm uses a
        random point near the midpoint of two points that has to be
        further sampled. Hence the same plots can appear slightly
        different.

    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        `n` samples a maximum of `2^{n}` points.

        If the ``adaptive`` flag is set to ``False``, this will be
        ignored.

    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points. If the ``adaptive``
        flag is set to ``True``, this will be ignored.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

    Multiple plots with single range.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)

    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)

    No adaptive sampling.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, adaptive=False, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)

    See Also
    ========

    Plot, LineOver1DRangeSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    params = kwargs.get("params", None)
    free = set()
    for p in plot_expr:
        if not isinstance(p[1][0], str):
            free |= {p[1][0]}
        else:
            free |= {Symbol(p[1][0])}
    if params:
        free = free.difference(params.keys())
    x = free.pop() if free else Symbol("x")
    kwargs.setdefault('xlabel', x)
    kwargs.setdefault('ylabel', Function('f')(x))

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _build_line_series(*plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric((expr_x, expr_y), range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``.

        If the range is not specified, then a default range of (-10, 10)
        is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``. Specifies the
        number of the points used for the uniform sampling.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import plot_parametric, symbols, cos, sin
       >>> u = symbols('u')

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-10.0, 10.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-10.0, 10.0)

    A parametric plot with multiple expressions with different ranges
    for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u), (u, -5, 5)),
       ...     (cos(u), u, (u, -5, 5)))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-5.0, 5.0)

    Notes
    =====

    The plotting uses an adaptive algorithm which samples recursively to
    accurately plot the curve. The adaptive algorithm uses a random point
    near the midpoint of two points that has to be further sampled.
    Hence, repeating the same plot command can give slightly different
    results because of the random sampling.

    If there are multiple plots, then the same optional arguments are
    applied to all the plots drawn in the same canvas. If you want to
    set these options separately, you can index the returned ``Plot``
    object and set it.

    For example, when you specify ``line_color`` once, it would be
    applied simultaneously to both series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')

    If you want to specify the line color for the specific series, you
    should index each item and apply the property manually.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> p[0].line_color = 'red'
        >>> p.show()

    See Also
    ========

    Plot, Parametric2DLineSeries
    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along x.

    expr_y : Expression representing the function along y.

    expr_z : Expression representing the function along z.

    range : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the parameter variable, e.g., (u, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    n : int
        The range is uniformly sampled at ``n`` number of points.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    Aesthetics:

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str
        The label to the plot. It will be used when called with ``legend=True``
        to denote the function with the given label in the plot.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5)),
       ...     (sin(u), u**2, u, (u, -5, 5)))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-5.0, 5.0)


    See Also
    ========

    Plot, Parametric3DLineSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def _plot3d_plot_contour_helper(Series, *args, **kwargs):
    """plot3d and plot_contour are structurally identical. Let's reduce
    code repetition.
    """
    # NOTE: if this import would be at the top-module level, it would trigger
    # SymPy's optional-dependencies tests to fail.
    from sympy.vector import BaseScalar

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    free_x = set()
    free_y = set()
    _types = (Symbol, BaseScalar, Indexed, AppliedUndef)
    for p in plot_expr:
        free_x |= {p[1][0]} if isinstance(p[1][0], _types) else {Symbol(p[1][0])}
        free_y |= {p[2][0]} if isinstance(p[2][0], _types) else {Symbol(p[2][0])}
    x = free_x.pop() if free_x else Symbol("x")
    y = free_y.pop() if free_y else Symbol("y")
    kwargs.setdefault("xlabel", x)
    kwargs.setdefault("ylabel", y)
    kwargs.setdefault("zlabel", Function('f')(x, y))

    # if a polar discretization is requested and automatic labelling has ben
    # applied, hide the labels on the x-y axis.
    if kwargs.get("is_polar", False):
        if callable(kwargs["xlabel"]):
            kwargs["xlabel"] = ""
        if callable(kwargs["ylabel"]):
            kwargs["ylabel"] = ""

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Series, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    plots = plot_factory(*series, **kwargs)
    if kwargs.get("show", True):
        plots.show()
    return plots


def plot3d(*args, show=True, **kwargs):
    """
    Plots a 3D surface plot.

    Usage
    =====

    Single plot

    ``plot3d(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot3d(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot3d((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``SurfaceOver2DRangeSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot.
        See :class:`~.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with same range

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: -x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d((x**2 + y**2, (x, -5, 5), (y, -5, 5)),
       ...     (x*y, (x, -3, 3), (y, -3, 3)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: x*y for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    See Also
    ========

    Plot, SurfaceOver2DRangeSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(
        SurfaceOver2DRangeSeries, *args, **kwargs)


def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along ``x``.

    expr_y : Expression representing the function along ``y``.

    expr_z : Expression representing the function along ``z``.

    range_u : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the u variable, e.g. (u, 0, 5).

    range_v : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the v variable, e.g. (v, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    n1 : int
        The ``u`` range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_u``, which should be
        considered deprecated.

    n2 : int
        The ``v`` range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_v``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`~Plot` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots

def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`sympy.plotting.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(ContourSeries, *args, **kwargs)


def check_arguments(args, expr_len, nb_of_free_symbols):
    """
    Checks the arguments and converts into tuples of the
    form (exprs, ranges).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import check_arguments
       >>> x = symbols('x')
       >>> check_arguments([cos(x), sin(x)], 2, 1)
           [(cos(x), sin(x), (x, -10, 10))]

       >>> check_arguments([x, x**2], 1, 1)
           [(x, (x, -10, 10)), (x**2, (x, -10, 10))]
    """
    if not args:
        return []
    if expr_len > 1 and isinstance(args[0], Expr):
        # Multiple expressions same range.
        # The arguments are tuples when the expression length is
        # greater than 1.
        if len(args) < expr_len:
            raise ValueError("len(args) should not be less than expr_len")
        for i in range(len(args)):
            if isinstance(args[i], Tuple):
                break
        else:
            i = len(args) + 1

        exprs = Tuple(*args[:i])
        free_symbols = list(set().union(*[e.free_symbols for e in exprs]))
        if len(args) == expr_len + nb_of_free_symbols:
            #Ranges given
            plots = [exprs + Tuple(*args[expr_len:])]
        else:
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(len(free_symbols) - nb_of_free_symbols):
                ranges.append(Tuple(Dummy()) + default_range)
            plots = [exprs + Tuple(*ranges)]
        return plots

    if isinstance(args[0], Expr) or (isinstance(args[0], Tuple) and
                                     len(args[0]) == expr_len and
                                     expr_len != 3):
        # Cannot handle expressions with number of expression = 3. It is
        # not possible to differentiate between expressions and ranges.
        #Series of plots with same range
        for i in range(len(args)):
            if isinstance(args[i], Tuple) and len(args[i]) != expr_len:
                break
            if not isinstance(args[i], Tuple):
                args[i] = Tuple(args[i])
        else:
            i = len(args) + 1

        exprs = args[:i]
        assert all(isinstance(e, Expr) for expr in exprs for e in expr)
        free_symbols = list(set().union(*[e.free_symbols for expr in exprs
                                        for e in expr]))

        if len(free_symbols) > nb_of_free_symbols:
            raise ValueError("The number of free_symbols in the expression "
                             "is greater than %d" % nb_of_free_symbols)
        if len(args) == i + nb_of_free_symbols and isinstance(args[i], Tuple):
            ranges = Tuple(*list(args[
                           i:i + nb_of_free_symbols]))
            plots = [expr + ranges for expr in exprs]
            return plots
        else:
            # Use default ranges.
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(nb_of_free_symbols - len(free_symbols)):
                ranges.append(Tuple(Dummy()) + default_range)
            ranges = Tuple(*ranges)
            plots = [expr + ranges for expr in exprs]
            return plots

    elif isinstance(args[0], Tuple) and len(args[0]) == expr_len + nb_of_free_symbols:
        # Multiple plots with different ranges.
        for arg in args:
            for i in range(expr_len):
                if not isinstance(arg[i], Expr):
                    raise ValueError("Expected an expression, given %s" %
                                     str(arg[i]))
            for i in range(nb_of_free_symbols):
                if not len(arg[i + expr_len]) == 3:
                    raise ValueError("The ranges should be a tuple of "
                                     "length 3, got %s" % str(arg[i + expr_len]))
        return args
