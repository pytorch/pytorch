from sympy.plotting.series import BaseSeries, GenericDataSeries
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {
    ('Plot.append', 'Plot.extend'): ['matplotlib'],
}


# Global variable
# Set to False when running tests / doctests so that the plots don't show.
_show = True

def unset_show():
    """
    Disable show(). For use in the tests.
    """
    global _show
    _show = False


def _deprecation_msg_m_a_r_f(attr):
    sympy_deprecation_warning(
        f"The `{attr}` property is deprecated. The `{attr}` keyword "
        "argument should be passed to a plotting function, which generates "
        "the appropriate data series. If needed, index the plot object to "
        "retrieve a specific data series.",
        deprecated_since_version="1.13",
        active_deprecations_target="deprecated-markers-annotations-fill-rectangles",
        stacklevel=4)


def _create_generic_data_series(**kwargs):
    keywords = ["annotations", "markers", "fill", "rectangles"]
    series = []
    for kw in keywords:
        dictionaries = kwargs.pop(kw, [])
        if dictionaries is None:
            dictionaries = []
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        for d in dictionaries:
            args = d.pop("args", [])
            series.append(GenericDataSeries(kw, *args, **d))
    return series


class Plot:
    """Base class for all backends. A backend represents the plotting library,
    which implements the necessary functionalities in order to use SymPy
    plotting functions.

    For interactive work the function :func:`plot` is better suited.

    This class permits the plotting of SymPy expressions using numerous
    backends (:external:mod:`matplotlib`, textplot, the old pyglet module for SymPy, Google
    charts api, etc).

    The figure can contain an arbitrary number of plots of SymPy expressions,
    lists of coordinates of points, etc. Plot has a private attribute _series that
    contains all data series to be plotted (expressions for lines or surfaces,
    lists of points, etc (all subclasses of BaseSeries)). Those data series are
    instances of classes not imported by ``from sympy import *``.

    The customization of the figure is on two levels. Global options that
    concern the figure as a whole (e.g. title, xlabel, scale, etc) and
    per-data series options (e.g. name) and aesthetics (e.g. color, point shape,
    line type, etc.).

    The difference between options and aesthetics is that an aesthetic can be
    a function of the coordinates (or parameters in a parametric plot). The
    supported values for an aesthetic are:

    - None (the backend uses default values)
    - a constant
    - a function of one variable (the first coordinate or parameter)
    - a function of two variables (the first and second coordinate or parameters)
    - a function of three variables (only in nonparametric 3D plots)

    Their implementation depends on the backend so they may not work in some
    backends.

    If the plot is parametric and the arity of the aesthetic function permits
    it the aesthetic is calculated over parameters and not over coordinates.
    If the arity does not permit calculation over parameters the calculation is
    done over coordinates.

    Only cartesian coordinates are supported for the moment, but you can use
    the parametric plots to plot in polar, spherical and cylindrical
    coordinates.

    The arguments for the constructor Plot must be subclasses of BaseSeries.

    Any global option can be specified as a keyword argument.

    The global options for a figure are:

    - title : str
    - xlabel : str or Symbol
    - ylabel : str or Symbol
    - zlabel : str or Symbol
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - axis : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - aspect_ratio : tuple of two floats or {'auto'}
    - autoscale : bool
    - margin : float in [0, 1]
    - backend : {'default', 'matplotlib', 'text'} or a subclass of BaseBackend
    - size : optional tuple of two floats, (width, height); default: None

    The per data series options and aesthetics are:
    There are none in the base series. See below for options for subclasses.

    Some data series support additional aesthetics or options:

    :class:`~.LineOver1DRangeSeries`, :class:`~.Parametric2DLineSeries`, and
    :class:`~.Parametric3DLineSeries` support the following:

    Aesthetics:

    - line_color : string, or float, or function, optional
        Specifies the color for the plot, which depends on the backend being
        used.

        For example, if ``MatplotlibBackend`` is being used, then
        Matplotlib string colors are acceptable (``"red"``, ``"r"``,
        ``"cyan"``, ``"c"``, ...).
        Alternatively, we can use a float number, 0 < color < 1, wrapped in a
        string (for example, ``line_color="0.5"``) to specify grayscale colors.
        Alternatively, We can specify a function returning a single
        float value: this will be used to apply a color-loop (for example,
        ``line_color=lambda x: math.cos(x)``).

        Note that by setting line_color, it would be applied simultaneously
        to all the series.

    Options:

    - label : str
    - steps : bool
    - integers_only : bool

    :class:`~.SurfaceOver2DRangeSeries` and :class:`~.ParametricSurfaceSeries`
    support the following:

    Aesthetics:

    - surface_color : function which returns a float.

    Notes
    =====

    How the plotting module works:

    1. Whenever a plotting function is called, the provided expressions are
       processed and a list of instances of the
       :class:`~sympy.plotting.series.BaseSeries` class is created, containing
       the necessary information to plot the expressions
       (e.g. the expression, ranges, series name, ...). Eventually, these
       objects will generate the numerical data to be plotted.
    2. A subclass of :class:`~.Plot` class is instantiaed (referred to as
       backend, from now on), which stores the list of series and the main
       attributes of the plot (e.g. axis labels, title, ...).
       The backend implements the logic to generate the actual figure with
       some plotting library.
    3. When the ``show`` command is executed, series are processed one by one
       to generate numerical data and add it to the figure. The backend is also
       going to set the axis labels, title, ..., according to the values stored
       in the Plot instance.

    The backend should check if it supports the data series that it is given
    (e.g. :class:`TextBackend` supports only
    :class:`~sympy.plotting.series.LineOver1DRangeSeries`).

    It is the backend responsibility to know how to use the class of data series
    that it's given. Note that the current implementation of the ``*Series``
    classes is "matplotlib-centric": the numerical data returned by the
    ``get_points`` and ``get_meshes`` methods is meant to be used directly by
    Matplotlib. Therefore, the new backend will have to pre-process the
    numerical data to make it compatible with the chosen plotting library.
    Keep in mind that future SymPy versions may improve the ``*Series`` classes
    in order to return numerical data "non-matplotlib-centric", hence if you code
    a new backend you have the responsibility to check if its working on each
    SymPy release.

    Please explore the :class:`MatplotlibBackend` source code to understand
    how a backend should be coded.

    In order to be used by SymPy plotting functions, a backend must implement
    the following methods:

    * show(self): used to loop over the data series, generate the numerical
        data, plot it and set the axis labels, title, ...
    * save(self, path): used to save the current plot to the specified file
        path.
    * close(self): used to close the current plot backend (note: some plotting
        library does not support this functionality. In that case, just raise a
        warning).
    """

    def __init__(self, *args,
        title=None, xlabel=None, ylabel=None, zlabel=None, aspect_ratio='auto',
        xlim=None, ylim=None, axis_center='auto', axis=True,
        xscale='linear', yscale='linear', legend=False, autoscale=True,
        margin=0, annotations=None, markers=None, rectangles=None,
        fill=None, backend='default', size=None, **kwargs):

        # Options for the graph as a whole.
        # The possible values for each option are described in the docstring of
        # Plot. They are based purely on convention, no checking is done.
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.aspect_ratio = aspect_ratio
        self.axis_center = axis_center
        self.axis = axis
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.autoscale = autoscale
        self.margin = margin
        self._annotations = annotations
        self._markers = markers
        self._rectangles = rectangles
        self._fill = fill

        # Contains the data objects to be plotted. The backend should be smart
        # enough to iterate over this list.
        self._series = []
        self._series.extend(args)
        self._series.extend(_create_generic_data_series(
            annotations=annotations, markers=markers, rectangles=rectangles,
            fill=fill))

        is_real = \
            lambda lim: all(getattr(i, 'is_real', True) for i in lim)
        is_finite = \
            lambda lim: all(getattr(i, 'is_finite', True) for i in lim)

        # reduce code repetition
        def check_and_set(t_name, t):
            if t:
                if not is_real(t):
                    raise ValueError(
                    "All numbers from {}={} must be real".format(t_name, t))
                if not is_finite(t):
                    raise ValueError(
                    "All numbers from {}={} must be finite".format(t_name, t))
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", xlim)
        self.ylim = None
        check_and_set("ylim", ylim)
        self.size = None
        check_and_set("size", size)

    @property
    def _backend(self):
        return self

    @property
    def backend(self):
        return type(self)

    def __str__(self):
        series_strs = [('[%d]: ' % i) + str(s)
                       for i, s in enumerate(self._series)]
        return 'Plot object containing:\n' + '\n'.join(series_strs)

    def __getitem__(self, index):
        return self._series[index]

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args

    def __delitem__(self, index):
        del self._series[index]

    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot's first series object to the first, use the
        ``append`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x*x, show=False)
           >>> p2 = plot(x, show=False)
           >>> p1.append(p2[0])
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        extend

        """
        if isinstance(arg, BaseSeries):
            self._series.append(arg)
        else:
            raise TypeError('Must specify element of plot to append.')

    def extend(self, arg):
        """Adds all series from another plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot to the first, use the ``extend`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x**2, show=False)
           >>> p2 = plot(x, -x, show=False)
           >>> p1.extend(p2)
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           [2]: cartesian line: -x for x over (-10.0, 10.0)
           >>> p1.show()

        """
        if isinstance(arg, Plot):
            self._series.extend(arg._series)
        elif is_sequence(arg):
            self._series.extend(arg)
        else:
            raise TypeError('Expecting Plot or sequence of BaseSeries')

    def show(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    # deprecations

    @property
    def markers(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        return self._markers

    @markers.setter
    def markers(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("markers")
        self._series.extend(_create_generic_data_series(markers=v))
        self._markers = v

    @property
    def annotations(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        return self._annotations

    @annotations.setter
    def annotations(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("annotations")
        self._series.extend(_create_generic_data_series(annotations=v))
        self._annotations = v

    @property
    def rectangles(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        return self._rectangles

    @rectangles.setter
    def rectangles(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("rectangles")
        self._series.extend(_create_generic_data_series(rectangles=v))
        self._rectangles = v

    @property
    def fill(self):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        return self._fill

    @fill.setter
    def fill(self, v):
        """.. deprecated:: 1.13"""
        _deprecation_msg_m_a_r_f("fill")
        self._series.extend(_create_generic_data_series(fill=v))
        self._fill = v
