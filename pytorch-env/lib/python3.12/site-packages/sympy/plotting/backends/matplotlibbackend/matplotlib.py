from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend
from sympy.printing.latex import latex


# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`


def _str_or_latex(label):
    if isinstance(label, Basic):
        return latex(label, mode='inline')
    return str(label)


def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start,
                          intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end,
                          intervaly.end, intervaly.start, None])
    else:
        #XXX Ugly hack. Matplotlib does not accept empty lists for ``fill``
        xlist.extend((None, None, None, None))
        ylist.extend((None, None, None, None))
    return xlist, ylist


# Don't have to check for the success of importing matplotlib in each case;
# we will only be using this backend if we can successfully import matploblib
class MatplotlibBackend(base_backend.Plot):
    """ This class implements the functionalities to use Matplotlib with SymPy
    plotting functions.
    """

    def __init__(self, *series, **kwargs):
        super().__init__(*series, **kwargs)
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.aspect = kwargs.get('aspect_ratio', 'auto')
        if self.aspect != 'auto':
            self.aspect = float(self.aspect[1]) / self.aspect[0]
        # PlotGrid can provide its figure and axes to be populated with
        # the data from the series.
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)

    def _create_figure(self):
        def set_spines(ax):
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if self._plotgrid_fig is not None:
            self.fig = self._plotgrid_fig
            self.ax = self._plotgrid_ax
            if not any(s.is_3D for s in self._series):
                set_spines(self.ax)
        else:
            self.fig = self.plt.figure(figsize=self.size)
            if any(s.is_3D for s in self._series):
                self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            else:
                self.ax = self.fig.add_subplot(1, 1, 1)
                set_spines(self.ax)

    @staticmethod
    def get_segments(x, y, z=None):
        """ Convert two list of coordinates to a list of segments to be used
        with Matplotlib's :external:class:`~matplotlib.collections.LineCollection`.

        Parameters
        ==========
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates

            z : list
                List of z-coordinates for a 3D line.
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _process_series(self, series, ax):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', import_kwargs={'fromlist': ['mplot3d']})

        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        for s in series:
            # Create the collections
            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                else:
                    x, y = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    segments = self.get_segments(x, y)
                    collection = self.LineCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    line, = ax.plot(x, y, label=lbl, color=s.line_color)
            elif s.is_contour:
                ax.contour(*s.get_data())
            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    art3d = mpl_toolkits.mplot3d.art3d
                    segments = self.get_segments(x, y, z)
                    collection = art3d.Line3DCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    ax.plot(x, y, z, label=lbl, color=s.line_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_3Dsurface:
                if s.is_parametric:
                    x, y, z, u, v = s.get_data()
                else:
                    x, y, z = s.get_data()
                collection = ax.plot_surface(x, y, z,
                    cmap=getattr(self.cm, 'viridis', self.cm.jet),
                    rstride=1, cstride=1, linewidth=0.1)
                if isinstance(s.surface_color, (float, int, Callable)):
                    color_array = s.get_color_array()
                    color_array = color_array.reshape(color_array.size)
                    collection.set_array(color_array)
                else:
                    collection.set_color(s.surface_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_implicit:
                points = s.get_data()
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    ax.fill(x, y, facecolor=s.line_color, edgecolor='None')
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    # XXX: ``contour`` plots multiple lines. Should be fixed.
                    ListedColormap = self.matplotlib.colors.ListedColormap
                    colormap = ListedColormap(["white", s.line_color])
                    xarray, yarray, zarray, plot_type = points
                    if plot_type == 'contour':
                        ax.contour(xarray, yarray, zarray, cmap=colormap)
                    else:
                        ax.contourf(xarray, yarray, zarray, cmap=colormap)
            elif s.is_generic:
                if s.type == "markers":
                    # s.rendering_kw["color"] = s.line_color
                    ax.plot(*s.args, **s.rendering_kw)
                elif s.type == "annotations":
                    ax.annotate(*s.args, **s.rendering_kw)
                elif s.type == "fill":
                    # s.rendering_kw["color"] = s.line_color
                    ax.fill_between(*s.args, **s.rendering_kw)
                elif s.type == "rectangles":
                    # s.rendering_kw["color"] = s.line_color
                    ax.add_patch(
                        self.matplotlib.patches.Rectangle(
                            *s.args, **s.rendering_kw))
            else:
                raise NotImplementedError(
                    '{} is not supported in the SymPy plotting module '
                    'with matplotlib backend. Please report this issue.'
                    .format(ax))

        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(ax, Axes3D):
            ax.autoscale_view(
                scalex=ax.get_autoscalex_on(),
                scaley=ax.get_autoscaley_on())
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
                ax.set_ylim(ylim)
            else:
                ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
                ax.set_zlim(zlim)
            else:
                ax.set_zlim([0, 1])

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(ax, Axes3D):
            ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(ax, Axes3D):
            ax.set_yscale(self.yscale)
        if not isinstance(ax, Axes3D) or self.matplotlib.__version__ >= '1.2.0':  # XXX in the distant future remove this check
            ax.set_autoscale_on(self.autoscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(ax, Axes3D):
                pass
            elif val == 'center':
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
            elif val == 'auto':
                xl, xh = ax.get_xlim()
                yl, yh = ax.get_ylim()
                pos_left = ('data', 0) if xl*xh <= 0 else 'center'
                pos_bottom = ('data', 0) if yl*yh <= 0 else 'center'
                ax.spines['left'].set_position(pos_left)
                ax.spines['bottom'].set_position(pos_bottom)
            else:
                ax.spines['left'].set_position(('data', val[0]))
                ax.spines['bottom'].set_position(('data', val[1]))
        if not self.axis:
            ax.set_axis_off()
        if self.legend:
            if ax.legend():
                ax.legend_.set_visible(self.legend)
        if self.margin:
            ax.set_xmargin(self.margin)
            ax.set_ymargin(self.margin)
        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            xlbl = _str_or_latex(self.xlabel)
            ax.set_xlabel(xlbl, position=(1, 0))
        if self.ylabel:
            ylbl = _str_or_latex(self.ylabel)
            ax.set_ylabel(ylbl, position=(0, 1))
        if isinstance(ax, Axes3D) and self.zlabel:
            zlbl = _str_or_latex(self.zlabel)
            ax.set_zlabel(zlbl, position=(0, 1))

        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        self.ax.set_aspect(self.aspect)


    def process_series(self):
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
        self._create_figure()
        self._process_series(self._series, self.ax)

    def show(self):
        self.process_series()
        #TODO after fixing https://github.com/ipython/ipython/issues/1255
        # you can uncomment the next line and remove the pyplot.show() call
        #self.fig.show()
        if base_backend._show:
            self.fig.tight_layout()
            self.plt.show()
        else:
            self.close()

    def save(self, path):
        self.process_series()
        self.fig.savefig(path)

    def close(self):
        self.plt.close(self.fig)
