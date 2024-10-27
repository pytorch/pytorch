### The base class for all series
from collections.abc import Callable
from sympy.calculus.util import continuous_domain
from sympy.concrete import Sum, Product
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions import atan2, zeta, frac, ceiling, floor, im
from sympy.core.relational import (Equality, GreaterThan,
    LessThan, Relational, Ne)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.utils import _get_free_symbols, extract_solution
from sympy.printing.latex import latex
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.sets.sets import Set, Interval, Union
from sympy.simplify.simplify import nsimplify
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.lambdify import lambdify
from .intervalmath import interval
import warnings


class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """
    def _print_And(self, expr):
        PREC = precedence(expr)
        return " & ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return " | ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))


def _uniform_eval(f1, f2, *args, modules=None,
    force_real_eval=False, has_sum=False):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
    np = import_module('numpy')

    def wrapper_func(func, *args):
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)

    # NOTE: np.vectorize is much slower than numpy vectorized operations.
    # However, this modules must be able to evaluate functions also with
    # mpmath or sympy.
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])

    def _eval_with_sympy(err=None):
        if f2 is None:
            msg = "Impossible to evaluate the provided numerical function"
            if err is None:
                msg += "."
            else:
                msg += "because the following exception was raised:\n"
                "{}: {}".format(type(err).__name__, err)
            raise RuntimeError(msg)
        if err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
        return wrapper_func(f2, *args)

    if modules == "sympy":
        return _eval_with_sympy()

    try:
        return wrapper_func(f1, *args)
    except Exception as err:
        return _eval_with_sympy(err)


def _adaptive_eval(f, x):
    """Evaluate f(x) with an adaptive algorithm. Post-process the result.
    If a symbolic expression is evaluated with SymPy, it might returns
    another symbolic expression, containing additions, ...
    Force evaluation to a float.

    Parameters
    ==========
    f : callable
    x : float
    """
    np = import_module('numpy')

    y = f(x)
    if isinstance(y, Expr) and (not y.is_Number):
        y = y.evalf()
    y = complex(y)
    if y.imag > 1e-08:
        return np.nan
    return y.real


def _get_wrapper_for_expr(ret):
    wrapper = "%s"
    if ret == "real":
        wrapper = "re(%s)"
    elif ret == "imag":
        wrapper = "im(%s)"
    elif ret == "abs":
        wrapper = "abs(%s)"
    elif ret == "arg":
        wrapper = "arg(%s)"
    return wrapper


class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that is given.
    (e.g. TextBackend supports only LineOver1DRangeSeries).
    It is the backend responsibility to know how to use the class of
    data series that is given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (e.g. LineOver1DRangeSeries belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).

    BaseSeries
    """

    # Some flags follow. The rationale for using flags instead of checking base
    # classes is that setting multiple flags is simpler than multiple
    # inheritance.

    is_2Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dsurface = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_contour = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_implicit = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x (1D array), mesh_y(1D array,
    #     mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes
    # Different from is_contour as the colormap in backend will be
    # different

    is_interactive = False
    # An interactive series can update its data.

    is_parametric = False
    # The calculation of aesthetics expects:
    #   - get_parameter_points returning one or two np.arrays (1D or 2D)
    # used for calculation aesthetics

    is_generic = False
    # Represent generic user-provided numerical data

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # Represents a 2D or 3D vector data series

    _N = 100
    # default number of discretization points for uniform sampling. Each
    # subclass can set its number.

    def __init__(self, *args, **kwargs):
        kwargs = _set_discretization_points(kwargs.copy(), type(self))
        # discretize the domain using only integer numbers
        self.only_integers = kwargs.get("only_integers", False)
        # represents the evaluation modules to be used by lambdify
        self.modules = kwargs.get("modules", None)
        # plot functions might create data series that might not be useful to
        # be shown on the legend, for example wireframe lines on 3D plots.
        self.show_in_legend = kwargs.get("show_in_legend", True)
        # line and surface series can show data with a colormap, hence a
        # colorbar is essential to understand the data. However, sometime it
        # is useful to hide it on series-by-series base. The following keyword
        # controls wheter the series should show a colorbar or not.
        self.colorbar = kwargs.get("colorbar", True)
        # Some series might use a colormap as default coloring. Setting this
        # attribute to False will inform the backends to use solid color.
        self.use_cm = kwargs.get("use_cm", False)
        # If True, the backend will attempt to render it on a polar-projection
        # axis, or using a polar discretization if a 3D plot is requested
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        # If True, the rendering will use points, not lines.
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        # some backend is able to render latex, other needs standard text
        self._label = self._latex_label = ""

        self._ranges = []
        self._n = [
            int(kwargs.get("n1", self._N)),
            int(kwargs.get("n2", self._N)),
            int(kwargs.get("n3", self._N))
        ]
        self._scales = [
            kwargs.get("xscale", "linear"),
            kwargs.get("yscale", "linear"),
            kwargs.get("zscale", "linear")
        ]

        # enable interactive widget plots
        self._params = kwargs.get("params", {})
        if not isinstance(self._params, dict):
            raise TypeError("`params` must be a dictionary mapping symbols "
                "to numeric values.")
        if len(self._params) > 0:
            self.is_interactive = True

        # contains keyword arguments that will be passed to the rendering
        # function of the chosen plotting library
        self.rendering_kw = kwargs.get("rendering_kw", {})

        # numerical transformation functions to be applied to the output data:
        # x, y, z (coordinates), p (parameter on parametric plots)
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        self._tp = kwargs.get("tp", None)
        if not all(callable(t) or (t is None) for t in
            [self._tx, self._ty, self._tz, self._tp]):
            raise TypeError("`tx`, `ty`, `tz`, `tp` must be functions.")

        # list of numerical functions representing the expressions to evaluate
        self._functions = []
        # signature for the numerical functions
        self._signature = []
        # some expressions don't like to be evaluated over complex data.
        # if that's the case, set this to True
        self._force_real_eval = kwargs.get("force_real_eval", None)
        # this attribute will eventually contain a dictionary with the
        # discretized ranges
        self._discretized_domain = None
        # wheter the series contains any interactive range, which is a range
        # where the minimum and maximum values can be changed with an
        # interactive widget
        self._interactive_ranges = False
        # NOTE: consider a generic summation, for example:
        #   s = Sum(cos(pi * x), (x, 1, y))
        # This gets lambdified to something:
        #   sum(cos(pi*x) for x in range(1, y+1))
        # Hence, y needs to be an integer, otherwise it raises:
        #   TypeError: 'complex' object cannot be interpreted as an integer
        # This list will contains symbols that are upper bound to summations
        # or products
        self._needs_to_be_int = []
        # a color function will be responsible to set the line/surface color
        # according to some logic. Each data series will et an appropriate
        # default value.
        self.color_func = None
        # NOTE: color_func usually receives numerical functions that are going
        # to be evaluated over the coordinates of the computed points (or the
        # discretized meshes).
        # However, if an expression is given to color_func, then it will be
        # lambdified with symbols in self._signature, and it will be evaluated
        # with the same data used to evaluate the plotted expression.
        self._eval_color_func_with_signature = False

    def _block_lambda_functions(self, *exprs):
        """Some data series can be used to plot numerical functions, others
        cannot. Execute this method inside the `__init__` to prevent the
        processing of numerical functions.
        """
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " requires a symbolic "
                "expression.")

    def _check_fs(self):
        """ Checks if there are enogh parameters and free symbols.
        """
        exprs, ranges = self.expr, self.ranges
        params, label = self.params, self.label
        exprs = exprs if hasattr(exprs, "__iter__") else [exprs]
        if any(callable(e) for e in exprs):
            return

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = _get_free_symbols(exprs)
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format(
                    (exprs, ranges, label) if ranges is not None else (exprs, label))
                + "params: {}\n".format(params)
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

        # verify that all symbols are known (they either represent plotting
        # ranges or parameters)
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any(t in fs for t in range_symbols):
                # ranges can't depend on each other, for example this are
                # not allowed:
                # (x, 0, y), (y, 0, 3)
                # (x, 0, y), (y, x + 2, 3)
                raise ValueError("Range symbols can't be included into "
                    "minimum and maximum of a range. "
                    "Received range: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError(
                    "Unkown symbols found in plotting range: %s. " % (r,) +
                    "Are the following parameters? %s" % remaining_fs)

    def _create_lambda_func(self):
        """Create the lambda functions to be used by the uniform meshing
        strategy.

        Notes
        =====
        The old sympy.plotting used experimental_lambdify. It created one
        lambda function each time an evaluation was requested. If that failed,
        it went on to create a different lambda function and evaluated it,
        and so on.

        This new module changes strategy: it creates right away the default
        lambda function as well as the backup one. The reason is that the
        series could be interactive, hence the numerical function will be
        evaluated multiple times. So, let's create the functions just once.

        This approach works fine for the majority of cases, in which the
        symbolic expression is relatively short, hence the lambdification
        is fast. If the expression is very long, this approach takes twice
        the time to create the lambda functions. Be aware of that!
        """
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if not any(callable(e) for e in exprs):
            fs = _get_free_symbols(exprs)
            self._signature = sorted(fs, key=lambda t: t.name)

            # Generate a list of lambda functions, two for each expression:
            # 1. the default one.
            # 2. the backup one, in case of failures with the default one.
            self._functions = []
            for e in exprs:
                # TODO: set cse=True once this issue is solved:
                # https://github.com/sympy/sympy/issues/24246
                self._functions.append([
                    lambdify(self._signature, e, modules=self.modules),
                    lambdify(self._signature, e, modules="sympy", dummify=True),
                ])
        else:
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            self._functions = [(e, None) for e in exprs]

        # deal with symbolic color_func
        if isinstance(self.color_func, Expr):
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        """If the value of a plotting range is a symbolic expression,
        substitute the parameters in order to get a numerical value.
        """
        if not self._interactive_ranges:
            return complex(t)
        return complex(t.subs(self.params))

    def _create_discretized_domain(self):
        """Discretize the ranges for uniform meshing strategy.
        """
        # NOTE: the goal is to create a dictionary stored in
        # self._discretized_domain, mapping symbols to a numpy array
        # representing the discretization
        discr_symbols = []
        discretizations = []

        # create a 1D discretization
        for i, r in enumerate(self.ranges):
            discr_symbols.append(r[0])
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.only_integers or (r[0] in self._needs_to_be_int)
            d = BaseSeries._discretize(start, end, self.n[i],
                scale=self.scales[i],
                only_integers=needs_integer_discr)

            if ((not self._force_real_eval) and (not needs_integer_discr) and
                (d.dtype != "complex")):
                d = d + 1j * c_start.imag

            if needs_integer_discr:
                d = d.astype(int)

            discretizations.append(d)

        # create 2D or 3D
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        """Create 2D or 3D discretized grids.

        Subclasses should override this method in order to implement a
        different behaviour.
        """
        np = import_module('numpy')

        # discretization suitable for 2D line plots, 3D surface plots,
        # contours plots, vector plots
        # NOTE: why indexing='ij'? Because it produces consistent results with
        # np.mgrid. This is important as Mayavi requires this indexing
        # to correctly compute 3D streamlines. While VTK is able to compute
        # streamlines regardless of the indexing, with indexing='xy' it
        # produces "strange" results with "voids" into the
        # discretization volume. indexing='ij' solves the problem.
        # Also note that matplotlib 2D streamlines requires indexing='xy'.
        indexing = "xy"
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = "ij"
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = dict(zip(discr_symbols, meshes))

    def _evaluate(self, cast_to_real=True):
        """Evaluation of the symbolic expression (or expressions) with the
        uniform meshing strategy, based on current values of the parameters.
        """
        np = import_module('numpy')

        # create lambda functions
        if not self._functions:
            self._create_lambda_func()
        # create (or update) the discretized domain
        if (not self._discretized_domain) or self._interactive_ranges:
            self._create_discretized_domain()
        # ensure that discretized domains are returned with the proper order
        discr = [self._discretized_domain[s[0]] for s in self.ranges]

        args = self._aggregate_args()

        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)
            # the evaluation might produce an int/float. Need this correction.
            r = self._correct_shape(np.array(r), discr[0])
            # sometime the evaluation is performed over arrays of type object.
            # hence, `result` might be of type object, which don't work well
            # with numpy real and imag functions.
            r = r.astype(complex)
            results.append(r)

        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        """Create a list of arguments to be passed to the lambda function,
        sorted accoring to self._signature.
        """
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(
                    int(self._params[s]) if s in self._needs_to_be_int else
                    self._params[s] if self._force_real_eval
                    else complex(self._params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    @property
    def expr(self):
        """Return the expression (or expressions) of the series."""
        return self._expr

    @expr.setter
    def expr(self, e):
        """Set the expression (or expressions) of the series."""
        is_iter = hasattr(e, "__iter__")
        is_callable = callable(e) if not is_iter else any(callable(t) for t in e)
        if is_callable:
            self._expr = e
        else:
            self._expr = sympify(e) if not is_iter else Tuple(*e)

            # look for the upper bound of summations and products
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)

            # list of sympy functions that when lambdified, the corresponding
            # numpy functions don't like complex-type arguments
            pf = [ceiling, floor, atan2, frac, zeta]
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                if self._force_real_eval and ((self.modules is None) or
                    (isinstance(self.modules, str) and "numpy" in self.modules)):
                    funcs = [f for f, c in zip(pf, check_res) if c]
                    warnings.warn("NumPy is unable to evaluate with complex "
                        "numbers some of the functions included in this "
                        "symbolic expression: %s. " % funcs +
                        "Hence, the evaluation will use real numbers. "
                        "If you believe the resulting plot is incorrect, "
                        "change the evaluation module by setting the "
                        "`modules` keyword argument.")
            if self._functions:
                # update lambda functions
                self._create_lambda_func()

    @property
    def is_3D(self):
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        """This method enables back-compatibility with old sympy.plotting"""
        # NOTE: color_func is set inside the init method of the series.
        # If line_color/surface_color is not a callable, then color_func will
        # be set to None.
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        self._line_surface_color("_line_color", val)

    @property
    def n(self):
        """Returns a list [n1, n2, n3] of numbers of discratization points.
        """
        return self._n

    @n.setter
    def n(self, v):
        """Set the numbers of discretization points. ``v`` must be an int or
        a list.

        Let ``s`` be a series. Then:

        * to set the number of discretization points along the x direction (or
          first parameter): ``s.n = 10``
        * to set the number of discretization points along the x and y
          directions (or first and second parameters): ``s.n = [10, 15]``
        * to set the number of discretization points along the x, y and z
          directions: ``s.n = [10, 15, 20]``

        The following is highly unreccomended, because it prevents
        the execution of necessary code in order to keep updated data:
        ``s.n[1] = 15``
        """
        if not hasattr(v, "__iter__"):
            self._n[0] = v
        else:
            self._n[:len(v)] = v
        if self._discretized_domain:
            # update the discretized domain
            self._create_discretized_domain()

    @property
    def params(self):
        """Get or set the current parameters dictionary.

        Parameters
        ==========

        p : dict

            * key: symbol associated to the parameter
            * val: the numeric value
        """
        return self._params

    @params.setter
    def params(self, p):
        self._params = p

    def _post_init(self):
        exprs = self.expr if hasattr(self.expr, "__iter__") else [self.expr]
        if any(callable(e) for e in exprs) and self.params:
            raise TypeError("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "user-provided numerical functions.")

        # if the expressions is a lambda function and no label has been
        # provided, then its better to do the following in order to avoid
        # suprises on the backend
        if any(callable(e) for e in exprs):
            if self._label == str(self.expr):
                self.label = ""

        self._check_fs()

        if hasattr(self, "adaptive") and self.adaptive and self.params:
            warnings.warn("`params` was provided, hence an interactive plot "
                "is expected. However, interactive plots do not support "
                "adaptive evaluation. Automatically switched to "
                "adaptive=False.")
            self.adaptive = False

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, v):
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        self._line_surface_color("_surface_color", val)

    @property
    def rendering_kw(self):
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            self._rendering_kw = {}
            if kwargs is not None:
                warnings.warn(
                    "`rendering_kw` must be a dictionary, instead an "
                    "object of type %s was received. " % type(kwargs) +
                    "Automatically setting `rendering_kw` to an empty "
                    "dictionary")

    @staticmethod
    def _discretize(start, end, N, scale="linear", only_integers=False):
        """Discretize a 1D domain.

        Returns
        =======

        domain : np.ndarray with dtype=float or complex
            The domain's dtype will be float or complex (depending on the
            type of start/end) even if only_integers=True. It is left for
            the downstream code to perform further casting, if necessary.
        """
        np = import_module('numpy')

        if only_integers is True:
            start, end = int(start), int(end)
            N = end - start + 1

        if scale == "linear":
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_shape(a, b):
        """Convert ``a`` to a np.ndarray of the same shape of ``b``.

        Parameters
        ==========

        a : int, float, complex, np.ndarray
            Usually, this is the result of a numerical evaluation of a
            symbolic expression. Even if a discretized domain was used to
            evaluate the function, the result can be a scalar (int, float,
            complex). Think for example to ``expr = Float(2)`` and
            ``f = lambdify(x, expr)``. No matter the shape of the numerical
            array representing x, the result of the evaluation will be
            a single value.

        b : np.ndarray
            It represents the correct shape that ``a`` should have.

        Returns
        =======
        new_a : np.ndarray
            An array with the correct shape.
        """
        np = import_module('numpy')

        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if a.shape != b.shape:
            if a.shape == ():
                a = a * np.ones_like(b)
            else:
                a = a.reshape(b.shape)
        return a

    def eval_color_func(self, *args):
        """Evaluate the color function.

        Parameters
        ==========

        args : tuple
            Arguments to be passed to the coloring function. Can be coordinates
            or parameters or both.

        Notes
        =====

        The backend will request the data series to generate the numerical
        data. Depending on the data series, either the data series itself or
        the backend will eventually execute this function to generate the
        appropriate coloring value.
        """
        np = import_module('numpy')
        if self.color_func is None:
            # NOTE: with the line_color and surface_color attributes
            # (back-compatibility with the old sympy.plotting module) it is
            # possible to create a plot with a callable line_color (or
            # surface_color). For example:
            # p = plot(sin(x), line_color=lambda x, y: -y)
            # This creates a ColoredLineOver1DRangeSeries with line_color=None
            # and color_func=lambda x, y: -y, which efffectively is a
            # parametric series. Later we could change it to a string value:
            # p[0].line_color = "red"
            # However, this sets ine_color="red" and color_func=None, but the
            # series is still ColoredLineOver1DRangeSeries (a parametric
            # series), which will render using a color_func...
            warnings.warn("This is likely not the result you were "
                "looking for. Please, re-execute the plot command, this time "
                "with the appropriate an appropriate value to line_color "
                "or surface_color.")
            return np.ones_like(args[0])

        if self._eval_color_func_with_signature:
            args = self._aggregate_args()
            color = self.color_func(*args)
            _re, _im = np.real(color), np.imag(color)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re

        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    # ColoredLineOver1DRangeSeries
                    return self._correct_shape(self.color_func(args[0]), args[0])
                # Parametric2DLineSeries
                return self._correct_shape(self.color_func(args[2]), args[2])
            elif self.is_3Dline and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            return self._correct_shape(self.color_func(args[0]), args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(*args[3:]), args[3])
            return self._correct_shape(self.color_func(*args[:2]), args[0])
        return self._correct_shape(self.color_func(*args[:nargs]), args[0])

    def get_data(self):
        """Compute and returns the numerical data.

        The number of parameters returned by this method depends on the
        specific instance. If ``s`` is the series, make sure to read
        ``help(s.get_data)`` to understand what it returns.
        """
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        """Given a latex representation of an expression, wrap it inside
        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".
        """
        return wrapper % label

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self.expr):
            # when the backend requests a latex label and user didn't provide
            # any label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

    @property
    def label(self):
        return self.get_label()

    @label.setter
    def label(self, val):
        """Set the labels associated to this series."""
        # NOTE: the init method of any series requires a label. If the user do
        # not provide it, the preprocessing function will set label=None, which
        # informs the series to initialize two attributes:
        # _label contains the string representation of the expression.
        # _latex_label contains the latex representation of the expression.
        self._label = self._latex_label = val

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals

    def _apply_transform(self, *args):
        """Apply transformations to the results of numerical evaluation.

        Parameters
        ==========
        args : tuple
            Results of numerical evaluation.

        Returns
        =======
        transformed_args : tuple
            Tuple containing the transformed results.
        """
        t = lambda x, transform: x if transform is None else transform(x)
        x, y, z = None, None, None
        if len(args) == 2:
            x, y = args
            return t(x, self._tx), t(y, self._ty)
        elif (len(args) == 3) and isinstance(self, Parametric2DLineSeries):
            x, y, u = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        elif len(args) == 3:
            x, y, z = args
            return t(x, self._tx), t(y, self._ty), t(z, self._tz)
        elif (len(args) == 4) and isinstance(self, Parametric3DLineSeries):
            x, y, z, u = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        elif len(args) == 4: # 2D vector plot
            x, y, u, v = args
            return (
                t(x, self._tx), t(y, self._ty),
                t(u, self._tx), t(v, self._ty)
            )
        elif (len(args) == 5) and isinstance(self, ParametricSurfaceSeries):
            x, y, z, u, v = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        elif (len(args) == 6) and self.is_3Dvector: # 3D vector plot
            x, y, z, u, v, w = args
            return (
                t(x, self._tx), t(y, self._ty), t(z, self._tz),
                t(u, self._tx), t(v, self._ty), t(w, self._tz)
            )
        elif len(args) == 6: # complex plot
            x, y, _abs, _arg, img, colors = args
            return (
                x, y, t(_abs, self._tz), _arg, img, colors)
        return args

    def _str_helper(self, s):
        pre, post = "", ""
        if self.is_interactive:
            pre = "interactive "
            post = " and parameters " + str(tuple(self.params.keys()))
        return pre + s + post


def _detect_poles_numerical_helper(x, y, eps=0.01, expr=None, symb=None, symbolic=False):
    """Compute the steepness of each segment. If it's greater than a
    threshold, set the right-point y-value non NaN and record the
    corresponding x-location for further processing.

    Returns
    =======
    x : np.ndarray
        Unchanged x-data.
    yy : np.ndarray
        Modified y-data with NaN values.
    """
    np = import_module('numpy')

    yy = y.copy()
    threshold = np.pi / 2 - eps
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = abs(y[i + 1] - y[i])
        angle = np.arctan(dy / dx)
        if abs(angle) >= threshold:
            yy[i + 1] = np.nan

    return x, yy

def _detect_poles_symbolic_helper(expr, symb, start, end):
    """Attempts to compute symbolic discontinuities.

    Returns
    =======
    pole : list
        List of symbolic poles, possibily empty.
    """
    poles = []
    interval = Interval(nsimplify(start), nsimplify(end))
    res = continuous_domain(expr, symb, interval)
    res = res.simplify()
    if res == interval:
        pass
    elif (isinstance(res, Union) and
        all(isinstance(t, Interval) for t in res.args)):
        poles = []
        for s in res.args:
            if s.left_open:
                poles.append(s.left)
            if s.right_open:
                poles.append(s.right)
        poles = list(set(poles))
    else:
        raise ValueError(
            f"Could not parse the following object: {res} .\n"
            "Please, submit this as a bug. Consider also to set "
            "`detect_poles=True`."
        )
    return poles


### 2D lines
class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """

    is_2Dline = True
    _dim = 2
    _N = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get("steps", False)
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.adaptive = kwargs.get("adaptive", False)
        self.depth = kwargs.get('depth', 12)
        self.use_cm = kwargs.get("use_cm", False)
        self.color_func = kwargs.get("color_func", None)
        self.line_color = kwargs.get("line_color", None)
        self.detect_poles = kwargs.get("detect_poles", False)
        self.eps = kwargs.get("eps", 0.01)
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.unwrap = kwargs.get("unwrap", False)
        # when detect_poles="symbolic", stores the location of poles so that
        # they can be appropriately rendered
        self.poles_locations = []
        exclude = kwargs.get("exclude", [])
        if isinstance(exclude, Set):
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, "__iter__"):
            exclude = [exclude]
        exclude = [float(e) for e in exclude]
        self.exclude = sorted(exclude)

    def get_data(self):
        """Return coordinates for plotting the line.

        Returns
        =======

        x: np.ndarray
            x-coordinates

        y: np.ndarray
            y-coordinates

        z: np.ndarray (optional)
            z-coordinates in case of Parametric3DLineSeries,
            Parametric3DLineInteractiveSeries

        param : np.ndarray (optional)
            The parameter in case of Parametric2DLineSeries,
            Parametric3DLineSeries or AbsArgLineSeries (and their
            corresponding interactive series).
        """
        np = import_module('numpy')
        points = self._get_data_helper()

        if (isinstance(self, LineOver1DRangeSeries) and
            (self.detect_poles == "symbolic")):
            poles = _detect_poles_symbolic_helper(
                self.expr.subs(self.params), *self.ranges[0])
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = t(np.array(poles), self._tx)

        # postprocessing
        points = self._apply_transform(*points)

        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                x, y = points
                x, y = _detect_poles_numerical_helper(
                    x, y, self.eps)
                points = (x, y)
            else:
                x, y, p = points
                x, y = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)

        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            if self.is_2Dline:
                if len(points) == 2:
                    x, y = points
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    x, y, p = points
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)

        if self.steps is True:
            if self.is_2Dline:
                x, y = points[0], points[1]
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)

        if len(self.exclude) > 0:
            points = self._insert_exclusions(points)
        return points

    def get_segments(self):
        sympy_deprecation_warning(
            """
            The Line2DBaseSeries.get_segments() method is deprecated.

            Instead, use the MatplotlibBackend.get_segments() method, or use
            The get_points() or get_data() methods.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-get-segments")

        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _insert_exclusions(self, points):
        """Add NaN to each of the exclusion point. Practically, this adds a
        NaN to the exlusion point, plus two other nearby points evaluated with
        the numerical functions associated to this data series.
        These nearby points are important when the number of discretization
        points is low, or the scale is logarithm.

        NOTE: it would be easier to just add exclusion points to the
        discretized domain before evaluation, then after evaluation add NaN
        to the exclusion points. But that's only work with adaptive=False.
        The following approach work even with adaptive=True.
        """
        np = import_module("numpy")
        points = list(points)
        n = len(points)
        # index of the x-coordinate (for 2d plots) or parameter (for 2d/3d
        # parametric plots)
        k = n - 1
        if n == 2:
            k = 0
        # indeces of the other coordinates
        j_indeces = sorted(set(range(n)).difference([k]))
        # TODO: for now, I assume that numpy functions are going to succeed
        funcs = [f[0] for f in self._functions]

        for e in self.exclude:
            res = points[k] - e >= 0
            # if res contains both True and False, ie, if e is found
            if any(res) and any(~res):
                idx = np.nanargmax(res)
                # select the previous point with respect to e
                idx -= 1
                # TODO: what if points[k][idx]==e or points[k][idx+1]==e?

                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])
                    delta_post = abs(e - points[k][idx + 1])
                    delta = min(delta_prev, delta_post) / 100
                    prev = e - delta
                    post = e + delta

                    # add points to the x-coord or the parameter
                    points[k] = np.concatenate(
                        (points[k][:idx], [prev, e, post], points[k][idx+1:]))

                    # add points to the other coordinates
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))
                        c += 1
                        points[j] = np.concatenate(
                            (points[j][:idx], [values[0], np.nan, values[1]], points[j][idx+1:]))
        return points

    @property
    def var(self):
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end(self):
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def xscale(self):
        return self._scales[0]

    @xscale.setter
    def xscale(self, v):
        self.scales = v

    def get_color_array(self):
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:  # only if the line is 3D (otherwise raises an error)
                    return f(*variables)
        else:
            return c*np.ones(self.nb_of_points)


class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label="", **kwargs):
        super().__init__(**kwargs)
        np = import_module('numpy')
        if len(list_x) != len(list_y):
            raise ValueError(
                "The two lists of coordinates must have the same "
                "number of elements.\n"
                "Received: len(list_x) = {} ".format(len(list_x)) +
                "and len(list_y) = {}".format(len(list_y))
            )
        self._block_lambda_functions(list_x, list_y)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists "
                    "are symbolic expressions, but the ``params`` dictionary "
                    "was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)
            self.list_y = np.array(list_y, dtype=np.float64)

        self._expr = (self.list_x, self.list_y)
        if not any(isinstance(t, np.ndarray) for t in [self.list_x, self.list_y]):
            self._check_fs()
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.label = label
        self.rendering_kw = kwargs.get("rendering_kw", {})
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError(
                    "%s don't support symbolic " % self.__class__.__name__ +
                    "expression for `color_func`.")

    def __str__(self):
        return "2D list plot"

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed."""
        lx, ly = self.list_x, self.list_y

        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)

        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._eval_color_func_and_return(lx, ly)

    def _eval_color_func_and_return(self, *data):
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]
        return data


class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        self._cast = complex
        # for complex-related data series, this determines what data to return
        # on the y-axis
        self._return = kwargs.get("return", None)
        self._post_init()

        if not self._interactive_ranges:
            # NOTE: the following check is only possible when the minimum and
            # maximum values of a plotting range are numeric
            start, end = [complex(t) for t in self.ranges[0][1:]]
            if im(start) != im(end):
                raise ValueError(
                    "%s requires the imaginary " % self.__class__.__name__ +
                    "part of the start and end values of the range "
                    "to be the same.")

        if self.adaptive and self._return:
            warnings.warn("The adaptive algorithm is unable to deal with "
                "complex numbers. Automatically switching to uniform meshing.")
            self.adaptive = False

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v

    def __str__(self):
        def f(t):
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = "interactive " if self.is_interactive else ""
        post = ""
        if self.is_interactive:
            post = " and parameters " + str(tuple(self.params.keys()))
        wrapper = _get_wrapper_for_expr(self._return)
        return pre + "cartesian line: %s for %s over %s" % (
            wrapper % self.expr,
            str(self.var),
            str((f(self.start), f(self.end))),
        ) + post

    def get_points(self):
        """Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates
        """
        return self._get_data_helper()

    def _adaptive_sampling(self):
        try:
            if callable(self.expr):
                f = self.expr
            else:
                f = lambdify([self.var], self.expr, self.modules)
            x, y = self._adaptive_sampling_helper(f)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f = lambdify([self.var], self.expr, "sympy")
            x, y = self._adaptive_sampling_helper(f)
        return x, y

    def _adaptive_sampling_helper(self, f):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
               Luiz Henrique de Figueiredo.
        """
        np = import_module('numpy')

        x_coords = []
        y_coords = []
        def sample(p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            random = 0.45 + np.random.rand() * 0.1
            if self.xscale == 'log':
                xnew = 10**(np.log10(p[0]) + random * (np.log10(q[0]) -
                                                        np.log10(p[0])))
            else:
                xnew = p[0] + random * (q[0] - p[0])
            ynew = _adaptive_eval(f, xnew)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])

            # Sample to depth of 6 (whether the line is flat or not)
            # without using linspace (to avoid aliasing).
            elif depth < 6:
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif p[1] is None and q[1] is None:
                if self.xscale == 'log':
                    xarray = np.logspace(p[0], q[0], 10)
                else:
                    xarray = np.linspace(p[0], q[0], 10)
                yarray = list(map(f, xarray))
                if not all(y is None for y in yarray):
                    for i in range(len(yarray) - 1):
                        if not (yarray[i] is None and yarray[i + 1] is None):
                            sample([xarray[i], yarray[i]],
                                [xarray[i + 1], yarray[i + 1]], depth + 1)

            # Sample further if one of the end points in None (i.e. a
            # complex value) or the three points are not almost collinear.
            elif (p[1] is None or q[1] is None or new_point[1] is None
                    or not flat(p, new_point, q)):
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])

        f_start = _adaptive_eval(f, self.start.real)
        f_end = _adaptive_eval(f, self.end.real)
        x_coords.append(self.start.real)
        y_coords.append(f_start)
        sample(np.array([self.start.real, f_start]),
                np.array([self.end.real, f_end]), 0)

        return (x_coords, y_coords)

    def _uniform_sampling(self):
        np = import_module('numpy')

        x, result = self._evaluate()
        _re, _im = np.real(result), np.imag(result)
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        return x, _re, _im

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        """
        np = import_module('numpy')
        if self.adaptive and (not self.only_integers):
            x, y = self._adaptive_sampling()
            return [np.array(t) for t in [x, y]]

        x, _re, _im = self._uniform_sampling()

        if self._return is None:
            # The evaluation could produce complex numbers. Set real elements
            # to NaN where there are non-zero imaginary elements
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == "real":
            pass
        elif self._return == "imag":
            _re = _im
        elif self._return == "abs":
            _re = np.sqrt(_re**2 + _im**2)
        elif self._return == "arg":
            _re = np.arctan2(_im, _re)
        else:
            raise ValueError("`_return` not recognized. "
                "Received: %s" % self._return)

        return x, _re


class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True

    def _set_parametric_line_label(self, label):
        """Logic to set the correct label to be shown on the plot.
        If `use_cm=True` there will be a colorbar, so we show the parameter.
        If `use_cm=False`, there might be a legend, so we show the expressions.

        Parameters
        ==========
        label : str
            label passed in by the pre-processor or the user
        """
        self._label = str(self.var) if label is None else label
        self._latex_label = latex(self.var) if label is None else label
        if (self.use_cm is False) and (self._label == str(self.var)):
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        # if the expressions is a lambda function and use_cm=False and no label
        # has been provided, then its better to do the following in order to
        # avoid suprises on the backend
        if any(callable(e) for e in self.expr) and (not self.use_cm):
            if self._label == str(self.expr):
                self._label = ""

    def get_label(self, use_latex=False, wrapper="$%s$"):
        # parametric lines returns the representation of the parameter to be
        # shown on the colorbar if `use_cm=True`, otherwise it returns the
        # representation of the expression to be placed on the legend.
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            # here the user has provided a custom label
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        """Returns coordinates that needs to be postprocessed.
        Depending on the `adaptive` option, this function will either use an
        adaptive algorithm or it will uniformly sample the expression over the
        provided range.
        """
        if self.adaptive:
            np = import_module("numpy")
            coords = self._adaptive_sampling()
            coords = [np.array(t) for t in coords]
        else:
            coords = self._uniform_sampling()

        if self.is_2Dline and self.is_polar:
            # when plot_polar is executed with polar_axis=True
            np = import_module('numpy')
            x, y, _ = coords
            r = np.sqrt(x**2 + y**2)
            t = np.arctan2(y, x)
            coords = [t, r, coords[-1]]

        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)

        return coords

    def _uniform_sampling(self):
        """Returns coordinates that needs to be postprocessed."""
        np = import_module('numpy')

        results = self._evaluate()
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        return [*results[1:], results[0]]

    def get_parameter_points(self):
        return self.get_data()[-1]

    def get_points(self):
        """ Return lists of coordinates for plotting. Depending on the
        ``adaptive`` option, this function will either use an adaptive algorithm
        or it will uniformly sample the expression over the provided range.

        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.

        Returns
        =======
            x : list
                List of x-coordinates
            y : list
                List of y-coordinates
            z : list
                List of z-coordinates, only for 3D parametric line plot.
        """
        return self._get_data_helper()[:-1]

    @property
    def nb_of_points(self):
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        self.n = v


class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""

    is_2Dline = True

    def __init__(self, expr_x, expr_y, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        return self._str_helper(
            "parametric cartesian line: (%s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.var),
            str((self.start, self.end))
        ))

    def _adaptive_sampling(self):
        try:
            if callable(self.expr_x) and callable(self.expr_y):
                f_x = self.expr_x
                f_y = self.expr_y
            else:
                f_x = lambdify([self.var], self.expr_x)
                f_y = lambdify([self.var], self.expr_y)
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        except Exception as err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not self.modules else self.modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
            f_x = lambdify([self.var], self.expr_x, "sympy")
            f_y = lambdify([self.var], self.expr_y, "sympy")
            x, y, p = self._adaptive_sampling_helper(f_x, f_y)
        return x, y, p

    def _adaptive_sampling_helper(self, f_x, f_y):
        """The adaptive sampling is done by recursively checking if three
        points are almost collinear. If they are not collinear, then more
        points are added between those points.

        References
        ==========

        .. [1] Adaptive polygonal approximation of parametric curves,
            Luiz Henrique de Figueiredo.
        """
        x_coords = []
        y_coords = []
        param = []

        def sample(param_p, param_q, p, q, depth):
            """ Samples recursively if three points are almost collinear.
            For depth < 6, points are added irrespective of whether they
            satisfy the collinearity condition or not. The maximum depth
            allowed is 12.
            """
            # Randomly sample to avoid aliasing.
            np = import_module('numpy')
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = _adaptive_eval(f_x, param_new)
            ynew = _adaptive_eval(f_y, param_new)
            new_point = np.array([xnew, ynew])

            # Maximum depth
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

            # Sample irrespective of whether the line is flat till the
            # depth of 6. We are not using linspace to avoid aliasing.
            elif depth < 6:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)

            # Sample ten points if complex values are encountered
            # at both ends. If there is a real value in between, then
            # sample those points further.
            elif ((p[0] is None and q[1] is None) or
                    (p[1] is None and q[1] is None)):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = [_adaptive_eval(f_x, t) for t in param_array]
                y_array = [_adaptive_eval(f_y, t) for t in param_array]
                if not all(x is None and y is None
                           for x, y in zip(x_array, y_array)):
                    for i in range(len(y_array) - 1):
                        if ((x_array[i] is not None and y_array[i] is not None) or
                                (x_array[i + 1] is not None and y_array[i + 1] is not None)):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(param_array[i], param_array[i], point_a,
                                   point_b, depth + 1)

            # Sample further if one of the end points in None (i.e. a complex
            # value) or the three points are not almost collinear.
            elif (p[0] is None or p[1] is None
                    or q[1] is None or q[0] is None
                    or not flat(p, new_point, q)):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)

        f_start_x = _adaptive_eval(f_x, self.start)
        f_start_y = _adaptive_eval(f_y, self.start)
        start = [f_start_x, f_start_y]
        f_end_x = _adaptive_eval(f_x, self.end)
        f_end_y = _adaptive_eval(f_y, self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(self.start)
        sample(self.start, self.end, start, end, 0)

        return x_coords, y_coords, param


### 3D lines
class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""

    is_2Dline = False
    is_3Dline = True
    _dim = 3

    def __init__(self):
        super().__init__()


class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""

    is_2Dline = False
    is_3Dline = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._cast = float
        self.adaptive = False
        self.use_cm = kwargs.get("use_cm", True)
        self._set_parametric_line_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def __str__(self):
        return self._str_helper(
            "3D parametric cartesian line: (%s, %s, %s) for %s over %s" % (
            str(self.expr_x),
            str(self.expr_y),
            str(self.expr_z),
            str(self.var),
            str((self.start, self.end))
        ))

    def get_data(self):
        # TODO: remove this
        np = import_module("numpy")
        x, y, z, p = super().get_data()
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        return x, y, z, p


### Surfaces
class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""

    is_3Dsurface = True

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.use_cm = kwargs.get("use_cm", False)
        # NOTE: why should SurfaceOver2DRangeSeries support is polar?
        # After all, the same result can be achieve with
        # ParametricSurfaceSeries. For example:
        # sin(r) for (r, 0, 2 * pi) and (theta, 0, pi/2) can be parameterized
        # as (r * cos(theta), r * sin(theta), sin(t)) for (r, 0, 2 * pi) and
        # (theta, 0, pi/2).
        # Because it is faster to evaluate (important for interactive plots).
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        self.surface_color = kwargs.get("surface_color", None)
        self.color_func = kwargs.get("color_func", lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        exprs = self.expr
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        # if the expressions is a lambda function and no label
        # has been provided, then its better to do the following to avoid
        # suprises on the backend
        is_lambda = (callable(exprs) if not hasattr(exprs, "__iter__")
            else any(callable(e) for e in exprs))
        if is_lambda and (self._label == str(exprs)):
                self._label = ""
                self._latex_label = ""

    def get_color_array(self):
        np = import_module('numpy')
        c = self.surface_color
        if isinstance(c, Callable):
            f = np.vectorize(c)
            nargs = arity(c)
            if self.is_parametric:
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            variables = list(map(centers_of_faces, self.get_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        else:
            if isinstance(self, SurfaceOver2DRangeSeries):
                return c*np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
            else:
                return c*np.ones(min(self.nb_of_points_u, self.nb_of_points_v))


class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._set_surface_label(label)
        self._post_init()
        # TODO: remove this
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    @property
    def var_x(self):
        return self.ranges[0][0]

    @property
    def var_y(self):
        return self.ranges[1][0]

    @property
    def start_x(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_x(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_y(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_y(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_x(self):
        return self.n[0]

    @nb_of_points_x.setter
    def nb_of_points_x(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_y(self):
        return self.n[1]

    @nb_of_points_y.setter
    def nb_of_points_y(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        series_type = "cartesian surface" if self.is_3Dsurface else "contour"
        return self._str_helper(
            series_type + ": %s for" " %s over %s and %s over %s" % (
            str(self.expr),
            str(self.var_x), str((self.start_x, self.end_x)),
            str(self.var_y), str((self.start_y, self.end_y)),
        ))

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        mesh_x : np.ndarray
            Discretized x-domain.
        mesh_y : np.ndarray
            Discretized y-domain.
        mesh_z : np.ndarray
            Results of the evaluation.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        x, y, z = results
        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)

        # TODO: remove this
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(x, y, z)


class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric SymPy
    expressions and a range."""

    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z,
        var_start_end_u, var_start_end_v, label="", **kwargs):
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.color_func = kwargs.get("color_func", lambda x, y, z, u, v: z)
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_u(self):
        return self.ranges[0][0]

    @property
    def var_v(self):
        return self.ranges[1][0]

    @property
    def start_u(self):
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_u(self):
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_v(self):
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_v(self):
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_u(self):
        return self.n[0]

    @nb_of_points_u.setter
    def nb_of_points_u(self, v):
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_v(self):
        return self.n[1]

    @nb_of_points_v.setter
    def nb_of_points_v(self, v):
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        return self._str_helper(
            "parametric cartesian surface: (%s, %s, %s) for"
            " %s over %s and %s over %s" % (
            str(self.expr_x), str(self.expr_y), str(self.expr_z),
            str(self.var_u), str((self.start_u, self.end_u)),
            str(self.var_v), str((self.start_v, self.end_v)),
        ))

    def get_parameter_meshes(self):
        return self.get_data()[3:]

    def get_meshes(self):
        """Return the x,y,z coordinates for plotting the surface.
        This function is available for back-compatibility purposes. Consider
        using ``get_data()`` instead.
        """
        return self.get_data()[:3]

    def get_data(self):
        """Return arrays of coordinates for plotting.

        Returns
        =======
        x : np.ndarray [n2 x n1]
            x-coordinates.
        y : np.ndarray [n2 x n1]
            y-coordinates.
        z : np.ndarray [n2 x n1]
            z-coordinates.
        mesh_u : np.ndarray [n2 x n1]
            Discretized u range.
        mesh_v : np.ndarray [n2 x n1]
            Discretized v range.
        """
        np = import_module('numpy')

        results = self._evaluate()
        # mask out complex values
        for i, r in enumerate(results):
            _re, _im = np.real(r), np.imag(r)
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re

        # TODO: remove this
        x, y, z = results[2:]
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))

        return self._apply_transform(*results[2:], *results[:2])


### Contours
class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""

    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get("is_filled", kwargs.get("fill", True))
        self.show_clabels = kwargs.get("clabels", True)

        # NOTE: contour plots are used by plot_contour, plot_vector and
        # plot_complex_vector. By implementing contour_kw we are able to
        # quickly target the contour plot.
        self.rendering_kw = kwargs.get("contour_kw",
            kwargs.get("rendering_kw", {}))


class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax[0]
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readibility. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    """
    is_generic = True

    def __init__(self, tp, *args, **kwargs):
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        return self.args


class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""

    is_implicit = True
    use_cm = False
    _N = 100

    def __init__(self, expr, var_start_end_x, var_start_end_y, label="", **kwargs):
        super().__init__(**kwargs)
        self.adaptive = kwargs.get("adaptive", False)
        self.expr = expr
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.ranges = [var_start_end_x, var_start_end_y]
        self.var_x, self.start_x, self.end_x = self.ranges[0]
        self.var_y, self.start_y, self.end_y = self.ranges[1]
        self._color = kwargs.get("color", kwargs.get("line_color", None))

        if self.is_interactive and self.adaptive:
            raise NotImplementedError("Interactive plot with `adaptive=True` "
                "is not supported.")

        # Check whether the depth is greater than 4 or less than 0.
        depth = kwargs.get("depth", 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
        self._post_init()

    @property
    def expr(self):
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    @expr.setter
    def expr(self, expr):
        self._block_lambda_functions(expr)
        # these are needed for adaptive evaluation
        expr, has_equality = self._has_equality(sympify(expr))
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self._label = str(expr)
        self._latex_label = latex(expr)

        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            self.adaptive = True
            msg = "contains Boolean functions. "
            if isinstance(expr, Ne):
                msg = "is an unequality. "
            warnings.warn(
                "The provided expression " + msg
                + "In order to plot the expression, the algorithm "
                + "automatically switched to an adaptive sampling."
            )

        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            # these are needed for uniform meshing evaluation
            expr, is_equality = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

    @property
    def line_color(self):
        return self._color

    @line_color.setter
    def line_color(self, v):
        self._color = v

    color = line_color

    def _has_equality(self, expr):
        # Represents whether the expression contains an Equality, GreaterThan
        # or LessThan
        has_equality = False

        def arg_expand(bool_expr):
            """Recursively expands the arguments of an Boolean Function"""
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)

        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            # Check whether there is an equality in the expression provided.
            if any(isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True

        return expr, has_equality

    def __str__(self):
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t

        return self._str_helper(
            "Implicit expression: %s for %s over %s and %s over %s") % (
            str(self._adaptive_expr),
            str(self.var_x),
            str((f(self.start_x), f(self.end_x))),
            str(self.var_y),
            str((f(self.start_y), f(self.end_y))),
        )

    def get_data(self):
        """Returns numerical data.

        Returns
        =======

        If the series is evaluated with the `adaptive=True` it returns:

        interval_list : list
            List of bounding rectangular intervals to be postprocessed and
            eventually used with Matplotlib's ``fill`` command.
        dummy : str
            A string containing ``"fill"``.

        Otherwise, it returns 2D numpy arrays to be used with Matplotlib's
        ``contour`` or ``contourf`` commands:

        x_array : np.ndarray
        y_array : np.ndarray
        z_array : np.ndarray
        plot_type : str
            A string specifying which plot command to use, ``"contour"``
            or ``"contourf"``.
        """
        if self.adaptive:
            data = self._adaptive_eval()
            if data is not None:
                return data

        return self._get_meshes_grid()

    def _adaptive_eval(self):
        """
        References
        ==========

        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
        Mathematical Formulae with Two Free Variables.

        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
        Arithmetic. Master's thesis. University of Toronto, 1996
        """
        import sympy.plotting.intervalmath.lib_interval as li

        user_functions = {}
        printer = IntervalMathPrinter({
            'fully_qualified_modules': False, 'inline': True,
            'allow_unknown_functions': True,
            'user_functions': user_functions})

        keys = [t for t in dir(li) if ("__" not in t) and (t not in ["import_module", "interval"])]
        vals = [getattr(li, k) for k in keys]
        d = dict(zip(keys, vals))
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)
        data = None

        try:
            data = self._get_raster_interval(func)
        except NameError as err:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression, as some functions are not yet implemented"
                " in the interval math module:\n\n"
                "NameError: %s\n\n" % err +
                "Proceeding with uniform meshing."
                )
            self.adaptive = False
        except TypeError:
            warnings.warn(
                "Adaptive meshing could not be applied to the"
                " expression. Using uniform meshing.")
            self.adaptive = False

        return data

    def _get_raster_interval(self, func):
        """Uses interval math to adaptively mesh and obtain the plot"""
        np = import_module('numpy')

        k = self.depth
        interval_list = []
        sx, sy = [float(t) for t in [self.start_x, self.start_y]]
        ex, ey = [float(t) for t in [self.end_x, self.end_y]]
        # Create initial 32 divisions
        xsample = np.linspace(sx, ex, 33)
        ysample = np.linspace(sy, ey, 33)

        # Add a small jitter so that there are no false positives for equality.
        # Ex: y==x becomes True for x interval(1, 2) and y interval(1, 2)
        # which will draw a rectangle.
        jitterx = (
            (np.random.rand(len(xsample)) * 2 - 1)
            * (ex - sx)
            / 2 ** 20
        )
        jittery = (
            (np.random.rand(len(ysample)) * 2 - 1)
            * (ey - sy)
            / 2 ** 20
        )
        xsample += jitterx
        ysample += jittery

        xinter = [interval(x1, x2) for x1, x2 in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for y1, y2 in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        # recursive call refinepixels which subdivides the intervals which are
        # neither True nor False according to the expression.
        def refine_pixels(interval_list):
            """Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:

                # Convert the array indices to x and y values
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                # The expression is valid in the interval. Change the contour
                # array values to 1.
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    # Subdivide
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return temp_interval_list, plot_list

        while k >= 0 and len(interval_list):
            interval_list, plot_list_temp = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        # Check whether the expression represents an equality
        # If it represents an equality, then none of the intervals
        # would have satisfied the expression due to floating point
        # differences. Add all the undecided values to the plot.
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return plot_list, "fill"

    def _get_meshes_grid(self):
        """Generates the mesh for generating a contour.

        In the case of equality, ``contour`` function of matplotlib can
        be used. In other cases, matplotlib's ``contourf`` is used.
        """
        np = import_module('numpy')

        xarray, yarray, z_grid = self._evaluate()
        _re, _im = np.real(z_grid), np.imag(z_grid)
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        if self._is_equality:
            return xarray, yarray, _re, 'contour'
        return xarray, yarray, _re, 'contourf'

    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        """If the expression is a Relational, rewrite it as a single
        expression.

        Returns
        =======

        expr : Expr
            The rewritten expression

        equality : Boolean
            Wheter the original expression was an Equality or not.
        """
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        elif isinstance(expr, Relational):
            expr = expr.gts - expr.lts
        elif not adaptive:
            raise NotImplementedError(
                "The expression is not supported for "
                "plotting in uniform meshed plot."
            )
        return expr, equality

    def get_label(self, use_latex=False, wrapper="$%s$"):
        """Return the label to be used to display the expression.

        Parameters
        ==========
        use_latex : bool
            If False, the string representation of the expression is returned.
            If True, the latex representation is returned.
        wrapper : str
            The backend might need the latex representation to be wrapped by
            some characters. Default to ``"$%s$"``.

        Returns
        =======
        label : str
        """
        if use_latex is False:
            return self._label
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label


##############################################################################
# Finding the centers of line segments or mesh faces
##############################################################################

def centers_of_segments(array):
    np = import_module('numpy')
    return np.mean(np.vstack((array[:-1], array[1:])), 0)


def centers_of_faces(array):
    np = import_module('numpy')
    return np.mean(np.dstack((array[:-1, :-1],
                             array[1:, :-1],
                             array[:-1, 1:],
                             array[:-1, :-1],
                             )), 2)


def flat(x, y, z, eps=1e-3):
    """Checks whether three points are almost collinear"""
    np = import_module('numpy')
    # Workaround plotting piecewise (#8577)
    vector_a = (x - y).astype(float)
    vector_b = (z - y).astype(float)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps


def _set_discretization_points(kwargs, pt):
    """Allow the use of the keyword arguments ``n, n1, n2`` to
    specify the number of discretization points in one and two
    directions, while keeping back-compatibility with older keyword arguments
    like, ``nb_of_points, nb_of_points_*, points``.

    Parameters
    ==========

    kwargs : dict
        Dictionary of keyword arguments passed into a plotting function.
    pt : type
        The type of the series, which indicates the kind of plot we are
        trying to create.
    """
    replace_old_keywords = {
        "nb_of_points": "n",
        "nb_of_points_x": "n1",
        "nb_of_points_y": "n2",
        "nb_of_points_u": "n1",
        "nb_of_points_v": "n2",
        "points": "n"
    }
    for k, v in replace_old_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)

    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries,
        Parametric3DLineSeries]:
        if "n" in kwargs.keys():
            kwargs["n1"] = kwargs["n"]
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 0):
                kwargs["n1"] = kwargs["n"][0]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries,
        ParametricSurfaceSeries, ImplicitSeries]:
        if "n" in kwargs.keys():
            if hasattr(kwargs["n"], "__iter__") and (len(kwargs["n"]) > 1):
                kwargs["n1"] = kwargs["n"][0]
                kwargs["n2"] = kwargs["n"][1]
            else:
                kwargs["n1"] = kwargs["n2"] = kwargs["n"]
    return kwargs
