from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence


class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """

    ## Class-level attributes
    ## used to register and lookup
    ## plot modes. See PlotModeBase
    ## for descriptions and usage.

    i_vars, d_vars = '', ''
    intervals = []
    aliases = []
    is_default = False

    ## Draw is the only method here which
    ## is meant to be overridden in child
    ## classes, and PlotModeBase provides
    ## a base implementation.
    def draw(self):
        raise NotImplementedError()

    ## Everything else in this file has to
    ## do with registration and retrieval
    ## of plot modes. This is where I've
    ## hidden much of the ugliness of automatic
    ## plot mode divination...

    ## Plot mode registry data structures
    _mode_alias_list = []
    _mode_map = {
        1: {1: {}, 2: {}},
        2: {1: {}, 2: {}},
        3: {1: {}, 2: {}},
    }  # [d][i][alias_str]: class
    _mode_default_map = {
        1: {},
        2: {},
        3: {},
    }  # [d][i]: class
    _i_var_max, _d_var_max = 2, 3

    def __new__(cls, *args, **kwargs):
        """
        This is the function which interprets
        arguments given to Plot.__init__ and
        Plot.__setattr__. Returns an initialized
        instance of the appropriate child class.
        """

        newargs, newkwargs = PlotMode._extract_options(args, kwargs)
        mode_arg = newkwargs.get('mode', '')

        # Interpret the arguments
        d_vars, intervals = PlotMode._interpret_args(newargs)
        i_vars = PlotMode._find_i_vars(d_vars, intervals)
        i, d = max([len(i_vars), len(intervals)]), len(d_vars)

        # Find the appropriate mode
        subcls = PlotMode._get_mode(mode_arg, i, d)

        # Create the object
        o = object.__new__(subcls)

        # Do some setup for the mode instance
        o.d_vars = d_vars
        o._fill_i_vars(i_vars)
        o._fill_intervals(intervals)
        o.options = newkwargs

        return o

    @staticmethod
    def _get_mode(mode_arg, i_var_count, d_var_count):
        """
        Tries to return an appropriate mode class.
        Intended to be called only by __new__.

        mode_arg
            Can be a string or a class. If it is a
            PlotMode subclass, it is simply returned.
            If it is a string, it can an alias for
            a mode or an empty string. In the latter
            case, we try to find a default mode for
            the i_var_count and d_var_count.

        i_var_count
            The number of independent variables
            needed to evaluate the d_vars.

        d_var_count
            The number of dependent variables;
            usually the number of functions to
            be evaluated in plotting.

        For example, a Cartesian function y = f(x) has
        one i_var (x) and one d_var (y). A parametric
        form x,y,z = f(u,v), f(u,v), f(u,v) has two
        two i_vars (u,v) and three d_vars (x,y,z).
        """
        # if the mode_arg is simply a PlotMode class,
        # check that the mode supports the numbers
        # of independent and dependent vars, then
        # return it
        try:
            m = None
            if issubclass(mode_arg, PlotMode):
                m = mode_arg
        except TypeError:
            pass
        if m:
            if not m._was_initialized:
                raise ValueError(("To use unregistered plot mode %s "
                                  "you must first call %s._init_mode().")
                                 % (m.__name__, m.__name__))
            if d_var_count != m.d_var_count:
                raise ValueError(("%s can only plot functions "
                                  "with %i dependent variables.")
                                 % (m.__name__,
                                     m.d_var_count))
            if i_var_count > m.i_var_count:
                raise ValueError(("%s cannot plot functions "
                                  "with more than %i independent "
                                  "variables.")
                                 % (m.__name__,
                                     m.i_var_count))
            return m
        # If it is a string, there are two possibilities.
        if isinstance(mode_arg, str):
            i, d = i_var_count, d_var_count
            if i > PlotMode._i_var_max:
                raise ValueError(var_count_error(True, True))
            if d > PlotMode._d_var_max:
                raise ValueError(var_count_error(False, True))
            # If the string is '', try to find a suitable
            # default mode
            if not mode_arg:
                return PlotMode._get_default_mode(i, d)
            # Otherwise, interpret the string as a mode
            # alias (e.g. 'cartesian', 'parametric', etc)
            else:
                return PlotMode._get_aliased_mode(mode_arg, i, d)
        else:
            raise ValueError("PlotMode argument must be "
                             "a class or a string")

    @staticmethod
    def _get_default_mode(i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        try:
            return PlotMode._mode_default_map[d][i]
        except KeyError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count until we
            # reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_default_mode(i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a default mode "
                                  "for %i independent and %i "
                                  "dependent variables.") % (i_vars, d))

    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        if alias not in PlotMode._mode_alias_list:
            raise ValueError(("Couldn't find a mode called"
                              " %s. Known modes: %s.")
                             % (alias, ", ".join(PlotMode._mode_alias_list)))
        try:
            return PlotMode._mode_map[d][i][alias]
        except TypeError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count and alias
            # until we reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a %s mode "
                                  "for %i independent and %i "
                                  "dependent variables.")
                                 % (alias, i_vars, d))

    @classmethod
    def _register(cls):
        """
        Called once for each user-usable plot mode.
        For Cartesian2D, it is invoked after the
        class definition: Cartesian2D._register()
        """
        name = cls.__name__
        cls._init_mode()

        try:
            i, d = cls.i_var_count, cls.d_var_count
            # Add the mode to _mode_map under all
            # given aliases
            for a in cls.aliases:
                if a not in PlotMode._mode_alias_list:
                    # Also track valid aliases, so
                    # we can quickly know when given
                    # an invalid one in _get_mode.
                    PlotMode._mode_alias_list.append(a)
                PlotMode._mode_map[d][i][a] = cls
            if cls.is_default:
                # If this mode was marked as the
                # default for this d,i combination,
                # also set that.
                PlotMode._mode_default_map[d][i] = cls

        except Exception as e:
            raise RuntimeError(("Failed to register "
                              "plot mode %s. Reason: %s")
                               % (name, (str(e))))

    @classmethod
    def _init_mode(cls):
        """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """
        def symbols_list(symbol_str):
            return [Symbol(s) for s in symbol_str]

        # Convert the vars strs into
        # lists of symbols.
        cls.i_vars = symbols_list(cls.i_vars)
        cls.d_vars = symbols_list(cls.d_vars)

        # Var count is used often, calculate
        # it once here
        cls.i_var_count = len(cls.i_vars)
        cls.d_var_count = len(cls.d_vars)

        if cls.i_var_count > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, False))
        if cls.d_var_count > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, False))

        # Try to use first alias as primary_alias
        if len(cls.aliases) > 0:
            cls.primary_alias = cls.aliases[0]
        else:
            cls.primary_alias = cls.__name__

        di = cls.intervals
        if len(di) != cls.i_var_count:
            raise ValueError("Plot mode must provide a "
                             "default interval for each i_var.")
        for i in range(cls.i_var_count):
            # default intervals must be given [min,max,steps]
            # (no var, but they must be in the same order as i_vars)
            if len(di[i]) != 3:
                raise ValueError("length should be equal to 3")

            # Initialize an incomplete interval,
            # to later be filled with a var when
            # the mode is instantiated.
            di[i] = PlotInterval(None, *di[i])

        # To prevent people from using modes
        # without these required fields set up.
        cls._was_initialized = True

    _was_initialized = False

    ## Initializer Helper Methods

    @staticmethod
    def _find_i_vars(functions, intervals):
        i_vars = []

        # First, collect i_vars in the
        # order they are given in any
        # intervals.
        for i in intervals:
            if i.v is None:
                continue
            elif i.v in i_vars:
                raise ValueError(("Multiple intervals given "
                                  "for %s.") % (str(i.v)))
            i_vars.append(i.v)

        # Then, find any remaining
        # i_vars in given functions
        # (aka d_vars)
        for f in functions:
            for a in f.free_symbols:
                if a not in i_vars:
                    i_vars.append(a)

        return i_vars

    def _fill_i_vars(self, i_vars):
        # copy default i_vars
        self.i_vars = [Symbol(str(i)) for i in self.i_vars]
        # replace with given i_vars
        for i in range(len(i_vars)):
            self.i_vars[i] = i_vars[i]

    def _fill_intervals(self, intervals):
        # copy default intervals
        self.intervals = [PlotInterval(i) for i in self.intervals]
        # track i_vars used so far
        v_used = []
        # fill copy of default
        # intervals with given info
        for i in range(len(intervals)):
            self.intervals[i].fill_from(intervals[i])
            if self.intervals[i].v is not None:
                v_used.append(self.intervals[i].v)
        # Find any orphan intervals and
        # assign them i_vars
        for i in range(len(self.intervals)):
            if self.intervals[i].v is None:
                u = [v for v in self.i_vars if v not in v_used]
                if len(u) == 0:
                    raise ValueError("length should not be equal to 0")
                self.intervals[i].v = u[0]
                v_used.append(u[0])

    @staticmethod
    def _interpret_args(args):
        interval_wrong_order = "PlotInterval %s was given before any function(s)."
        interpret_error = "Could not interpret %s as a function or interval."

        functions, intervals = [], []
        if isinstance(args[0], GeometryEntity):
            for coords in list(args[0].arbitrary_point()):
                functions.append(coords)
            intervals.append(PlotInterval.try_parse(args[0].plot_interval()))
        else:
            for a in args:
                i = PlotInterval.try_parse(a)
                if i is not None:
                    if len(functions) == 0:
                        raise ValueError(interval_wrong_order % (str(i)))
                    else:
                        intervals.append(i)
                else:
                    if is_sequence(a, include=str):
                        raise ValueError(interpret_error % (str(a)))
                    try:
                        f = sympify(a)
                        functions.append(f)
                    except TypeError:
                        raise ValueError(interpret_error % str(a))

        return functions, intervals

    @staticmethod
    def _extract_options(args, kwargs):
        newkwargs, newargs = {}, []
        for a in args:
            if isinstance(a, str):
                newkwargs = dict(newkwargs, **parse_option_string(a))
            else:
                newargs.append(a)
        newkwargs = dict(newkwargs, **kwargs)
        return newargs, newkwargs


def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """
    if is_plotting:
        v = "Plotting"
    else:
        v = "Registering plot modes"
    if is_independent:
        n, s = PlotMode._i_var_max, "independent"
    else:
        n, s = PlotMode._d_var_max, "dependent"
    return ("%s with more than %i %s variables "
            "is not supported.") % (v, n, s)
