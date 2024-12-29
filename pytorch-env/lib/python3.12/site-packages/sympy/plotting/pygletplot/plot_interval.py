from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer


class PlotInterval:
    """
    """
    _v, _v_min, _v_max, _v_steps = None, None, None, None

    def require_all_args(f):
        def check(self, *args, **kwargs):
            for g in [self._v, self._v_min, self._v_max, self._v_steps]:
                if g is None:
                    raise ValueError("PlotInterval is incomplete.")
            return f(self, *args, **kwargs)
        return check

    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], PlotInterval):
                self.fill_from(args[0])
                return
            elif isinstance(args[0], str):
                try:
                    args = eval(args[0])
                except TypeError:
                    s_eval_error = "Could not interpret string %s."
                    raise ValueError(s_eval_error % (args[0]))
            elif isinstance(args[0], (tuple, list)):
                args = args[0]
            else:
                raise ValueError("Not an interval.")
        if not isinstance(args, (tuple, list)) or len(args) > 4:
            f_error = "PlotInterval must be a tuple or list of length 4 or less."
            raise ValueError(f_error)

        args = list(args)
        if len(args) > 0 and (args[0] is None or isinstance(args[0], Symbol)):
            self.v = args.pop(0)
        if len(args) in [2, 3]:
            self.v_min = args.pop(0)
            self.v_max = args.pop(0)
            if len(args) == 1:
                self.v_steps = args.pop(0)
        elif len(args) == 1:
            self.v_steps = args.pop(0)

    def get_v(self):
        return self._v

    def set_v(self, v):
        if v is None:
            self._v = None
            return
        if not isinstance(v, Symbol):
            raise ValueError("v must be a SymPy Symbol.")
        self._v = v

    def get_v_min(self):
        return self._v_min

    def set_v_min(self, v_min):
        if v_min is None:
            self._v_min = None
            return
        try:
            self._v_min = sympify(v_min)
            float(self._v_min.evalf())
        except TypeError:
            raise ValueError("v_min could not be interpreted as a number.")

    def get_v_max(self):
        return self._v_max

    def set_v_max(self, v_max):
        if v_max is None:
            self._v_max = None
            return
        try:
            self._v_max = sympify(v_max)
            float(self._v_max.evalf())
        except TypeError:
            raise ValueError("v_max could not be interpreted as a number.")

    def get_v_steps(self):
        return self._v_steps

    def set_v_steps(self, v_steps):
        if v_steps is None:
            self._v_steps = None
            return
        if isinstance(v_steps, int):
            v_steps = Integer(v_steps)
        elif not isinstance(v_steps, Integer):
            raise ValueError("v_steps must be an int or SymPy Integer.")
        if v_steps <= S.Zero:
            raise ValueError("v_steps must be positive.")
        self._v_steps = v_steps

    @require_all_args
    def get_v_len(self):
        return self.v_steps + 1

    v = property(get_v, set_v)
    v_min = property(get_v_min, set_v_min)
    v_max = property(get_v_max, set_v_max)
    v_steps = property(get_v_steps, set_v_steps)
    v_len = property(get_v_len)

    def fill_from(self, b):
        if b.v is not None:
            self.v = b.v
        if b.v_min is not None:
            self.v_min = b.v_min
        if b.v_max is not None:
            self.v_max = b.v_max
        if b.v_steps is not None:
            self.v_steps = b.v_steps

    @staticmethod
    def try_parse(*args):
        """
        Returns a PlotInterval if args can be interpreted
        as such, otherwise None.
        """
        if len(args) == 1 and isinstance(args[0], PlotInterval):
            return args[0]
        try:
            return PlotInterval(*args)
        except ValueError:
            return None

    def _str_base(self):
        return ",".join([str(self.v), str(self.v_min),
                         str(self.v_max), str(self.v_steps)])

    def __repr__(self):
        """
        A string representing the interval in class constructor form.
        """
        return "PlotInterval(%s)" % (self._str_base())

    def __str__(self):
        """
        A string representing the interval in list form.
        """
        return "[%s]" % (self._str_base())

    @require_all_args
    def assert_complete(self):
        pass

    @require_all_args
    def vrange(self):
        """
        Yields v_steps+1 SymPy numbers ranging from
        v_min to v_max.
        """
        d = (self.v_max - self.v_min) / self.v_steps
        for i in range(self.v_steps + 1):
            a = self.v_min + (d * Integer(i))
            yield a

    @require_all_args
    def vrange2(self):
        """
        Yields v_steps pairs of SymPy numbers ranging from
        (v_min, v_min + step) to (v_max - step, v_max).
        """
        d = (self.v_max - self.v_min) / self.v_steps
        a = self.v_min + (d * S.Zero)
        for i in range(self.v_steps):
            b = self.v_min + (d * Integer(i + 1))
            yield a, b
            a = b

    def frange(self):
        for i in self.vrange():
            yield float(i.evalf())
