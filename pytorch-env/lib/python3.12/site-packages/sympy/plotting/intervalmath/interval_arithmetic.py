"""
Interval Arithmetic for plotting.
This module does not implement interval arithmetic accurately and
hence cannot be used for purposes other than plotting. If you want
to use interval arithmetic, use mpmath's interval arithmetic.

The module implements interval arithmetic using numpy and
python floating points. The rounding up and down is not handled
and hence this is not an accurate implementation of interval
arithmetic.

The module uses numpy for speed which cannot be achieved with mpmath.
"""

# Q: Why use numpy? Why not simply use mpmath's interval arithmetic?
# A: mpmath's interval arithmetic simulates a floating point unit
# and hence is slow, while numpy evaluations are orders of magnitude
# faster.

# Q: Why create a separate class for intervals? Why not use SymPy's
# Interval Sets?
# A: The functionalities that will be required for plotting is quite
# different from what Interval Sets implement.

# Q: Why is rounding up and down according to IEEE754 not handled?
# A: It is not possible to do it in both numpy and python. An external
# library has to used, which defeats the whole purpose i.e., speed. Also
# rounding is handled for very few functions in those libraries.

# Q Will my plots be affected?
# A It will not affect most of the plots. The interval arithmetic
# module based suffers the same problems as that of floating point
# arithmetic.

from sympy.core.numbers import int_valued
from sympy.core.logic import fuzzy_and
from sympy.simplify.simplify import nsimplify

from .interval_membership import intervalMembership


class interval:
    """ Represents an interval containing floating points as start and
    end of the interval
    The is_valid variable tracks whether the interval obtained as the
    result of the function is in the domain and is continuous.
    - True: Represents the interval result of a function is continuous and
            in the domain of the function.
    - False: The interval argument of the function was not in the domain of
             the function, hence the is_valid of the result interval is False
    - None: The function was not continuous over the interval or
            the function's argument interval is partly in the domain of the
            function

    A comparison between an interval and a real number, or a
    comparison between two intervals may return ``intervalMembership``
    of two 3-valued logic values.
    """

    def __init__(self, *args, is_valid=True, **kwargs):
        self.is_valid = is_valid
        if len(args) == 1:
            if isinstance(args[0], interval):
                self.start, self.end = args[0].start, args[0].end
            else:
                self.start = float(args[0])
                self.end = float(args[0])
        elif len(args) == 2:
            if args[0] < args[1]:
                self.start = float(args[0])
                self.end = float(args[1])
            else:
                self.start = float(args[1])
                self.end = float(args[0])

        else:
            raise ValueError("interval takes a maximum of two float values "
                            "as arguments")

    @property
    def mid(self):
        return (self.start + self.end) / 2.0

    @property
    def width(self):
        return self.end - self.start

    def __repr__(self):
        return "interval(%f, %f)" % (self.start, self.end)

    def __str__(self):
        return "[%f, %f]" % (self.start, self.end)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            if self.end < other:
                return intervalMembership(True, self.is_valid)
            elif self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        elif isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end < other. start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            if self.start > other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__lt__(self)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(True, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(False, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(True, valid)
            elif self.__lt__(other)[0] is not None:
                return intervalMembership(False, valid)
            else:
                return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(False, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(True, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(False, valid)
            if not self.__lt__(other)[0] is None:
                return intervalMembership(True, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            if self.end <= other:
                return intervalMembership(True, self.is_valid)
            if self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end <= other.start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            if self.start >= other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__le__(self)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.is_valid:
                return interval(self.start + other, self.end + other)
            else:
                start = self.start + other
                end = self.end + other
                return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start + other.start
            end = self.end + other.end
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            start = self.start - other
            end = self.end - other
            return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start - other.end
            end = self.end - other.start
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            start = other - self.end
            end = other - self.start
            return interval(start, end, is_valid=self.is_valid)
        elif isinstance(other, interval):
            return other.__sub__(self)
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_valid:
            return interval(-self.end, -self.start)
        else:
            return interval(-self.end, -self.start, is_valid=self.is_valid)

    def __mul__(self, other):
        if isinstance(other, interval):
            if self.is_valid is False or other.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif self.is_valid is None or other.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
                inters = []
                inters.append(self.start * other.start)
                inters.append(self.end * other.start)
                inters.append(self.start * other.end)
                inters.append(self.end * other.end)
                start = min(inters)
                end = max(inters)
                return interval(start, end)
        elif isinstance(other, (int, float)):
            return interval(self.start*other, self.end*other, is_valid=self.is_valid)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __contains__(self, other):
        if isinstance(other, (int, float)):
            return self.start <= other and self.end >= other
        else:
            return self.start <= other.start and other.end <= self.end

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = interval(other)
            return other.__truediv__(self)
        elif isinstance(other, interval):
            return other.__truediv__(self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        # Both None and False are handled
        if not self.is_valid:
            # Don't divide as the value is not valid
            return interval(-float('inf'), float('inf'), is_valid=self.is_valid)
        if isinstance(other, (int, float)):
            if other == 0:
                # Divide by zero encountered. valid nowhere
                return interval(-float('inf'), float('inf'), is_valid=False)
            else:
                return interval(self.start / other, self.end / other)

        elif isinstance(other, interval):
            if other.is_valid is False or self.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif other.is_valid is None or self.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
               # denominator contains both signs, i.e. being divided by zero
               # return the whole real line with is_valid = None
                if 0 in other:
                    return interval(-float('inf'), float('inf'), is_valid=None)

                # denominator negative
                this = self
                if other.end < 0:
                    this = -this
                    other = -other

                # denominator positive
                inters = []
                inters.append(this.start / other.start)
                inters.append(this.end / other.start)
                inters.append(this.start / other.end)
                inters.append(this.end / other.end)
                start = max(inters)
                end = min(inters)
                return interval(start, end)
        else:
            return NotImplemented

    def __pow__(self, other):
        # Implements only power to an integer.
        from .lib_interval import exp, log
        if not self.is_valid:
            return self
        if isinstance(other, interval):
            return exp(other * log(self))
        elif isinstance(other, (float, int)):
            if other < 0:
                return 1 / self.__pow__(abs(other))
            else:
                if int_valued(other):
                    return _pow_int(self, other)
                else:
                    return _pow_float(self, other)
        else:
            return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            if not self.is_valid:
                #Don't do anything
                return self
            elif other < 0:
                if self.width > 0:
                    return interval(-float('inf'), float('inf'), is_valid=False)
                else:
                    power_rational = nsimplify(self.start)
                    num, denom = power_rational.as_numer_denom()
                    if denom % 2 == 0:
                        return interval(-float('inf'), float('inf'),
                                        is_valid=False)
                    else:
                        start = -abs(other)**self.start
                        end = start
                        return interval(start, end)
            else:
                return interval(other**self.start, other**self.end)
        elif isinstance(other, interval):
            return other.__pow__(self)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.is_valid, self.start, self.end))


def _pow_float(inter, power):
    """Evaluates an interval raised to a floating point."""
    power_rational = nsimplify(power)
    num, denom = power_rational.as_numer_denom()
    if num % 2 == 0:
        start = abs(inter.start)**power
        end = abs(inter.end)**power
        if start < 0:
            ret = interval(0, max(start, end))
        else:
            ret = interval(start, end)
        return ret
    elif denom % 2 == 0:
        if inter.end < 0:
            return interval(-float('inf'), float('inf'), is_valid=False)
        elif inter.start < 0:
            return interval(0, inter.end**power, is_valid=None)
        else:
            return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0:
            start = -abs(inter.start)**power
        else:
            start = inter.start**power

        if inter.end < 0:
            end = -abs(inter.end)**power
        else:
            end = inter.end**power

        return interval(start, end, is_valid=inter.is_valid)


def _pow_int(inter, power):
    """Evaluates an interval raised to an integer power"""
    power = int(power)
    if power & 1:
        return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0 and inter.end > 0:
            start = 0
            end = max(inter.start**power, inter.end**power)
            return interval(start, end)
        else:
            return interval(inter.start**power, inter.end**power)
