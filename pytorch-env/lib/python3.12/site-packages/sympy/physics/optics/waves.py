"""
This module has all the classes and functions related to waves in optics.

**Contains**

* TWave
"""

__all__ = ['TWave']

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.numbers import (Number, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.physics.units import speed_of_light, meter, second


c = speed_of_light.convert_to(meter/second)


class TWave(Expr):

    r"""
    This is a simple transverse sine wave travelling in a one-dimensional space.
    Basic properties are required at the time of creation of the object,
    but they can be changed later with respective methods provided.

    Explanation
    ===========

    It is represented as :math:`A \times cos(k*x - \omega \times t + \phi )`,
    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency,
    :math:`k` is the wavenumber (spatial frequency), :math:`x` is a spatial variable
    to represent the position on the dimension on which the wave propagates,
    and :math:`\phi` is the phase angle of the wave.


    Arguments
    =========

    amplitude : Sympifyable
        Amplitude of the wave.
    frequency : Sympifyable
        Frequency of the wave.
    phase : Sympifyable
        Phase angle of the wave.
    time_period : Sympifyable
        Time period of the wave.
    n : Sympifyable
        Refractive index of the medium.

    Raises
    =======

    ValueError : When neither frequency nor time period is provided
        or they are not consistent.
    TypeError : When anything other than TWave objects is added.


    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.optics import TWave
    >>> A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    >>> w1 = TWave(A1, f, phi1)
    >>> w2 = TWave(A2, f, phi2)
    >>> w3 = w1 + w2  # Superposition of two waves
    >>> w3
    TWave(sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2), f,
        atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)), 1/f, n)
    >>> w3.amplitude
    sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    >>> w3.phase
    atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    >>> w3.speed
    299792458*meter/(second*n)
    >>> w3.angular_velocity
    2*pi*f

    """

    def __new__(
            cls,
            amplitude,
            frequency=None,
            phase=S.Zero,
            time_period=None,
            n=Symbol('n')):
        if time_period is not None:
            time_period = _sympify(time_period)
            _frequency = S.One/time_period
        if frequency is not None:
            frequency = _sympify(frequency)
            _time_period = S.One/frequency
            if time_period is not None:
                if frequency != S.One/time_period:
                    raise ValueError("frequency and time_period should be consistent.")
        if frequency is None and time_period is None:
            raise ValueError("Either frequency or time period is needed.")
        if frequency is None:
            frequency = _frequency
        if time_period is None:
            time_period = _time_period

        amplitude = _sympify(amplitude)
        phase = _sympify(phase)
        n = sympify(n)
        obj = Basic.__new__(cls, amplitude, frequency, phase, time_period, n)
        return obj

    @property
    def amplitude(self):
        """
        Returns the amplitude of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.amplitude
        A
        """
        return self.args[0]

    @property
    def frequency(self):
        """
        Returns the frequency of the wave,
        in cycles per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.frequency
        f
        """
        return self.args[1]

    @property
    def phase(self):
        """
        Returns the phase angle of the wave,
        in radians.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.phase
        phi
        """
        return self.args[2]

    @property
    def time_period(self):
        """
        Returns the temporal period of the wave,
        in seconds per cycle.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.time_period
        1/f
        """
        return self.args[3]

    @property
    def n(self):
        """
        Returns the refractive index of the medium
        """
        return self.args[4]

    @property
    def wavelength(self):
        """
        Returns the wavelength (spatial period) of the wave,
        in meters per cycle.
        It depends on the medium of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavelength
        299792458*meter/(second*f*n)
        """
        return c/(self.frequency*self.n)


    @property
    def speed(self):
        """
        Returns the propagation speed of the wave,
        in meters per second.
        It is dependent on the propagation medium.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.speed
        299792458*meter/(second*n)
        """
        return self.wavelength*self.frequency

    @property
    def angular_velocity(self):
        """
        Returns the angular velocity of the wave,
        in radians per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.angular_velocity
        2*pi*f
        """
        return 2*pi*self.frequency

    @property
    def wavenumber(self):
        """
        Returns the wavenumber of the wave,
        in radians per meter.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavenumber
        pi*second*f*n/(149896229*meter)
        """
        return 2*pi/self.wavelength

    def __str__(self):
        """String representation of a TWave."""
        from sympy.printing import sstr
        return type(self).__name__ + sstr(self.args)

    __repr__ = __str__

    def __add__(self, other):
        """
        Addition of two waves will result in their superposition.
        The type of interference will depend on their phase angles.
        """
        if isinstance(other, TWave):
            if self.frequency == other.frequency and self.wavelength == other.wavelength:
                return TWave(sqrt(self.amplitude**2 + other.amplitude**2 + 2 *
                                  self.amplitude*other.amplitude*cos(
                                      self.phase - other.phase)),
                             self.frequency,
                             atan2(self.amplitude*sin(self.phase)
                             + other.amplitude*sin(other.phase),
                             self.amplitude*cos(self.phase)
                             + other.amplitude*cos(other.phase))
                             )
            else:
                raise NotImplementedError("Interference of waves with different frequencies"
                    " has not been implemented.")
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be added.")

    def __mul__(self, other):
        """
        Multiplying a wave by a scalar rescales the amplitude of the wave.
        """
        other = sympify(other)
        if isinstance(other, Number):
            return TWave(self.amplitude*other, *self.args[1:])
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be multiplied.")

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __neg__(self):
        return self.__mul__(-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def _eval_rewrite_as_sin(self, *args, **kwargs):
        return self.amplitude*sin(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase + pi/2, evaluate=False)

    def _eval_rewrite_as_cos(self, *args, **kwargs):
        return self.amplitude*cos(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase)

    def _eval_rewrite_as_pde(self, *args, **kwargs):
        mu, epsilon, x, t = symbols('mu, epsilon, x, t')
        E = Function('E')
        return Derivative(E(x, t), x, 2) + mu*epsilon*Derivative(E(x, t), t, 2)

    def _eval_rewrite_as_exp(self, *args, **kwargs):
        return self.amplitude*exp(I*(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase))
