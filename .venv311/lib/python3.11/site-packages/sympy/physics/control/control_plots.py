from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.plot import plot_parametric
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
from sympy.geometry.polygon import deg

__all__ = ['pole_zero_numerical_data', 'pole_zero_plot',
    'step_response_numerical_data', 'step_response_plot',
    'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot',
    'nyquist_plot_expr', 'nyquist_plot', 'nichols_plot_expr',
    'nichols_plot']


matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

if matplotlib:
    plt = matplotlib.pyplot


def _check_system(system):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError("Only SISO LTI systems are currently supported.")
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")
    if sys.has(exp):
        # Should test that exp is not part of a constant, in which case
        # no exception is required, compare exp(s) with s*exp(1)
        raise NotImplementedError("Time delay terms are not supported.")


def _poly_roots(poly):
    """Function to get the roots of a polynomial."""
    def _eval(l):
        return [float(i) if i.is_real else complex(i) for i in l]
    if poly.domain in (QQ, ZZ):
        return _eval(poly.all_roots())
    # XXX: Use all_roots() for irrational coefficients when possible
    # See https://github.com/sympy/sympy/issues/22943
    return _eval(poly.nroots())


def pole_zero_numerical_data(system):
    """
    Returns the numerical data of poles and zeros of the system.
    It is internally used by ``pole_zero_plot`` to get the data
    for plotting poles and zeros. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the pole-zero data is to be computed.

    Returns
    =======

    tuple : (zeros, poles)
        zeros = Zeros of the system as a list of Python float/complex.
        poles = Poles of the system as a list of Python float/complex.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import pole_zero_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> pole_zero_numerical_data(tf1)
    ([-1j, 1j], [-2.0, -1.0, (-0.5-0.8660254037844386j), (-0.5+0.8660254037844386j)])

    See Also
    ========

    pole_zero_plot

    """
    _check_system(system)
    system = system.doit()  # Get the equivalent TransferFunction object.

    num_poly = Poly(system.num, system.var)
    den_poly = Poly(system.den, system.var)

    return _poly_roots(num_poly), _poly_roots(den_poly)


def pole_zero_plot(system, pole_color='blue', pole_markersize=10,
    zero_color='orange', zero_markersize=7, grid=True, show_axes=True,
    show=True, **kwargs):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
    pole_color : str, tuple, optional
        The color of the pole points on the plot. Default color
        is blue. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot. Default color
        is orange. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import pole_zero_plot
        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> pole_zero_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    pole_zero_numerical_data

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot

    """
    zeros, poles = pole_zero_numerical_data(system)

    zero_real = [i.real for i in zeros]
    zero_imag = [i.imag for i in zeros]

    pole_real = [i.real for i in poles]
    pole_imag = [i.imag for i in poles]

    plt.plot(pole_real, pole_imag, 'x', mfc='none',
        markersize=pole_markersize, color=pole_color)
    plt.plot(zero_real, zero_imag, 'o', markersize=zero_markersize,
        color=zero_color)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title(f'Poles and Zeros of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def step_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the step response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the unit step response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the step response. NumPy array.
        y = Amplitude-axis values of the points in the step response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import step_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> step_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.025413462339411542, 0.0484508722725343, ... , 9.670250533855183, 9.844291913708725, 10.0],
    [0.0, 0.023844582399907256, 0.042894276802320226, ..., 6.828770759094287e-12, 6.456457160755703e-12])

    See Also
    ========

    step_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()/(system.var)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def step_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import step_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> step_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    x, y = step_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Unit Step Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def impulse_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the impulse response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the impulse response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the impulse response. NumPy array.
        y = Amplitude-axis values of the points in the impulse response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import impulse_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> impulse_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.06616480200395854,... , 9.854500743565858, 10.0],
    [0.9999999799999999, 0.7042848373025861,...,7.170748906965121e-13, -5.1901263495547205e-12])

    See Also
    ========

    impulse_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def impulse_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Impulse Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import impulse_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> impulse_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/dynamicsystem.impulse.html

    """
    x, y = impulse_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Impulse Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def ramp_response_numerical_data(system, slope=1, prec=8,
    lower_limit=0, upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the ramp response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the ramp response data is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the ramp response plot. NumPy array.
        y = Amplitude-axis values of the points in the ramp response plot. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

        When ``slope`` is negative.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import ramp_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> ramp_response_numerical_data(tf1)   # doctest: +SKIP
    (([0.0, 0.12166980856813935,..., 9.861246379582118, 10.0],
    [1.4504508011325967e-09, 0.006046440489058766,..., 0.12499999999568202, 0.12499999999661349]))

    See Also
    ========

    ramp_response_plot

    """
    if slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = (slope*system.to_expr())/((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    x, y = ramp_response_numerical_data(system, slope=slope, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_magnitude_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', **kwargs):
    """
    Returns the numerical data of the Bode magnitude plot of the system.
    It is internally used by ``bode_magnitude_plot`` to get the data
    for plotting Bode magnitude plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode magnitude plot.
        y = y-axis values of the Bode magnitude plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_magnitude_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_magnitude_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.5148378120533502e-05,..., 68437.36188804005, 100000.0],
    [-6.020599914256786, -6.0205999155219505,..., -193.4117304087953, -200.00000000260573])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)

    x, y = LineOver1DRangeSeries(mag,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    return x, y


def bode_magnitude_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', **kwargs):
    r"""
    Returns the Bode magnitude plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_magnitude_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')


    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Bode Plot (Magnitude) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_phase_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', phase_unit='rad', phase_unwrap = True, **kwargs):
    """
    Returns the numerical data of the Bode phase plot of the system.
    It is internally used by ``bode_phase_plot`` to get the data
    for plotting Bode phase plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the Bode phase plot data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and '``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.
    phase_unwrap : bool, optional
        Set to ``True`` by default.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode phase plot.
        y = y-axis values of the Bode phase plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency or phase units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_phase_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_phase_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.4472354033813751e-05, 2.035581932165858e-05,..., 47577.3248186011, 67884.09326036123, 100000.0],
    [-2.5000000000291665e-05, -3.6180885085e-05, -5.08895483066e-05,...,-3.1415085799262523, -3.14155265358979])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    if phase_unit == 'deg':
        phase = arg(w_expr)*180/pi
    else:
        phase = arg(w_expr)

    x, y = LineOver1DRangeSeries(phase,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    half = None
    if phase_unwrap:
        if(phase_unit == 'rad'):
            half = pi
        elif(phase_unit == 'deg'):
            half = 180
    if half:
        unit = 2*half
        for i in range(1, len(y)):
            diff = y[i] - y[i - 1]
            if diff > half:      # Jump from -half to half
                y[i] = (y[i] - unit)
            elif diff < -half:   # Jump from half to -half
                y[i] = (y[i] + unit)

    return x, y


def bode_phase_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_phase_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')

    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Phase (%s)' % phase_unit)
    plt.title(f'Bode Plot (Phase) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_plot(system, initial_exp=-5, final_exp=5,
    grid=True, show_axes=False, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import bode_plot
        >>> tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> bode_plot(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP

    See Also
    ========

    bode_magnitude_plot, bode_phase_plot

    """
    plt.subplot(211)
    mag = bode_magnitude_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes,
        freq_unit=freq_unit, **kwargs)
    mag.title(f'Bode Plot of ${latex(system)}$', pad=20)
    mag.xlabel(None)
    plt.subplot(212)
    bode_phase_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap, **kwargs).title(None)

    if show:
        plt.show()
        return

    return plt


def nyquist_plot_expr(system):
    """Function to get the expression for Nyquist plot."""
    s = system.var
    w = Dummy('w', real=True)
    repl = I * w
    expr = system.to_expr()
    w_expr = expr.subs({s: repl})
    w_expr = w_expr.as_real_imag()
    real_expr = w_expr[0]
    imag_expr = w_expr[1]
    return real_expr, imag_expr, w


def nichols_plot_expr(system):
    """Function to get the expression for Nichols plot."""
    s = system.var
    w = Dummy('w', real=True)
    sys_expr = system.to_expr()
    H_jw = sys_expr.subs(s, I*w)
    mag_expr = Abs(H_jw)
    mag_dB_expr = 20*log(mag_expr, 10)
    phase_expr = arg(H_jw)
    phase_deg_expr = deg(phase_expr)
    return mag_dB_expr, phase_deg_expr, w


def nyquist_plot(system, initial_omega=0.01, final_omega=100, show=True,
                color='b', **kwargs):
    r"""
    Generates the Nyquist plot for a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nyquist_plot
        >>> tf1 = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
        >>> nyquist_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nichols_plot, bode_plot

    """
    _check_system(system)
    real_expr, imag_expr, w = nyquist_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (real_expr, imag_expr),   # The curve
        (real_expr, -imag_expr),  # Its mirror image
        *w_values,
        show=False,
        line_color=color,
        adaptive=True,
        title=f'Nyquist Plot of ${latex(system)}$',
        xlabel='Real Axis',
        ylabel='Imaginary Axis',
        size=(6, 5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p


def nichols_plot(system, initial_omega=0.01, final_omega=100, show=True, color='b', **kwargs):
    r"""
    Generates the Nichols plot for a LTI system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nichols_plot
        >>> tf1 = TransferFunction(1.5, s**2+14*s+40.02, s)
        >>> nichols_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nyquist_plot, bode_plot

    """
    _check_system(system)
    magnitude_dB_expr, phase_deg_expr, w = nichols_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (phase_deg_expr, magnitude_dB_expr),
        *w_values,
        show=False,
        line_color=color,
        title=f'Nichols Plot of ${latex(system)}$',
        xlabel='Phase [deg]',
        ylabel='Magnitude [dB]',
        size=(6,5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p
