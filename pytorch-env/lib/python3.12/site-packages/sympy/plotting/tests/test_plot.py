import os
from tempfile import TemporaryDirectory
import pytest
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.elementary.miscellaneous import Min
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
    Plot, plot, plot_parametric, plot3d_parametric_line, plot3d,
    plot3d_parametric_surface)
from sympy.plotting.plot import (
    unset_show, plot_contour, PlotGrid, MatplotlibBackend, TextBackend)
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries)
from sympy.testing.pytest import skip, warns, raises, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings

unset_show()


matplotlib = import_module(
    'matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))


class DummyBackendNotOk(Plot):
    """ Used to verify if users can create their own backends.
    This backend is meant to raise NotImplementedError for methods `show`,
    `save`, `close`.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


class DummyBackendOk(Plot):
    """ Used to verify if users can create their own backends.
    This backend is meant to pass all tests.
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def show(self):
        pass

    def save(self):
        pass

    def close(self):
        pass

def test_basic_plotting_backend():
    x = Symbol('x')
    plot(x, (x, 0, 3), backend='text')
    plot(x**2 + 1, (x, 0, 3), backend='text')

@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_1(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        ###
        # Examples from the 'introduction' notebook
        ###
        p = plot(x, legend=True, label='f1', adaptive=adaptive, n=10)
        p = plot(x*sin(x), x*cos(x), label='f2', adaptive=adaptive, n=10)
        p.extend(p)
        p[0].line_color = lambda a: a
        p[1].line_color = 'b'
        p.title = 'Big title'
        p.xlabel = 'the x axis'
        p[1].label = 'straight line'
        p.legend = True
        p.aspect_ratio = (1, 1)
        p.xlim = (-15, 20)
        filename = 'test_basic_options_and_colors.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p.extend(plot(x + 1, adaptive=adaptive, n=10))
        p.append(plot(x + 3, x**2, adaptive=adaptive, n=10)[1])
        filename = 'test_plot_extend_append.png'
        p.save(os.path.join(tmpdir, filename))

        p[2] = plot(x**2, (x, -2, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_setitem.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(sin(x), (x, -2*pi, 4*pi), adaptive=adaptive, n=10)
        filename = 'test_line_explicit.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(sin(x), adaptive=adaptive, n=10)
        filename = 'test_line_default_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot((x**2, (x, -5, 5)), (x**3, (x, -3, 3)), adaptive=adaptive, n=10)
        filename = 'test_line_multiple_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        raises(ValueError, lambda: plot(x, y))

        #Piecewise plots
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(Piecewise((x, x < 1), (x**2, True)), (x, -3, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise_2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # test issue 7471
        p1 = plot(x, adaptive=adaptive, n=10)
        p2 = plot(3, adaptive=adaptive, n=10)
        p1.extend(p2)
        filename = 'test_horizontal_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # test issue 10925
        f = Piecewise((-1, x < -1), (x, And(-1 <= x, x < 0)), \
            (x**2, And(0 <= x, x < 1)), (x**3, x >= 1))
        p = plot(f, (x, -3, 3), adaptive=adaptive, n=10)
        filename = 'test_plot_piecewise_3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_2(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        #parametric 2d plots.
        #Single plot with default range.
        p = plot_parametric(sin(x), cos(x), adaptive=adaptive, n=10)
        filename = 'test_parametric.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Single plot with range.
        p = plot_parametric(
            sin(x), cos(x), (x, -5, 5), legend=True, label='parametric_plot',
            adaptive=adaptive, n=10)
        filename = 'test_parametric_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Multiple plots with same range.
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #Multiple plots with different ranges.
        p = plot_parametric(
            (sin(x), cos(x), (x, -3, 3)), (x, sin(x), (x, -5, 5)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #depth of recursion specified.
        p = plot_parametric(x, sin(x), depth=13,
            adaptive=adaptive, n=10)
        filename = 'test_recursion_depth.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #No adaptive sampling.
        p = plot_parametric(cos(x), sin(x), adaptive=False, n=500)
        filename = 'test_adaptive.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        #3d parametric plots
        p = plot3d_parametric_line(
            sin(x), cos(x), x, legend=True, label='3d_parametric_plot',
            adaptive=adaptive, n=10)
        filename = 'test_3d_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(
            (sin(x), cos(x), x, (x, -5, 5)), (cos(x), sin(x), x, (x, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_3d_line_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(sin(x), cos(x), x, n=30,
            adaptive=adaptive)
        filename = 'test_3d_line_points.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # 3d surface single plot.
        p = plot3d(x * y, adaptive=adaptive, n=10)
        filename = 'test_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple 3D plots with same range.
        p = plot3d(-x * y, x * y, (x, -5, 5), adaptive=adaptive, n=10)
        filename = 'test_surface_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple 3D plots with different ranges.
        p = plot3d(
            (x * y, (x, -3, 3), (y, -3, 3)), (-x * y, (x, -3, 3), (y, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_surface_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Single Parametric 3D plot
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y,
            adaptive=adaptive, n=10)
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Parametric 3D plots.
        p = plot3d_parametric_surface(
            (x*sin(z), x*cos(z), z, (x, -5, 5), (z, -5, 5)),
            (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)),
            adaptive=adaptive, n=10)
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Single Contour plot.
        p = plot_contour(sin(x)*sin(y), (x, -5, 5), (y, -5, 5),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Contour plots with same range.
        p = plot_contour(x**2 + y**2, x**3 + y**3, (x, -5, 5), (y, -5, 5),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # Multiple Contour plots with different range.
        p = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)),
            adaptive=adaptive, n=10)
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_3(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        ###
        # Examples from the 'colors' notebook
        ###

        p = plot(sin(x), adaptive=adaptive, n=10)
        p[0].line_color = lambda a: a
        filename = 'test_colors_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: b
        filename = 'test_colors_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(x*sin(x), x*cos(x), (x, 0, 10), adaptive=adaptive, n=10)
        p[0].line_color = lambda a: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: a
        filename = 'test_colors_param_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))

        p[0].line_color = lambda a, b: b
        filename = 'test_colors_param_line_arity2b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_line(
            sin(x) + 0.1*sin(x)*cos(7*x),
            cos(x) + 0.1*cos(x)*cos(7*x),
            0.1*sin(7*x),
            (x, 0, 2*pi), adaptive=adaptive, n=10)
        p[0].line_color = lambdify_(x, sin(4*x))
        filename = 'test_colors_3d_line_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b: b
        filename = 'test_colors_3d_line_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].line_color = lambda a, b, c: c
        filename = 'test_colors_3d_line_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d(sin(x)*y, (x, 0, 6*pi), (y, -5, 5), adaptive=adaptive, n=10)
        p[0].surface_color = lambda a: a
        filename = 'test_colors_surface_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: b
        filename = 'test_colors_surface_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b, c: c
        filename = 'test_colors_surface_arity3a.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt((x - 3*pi)**2 + y**2))
        filename = 'test_colors_surface_arity3b.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot3d_parametric_surface(x * cos(4 * y), x * sin(4 * y), y,
                (x, -1, 1), (y, -1, 1), adaptive=adaptive, n=10)
        p[0].surface_color = lambda a: a
        filename = 'test_colors_param_surf_arity1.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambda a, b: a*b
        filename = 'test_colors_param_surf_arity2.png'
        p.save(os.path.join(tmpdir, filename))
        p[0].surface_color = lambdify_((x, y, z), sqrt(x**2 + y**2 + z**2))
        filename = 'test_colors_param_surf_arity3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True])
def test_plot_and_save_4(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    ###
    # Examples from the 'advanced' notebook
    ###

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        i = Integral(log((sin(x)**2 + 1)*sqrt(x**2 + 1)), (x, 0, y))
        p = plot(i, (y, 1, 5), adaptive=adaptive, n=10, force_real_eval=True)
        filename = 'test_advanced_integral.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_5(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        s = Sum(1/x**y, (x, 1, oo))
        p = plot(s, (y, 2, 10), adaptive=adaptive, n=10)
        filename = 'test_advanced_inf_sum.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p = plot(Sum(1/x, (x, 1, y)), (y, 2, 10), show=False,
            adaptive=adaptive, n=10)
        p[0].only_integers = True
        p[0].steps = True
        filename = 'test_advanced_fin_sum.png'

        # XXX: This should be fixed in experimental_lambdify or by using
        # ordinary lambdify so that it doesn't warn. The error results from
        # passing an array of values as the integration limit.
        #
        # UserWarning: The evaluation of the expression is problematic. We are
        # trying a failback method that may still work. Please report this as a
        # bug.
        with ignore_warnings(UserWarning):
            p.save(os.path.join(tmpdir, filename))

        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_and_save_6(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        filename = 'test.png'
        ###
        # Test expressions that can not be translated to np and generate complex
        # results.
        ###
        p = plot(sin(x) + I*cos(x))
        p.save(os.path.join(tmpdir, filename))

        with ignore_warnings(RuntimeWarning):
            p = plot(sqrt(sqrt(-x)))
            p.save(os.path.join(tmpdir, filename))

        p = plot(LambertW(x))
        p.save(os.path.join(tmpdir, filename))
        p = plot(sqrt(LambertW(x)))
        p.save(os.path.join(tmpdir, filename))

        #Characteristic function of a StudentT distribution with nu=10
        x1 = 5 * x**2 * exp_polar(-I*pi)/2
        m1 = meijerg(((1 / 2,), ()), ((5, 0, 1 / 2), ()), x1)
        x2 = 5*x**2 * exp_polar(I*pi)/2
        m2 = meijerg(((1/2,), ()), ((5, 0, 1/2), ()), x2)
        expr = (m1 + m2) / (48 * pi)
        with warns(
            UserWarning,
            match="The evaluation with NumPy/SciPy failed",
            test_stacklevel=False,
        ):
            p = plot(expr, (x, 1e-6, 1e-2), adaptive=adaptive, n=10)
            p.save(os.path.join(tmpdir, filename))


@pytest.mark.parametrize("adaptive", [True, False])
def test_plotgrid_and_save(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')

    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p1 = plot(x, adaptive=adaptive, n=10)
        p2 = plot_parametric((sin(x), cos(x)), (x, sin(x)), show=False,
            adaptive=adaptive, n=10)
        p3 = plot_parametric(
            cos(x), sin(x), adaptive=adaptive, n=10, show=False)
        p4 = plot3d_parametric_line(sin(x), cos(x), x, show=False,
            adaptive=adaptive, n=10)
        # symmetric grid
        p = PlotGrid(2, 2, p1, p2, p3, p4)
        filename = 'test_grid1.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        # grid size greater than the number of subplots
        p = PlotGrid(3, 4, p1, p2, p3, p4)
        filename = 'test_grid2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()

        p5 = plot(cos(x),(x, -pi, pi), show=False, adaptive=adaptive, n=10)
        p5[0].line_color = lambda a: a
        p6 = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1), show=False,
            adaptive=adaptive, n=10)
        p7 = plot_contour(
            (x**2 + y**2, (x, -5, 5), (y, -5, 5)),
            (x**3 + y**3, (x, -3, 3), (y, -3, 3)), show=False,
            adaptive=adaptive, n=10)
        # unsymmetric grid (subplots in one line)
        p = PlotGrid(1, 3, p5, p6, p7)
        filename = 'test_grid3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()


@pytest.mark.parametrize("adaptive", [True, False])
def test_append_issue_7140(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p1 = plot(x, adaptive=adaptive, n=10)
    p2 = plot(x**2, adaptive=adaptive, n=10)
    plot(x + 2, adaptive=adaptive, n=10)

    # append a series
    p2.append(p1[0])
    assert len(p2._series) == 2

    with raises(TypeError):
        p1.append(p2)

    with raises(TypeError):
        p1.append(p2._series)


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_15265(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    eqn = sin(x)

    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1), adaptive=adaptive, n=10)
    p._backend.close()

    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi), adaptive=adaptive, n=10)
    p._backend.close()

    p = plot(eqn, xlim=(-1, 1), adaptive=adaptive, n=10,
        ylim=(sympify('-3.14'), sympify('3.14')))
    p._backend.close()

    p = plot(eqn, adaptive=adaptive, n=10,
        xlim=(sympify('-3.14'), sympify('3.14')), ylim=(-1, 1))
    p._backend.close()

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))

    raises(ValueError,
        lambda: plot(eqn, adaptive=adaptive, n=10,
            xlim=(-1, 1), ylim=(-1, S.Infinity)))


def test_empty_Plot():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # No exception showing an empty plot
    plot()
    # Plot is only a base class: doesn't implement any logic for showing
    # images
    p = Plot()
    raises(NotImplementedError, lambda: p.show())


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_17405(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    f = x**0.3 - 10*x**3 + x**2
    p = plot(f, (x, -10, 10), adaptive=adaptive, n=30, show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present

    # RuntimeWarning: invalid value encountered in double_scalars
    with ignore_warnings(RuntimeWarning):
        assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_logplot_PR_16796(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(x, (x, .001, 100), adaptive=adaptive, n=30,
        xscale='log', show=False)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30
    assert p[0].end == 100.0
    assert p[0].start == .001


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_16572(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(LambertW(x), show=False, adaptive=adaptive, n=30)
    # Random number of segments, probably more than 50, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11865(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    k = Symbol('k', integer=True)
    f = Piecewise((-I*exp(I*pi*k)/k + I*exp(-I*pi*k)/k, Ne(k, 0)), (2*pi, True))
    p = plot(f, show=False, adaptive=adaptive, n=30)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    # and that there are no exceptions.
    assert len(p[0].get_data()[0]) >= 30


def test_issue_11461():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(real_root((log(x/(x-2))), 3), show=False, adaptive=True)
    with warns(
        RuntimeWarning,
        match="invalid value encountered in",
        test_stacklevel=False,
    ):
        # Random number of segments, probably more than 100, but we want to see
        # that there are segments generated, as opposed to when the bug was present
        # and that there are no exceptions.
        assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_11764(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot_parametric(cos(x), sin(x), (x, 0, 2 * pi),
        aspect_ratio=(1,1), show=False, adaptive=adaptive, n=30)
    assert p.aspect_ratio == (1, 1)
    # Random number of segments, probably more than 100, but we want to see
    # that there are segments generated, as opposed to when the bug was present
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_issue_13516(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    pm = plot(sin(x), backend="matplotlib", show=False, adaptive=adaptive, n=30)
    assert pm.backend == MatplotlibBackend
    assert len(pm[0].get_data()[0]) >= 30

    pt = plot(sin(x), backend="text", show=False, adaptive=adaptive, n=30)
    assert pt.backend == TextBackend
    assert len(pt[0].get_data()[0]) >= 30

    pd = plot(sin(x), backend="default", show=False, adaptive=adaptive, n=30)
    assert pd.backend == MatplotlibBackend
    assert len(pd[0].get_data()[0]) >= 30

    p = plot(sin(x), show=False, adaptive=adaptive, n=30)
    assert p.backend == MatplotlibBackend
    assert len(p[0].get_data()[0]) >= 30


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_limits(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(x, x**2, (x, -10, 10), adaptive=adaptive, n=10)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 10) < 2
    assert abs(xmax - 10) < 2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 10) < 10
    assert abs(ymax - 100) < 10


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot3d_parametric_line_limits(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    v1 = (2*cos(x), 2*sin(x), 2*x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2, adaptive=adaptive, n=60)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2

    p = plot3d_parametric_line(v2, v1, adaptive=adaptive, n=60)
    backend = p._backend

    xmin, xmax = backend.ax.get_xlim()
    assert abs(xmin + 2) < 1e-2
    assert abs(xmax - 2) < 1e-2
    ymin, ymax = backend.ax.get_ylim()
    assert abs(ymin + 2) < 1e-2
    assert abs(ymax - 2) < 1e-2
    zmin, zmax = backend.ax.get_zlim()
    assert abs(zmin + 10) < 1e-2
    assert abs(zmax - 10) < 1e-2


@pytest.mark.parametrize("adaptive", [True, False])
def test_plot_size(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    p1 = plot(sin(x), backend="matplotlib", size=(8, 4),
        adaptive=adaptive, n=10)
    s1 = p1._backend.fig.get_size_inches()
    assert (s1[0] == 8) and (s1[1] == 4)
    p2 = plot(sin(x), backend="matplotlib", size=(5, 10),
        adaptive=adaptive, n=10)
    s2 = p2._backend.fig.get_size_inches()
    assert (s2[0] == 5) and (s2[1] == 10)
    p3 = PlotGrid(2, 1, p1, p2, size=(6, 2),
        adaptive=adaptive, n=10)
    s3 = p3._backend.fig.get_size_inches()
    assert (s3[0] == 6) and (s3[1] == 2)

    with raises(ValueError):
        plot(sin(x), backend="matplotlib", size=(-1, 3))


def test_issue_20113():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')

    # verify the capability to use custom backends
    plot(sin(x), backend=Plot, show=False)
    p2 = plot(sin(x), backend=MatplotlibBackend, show=False)
    assert p2.backend == MatplotlibBackend
    assert len(p2[0].get_data()[0]) >= 30
    p3 = plot(sin(x), backend=DummyBackendOk, show=False)
    assert p3.backend == DummyBackendOk
    assert len(p3[0].get_data()[0]) >= 30

    # test for an improper coded backend
    p4 = plot(sin(x), backend=DummyBackendNotOk, show=False)
    assert p4.backend == DummyBackendNotOk
    assert len(p4[0].get_data()[0]) >= 30
    with raises(NotImplementedError):
        p4.show()
    with raises(NotImplementedError):
        p4.save("test/path")
    with raises(NotImplementedError):
        p4._backend.close()


def test_custom_coloring():
    x = Symbol('x')
    y = Symbol('y')
    plot(cos(x), line_color=lambda a: a)
    plot(cos(x), line_color=1)
    plot(cos(x), line_color="r")
    plot_parametric(cos(x), sin(x), line_color=lambda a: a)
    plot_parametric(cos(x), sin(x), line_color=1)
    plot_parametric(cos(x), sin(x), line_color="r")
    plot3d_parametric_line(cos(x), sin(x), x, line_color=lambda a: a)
    plot3d_parametric_line(cos(x), sin(x), x, line_color=1)
    plot3d_parametric_line(cos(x), sin(x), x, line_color="r")
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color=1)
    plot3d_parametric_surface(cos(x + y), sin(x - y), x - y,
            (x, -5, 5), (y, -5, 5),
            surface_color="r")
    plot3d(x*y, (x, -5, 5), (y, -5, 5),
            surface_color=lambda a, b: a**2 + b**2)
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color=1)
    plot3d(x*y, (x, -5, 5), (y, -5, 5), surface_color="r")


@pytest.mark.parametrize("adaptive", [True, False])
def test_deprecated_get_segments(adaptive):
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    f = sin(x)
    p = plot(f, (x, -10, 10), show=False, adaptive=adaptive, n=10)
    with warns_deprecated_sympy():
        p[0].get_segments()


@pytest.mark.parametrize("adaptive", [True, False])
def test_generic_data_series(adaptive):
    # verify that no errors are raised when generic data series are used
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol("x")
    p = plot(x,
        markers=[{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}],
        annotations=[{"text": "test", "xy": (0, 0)}],
        fill={"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]},
        rectangles=[{"xy": (0, 0), "width": 5, "height": 1}],
        adaptive=adaptive, n=10)
    assert len(p._backend.ax.collections) == 1
    assert len(p._backend.ax.patches) == 1
    assert len(p._backend.ax.lines) == 2
    assert len(p._backend.ax.texts) == 1


def test_deprecated_markers_annotations_rectangles_fill():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    p = plot(sin(x), (x, -10, 10), show=False)
    with warns_deprecated_sympy():
        p.markers = [{"args":[[0, 1], [0, 1]], "marker": "*", "linestyle": "none"}]
    assert len(p._series) == 2
    with warns_deprecated_sympy():
        p.annotations = [{"text": "test", "xy": (0, 0)}]
    assert len(p._series) == 3
    with warns_deprecated_sympy():
        p.fill = {"x": [0, 1, 2, 3], "y1": [0, 1, 2, 3]}
    assert len(p._series) == 4
    with warns_deprecated_sympy():
        p.rectangles = [{"xy": (0, 0), "width": 5, "height": 1}]
    assert len(p._series) == 5


def test_back_compatibility():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x = Symbol('x')
    y = Symbol('y')
    p = plot(sin(x), adaptive=False, n=5)
    assert len(p[0].get_points()) == 2
    assert len(p[0].get_data()) == 2
    p = plot_parametric(cos(x), sin(x), (x, 0, 2), adaptive=False, n=5)
    assert len(p[0].get_points()) == 2
    assert len(p[0].get_data()) == 3
    p = plot3d_parametric_line(cos(x), sin(x), x, (x, 0, 2),
        adaptive=False, n=5)
    assert len(p[0].get_points()) == 3
    assert len(p[0].get_data()) == 4
    p = plot3d(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 3
    p = plot_contour(cos(x**2 + y**2), (x, -pi, pi), (y, -pi, pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 3
    p = plot3d_parametric_surface(x * cos(y), x * sin(y), x * cos(4 * y) / 2,
        (x, 0, pi), (y, 0, 2*pi), n=5)
    assert len(p[0].get_meshes()) == 3
    assert len(p[0].get_data()) == 5


def test_plot_arguments():
    ### Test arguments for plot()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single expressions
    p = plot(x + 1)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}

    # single expressions custom label
    p = plot(x + 1, "label")
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "label"
    assert p[0].rendering_kw == {}

    # single expressions with range
    p = plot(x + 1, (x, -2, 2))
    assert p[0].ranges == [(x, -2, 2)]

    # single expressions with range, label and rendering-kw dictionary
    p = plot(x + 1, (x, -2, 2), "test", {"color": "r"})
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"color": "r"}

    # multiple expressions
    p = plot(x + 1, x**2)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x**2"
    assert p[1].rendering_kw == {}

    # multiple expressions over the same range
    p = plot(x + 1, x**2, (x, 0, 5))
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]

    # multiple expressions over the same range with the same rendering kws
    p = plot(x + 1, x**2, (x, 0, 5), {"color": "r"})
    assert p[0].ranges == [(x, 0, 5)]
    assert p[1].ranges == [(x, 0, 5)]
    assert p[0].rendering_kw == {"color": "r"}
    assert p[1].rendering_kw == {"color": "r"}

    # multiple expressions with different ranges, labels and rendering kws
    p = plot(
        (x + 1, (x, 0, 5)),
        (x**2, (x, -2, 2), "test", {"color": "r"}))
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert p[0].expr == x + 1
    assert p[0].ranges == [(x, 0, 5)]
    assert p[0].get_label(False) == "x + 1"
    assert p[0].rendering_kw == {}
    assert isinstance(p[1], LineOver1DRangeSeries)
    assert p[1].expr == x**2
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"color": "r"}

    # single argument: lambda function
    f = lambda t: t
    p = plot(lambda t: t)
    assert isinstance(p[0], LineOver1DRangeSeries)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range and label
    p = plot(f, ("t", -5, 6), "test")
    assert p[0].ranges[0][1:] == (-5, 6)
    assert p[0].get_label(False) == "test"


def test_plot_parametric_arguments():
    ### Test arguments for plot_parametric()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot_parametric(x + 1, x)
    assert isinstance(p[0], Parametric2DLineSeries)
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot_parametric(x + 1, x, (x, -2, 2), "test",
        {"cmap": "Reds"})
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot_parametric((x + 1, x), (x, -2, 2), "test")
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expressions same symbol
    p = plot_parametric((x + 1, x), (x ** 2, x + 1))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions different symbols
    p = plot_parametric((x + 1, x), (y ** 2, y + 1, "test"))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, y + 1)
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions same range
    p = plot_parametric((x + 1, x), (x ** 2, x + 1), (x, -2, 2))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot_parametric(
        (x + 1, x, (x, -2, 2), "test1"),
        (x ** 2, x + 1, (x, -3, 3), "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x)
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test1"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, x + 1)
    assert p[1].ranges == [(x, -3, 3)]
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    p = plot_parametric(fx, fy)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot_parametric(fx, fy, ("t", 0, 2), "test")
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


def test_plot3d_parametric_line_arguments():
    ### Test arguments for plot3d_parametric_line()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_line(x + 1, x, sin(x))
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}

    # single parametric expression with custom range, label and rendering kws
    p = plot3d_parametric_line(x + 1, x, sin(x), (x, -2, 2),
        "test", {"cmap": "Reds"})
    assert isinstance(p[0], Parametric3DLineSeries)
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d_parametric_line((x + 1, x, sin(x)), (x, -2, 2), "test")
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -2, 2)]
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # multiple parametric expression same symbol
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)), (x ** 2, 1, cos(x), {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
    assert p[1].ranges == [(x, -10, 10)]
    assert p[1].get_label(False) == "x"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expression different symbols
    p = plot3d_parametric_line((x + 1, x, sin(x)), (y ** 2, 1, cos(y)))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (y ** 2, 1, cos(y))
    assert p[1].ranges == [(y, -10, 10)]
    assert p[1].get_label(False) == "y"
    assert p[1].rendering_kw == {}

    # multiple parametric expression, custom ranges and labels
    p = plot3d_parametric_line(
        (x + 1, x, sin(x)),
        (x ** 2, 1, cos(x), (x, -2, 2), "test", {"cmap": "Reds"}))
    assert p[0].expr == (x + 1, x, sin(x))
    assert p[0].ranges == [(x, -10, 10)]
    assert p[0].get_label(False) == "x"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x ** 2, 1, cos(x))
    assert p[1].ranges == [(x, -2, 2)]
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single argument: lambda function
    fx = lambda t: t
    fy = lambda t: 2 * t
    fz = lambda t: 3 * t
    p = plot3d_parametric_line(fx, fy, fz)
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert "Dummy" in p[0].get_label(False)
    assert p[0].rendering_kw == {}

    # single argument: lambda function + custom range + label
    p = plot3d_parametric_line(fx, fy, fz, ("t", 0, 2), "test")
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}


def test_plot3d_plot_contour_arguments():
    ### Test arguments for plot3d() and plot_contour()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single expression
    p = plot3d(x + y)
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}

    # single expression, custom range, label and rendering kws
    p = plot3d(x + y, (x, -2, 2), "test", {"cmap": "Reds"})
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    p = plot3d(x + y, (x, -2, 2), (y, -4, 4), "test")
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)

    # multiple expressions
    p = plot3d(x + y, x * y)
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, same custom ranges
    p = plot3d(x + y, x * y, (x, -2, 2), (y, -4, 4))
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -2, 2)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "x*y"
    assert p[1].rendering_kw == {}

    # multiple expressions, custom ranges, labels and rendering kws
    p = plot3d(
        (x + y, (x, -2, 2), (y, -4, 4)),
        (x * y, (x, -3, 3), (y, -6, 6), "test", {"cmap": "Reds"}))
    assert p[0].expr == x + y
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "x + y"
    assert p[0].rendering_kw == {}
    assert p[1].expr == x * y
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -6, 6)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # single expression: lambda function
    f = lambda x, y: x + y
    p = plot3d(f)
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-10, 10)
    assert p[0].ranges[1][1:] == (-10, 10)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # single expression: lambda function + custom ranges + label
    p = plot3d(f, ("a", -5, 3), ("b", -2, 1), "test")
    assert callable(p[0].expr)
    assert p[0].ranges[0][1:] == (-5, 3)
    assert p[0].ranges[1][1:] == (-2, 1)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}

    # test issue 25818
    # single expression, custom range, min/max functions
    p = plot3d(Min(x, y), (x, 0, 10), (y, 0, 10))
    assert isinstance(p[0], SurfaceOver2DRangeSeries)
    assert p[0].expr == Min(x, y)
    assert p[0].ranges[0] == (x, 0, 10)
    assert p[0].ranges[1] == (y, 0, 10)
    assert p[0].get_label(False) == "Min(x, y)"
    assert p[0].rendering_kw == {}


def test_plot3d_parametric_surface_arguments():
    ### Test arguments for plot3d_parametric_surface()
    if not matplotlib:
        skip("Matplotlib not the default backend")

    x, y = symbols("x, y")

    # single parametric expression
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y))
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}

    # single parametric expression, custom ranges, labels and rendering kws
    p = plot3d_parametric_surface(x + y, cos(x + y), sin(x + y),
        (x, -2, 2), (y, -4, 4), "test", {"cmap": "Reds"})
    assert isinstance(p[0], ParametricSurfaceSeries)
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -4, 4)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {"cmap": "Reds"}

    # multiple parametric expressions
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y)),
        (x - y, cos(x - y), sin(x - y), "test"))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[0].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[0].get_label(False) == "(x + y, cos(x + y), sin(x + y))"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -10, 10) or (y, -10, 10)
    assert p[1].ranges[1] == (x, -10, 10) or (y, -10, 10)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}

    # multiple parametric expressions, custom ranges and labels
    p = plot3d_parametric_surface(
        (x + y, cos(x + y), sin(x + y), (x, -2, 2), "test"),
        (x - y, cos(x - y), sin(x - y), (x, -3, 3), (y, -4, 4),
            "test2", {"cmap": "Reds"}))
    assert p[0].expr == (x + y, cos(x + y), sin(x + y))
    assert p[0].ranges[0] == (x, -2, 2)
    assert p[0].ranges[1] == (y, -10, 10)
    assert p[0].get_label(False) == "test"
    assert p[0].rendering_kw == {}
    assert p[1].expr == (x - y, cos(x - y), sin(x - y))
    assert p[1].ranges[0] == (x, -3, 3)
    assert p[1].ranges[1] == (y, -4, 4)
    assert p[1].get_label(False) == "test2"
    assert p[1].rendering_kw == {"cmap": "Reds"}

    # lambda functions instead of symbolic expressions for a single 3D
    # parametric surface
    p = plot3d_parametric_surface(
        lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4))
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (-0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}

    # lambda functions instead of symbolic expressions for multiple 3D
    # parametric surfaces
    p = plot3d_parametric_surface(
        (lambda u, v: u, lambda u, v: v, lambda u, v: u + v,
        ("u", 0, 2), ("v", -3, 4)),
        (lambda u, v: v, lambda u, v: u, lambda u, v: u - v,
        ("u", -2, 3), ("v", -4, 5), "test"))
    assert all(callable(t) for t in p[0].expr)
    assert p[0].ranges[0][1:] == (0, 2)
    assert p[0].ranges[1][1:] == (-3, 4)
    assert p[0].get_label(False) == ""
    assert p[0].rendering_kw == {}
    assert all(callable(t) for t in p[1].expr)
    assert p[1].ranges[0][1:] == (-2, 3)
    assert p[1].ranges[1][1:] == (-4, 5)
    assert p[1].get_label(False) == "test"
    assert p[1].rendering_kw == {}
