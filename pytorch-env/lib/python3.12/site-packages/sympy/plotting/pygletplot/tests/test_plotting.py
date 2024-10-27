from sympy.external.importtools import import_module

disabled = False

# if pyglet.gl fails to import, e.g. opengl is missing, we disable the tests
pyglet_gl = import_module("pyglet.gl", catch=(OSError,))
pyglet_window = import_module("pyglet.window", catch=(OSError,))
if not pyglet_gl or not pyglet_window:
    disabled = True


from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
x, y, z = symbols('x, y, z')


def test_plot_2d():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(x, [x, -5, 5, 4], visible=False)
    p.wait_for_calculations()


def test_plot_2d_discontinuous():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -1, 1, 2], visible=False)
    p.wait_for_calculations()


def test_plot_3d():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(x*y, [x, -5, 5, 5], [y, -5, 5, 5], visible=False)
    p.wait_for_calculations()


def test_plot_3d_discontinuous():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -3, 3, 6], [y, -1, 1, 1], visible=False)
    p.wait_for_calculations()


def test_plot_2d_polar():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -1, 1, 4], 'mode=polar', visible=False)
    p.wait_for_calculations()


def test_plot_3d_cylinder():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(
        1/y, [x, 0, 6.282, 4], [y, -1, 1, 4], 'mode=polar;style=solid',
        visible=False)
    p.wait_for_calculations()


def test_plot_3d_spherical():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(
        1, [x, 0, 6.282, 4], [y, 0, 3.141,
            4], 'mode=spherical;style=wireframe',
        visible=False)
    p.wait_for_calculations()


def test_plot_2d_parametric():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(sin(x), cos(x), [x, 0, 6.282, 4], visible=False)
    p.wait_for_calculations()


def test_plot_3d_parametric():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(sin(x), cos(x), x/5.0, [x, 0, 6.282, 4], visible=False)
    p.wait_for_calculations()


def _test_plot_log():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(log(x), [x, 0, 6.282, 4], 'mode=polar', visible=False)
    p.wait_for_calculations()


def test_plot_integral():
    # Make sure it doesn't treat x as an independent variable
    from sympy.plotting.pygletplot import PygletPlot
    from sympy.integrals.integrals import Integral
    p = PygletPlot(Integral(z*x, (x, 1, z), (z, 1, y)), visible=False)
    p.wait_for_calculations()
