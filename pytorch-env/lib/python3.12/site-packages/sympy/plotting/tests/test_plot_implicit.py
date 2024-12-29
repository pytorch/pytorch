from sympy.core.numbers import (I, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import (And, Or)
from sympy.plotting.plot_implicit import plot_implicit
from sympy.plotting.plot import unset_show
from tempfile import NamedTemporaryFile, mkdtemp
from sympy.testing.pytest import skip, warns, XFAIL
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager

import os

#Set plots not to show
unset_show()

def tmp_file(dir=None, name=''):
    return NamedTemporaryFile(
    suffix='.png', dir=dir, delete=False).name

def plot_and_save(expr, *args, name='', dir=None, **kwargs):
    p = plot_implicit(expr, *args, **kwargs)
    p.save(tmp_file(dir=dir, name=name))
    # Close the plot to avoid a warning from matplotlib
    p._backend.close()

def plot_implicit_tests(name):
    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    x = Symbol('x')
    y = Symbol('y')
    #implicit plot tests
    plot_and_save(Eq(y, cos(x)), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), (x, -5, 5),
            (y, -4, 4), name=name, dir=temp_dir)
    plot_and_save(y > 1 / x, (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y < 1 / tan(x), (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y >= 2 * sin(x) * cos(x), (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y <= x**2, (x, -3, 3),
            (y, -1, 5), name=name, dir=temp_dir)

    #Test all input args for plot_implicit
    plot_and_save(Eq(y**2, x**3 - x), dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, n=500, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), dir=temp_dir)
    plot_and_save(And(y > exp(x), y > x + 2), dir=temp_dir)
    plot_and_save(Or(y > x, y > -x), dir=temp_dir)
    plot_and_save(x**2 - 1, (x, -5, 5), dir=temp_dir)
    plot_and_save(x**2 - 1, dir=temp_dir)
    plot_and_save(y > x, depth=-5, dir=temp_dir)
    plot_and_save(y > x, depth=5, dir=temp_dir)
    plot_and_save(y > cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(y < cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(And(y > cos(x), Or(y > x, Eq(y, x))), dir=temp_dir)
    plot_and_save(y - cos(pi / x), dir=temp_dir)

    plot_and_save(x**2 - 1, title='An implicit plot', dir=temp_dir)

@XFAIL
def test_no_adaptive_meshing():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        try:
            temp_dir = mkdtemp()
            TmpFileManager.tmp_folder(temp_dir)
            x = Symbol('x')
            y = Symbol('y')
            # Test plots which cannot be rendered using the adaptive algorithm

            # This works, but it triggers a deprecation warning from sympify(). The
            # code needs to be updated to detect if interval math is supported without
            # relying on random AttributeErrors.
            with warns(UserWarning, match="Adaptive meshing could not be applied"):
                plot_and_save(Eq(y, re(cos(x) + I*sin(x))), name='test', dir=temp_dir)
        finally:
            TmpFileManager.cleanup()
    else:
        skip("Matplotlib not the default backend")
def test_line_color():
    x, y = symbols('x, y')
    p = plot_implicit(x**2 + y**2 - 1, line_color="green", show=False)
    assert p._series[0].line_color == "green"
    p = plot_implicit(x**2 + y**2 - 1, line_color='r', show=False)
    assert p._series[0].line_color == "r"

def test_matplotlib():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        try:
            plot_implicit_tests('test')
            test_line_color()
        finally:
            TmpFileManager.cleanup()
    else:
        skip("Matplotlib not the default backend")


def test_region_and():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if not matplotlib:
        skip("Matplotlib not the default backend")

    from matplotlib.testing.compare import compare_images
    test_directory = os.path.dirname(os.path.abspath(__file__))

    try:
        temp_dir = mkdtemp()
        TmpFileManager.tmp_folder(temp_dir)

        x, y = symbols('x y')

        r1 = (x - 1)**2 + y**2 < 2
        r2 = (x + 1)**2 + y**2 < 2

        test_filename = tmp_file(dir=temp_dir, name="test_region_and")
        cmp_filename = os.path.join(test_directory, "test_region_and.png")
        p = plot_implicit(r1 & r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_or")
        cmp_filename = os.path.join(test_directory, "test_region_or.png")
        p = plot_implicit(r1 | r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_not")
        cmp_filename = os.path.join(test_directory, "test_region_not.png")
        p = plot_implicit(~r1, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_xor")
        cmp_filename = os.path.join(test_directory, "test_region_xor.png")
        p = plot_implicit(r1 ^ r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
    finally:
        TmpFileManager.cleanup()
