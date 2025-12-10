from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.stats.error_prop import variance_prop
from sympy.stats.symbolic_probability import (RandomSymbol, Variance,
        Covariance)


def test_variance_prop():
    x, y, z = symbols('x y z')
    phi, t = consts = symbols('phi t')
    a = RandomSymbol(x)
    var_x = Variance(a)
    var_y = Variance(RandomSymbol(y))
    var_z = Variance(RandomSymbol(z))
    f = Function('f')(x)
    cases = {
        x + y: var_x + var_y,
        a + y: var_x + var_y,
        x + y + z: var_x + var_y + var_z,
        2*x: 4*var_x,
        x*y: var_x*y**2 + var_y*x**2,
        1/x: var_x/x**4,
        x/y: (var_x*y**2 + var_y*x**2)/y**4,
        exp(x): var_x*exp(2*x),
        exp(2*x): 4*var_x*exp(4*x),
        exp(-x*t): t**2*var_x*exp(-2*t*x),
        f: Variance(f),
        }
    for inp, out in cases.items():
        obs = variance_prop(inp, consts=consts)
        assert out == obs

def test_variance_prop_with_covar():
    x, y, z = symbols('x y z')
    phi, t = consts = symbols('phi t')
    a = RandomSymbol(x)
    var_x = Variance(a)
    b = RandomSymbol(y)
    var_y = Variance(b)
    c = RandomSymbol(z)
    var_z = Variance(c)
    covar_x_y = Covariance(a, b)
    covar_x_z = Covariance(a, c)
    covar_y_z = Covariance(b, c)
    cases = {
        x + y: var_x + var_y + 2*covar_x_y,
        a + y: var_x + var_y + 2*covar_x_y,
        x + y + z: var_x + var_y + var_z + \
                   2*covar_x_y + 2*covar_x_z + 2*covar_y_z,
        2*x: 4*var_x,
        x*y: var_x*y**2 + var_y*x**2 + 2*covar_x_y/(x*y),
        1/x: var_x/x**4,
        exp(x): var_x*exp(2*x),
        exp(2*x): 4*var_x*exp(4*x),
        exp(-x*t): t**2*var_x*exp(-2*t*x),
        }
    for inp, out in cases.items():
        obs = variance_prop(inp, consts=consts, include_covar=True)
        assert out == obs
