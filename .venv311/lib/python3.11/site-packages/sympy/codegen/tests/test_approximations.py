import math
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.codegen.rewriting import optimize
from sympy.codegen.approximations import SumApprox, SeriesApprox


def test_SumApprox_trivial():
    x = symbols('x')
    expr1 = 1 + x
    sum_approx = SumApprox(bounds={x: (-1e-20, 1e-20)}, reltol=1e-16)
    apx1 = optimize(expr1, [sum_approx])
    assert apx1 - 1 == 0


def test_SumApprox_monotone_terms():
    x, y, z = symbols('x y z')
    expr1 = exp(z)*(x**2 + y**2 + 1)
    bnds1 = {x: (0, 1e-3), y: (100, 1000)}
    sum_approx_m2 = SumApprox(bounds=bnds1, reltol=1e-2)
    sum_approx_m5 = SumApprox(bounds=bnds1, reltol=1e-5)
    sum_approx_m11 = SumApprox(bounds=bnds1, reltol=1e-11)
    assert (optimize(expr1, [sum_approx_m2])/exp(z) - (y**2)).simplify() == 0
    assert (optimize(expr1, [sum_approx_m5])/exp(z) - (y**2 + 1)).simplify() == 0
    assert (optimize(expr1, [sum_approx_m11])/exp(z) - (y**2 + 1 + x**2)).simplify() == 0


def test_SeriesApprox_trivial():
    x, z = symbols('x z')
    for factor in [1, exp(z)]:
        x = symbols('x')
        expr1 = exp(x)*factor
        bnds1 = {x: (-1, 1)}
        series_approx_50 = SeriesApprox(bounds=bnds1, reltol=0.50)
        series_approx_10 = SeriesApprox(bounds=bnds1, reltol=0.10)
        series_approx_05 = SeriesApprox(bounds=bnds1, reltol=0.05)
        c = (bnds1[x][1] + bnds1[x][0])/2  # 0.0
        f0 = math.exp(c)  # 1.0

        ref_50 = f0 + x + x**2/2
        ref_10 = f0 + x + x**2/2 + x**3/6
        ref_05 = f0 + x + x**2/2 + x**3/6 + x**4/24

        res_50 = optimize(expr1, [series_approx_50])
        res_10 = optimize(expr1, [series_approx_10])
        res_05 = optimize(expr1, [series_approx_05])

        assert (res_50/factor - ref_50).simplify() == 0
        assert (res_10/factor - ref_10).simplify() == 0
        assert (res_05/factor - ref_05).simplify() == 0

        max_ord3 = SeriesApprox(bounds=bnds1, reltol=0.05, max_order=3)
        assert optimize(expr1, [max_ord3]) == expr1
