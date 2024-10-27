from math import isclose
from sympy.core.numbers import I
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.abc import s, p, a
from sympy.external import import_module
from sympy.physics.control.control_plots import \
    (pole_zero_numerical_data, pole_zero_plot, step_response_numerical_data,
    step_response_plot, impulse_response_numerical_data,
    impulse_response_plot, ramp_response_numerical_data,
    ramp_response_plot, bode_magnitude_numerical_data,
    bode_phase_numerical_data, bode_plot)
from sympy.physics.control.lti import (TransferFunction,
    Series, Parallel, TransferFunctionMatrix)
from sympy.testing.pytest import raises, skip

matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

numpy = import_module('numpy')

tf1 = TransferFunction(1, p**2 + 0.5*p + 2, p)
tf2 = TransferFunction(p, 6*p**2 + 3*p + 1, p)
tf3 = TransferFunction(p, p**3 - 1, p)
tf4 = TransferFunction(10, p**3, p)
tf5 = TransferFunction(5, s**2 + 2*s + 10, s)
tf6 = TransferFunction(1, 1, s)
tf7 = TransferFunction(4*s*3 + 9*s**2 + 0.1*s + 11, 8*s**6 + 9*s**4 + 11, s)
tf8 = TransferFunction(5, s**2 + (2+I)*s + 10, s)

ser1 = Series(tf4, TransferFunction(1, p - 5, p))
ser2 = Series(tf3, TransferFunction(p, p + 2, p))

par1 = Parallel(tf1, tf2)


def _to_tuple(a, b):
    return tuple(a), tuple(b)

def _trim_tuple(a, b):
    a, b = _to_tuple(a, b)
    return tuple(a[0: 2] + a[len(a)//2 : len(a)//2 + 1] + a[-2:]), \
        tuple(b[0: 2] + b[len(b)//2 : len(b)//2 + 1] + b[-2:])

def y_coordinate_equality(plot_data_func, evalf_func, system):
    """Checks whether the y-coordinate value of the plotted
    data point is equal to the value of the function at a
    particular x."""
    x, y = plot_data_func(system)
    x, y = _trim_tuple(x, y)
    y_exp = tuple(evalf_func(system, x_i) for x_i in x)
    return all(Abs(y_exp_i - y_i) < 1e-8 for y_exp_i, y_i in zip(y_exp, y))


def test_errors():
    if not matplotlib:
        skip("Matplotlib not the default backend")

    # Invalid `system` check
    tfm = TransferFunctionMatrix([[tf6, tf5], [tf5, tf6]])
    expr = 1/(s**2 - 1)
    raises(NotImplementedError, lambda: pole_zero_plot(tfm))
    raises(NotImplementedError, lambda: pole_zero_numerical_data(expr))
    raises(NotImplementedError, lambda: impulse_response_plot(expr))
    raises(NotImplementedError, lambda: impulse_response_numerical_data(tfm))
    raises(NotImplementedError, lambda: step_response_plot(tfm))
    raises(NotImplementedError, lambda: step_response_numerical_data(expr))
    raises(NotImplementedError, lambda: ramp_response_plot(expr))
    raises(NotImplementedError, lambda: ramp_response_numerical_data(tfm))
    raises(NotImplementedError, lambda: bode_plot(tfm))

    # More than 1 variables
    tf_a = TransferFunction(a, s + 1, s)
    raises(ValueError, lambda: pole_zero_plot(tf_a))
    raises(ValueError, lambda: pole_zero_numerical_data(tf_a))
    raises(ValueError, lambda: impulse_response_plot(tf_a))
    raises(ValueError, lambda: impulse_response_numerical_data(tf_a))
    raises(ValueError, lambda: step_response_plot(tf_a))
    raises(ValueError, lambda: step_response_numerical_data(tf_a))
    raises(ValueError, lambda: ramp_response_plot(tf_a))
    raises(ValueError, lambda: ramp_response_numerical_data(tf_a))
    raises(ValueError, lambda: bode_plot(tf_a))

    # lower_limit > 0 for response plots
    raises(ValueError, lambda: impulse_response_plot(tf1, lower_limit=-1))
    raises(ValueError, lambda: step_response_plot(tf1, lower_limit=-0.1))
    raises(ValueError, lambda: ramp_response_plot(tf1, lower_limit=-4/3))

    # slope in ramp_response_plot() is negative
    raises(ValueError, lambda: ramp_response_plot(tf1, slope=-0.1))

    # incorrect frequency or phase unit
    raises(ValueError, lambda: bode_plot(tf1,freq_unit = 'hz'))
    raises(ValueError, lambda: bode_plot(tf1,phase_unit = 'degree'))


def test_pole_zero():
    if not numpy:
        skip("NumPy is required for this test")

    def pz_tester(sys, expected_value):
        z, p = pole_zero_numerical_data(sys)
        z_check = numpy.allclose(z, expected_value[0])
        p_check = numpy.allclose(p, expected_value[1])
        return p_check and z_check

    exp1 = [[], [-0.24999999999999994+1.3919410907075054j, -0.24999999999999994-1.3919410907075054j]]
    exp2 = [[0.0], [-0.25+0.3227486121839514j, -0.25-0.3227486121839514j]]
    exp3 = [[0.0], [-0.5000000000000004+0.8660254037844395j,
        -0.5000000000000004-0.8660254037844395j, 0.9999999999999998+0j]]
    exp4 = [[], [5.0, 0.0, 0.0, 0.0]]
    exp5 = [[-5.645751311064592, -0.5000000000000008, -0.3542486889354093],
        [-0.24999999999999986+1.3919410907075052j,
        -0.24999999999999986-1.3919410907075052j, -0.2499999999999998+0.32274861218395134j,
        -0.2499999999999998-0.32274861218395134j]]
    exp6 = [[], [-1.1641600331447917-3.545808351896439j,
          -0.8358399668552097+2.5458083518964383j]]

    assert pz_tester(tf1, exp1)
    assert pz_tester(tf2, exp2)
    assert pz_tester(tf3, exp3)
    assert pz_tester(ser1, exp4)
    assert pz_tester(par1, exp5)
    assert pz_tester(tf8, exp6)


def test_bode():
    if not numpy:
        skip("NumPy is required for this test")

    def bode_phase_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return arg(w_expr).subs({_w: point}).evalf()

    def bode_mag_evalf(system, point):
        expr = system.to_expr()
        _w = Dummy("w", real=True)
        w_expr = expr.subs({system.var: I*_w})
        return 20*log(Abs(w_expr), 10).subs({_w: point}).evalf()

    def test_bode_data(sys):
        return y_coordinate_equality(bode_magnitude_numerical_data, bode_mag_evalf, sys) \
            and y_coordinate_equality(bode_phase_numerical_data, bode_phase_evalf, sys)

    assert test_bode_data(tf1)
    assert test_bode_data(tf2)
    assert test_bode_data(tf3)
    assert test_bode_data(tf4)
    assert test_bode_data(tf5)


def check_point_accuracy(a, b):
    return all(isclose(*_, rel_tol=1e-1, abs_tol=1e-6
        ) for _ in zip(a, b))


def test_impulse_response():
    if not numpy:
        skip("NumPy is required for this test")

    def impulse_res_tester(sys, expected_value):
        x, y = _to_tuple(*impulse_response_numerical_data(sys,
            adaptive=False, n=10))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.544019738507865, 0.01993849743234938, -0.31140243360893216, -0.022852779906491996, 0.1778306498155759,
        0.01962941084328499, -0.1013115194573652, -0.014975541213105696, 0.0575789724730714))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.1666666675, 0.08389223412935855,
        0.02338051973475047, -0.014966807776379383, -0.034645954223054234, -0.040560075735512804,
        -0.037658628907103885, -0.030149507719590022, -0.021162090730736834, -0.012721292737437523))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (4.369893391586999e-09, 1.1750333000630964,
        3.2922404058312473, 9.432290008148343, 28.37098083007151, 86.18577464367974, 261.90356653762115,
        795.6538758627842, 2416.9920942096983, 7342.159505206647))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 6.17283950617284, 24.69135802469136,
        55.555555555555564, 98.76543209876544, 154.320987654321, 222.22222222222226, 302.46913580246917,
        395.0617283950618, 500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, -0.10455606138085417,
        0.06757671513476461, -0.03234567568833768, 0.013582514927757873, -0.005273419510705473,
        0.0019364083003354075, -0.000680070134067832, 0.00022969845960406913, -7.476094359583917e-05))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-6.016699583000218e-09, 0.35039802056107394, 3.3728423827689884, 12.119846079276684,
        25.86101014293389, 29.352480635282088, -30.49475907497664, -273.8717189554019, -863.2381702029659,
        -1747.0262164682233))
    exp7 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335,
        4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779,
        8.88888888888889, 10.0), (0.0, 18.934638095560974, 5346.93244680907, 1384609.8718249386,
        358161126.65801865, 92645770015.70108, 23964739753087.42, 6198974342083139.0, 1.603492601616059e+18,
        4.147764422869658e+20))

    assert impulse_res_tester(tf1, exp1)
    assert impulse_res_tester(tf2, exp2)
    assert impulse_res_tester(tf3, exp3)
    assert impulse_res_tester(tf4, exp4)
    assert impulse_res_tester(tf5, exp5)
    assert impulse_res_tester(tf7, exp6)
    assert impulse_res_tester(ser1, exp7)


def test_step_response():
    if not numpy:
        skip("NumPy is required for this test")

    def step_res_tester(sys, expected_value):
        x, y = _to_tuple(*step_response_numerical_data(sys,
            adaptive=False, n=10))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.9193285738516863e-08, 0.42283495488246126, 0.7840485977945262, 0.5546841805655717,
        0.33903033806932087, 0.4627251747410237, 0.5909907598988051, 0.5247213989553071,
        0.4486997874319281, 0.4839358435839171))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.13728409095645816, 0.19474559355325086, 0.1974909129243011, 0.16841657696573073,
        0.12559777736159378, 0.08153828016664713, 0.04360471317348958, 0.015072994568868221,
        -0.003636420058445484))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 0.6314542141914303, 2.9356520038101035, 9.37731009663807, 28.452300356688376,
        86.25721933273988, 261.9236645044672, 795.6435410577224, 2416.9786984578764, 7342.154119725917))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (0.0, 2.286236899862826, 18.28989519890261, 61.72839629629631, 146.31916159122088, 285.7796124828532,
        493.8271703703705, 784.1792566529494, 1170.553292729767, 1666.6667))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-3.999999997894577e-09, 0.6720357068882895, 0.4429938256137113, 0.5182010838004518,
        0.4944139147159695, 0.5016379853883338, 0.4995466896527733, 0.5001154784851325,
        0.49997448824584123, 0.5000039745919259))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (-1.5433688493882158e-09, 0.3428705539937336, 1.1253619102202777, 3.1849962651016517,
        9.47532757182671, 28.727231099148135, 87.29426924860557, 265.2138681048606, 805.6636260007757,
        2447.387582370878))

    assert step_res_tester(tf1, exp1)
    assert step_res_tester(tf2, exp2)
    assert step_res_tester(tf3, exp3)
    assert step_res_tester(tf4, exp4)
    assert step_res_tester(tf5, exp5)
    assert step_res_tester(ser2, exp6)


def test_ramp_response():
    if not numpy:
        skip("NumPy is required for this test")

    def ramp_res_tester(sys, num_points, expected_value, slope=1):
        x, y = _to_tuple(*ramp_response_numerical_data(sys,
            slope=slope, adaptive=False, n=num_points))
        x_check = check_point_accuracy(x, expected_value[0])
        y_check = check_point_accuracy(y, expected_value[1])
        return x_check and y_check

    exp1 = ((0.0, 2.0, 4.0, 6.0, 8.0, 10.0), (0.0, 0.7324667795033895, 1.9909720978650398,
        2.7956587704217783, 3.9224897567931514, 4.85022655284895))
    exp2 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445,
        5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0),
        (2.4360213402019326e-08, 0.10175320182493253, 0.33057612497658406, 0.5967937263298935,
        0.8431511866718248, 1.0398805391471613, 1.1776043125035738, 1.2600994825747305, 1.2981042689274653,
        1.304684417610106))
    exp3 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (-3.9329040468771836e-08,
        0.34686634635794555, 2.9998828170537903, 12.33303690737476, 40.993913948137795, 127.84145222317912,
        391.41713691996, 1192.0006858708389, 3623.9808672503405, 11011.728034546572))
    exp4 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.9051973784484078, 30.483158055174524,
        154.32098765432104, 487.7305288827924, 1190.7483615302544, 2469.1358024691367, 4574.3789056546275,
        7803.688462124678, 12500.0))
    exp5 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 3.8844361856975635, 9.141792069209865,
        14.096349157657231, 19.09783068994694, 24.10179770390321, 29.09907319114121, 34.10040420185154,
        39.09983919254265, 44.10006013058409))
    exp6 = ((0.0, 1.1111111111111112, 2.2222222222222223, 3.3333333333333335, 4.444444444444445, 5.555555555555555,
        6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0), (0.0, 1.1111111111111112, 2.2222222222222223,
        3.3333333333333335, 4.444444444444445, 5.555555555555555, 6.666666666666667, 7.777777777777779, 8.88888888888889, 10.0))

    assert ramp_res_tester(tf1, 6, exp1)
    assert ramp_res_tester(tf2, 10, exp2, 1.2)
    assert ramp_res_tester(tf3, 10, exp3, 1.5)
    assert ramp_res_tester(tf4, 10, exp4, 3)
    assert ramp_res_tester(tf5, 10, exp5, 9)
    assert ramp_res_tester(tf6, 10, exp6)
