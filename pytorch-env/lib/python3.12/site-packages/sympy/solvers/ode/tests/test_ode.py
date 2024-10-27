from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect

from sympy.solvers.ode import (classify_ode,
    homogeneous_order, dsolve)

from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
    constant_renumber, constantsimp, get_numbered_constants, solve_ics)

from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow, SKIP
from sympy.utilities.misc import filldedent


C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = symbols('C0:11')
u, x, y, z = symbols('u,x:z', real=True)
f = Function('f')
g = Function('g')
h = Function('h')

# Note: Examples which were specifically testing Single ODE solver are moved to test_single.py
# and all the system of ode examples are moved to test_systems.py
# Note: the tests below may fail (but still be correct) if ODE solver,
# the integral engine, solve(), or even simplify() changes. Also, in
# differently formatted solutions, the arbitrary constants might not be
# equal.  Using specific hints in tests can help to avoid this.

# Tests of order higher than 1 should run the solutions through
# constant_renumber because it will normalize it (constant_renumber causes
# dsolve() to return different results on different machines)


def test_get_numbered_constants():
    with raises(ValueError):
        get_numbered_constants(None)


def test_dsolve_all_hint():
    eq = f(x).diff(x)
    output = dsolve(eq, hint='all')

    # Match the Dummy variables:
    sol1 = output['separable_Integral']
    _y = sol1.lhs.args[1][0]
    sol1 = output['1st_homogeneous_coeff_subs_dep_div_indep_Integral']
    _u1 = sol1.rhs.args[1].args[1][0]

    expected = {'Bernoulli_Integral': Eq(f(x), C1 + Integral(0, x)),
        '1st_homogeneous_coeff_best': Eq(f(x), C1),
        'Bernoulli': Eq(f(x), C1),
        'nth_algebraic': Eq(f(x), C1),
        'nth_linear_euler_eq_homogeneous': Eq(f(x), C1),
        'nth_linear_constant_coeff_homogeneous': Eq(f(x), C1),
        'separable': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_indep_div_dep': Eq(f(x), C1),
        'nth_algebraic_Integral': Eq(f(x), C1),
        '1st_linear': Eq(f(x), C1),
        '1st_linear_Integral': Eq(f(x), C1 + Integral(0, x)),
        '1st_exact': Eq(f(x), C1),
        '1st_exact_Integral': Eq(Subs(Integral(0, x) + Integral(1, _y), _y, f(x)), C1),
        'lie_group': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_dep_div_indep': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral': Eq(log(x), C1 + Integral(-1/_u1, (_u1, f(x)/x))),
        '1st_power_series': Eq(f(x), C1),
        'separable_Integral': Eq(Integral(1, (_y, f(x))), C1 + Integral(0, x)),
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral': Eq(f(x), C1),
        'best': Eq(f(x), C1),
        'best_hint': 'nth_algebraic',
        'default': 'nth_algebraic',
        'order': 1}
    assert output == expected

    assert dsolve(eq, hint='best') == Eq(f(x), C1)


def test_dsolve_ics():
    # Maybe this should just use one of the solutions instead of raising...
    with raises(NotImplementedError):
        dsolve(f(x).diff(x) - sqrt(f(x)), ics={f(1):1})


@slow
def test_dsolve_options():
    eq = x*f(x).diff(x) + f(x)
    a = dsolve(eq, hint='all')
    b = dsolve(eq, hint='all', simplify=False)
    c = dsolve(eq, hint='all_Integral')
    keys = ['1st_exact', '1st_exact_Integral', '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_linear',
        '1st_linear_Integral', 'Bernoulli', 'Bernoulli_Integral',
        'almost_linear', 'almost_linear_Integral', 'best', 'best_hint',
        'default', 'factorable', 'lie_group',
        'nth_linear_euler_eq_homogeneous', 'order',
        'separable', 'separable_Integral']
    Integral_keys = ['1st_exact_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_linear_Integral',
        'Bernoulli_Integral', 'almost_linear_Integral', 'best', 'best_hint', 'default',
        'factorable', 'nth_linear_euler_eq_homogeneous',
        'order', 'separable_Integral']
    assert sorted(a.keys()) == keys
    assert a['order'] == ode_order(eq, f(x))
    assert a['best'] == Eq(f(x), C1/x)
    assert dsolve(eq, hint='best') == Eq(f(x), C1/x)
    assert a['default'] == 'factorable'
    assert a['best_hint'] == 'factorable'
    assert not a['1st_exact'].has(Integral)
    assert not a['separable'].has(Integral)
    assert not a['1st_homogeneous_coeff_best'].has(Integral)
    assert not a['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    assert not a['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    assert not a['1st_linear'].has(Integral)
    assert a['1st_linear_Integral'].has(Integral)
    assert a['1st_exact_Integral'].has(Integral)
    assert a['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    assert a['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    assert a['separable_Integral'].has(Integral)
    assert sorted(b.keys()) == keys
    assert b['order'] == ode_order(eq, f(x))
    assert b['best'] == Eq(f(x), C1/x)
    assert dsolve(eq, hint='best', simplify=False) == Eq(f(x), C1/x)
    assert b['default'] == 'factorable'
    assert b['best_hint'] == 'factorable'
    assert a['separable'] != b['separable']
    assert a['1st_homogeneous_coeff_subs_dep_div_indep'] != \
        b['1st_homogeneous_coeff_subs_dep_div_indep']
    assert a['1st_homogeneous_coeff_subs_indep_div_dep'] != \
        b['1st_homogeneous_coeff_subs_indep_div_dep']
    assert not b['1st_exact'].has(Integral)
    assert not b['separable'].has(Integral)
    assert not b['1st_homogeneous_coeff_best'].has(Integral)
    assert not b['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    assert not b['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    assert not b['1st_linear'].has(Integral)
    assert b['1st_linear_Integral'].has(Integral)
    assert b['1st_exact_Integral'].has(Integral)
    assert b['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    assert b['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    assert b['separable_Integral'].has(Integral)
    assert sorted(c.keys()) == Integral_keys
    raises(ValueError, lambda: dsolve(eq, hint='notarealhint'))
    raises(ValueError, lambda: dsolve(eq, hint='Liouville'))
    assert dsolve(f(x).diff(x) - 1/f(x)**2, hint='all')['best'] == \
        dsolve(f(x).diff(x) - 1/f(x)**2, hint='best')
    assert dsolve(f(x) + f(x).diff(x) + sin(x).diff(x) + 1, f(x),
                  hint="1st_linear_Integral") == \
        Eq(f(x), (C1 + Integral((-sin(x).diff(x) - 1)*
                exp(Integral(1, x)), x))*exp(-Integral(1, x)))


def test_classify_ode():
    assert classify_ode(f(x).diff(x, 2), f(x)) == \
        (
        'nth_algebraic',
        'nth_linear_constant_coeff_homogeneous',
        'nth_linear_euler_eq_homogeneous',
        'Liouville',
        '2nd_power_series_ordinary',
        'nth_algebraic_Integral',
        'Liouville_Integral',
        )
    assert classify_ode(f(x), f(x)) == ('nth_algebraic', 'nth_algebraic_Integral')
    assert classify_ode(Eq(f(x).diff(x), 0), f(x)) == (
        'nth_algebraic',
        'separable',
        '1st_exact',
        '1st_linear',
        'Bernoulli',
        '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_power_series', 'lie_group',
        'nth_linear_constant_coeff_homogeneous',
        'nth_linear_euler_eq_homogeneous',
        'nth_algebraic_Integral',
        'separable_Integral',
        '1st_exact_Integral',
        '1st_linear_Integral',
        'Bernoulli_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral')
    assert classify_ode(f(x).diff(x)**2, f(x)) == ('factorable',
         'nth_algebraic',
         'separable',
         '1st_exact',
         '1st_linear',
         'Bernoulli',
         '1st_homogeneous_coeff_best',
         '1st_homogeneous_coeff_subs_indep_div_dep',
         '1st_homogeneous_coeff_subs_dep_div_indep',
         '1st_power_series',
         'lie_group',
         'nth_linear_euler_eq_homogeneous',
         'nth_algebraic_Integral',
         'separable_Integral',
         '1st_exact_Integral',
         '1st_linear_Integral',
         'Bernoulli_Integral',
         '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
         '1st_homogeneous_coeff_subs_dep_div_indep_Integral')
    # issue 4749: f(x) should be cleared from highest derivative before classifying
    a = classify_ode(Eq(f(x).diff(x) + f(x), x), f(x))
    b = classify_ode(f(x).diff(x)*f(x) + f(x)*f(x) - x*f(x), f(x))
    c = classify_ode(f(x).diff(x)/f(x) + f(x)/f(x) - x/f(x), f(x))
    assert a == ('1st_exact',
        '1st_linear',
        'Bernoulli',
        'almost_linear',
        '1st_power_series', "lie_group",
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        '1st_exact_Integral',
        '1st_linear_Integral',
        'Bernoulli_Integral',
        'almost_linear_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral')
    assert b == ('factorable',
         '1st_linear',
         'Bernoulli',
         '1st_power_series',
         'lie_group',
         'nth_linear_constant_coeff_undetermined_coefficients',
         'nth_linear_constant_coeff_variation_of_parameters',
         '1st_linear_Integral',
         'Bernoulli_Integral',
         'nth_linear_constant_coeff_variation_of_parameters_Integral')
    assert c == ('factorable',
         '1st_linear',
         'Bernoulli',
         '1st_power_series',
         'lie_group',
         'nth_linear_constant_coeff_undetermined_coefficients',
         'nth_linear_constant_coeff_variation_of_parameters',
         '1st_linear_Integral',
         'Bernoulli_Integral',
         'nth_linear_constant_coeff_variation_of_parameters_Integral')

    assert classify_ode(
        2*x*f(x)*f(x).diff(x) + (1 + x)*f(x)**2 - exp(x), f(x)
    ) == ('factorable', '1st_exact', 'Bernoulli', 'almost_linear', 'lie_group',
        '1st_exact_Integral', 'Bernoulli_Integral', 'almost_linear_Integral')
    assert 'Riccati_special_minus2' in \
        classify_ode(2*f(x).diff(x) + f(x)**2 - f(x)/x + 3*x**(-2), f(x))
    raises(ValueError, lambda: classify_ode(x + f(x, y).diff(x).diff(
        y), f(x, y)))
    # issue 5176
    k = Symbol('k')
    assert classify_ode(f(x).diff(x)/(k*f(x) + k*x*f(x)) + 2*f(x)/(k*f(x) +
        k*x*f(x)) + x*f(x).diff(x)/(k*f(x) + k*x*f(x)) + z, f(x)) == \
        ('factorable', 'separable', '1st_exact', '1st_linear', 'Bernoulli',
        '1st_power_series', 'lie_group', 'separable_Integral', '1st_exact_Integral',
        '1st_linear_Integral', 'Bernoulli_Integral')
    # preprocessing
    ans = ('factorable', 'nth_algebraic', 'separable', '1st_exact', '1st_linear', 'Bernoulli',
        '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_power_series', 'lie_group',
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters',
        'nth_algebraic_Integral',
        'separable_Integral', '1st_exact_Integral',
        '1st_linear_Integral',
        'Bernoulli_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral')
    #     w/o f(x) given
    assert classify_ode(diff(f(x) + x, x) + diff(f(x), x)) == ans
    #     w/ f(x) and prep=True
    assert classify_ode(diff(f(x) + x, x) + diff(f(x), x), f(x),
                        prep=True) == ans

    assert classify_ode(Eq(2*x**3*f(x).diff(x), 0), f(x)) == \
        ('factorable', 'nth_algebraic', 'separable', '1st_exact',
         '1st_linear', 'Bernoulli', '1st_power_series',
         'lie_group', 'nth_linear_euler_eq_homogeneous',
         'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral',
         '1st_linear_Integral', 'Bernoulli_Integral')


    assert classify_ode(Eq(2*f(x)**3*f(x).diff(x), 0), f(x)) == \
        ('factorable', 'nth_algebraic', 'separable', '1st_exact', '1st_linear',
         'Bernoulli', '1st_power_series', 'lie_group', 'nth_algebraic_Integral',
         'separable_Integral', '1st_exact_Integral', '1st_linear_Integral',
         'Bernoulli_Integral')
    # test issue 13864
    assert classify_ode(Eq(diff(f(x), x) - f(x)**x, 0), f(x)) == \
        ('1st_power_series', 'lie_group')
    assert isinstance(classify_ode(Eq(f(x), 5), f(x), dict=True), dict)

    #This is for new behavior of classify_ode when called internally with default, It should
    # return the first hint which matches therefore, 'ordered_hints' key will not be there.
    assert sorted(classify_ode(Eq(f(x).diff(x), 0), f(x), dict=True).keys()) == \
        ['default', 'nth_linear_constant_coeff_homogeneous', 'order']
    a = classify_ode(2*x*f(x)*f(x).diff(x) + (1 + x)*f(x)**2 - exp(x), f(x), dict=True, hint='Bernoulli')
    assert sorted(a.keys()) == ['Bernoulli', 'Bernoulli_Integral', 'default', 'order', 'ordered_hints']

    # test issue 22155
    a = classify_ode(f(x).diff(x) - exp(f(x) - x), f(x))
    assert a == ('separable',
        '1st_exact', '1st_power_series',
        'lie_group', 'separable_Integral',
        '1st_exact_Integral')


def test_classify_ode_ics():
    # Dummy
    eq = f(x).diff(x, x) - f(x)

    # Not f(0) or f'(0)
    ics = {x: 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))


    ############################
    # f(0) type (AppliedUndef) #
    ############################


    # Wrong function
    ics = {g(0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Contains x
    ics = {f(x): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Too many args
    ics = {f(0, 0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # point contains x
    ics = {f(0): f(x)}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Does not raise
    ics = {f(0): f(0)}
    classify_ode(eq, f(x), ics=ics)

    # Does not raise
    ics = {f(0): 1}
    classify_ode(eq, f(x), ics=ics)


    #####################
    # f'(0) type (Subs) #
    #####################

    # Wrong function
    ics = {g(x).diff(x).subs(x, 0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Contains x
    ics = {f(y).diff(y).subs(y, x): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Wrong variable
    ics = {f(y).diff(y).subs(y, 0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Too many args
    ics = {f(x, y).diff(x).subs(x, 0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Derivative wrt wrong vars
    ics = {Derivative(f(x), x, y).subs(x, 0): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # point contains x
    ics = {f(x).diff(x).subs(x, 0): f(x)}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Does not raise
    ics = {f(x).diff(x).subs(x, 0): f(x).diff(x).subs(x, 0)}
    classify_ode(eq, f(x), ics=ics)

    # Does not raise
    ics = {f(x).diff(x).subs(x, 0): 1}
    classify_ode(eq, f(x), ics=ics)

    ###########################
    # f'(y) type (Derivative) #
    ###########################

    # Wrong function
    ics = {g(x).diff(x).subs(x, y): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Contains x
    ics = {f(y).diff(y).subs(y, x): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Too many args
    ics = {f(x, y).diff(x).subs(x, y): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Derivative wrt wrong vars
    ics = {Derivative(f(x), x, z).subs(x, y): 1}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # point contains x
    ics = {f(x).diff(x).subs(x, y): f(x)}
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # Does not raise
    ics = {f(x).diff(x).subs(x, 0): f(0)}
    classify_ode(eq, f(x), ics=ics)

    # Does not raise
    ics = {f(x).diff(x).subs(x, y): 1}
    classify_ode(eq, f(x), ics=ics)

def test_classify_sysode():
    # Here x is assumed to be x(t) and y as y(t) for simplicity.
    # Similarly diff(x,t) and diff(y,y) is assumed to be x1 and y1 respectively.
    k, l, m, n = symbols('k, l, m, n', Integer=True)
    k1, k2, k3, l1, l2, l3, m1, m2, m3 = symbols('k1, k2, k3, l1, l2, l3, m1, m2, m3', Integer=True)
    P, Q, R, p, q, r = symbols('P, Q, R, p, q, r', cls=Function)
    P1, P2, P3, Q1, Q2, R1, R2 = symbols('P1, P2, P3, Q1, Q2, R1, R2', cls=Function)
    x, y, z = symbols('x, y, z', cls=Function)
    t = symbols('t')
    x1 = diff(x(t),t) ; y1 = diff(y(t),t) ;

    eq6 = (Eq(x1, exp(k*x(t))*P(x(t),y(t))), Eq(y1,r(y(t))*P(x(t),y(t))))
    sol6 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): 0, (1, x(t), 1): 0, (0, x(t), 1): 1, (1, y(t), 0): 0, \
    (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): 0, (1, y(t), 1): 1}, 'type_of_equation': 'type2', 'func': \
    [x(t), y(t)], 'is_linear': False, 'eq': [-P(x(t), y(t))*exp(k*x(t)) + Derivative(x(t), t), -P(x(t), \
    y(t))*r(y(t)) + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    assert classify_sysode(eq6) == sol6

    eq7 = (Eq(x1, x(t)**2+y(t)/x(t)), Eq(y1, x(t)/y(t)))
    sol7 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): 0, (1, x(t), 1): 0, (0, x(t), 1): 1, (1, y(t), 0): 0, \
    (1, x(t), 0): -1/y(t), (0, y(t), 1): 0, (0, y(t), 0): -1/x(t), (1, y(t), 1): 1}, 'type_of_equation': 'type3', \
    'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)**2 + Derivative(x(t), t) - y(t)/x(t), -x(t)/y(t) + \
    Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    assert classify_sysode(eq7) == sol7

    eq8 = (Eq(x1, P1(x(t))*Q1(y(t))*R(x(t),y(t),t)), Eq(y1, P1(x(t))*Q1(y(t))*R(x(t),y(t),t)))
    sol8 = {'func': [x(t), y(t)], 'is_linear': False, 'type_of_equation': 'type4', 'eq': \
    [-P1(x(t))*Q1(y(t))*R(x(t), y(t), t) + Derivative(x(t), t), -P1(x(t))*Q1(y(t))*R(x(t), y(t), t) + \
    Derivative(y(t), t)], 'func_coeff': {(0, y(t), 1): 0, (1, y(t), 1): 1, (1, x(t), 1): 0, (0, y(t), 0): 0, \
    (1, x(t), 0): 0, (0, x(t), 0): 0, (1, y(t), 0): 0, (0, x(t), 1): 1}, 'order': {y(t): 1, x(t): 1}, 'no_of_equation': 2}
    assert classify_sysode(eq8) == sol8

    eq11 = (Eq(x1,x(t)*y(t)**3), Eq(y1,y(t)**5))
    sol11 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): -y(t)**3, (1, x(t), 1): 0, (0, x(t), 1): 1, \
    (1, y(t), 0): 0, (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): 0, (1, y(t), 1): 1}, 'type_of_equation': \
    'type1', 'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)*y(t)**3 + Derivative(x(t), t), \
    -y(t)**5 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    assert classify_sysode(eq11) == sol11

    eq13 = (Eq(x1,x(t)*y(t)*sin(t)**2), Eq(y1,y(t)**2*sin(t)**2))
    sol13 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): -y(t)*sin(t)**2, (1, x(t), 1): 0, (0, x(t), 1): 1, \
    (1, y(t), 0): 0, (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): -x(t)*sin(t)**2, (1, y(t), 1): 1}, \
    'type_of_equation': 'type4', 'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)*y(t)*sin(t)**2 + \
    Derivative(x(t), t), -y(t)**2*sin(t)**2 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    assert classify_sysode(eq13) == sol13


def test_solve_ics():
    # Basic tests that things work from dsolve.
    assert dsolve(f(x).diff(x) - 1/f(x), f(x), ics={f(1): 2}) == \
        Eq(f(x), sqrt(2 * x + 2))
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(0): 1}) == Eq(f(x), exp(x))
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), exp(x))
    assert dsolve(f(x).diff(x, x) + f(x), f(x), ics={f(0): 1,
        f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), sin(x) + cos(x))
    assert dsolve([f(x).diff(x) - f(x) + g(x), g(x).diff(x) - g(x) - f(x)],
        [f(x), g(x)], ics={f(0): 1, g(0): 0}) == [Eq(f(x), exp(x)*cos(x)), Eq(g(x), exp(x)*sin(x))]

    # Test cases where dsolve returns two solutions.
    eq = (x**2*f(x)**2 - x).diff(x)
    assert dsolve(eq, f(x), ics={f(1): 0}) == [Eq(f(x),
        -sqrt(x - 1)/x), Eq(f(x), sqrt(x - 1)/x)]
    assert dsolve(eq, f(x), ics={f(x).diff(x).subs(x, 1): 0}) == [Eq(f(x),
        -sqrt(x - S.Half)/x), Eq(f(x), sqrt(x - S.Half)/x)]

    eq = cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x)
    assert dsolve(eq, f(x),
        ics={f(0):1}, hint='1st_exact', simplify=False) == Eq(x*cos(f(x)) + f(x)**3/3, Rational(1, 3))
    assert dsolve(eq, f(x),
        ics={f(0):1}, hint='1st_exact', simplify=True) == Eq(x*cos(f(x)) + f(x)**3/3, Rational(1, 3))

    assert solve_ics([Eq(f(x), C1*exp(x))], [f(x)], [C1], {f(0): 1}) == {C1: 1}
    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2],
        {f(0): 1, f(pi/2): 1}) == {C1: 1, C2: 1}

    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2],
        {f(0): 1, f(x).diff(x).subs(x, 0): 1}) == {C1: 1, C2: 1}

    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2], {f(0): 1}) == \
        {C2: 1}

    # Some more complicated tests Refer to PR #16098

    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(0):0, f(x).diff(x).subs(x, 1):0})) == \
        {Eq(f(x), 0), Eq(f(x), x ** 3 / 6 - x / 2)}
    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(0):0})) == \
        {Eq(f(x), 0), Eq(f(x), C2*x + x**3/6)}

    K, r, f0 = symbols('K r f0')
    sol = Eq(f(x), K*f0*exp(r*x)/((-K + f0)*(f0*exp(r*x)/(-K + f0) - 1)))
    assert (dsolve(Eq(f(x).diff(x), r * f(x) * (1 - f(x) / K)), f(x), ics={f(0): f0})) == sol


    #Order dependent issues Refer to PR #16098
    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(x).diff(x).subs(x,0):0, f(0):0})) == \
        {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}
    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(0):0, f(x).diff(x).subs(x,0):0})) == \
        {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}

    # XXX: Ought to be ValueError
    raises(ValueError, lambda: solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2], {f(0): 1, f(pi): 1}))

    # Degenerate case. f'(0) is identically 0.
    raises(ValueError, lambda: solve_ics([Eq(f(x), sqrt(C1 - x**2))], [f(x)], [C1], {f(x).diff(x).subs(x, 0): 0}))

    EI, q, L = symbols('EI q L')

    # eq = Eq(EI*diff(f(x), x, 4), q)
    sols = [Eq(f(x), C1 + C2*x + C3*x**2 + C4*x**3 + q*x**4/(24*EI))]
    funcs = [f(x)]
    constants = [C1, C2, C3, C4]
    # Test both cases, Derivative (the default from f(x).diff(x).subs(x, L)),
    # and Subs
    ics1 = {f(0): 0,
        f(x).diff(x).subs(x, 0): 0,
        f(L).diff(L, 2): 0,
        f(L).diff(L, 3): 0}
    ics2 = {f(0): 0,
        f(x).diff(x).subs(x, 0): 0,
        Subs(f(x).diff(x, 2), x, L): 0,
        Subs(f(x).diff(x, 3), x, L): 0}

    solved_constants1 = solve_ics(sols, funcs, constants, ics1)
    solved_constants2 = solve_ics(sols, funcs, constants, ics2)
    assert solved_constants1 == solved_constants2 == {
        C1: 0,
        C2: 0,
        C3: L**2*q/(4*EI),
        C4: -L*q/(6*EI)}

    # Allow the ics to refer to f
    ics = {f(0): f(0)}
    assert dsolve(f(x).diff(x) - f(x), f(x), ics=ics) == Eq(f(x), f(0)*exp(x))

    ics = {f(x).diff(x).subs(x, 0): f(x).diff(x).subs(x, 0), f(0): f(0)}
    assert dsolve(f(x).diff(x, x) + f(x), f(x), ics=ics) == \
        Eq(f(x), f(0)*cos(x) + f(x).diff(x).subs(x, 0)*sin(x))

def test_ode_order():
    f = Function('f')
    g = Function('g')
    x = Symbol('x')
    assert ode_order(3*x*exp(f(x)), f(x)) == 0
    assert ode_order(x*diff(f(x), x) + 3*x*f(x) - sin(x)/x, f(x)) == 1
    assert ode_order(x**2*f(x).diff(x, x) + x*diff(f(x), x) - f(x), f(x)) == 2
    assert ode_order(diff(x*exp(f(x)), x, x), f(x)) == 2
    assert ode_order(diff(x*diff(x*exp(f(x)), x, x), x), f(x)) == 3
    assert ode_order(diff(f(x), x, x), g(x)) == 0
    assert ode_order(diff(f(x), x, x)*diff(g(x), x), f(x)) == 2
    assert ode_order(diff(f(x), x, x)*diff(g(x), x), g(x)) == 1
    assert ode_order(diff(x*diff(x*exp(f(x)), x, x), x), g(x)) == 0
    # issue 5835: ode_order has to also work for unevaluated derivatives
    # (ie, without using doit()).
    assert ode_order(Derivative(x*f(x), x), f(x)) == 1
    assert ode_order(x*sin(Derivative(x*f(x)**2, x, x)), f(x)) == 2
    assert ode_order(Derivative(x*Derivative(x*exp(f(x)), x, x), x), g(x)) == 0
    assert ode_order(Derivative(f(x), x, x), g(x)) == 0
    assert ode_order(Derivative(x*exp(f(x)), x, x), f(x)) == 2
    assert ode_order(Derivative(f(x), x, x)*Derivative(g(x), x), g(x)) == 1
    assert ode_order(Derivative(x*Derivative(f(x), x, x), x), f(x)) == 3
    assert ode_order(
        x*sin(Derivative(x*Derivative(f(x), x)**2, x, x)), f(x)) == 3


def test_homogeneous_order():
    assert homogeneous_order(exp(y/x) + tan(y/x), x, y) == 0
    assert homogeneous_order(x**2 + sin(x)*cos(y), x, y) is None
    assert homogeneous_order(x - y - x*sin(y/x), x, y) == 1
    assert homogeneous_order((x*y + sqrt(x**4 + y**4) + x**2*(log(x) - log(y)))/
        (pi*x**Rational(2, 3)*sqrt(y)**3), x, y) == Rational(-1, 6)
    assert homogeneous_order(y/x*cos(y/x) - x/y*sin(y/x) + cos(y/x), x, y) == 0
    assert homogeneous_order(f(x), x, f(x)) == 1
    assert homogeneous_order(f(x)**2, x, f(x)) == 2
    assert homogeneous_order(x*y*z, x, y) == 2
    assert homogeneous_order(x*y*z, x, y, z) == 3
    assert homogeneous_order(x**2*f(x)/sqrt(x**2 + f(x)**2), f(x)) is None
    assert homogeneous_order(f(x, y)**2, x, f(x, y), y) == 2
    assert homogeneous_order(f(x, y)**2, x, f(x), y) is None
    assert homogeneous_order(f(x, y)**2, x, f(x, y)) is None
    assert homogeneous_order(f(y, x)**2, x, y, f(x, y)) is None
    assert homogeneous_order(f(y), f(x), x) is None
    assert homogeneous_order(-f(x)/x + 1/sin(f(x)/ x), f(x), x) == 0
    assert homogeneous_order(log(1/y) + log(x**2), x, y) is None
    assert homogeneous_order(log(1/y) + log(x), x, y) == 0
    assert homogeneous_order(log(x/y), x, y) == 0
    assert homogeneous_order(2*log(1/y) + 2*log(x), x, y) == 0
    a = Symbol('a')
    assert homogeneous_order(a*log(1/y) + a*log(x), x, y) == 0
    assert homogeneous_order(f(x).diff(x), x, y) is None
    assert homogeneous_order(-f(x).diff(x) + x, x, y) is None
    assert homogeneous_order(O(x), x, y) is None
    assert homogeneous_order(x + O(x**2), x, y) is None
    assert homogeneous_order(x**pi, x) == pi
    assert homogeneous_order(x**x, x) is None
    raises(ValueError, lambda: homogeneous_order(x*y))


@XFAIL
def test_noncircularized_real_imaginary_parts():
    # If this passes, lines numbered 3878-3882 (at the time of this commit)
    # of sympy/solvers/ode.py for nth_linear_constant_coeff_homogeneous
    # should be removed.
    y = sqrt(1+x)
    i, r = im(y), re(y)
    assert not (i.has(atan2) and r.has(atan2))


def test_collect_respecting_exponentials():
    # If this test passes, lines 1306-1311 (at the time of this commit)
    # of sympy/solvers/ode.py should be removed.
    sol = 1 + exp(x/2)
    assert sol == collect( sol, exp(x/3))


def test_undetermined_coefficients_match():
    assert _undetermined_coefficients_match(g(x), x) == {'test': False}
    assert _undetermined_coefficients_match(sin(2*x + sqrt(5)), x) == \
        {'test': True, 'trialset':
            {cos(2*x + sqrt(5)), sin(2*x + sqrt(5))}}
    assert _undetermined_coefficients_match(sin(x)*cos(x), x) == \
        {'test': False}
    s = {cos(x), x*cos(x), x**2*cos(x), x**2*sin(x), x*sin(x), sin(x)}
    assert _undetermined_coefficients_match(sin(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(
        sin(x)*x**2 + sin(x)*x + sin(x), x) == {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(
        exp(2*x)*sin(x)*(x**2 + x + 1), x
    ) == {
        'test': True, 'trialset': {exp(2*x)*sin(x), x**2*exp(2*x)*sin(x),
        cos(x)*exp(2*x), x**2*cos(x)*exp(2*x), x*cos(x)*exp(2*x),
        x*exp(2*x)*sin(x)}}
    assert _undetermined_coefficients_match(1/sin(x), x) == {'test': False}
    assert _undetermined_coefficients_match(log(x), x) == {'test': False}
    assert _undetermined_coefficients_match(2**(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': {2**x, x*2**x, x**2*2**x}}
    assert _undetermined_coefficients_match(x**y, x) == {'test': False}
    assert _undetermined_coefficients_match(exp(x)*exp(2*x + 1), x) == \
        {'test': True, 'trialset': {exp(1 + 3*x)}}
    assert _undetermined_coefficients_match(sin(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': {x*cos(x), x*sin(x), x**2*cos(x),
        x**2*sin(x), cos(x), sin(x)}}
    assert _undetermined_coefficients_match(sin(x)*(x + sin(x)), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(sin(x)*(x + sin(2*x)), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(sin(x)*tan(x), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(
        x**2*sin(x)*exp(x) + x*sin(x) + x, x
    ) == {
        'test': True, 'trialset': {x**2*cos(x)*exp(x), x, cos(x), S.One,
        exp(x)*sin(x), sin(x), x*exp(x)*sin(x), x*cos(x), x*cos(x)*exp(x),
        x*sin(x), cos(x)*exp(x), x**2*exp(x)*sin(x)}}
    assert _undetermined_coefficients_match(4*x*sin(x - 2), x) == {
        'trialset': {x*cos(x - 2), x*sin(x - 2), cos(x - 2), sin(x - 2)},
        'test': True,
    }
    assert _undetermined_coefficients_match(2**x*x, x) == \
        {'test': True, 'trialset': {2**x, x*2**x}}
    assert _undetermined_coefficients_match(2**x*exp(2*x), x) == \
        {'test': True, 'trialset': {2**x*exp(2*x)}}
    assert _undetermined_coefficients_match(exp(-x)/x, x) == \
        {'test': False}
    # Below are from Ordinary Differential Equations,
    #                Tenenbaum and Pollard, pg. 231
    assert _undetermined_coefficients_match(S(4), x) == \
        {'test': True, 'trialset': {S.One}}
    assert _undetermined_coefficients_match(12*exp(x), x) == \
        {'test': True, 'trialset': {exp(x)}}
    assert _undetermined_coefficients_match(exp(I*x), x) == \
        {'test': True, 'trialset': {exp(I*x)}}
    assert _undetermined_coefficients_match(sin(x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x)}}
    assert _undetermined_coefficients_match(cos(x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x)}}
    assert _undetermined_coefficients_match(8 + 6*exp(x) + 2*sin(x), x) == \
        {'test': True, 'trialset': {S.One, cos(x), sin(x), exp(x)}}
    assert _undetermined_coefficients_match(x**2, x) == \
        {'test': True, 'trialset': {S.One, x, x**2}}
    assert _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(x), exp(x), exp(-x)}}
    assert _undetermined_coefficients_match(2*exp(2*x)*sin(x), x) == \
        {'test': True, 'trialset': {exp(2*x)*sin(x), cos(x)*exp(2*x)}}
    assert _undetermined_coefficients_match(x - sin(x), x) == \
        {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    assert _undetermined_coefficients_match(x**2 + 2*x, x) == \
        {'test': True, 'trialset': {S.One, x, x**2}}
    assert _undetermined_coefficients_match(4*x*sin(x), x) == \
        {'test': True, 'trialset': {x*cos(x), x*sin(x), cos(x), sin(x)}}
    assert _undetermined_coefficients_match(x*sin(2*x), x) == \
        {'test': True, 'trialset':
            {x*cos(2*x), x*sin(2*x), cos(2*x), sin(2*x)}}
    assert _undetermined_coefficients_match(x**2*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), x**2*exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(2*exp(-x) - x**2*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), x**2*exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(exp(-2*x) + x**2, x) == \
        {'test': True, 'trialset': {S.One, x, x**2, exp(-2*x)}}
    assert _undetermined_coefficients_match(x*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), exp(-x)}}
    assert _undetermined_coefficients_match(x + exp(2*x), x) == \
        {'test': True, 'trialset': {S.One, x, exp(2*x)}}
    assert _undetermined_coefficients_match(sin(x) + exp(-x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x), exp(-x)}}
    assert _undetermined_coefficients_match(exp(x), x) == \
        {'test': True, 'trialset': {exp(x)}}
    # converted from sin(x)**2
    assert _undetermined_coefficients_match(S.Half - cos(2*x)/2, x) == \
        {'test': True, 'trialset': {S.One, cos(2*x), sin(2*x)}}
    # converted from exp(2*x)*sin(x)**2
    assert _undetermined_coefficients_match(
        exp(2*x)*(S.Half + cos(2*x)/2), x
    ) == {
        'test': True, 'trialset': {exp(2*x)*sin(2*x), cos(2*x)*exp(2*x),
        exp(2*x)}}
    assert _undetermined_coefficients_match(2*x + sin(x) + cos(x), x) == \
        {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    # converted from sin(2*x)*sin(x)
    assert _undetermined_coefficients_match(cos(x)/2 - cos(3*x)/2, x) == \
        {'test': True, 'trialset': {cos(x), cos(3*x), sin(x), sin(3*x)}}
    assert _undetermined_coefficients_match(cos(x**2), x) == {'test': False}
    assert _undetermined_coefficients_match(2**(x**2), x) == {'test': False}


def test_issue_4785_22462():
    from sympy.abc import A
    eq = x + A*(x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact', '1st_linear',
        'Bernoulli', 'almost_linear', '1st_power_series', 'lie_group',
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        '1st_exact_Integral', '1st_linear_Integral', 'Bernoulli_Integral',
        'almost_linear_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral')
    # issue 4864
    eq = (x**2 + f(x)**2)*f(x).diff(x) - 2*x*f(x)
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact',
        '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_power_series',
        'lie_group', '1st_exact_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral')


def test_issue_4825():
    raises(ValueError, lambda: dsolve(f(x, y).diff(x) - y*f(x, y), f(x)))
    assert classify_ode(f(x, y).diff(x) - y*f(x, y), f(x), dict=True) == \
        {'order': 0, 'default': None, 'ordered_hints': ()}
    # See also issue 3793, test Z13.
    raises(ValueError, lambda: dsolve(f(x).diff(x), f(y)))
    assert classify_ode(f(x).diff(x), f(y), dict=True) == \
        {'order': 0, 'default': None, 'ordered_hints': ()}


def test_constant_renumber_order_issue_5308():
    from sympy.utilities.iterables import variations

    assert constant_renumber(C1*x + C2*y) == \
        constant_renumber(C1*y + C2*x) == \
        C1*x + C2*y
    e = C1*(C2 + x)*(C3 + y)
    for a, b, c in variations([C1, C2, C3], 3):
        assert constant_renumber(a*(b + x)*(c + y)) == e


def test_constant_renumber():
    e1, e2, x, y = symbols("e1:3 x y")
    exprs = [e2*x, e1*x + e2*y]

    assert constant_renumber(exprs[0]) == e2*x
    assert constant_renumber(exprs[0], variables=[x]) == C1*x
    assert constant_renumber(exprs[0], variables=[x], newconstants=[C2]) == C2*x
    assert constant_renumber(exprs, variables=[x, y]) == [C1*x, C1*y + C2*x]
    assert constant_renumber(exprs, variables=[x, y], newconstants=symbols("C3:5")) == [C3*x, C3*y + C4*x]


def test_issue_5770():
    k = Symbol("k", real=True)
    t = Symbol('t')
    w = Function('w')
    sol = dsolve(w(t).diff(t, 6) - k**6*w(t), w(t))
    assert len([s for s in sol.free_symbols if s.name.startswith('C')]) == 6
    assert constantsimp((C1*cos(x) + C2*cos(x))*exp(x), {C1, C2}) == \
        C1*cos(x)*exp(x)
    assert constantsimp(C1*cos(x) + C2*cos(x) + C3*sin(x), {C1, C2, C3}) == \
        C1*cos(x) + C3*sin(x)
    assert constantsimp(exp(C1 + x), {C1}) == C1*exp(x)
    assert constantsimp(x + C1 + y, {C1, y}) == C1 + x
    assert constantsimp(x + C1 + Integral(x, (x, 1, 2)), {C1}) == C1 + x


def test_issue_5112_5430():
    assert homogeneous_order(-log(x) + acosh(x), x) is None
    assert homogeneous_order(y - log(x), x, y) is None


def test_issue_5095():
    f = Function('f')
    raises(ValueError, lambda: dsolve(f(x).diff(x)**2, f(x), 'fdsjf'))


def test_homogeneous_function():
    f = Function('f')
    eq1 = tan(x + f(x))
    eq2 = sin((3*x)/(4*f(x)))
    eq3 = cos(x*f(x)*Rational(3, 4))
    eq4 = log((3*x + 4*f(x))/(5*f(x) + 7*x))
    eq5 = exp((2*x**2)/(3*f(x)**2))
    eq6 = log((3*x + 4*f(x))/(5*f(x) + 7*x) + exp((2*x**2)/(3*f(x)**2)))
    eq7 = sin((3*x)/(5*f(x) + x**2))
    assert homogeneous_order(eq1, x, f(x)) == None
    assert homogeneous_order(eq2, x, f(x)) == 0
    assert homogeneous_order(eq3, x, f(x)) == None
    assert homogeneous_order(eq4, x, f(x)) == 0
    assert homogeneous_order(eq5, x, f(x)) == 0
    assert homogeneous_order(eq6, x, f(x)) == 0
    assert homogeneous_order(eq7, x, f(x)) == None


def test_linear_coeff_match():
    n, d = z*(2*x + 3*f(x) + 5), z*(7*x + 9*f(x) + 11)
    rat = n/d
    eq1 = sin(rat) + cos(rat.expand())
    obj1 = LinearCoefficients(eq1)
    eq2 = rat
    obj2 = LinearCoefficients(eq2)
    eq3 = log(sin(rat))
    obj3 = LinearCoefficients(eq3)
    ans = (4, Rational(-13, 3))
    assert obj1._linear_coeff_match(eq1, f(x)) == ans
    assert obj2._linear_coeff_match(eq2, f(x)) == ans
    assert obj3._linear_coeff_match(eq3, f(x)) == ans

    # no c
    eq4 = (3*x)/f(x)
    obj4 = LinearCoefficients(eq4)
    # not x and f(x)
    eq5 = (3*x + 2)/x
    obj5 = LinearCoefficients(eq5)
    # denom will be zero
    eq6 = (3*x + 2*f(x) + 1)/(3*x + 2*f(x) + 5)
    obj6 = LinearCoefficients(eq6)
    # not rational coefficient
    eq7 = (3*x + 2*f(x) + sqrt(2))/(3*x + 2*f(x) + 5)
    obj7 = LinearCoefficients(eq7)
    assert obj4._linear_coeff_match(eq4, f(x)) is None
    assert obj5._linear_coeff_match(eq5, f(x)) is None
    assert obj6._linear_coeff_match(eq6, f(x)) is None
    assert obj7._linear_coeff_match(eq7, f(x)) is None


def test_constantsimp_take_problem():
    c = exp(C1) + 2
    assert len(Poly(constantsimp(exp(C1) + c + c*x, [C1])).gens) == 2


def test_series():
    C1 = Symbol("C1")
    eq = f(x).diff(x) - f(x)
    sol = Eq(f(x), C1 + C1*x + C1*x**2/2 + C1*x**3/6 + C1*x**4/24 +
            C1*x**5/120 + O(x**6))
    assert dsolve(eq, hint='1st_power_series') == sol
    assert checkodesol(eq, sol, order=1)[0]

    eq = f(x).diff(x) - x*f(x)
    sol = Eq(f(x), C1*x**4/8 + C1*x**2/2 + C1 + O(x**6))
    assert dsolve(eq, hint='1st_power_series') == sol
    assert checkodesol(eq, sol, order=1)[0]

    eq = f(x).diff(x) - sin(x*f(x))
    sol = Eq(f(x), (x - 2)**2*(1+ sin(4))*cos(4) + (x - 2)*sin(4) + 2 + O(x**3))
    assert dsolve(eq, hint='1st_power_series', ics={f(2): 2}, n=3) == sol
    # FIXME: The solution here should be O((x-2)**3) so is incorrect
    #assert checkodesol(eq, sol, order=1)[0]


@slow
def test_2nd_power_series_ordinary():
    C1, C2 = symbols("C1 C2")

    eq = f(x).diff(x, 2) - x*f(x)
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2*(x**3/6 + 1) + C1*x*(x**3/12 + 1) + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    assert checkodesol(eq, sol) == (True, 0)

    sol = Eq(f(x), C2*((x + 2)**4/6 + (x + 2)**3/6 - (x + 2)**2 + 1)
        + C1*(x + (x + 2)**4/12 - (x + 2)**3/3 + S(2))
        + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_ordinary', x0=-2) == sol
    # FIXME: Solution should be O((x+2)**6)
    # assert checkodesol(eq, sol) == (True, 0)

    sol = Eq(f(x), C2*x + C1 + O(x**2))
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=2) == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = (1 + x**2)*(f(x).diff(x, 2)) + 2*x*(f(x).diff(x)) -2*f(x)
    assert classify_ode(eq) == ('factorable', '2nd_hypergeometric', '2nd_hypergeometric_Integral',
    '2nd_power_series_ordinary')

    sol = Eq(f(x), C2*(-x**4/3 + x**2 + 1) + C1*x + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = f(x).diff(x, 2) + x*(f(x).diff(x)) + f(x)
    assert classify_ode(eq) == ('factorable', '2nd_power_series_ordinary',)
    sol = Eq(f(x), C2*(x**4/8 - x**2/2 + 1) + C1*x*(-x**2/3 + 1) + O(x**6))
    assert dsolve(eq) == sol
    # FIXME: checkodesol fails for this solution...
    # assert checkodesol(eq, sol) == (True, 0)

    eq = f(x).diff(x, 2) + f(x).diff(x) - x*f(x)
    assert classify_ode(eq) == ('2nd_power_series_ordinary',)
    sol = Eq(f(x), C2*(-x**4/24 + x**3/6 + 1)
            + C1*x*(x**3/24 + x**2/6 - x/2 + 1) + O(x**6))
    assert dsolve(eq) == sol
    # FIXME: checkodesol fails for this solution...
    # assert checkodesol(eq, sol) == (True, 0)

    eq = f(x).diff(x, 2) + x*f(x)
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2*(x**6/180 - x**3/6 + 1) + C1*x*(-x**3/12 + 1) + O(x**7))
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=7) == sol
    assert checkodesol(eq, sol) == (True, 0)


def test_2nd_power_series_regular():
    C1, C2, a = symbols("C1 C2 a")
    eq = x**2*(f(x).diff(x, 2)) - 3*x*(f(x).diff(x)) + (4*x + 4)*f(x)
    sol = Eq(f(x), C1*x**2*(-16*x**3/9 + 4*x**2 - 4*x + 1) + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = 4*x**2*(f(x).diff(x, 2)) -8*x**2*(f(x).diff(x)) + (4*x**2 +
        1)*f(x)
    sol = Eq(f(x), C1*sqrt(x)*(x**4/24 + x**3/6 + x**2/2 + x + 1) + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = x**2*(f(x).diff(x, 2)) - x**2*(f(x).diff(x)) + (
        x**2 - 2)*f(x)
    sol = Eq(f(x), C1*(-x**6/720 - 3*x**5/80 - x**4/8 + x**2/2 + x/2 + 1)/x +
            C2*x**2*(-x**3/60 + x**2/20 + x/2 + 1) + O(x**6))
    assert dsolve(eq) == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - Rational(1, 4))*f(x)
    sol = Eq(f(x), C1*(x**4/24 - x**2/2 + 1)/sqrt(x) +
        C2*sqrt(x)*(x**4/120 - x**2/6 + 1) + O(x**6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = x*f(x).diff(x, 2) + f(x).diff(x) - a*x*f(x)
    sol = Eq(f(x), C1*(a**2*x**4/64 + a*x**2/4 + 1) + O(x**6))
    assert dsolve(eq, f(x), hint="2nd_power_series_regular") == sol
    assert checkodesol(eq, sol) == (True, 0)

    eq = f(x).diff(x, 2) + ((1 - x)/x)*f(x).diff(x) + (a/x)*f(x)
    sol = Eq(f(x), C1*(-a*x**5*(a - 4)*(a - 3)*(a - 2)*(a - 1)/14400 + \
        a*x**4*(a - 3)*(a - 2)*(a - 1)/576 - a*x**3*(a - 2)*(a - 1)/36 + \
        a*x**2*(a - 1)/4 - a*x + 1) + O(x**6))
    assert dsolve(eq, f(x), hint="2nd_power_series_regular") == sol
    assert checkodesol(eq, sol) == (True, 0)


def test_issue_15056():
    t = Symbol('t')
    C3 = Symbol('C3')
    assert get_numbered_constants(Symbol('C1') * Function('C2')(t)) == C3


def test_issue_15913():
    eq = -C1/x - 2*x*f(x) - f(x) + Derivative(f(x), x)
    sol = C2*exp(x**2 + x) + exp(x**2 + x)*Integral(C1*exp(-x**2 - x)/x, x)
    assert checkodesol(eq, sol) == (True, 0)
    sol = C1 + C2*exp(-x*y)
    eq = Derivative(y*f(x), x) + f(x).diff(x, 2)
    assert checkodesol(eq, sol, f(x)) == (True, 0)


def test_issue_16146():
    raises(ValueError, lambda: dsolve([f(x).diff(x), g(x).diff(x)], [f(x), g(x), h(x)]))
    raises(ValueError, lambda: dsolve([f(x).diff(x), g(x).diff(x)], [f(x)]))


def test_dsolve_remove_redundant_solutions():

    eq = (f(x)-2)*f(x).diff(x)
    sol = Eq(f(x), C1)
    assert dsolve(eq) == sol

    eq = (f(x)-sin(x))*(f(x).diff(x, 2))
    sol = {Eq(f(x), C1 + C2*x), Eq(f(x), sin(x))}
    assert set(dsolve(eq)) == sol

    eq = (f(x)**2-2*f(x)+1)*f(x).diff(x, 3)
    sol = Eq(f(x), C1 + C2*x + C3*x**2)
    assert dsolve(eq) == sol


def test_issue_13060():
    A, B = symbols("A B", cls=Function)
    t = Symbol("t")
    eq = [Eq(Derivative(A(t), t), A(t)*B(t)), Eq(Derivative(B(t), t), A(t)*B(t))]
    sol = dsolve(eq)
    assert checkodesol(eq, sol) == (True, [0, 0])


def test_issue_22523():
    N, s = symbols('N s')
    rho = Function('rho')
    # intentionally use 4.0 to confirm issue with nfloat
    # works here
    eqn = 4.0*N*sqrt(N - 1)*rho(s) + (4*s**2*(N - 1) + (N - 2*s*(N - 1))**2
        )*Derivative(rho(s), (s, 2))
    match = classify_ode(eqn, dict=True, hint='all')
    assert match['2nd_power_series_ordinary']['terms'] == 5
    C1, C2 = symbols('C1,C2')
    sol = dsolve(eqn, hint='2nd_power_series_ordinary')
    # there is no r(2.0) in this result
    assert filldedent(sol) == filldedent(str('''
        Eq(rho(s), C2*(1 - 4.0*s**4*sqrt(N - 1.0)/N + 0.666666666666667*s**4/N
        - 2.66666666666667*s**3*sqrt(N - 1.0)/N - 2.0*s**2*sqrt(N - 1.0)/N +
        9.33333333333333*s**4*sqrt(N - 1.0)/N**2 - 0.666666666666667*s**4/N**2
        + 2.66666666666667*s**3*sqrt(N - 1.0)/N**2 -
        5.33333333333333*s**4*sqrt(N - 1.0)/N**3) + C1*s*(1.0 -
        1.33333333333333*s**3*sqrt(N - 1.0)/N - 0.666666666666667*s**2*sqrt(N
        - 1.0)/N + 1.33333333333333*s**3*sqrt(N - 1.0)/N**2) + O(s**6))'''))


def test_issue_22604():
    x1, x2 = symbols('x1, x2', cls = Function)
    t, k1, k2, m1, m2 = symbols('t k1 k2 m1 m2', real = True)
    k1, k2, m1, m2 = 1, 1, 1, 1
    eq1 = Eq(m1*diff(x1(t), t, 2) + k1*x1(t) - k2*(x2(t) - x1(t)), 0)
    eq2 = Eq(m2*diff(x2(t), t, 2) + k2*(x2(t) - x1(t)), 0)
    eqs = [eq1, eq2]
    [x1sol, x2sol] = dsolve(eqs, [x1(t), x2(t)], ics = {x1(0):0, x1(t).diff().subs(t,0):0, \
                                                        x2(0):1, x2(t).diff().subs(t,0):0})
    assert x1sol == Eq(x1(t), sqrt(3 - sqrt(5))*(sqrt(10) + 5*sqrt(2))*cos(sqrt(2)*t*sqrt(3 - sqrt(5))/2)/20 + \
                       (-5*sqrt(2) + sqrt(10))*sqrt(sqrt(5) + 3)*cos(sqrt(2)*t*sqrt(sqrt(5) + 3)/2)/20)
    assert x2sol == Eq(x2(t), (sqrt(5) + 5)*cos(sqrt(2)*t*sqrt(3 - sqrt(5))/2)/10 + (5 - sqrt(5))*cos(sqrt(2)*t*sqrt(sqrt(5) + 3)/2)/10)


def test_issue_22462():
    for de in [
            Eq(f(x).diff(x), -20*f(x)**2 - 500*f(x)/7200),
            Eq(f(x).diff(x), -2*f(x)**2 - 5*f(x)/7)]:
        assert 'Bernoulli' in classify_ode(de, f(x))


def test_issue_23425():
    x = symbols('x')
    y = Function('y')
    eq = Eq(-E**x*y(x).diff().diff() + y(x).diff(), 0)
    assert classify_ode(eq) == \
        ('Liouville', 'nth_order_reducible', \
        '2nd_power_series_ordinary', 'Liouville_Integral')


@SKIP("too slow for @slow")
def test_issue_25820():
    x = Symbol('x')
    y = Function('y')
    eq = y(x)**3*y(x).diff(x, 2) + 49
    assert dsolve(eq, y(x)) is not None  # doesn't raise
