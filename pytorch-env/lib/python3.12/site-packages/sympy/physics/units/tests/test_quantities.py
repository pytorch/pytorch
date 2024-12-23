import warnings

from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
                                 volume, kilometer, joule, molar_gas_constant,
                                 vacuum_permittivity, elementary_charge, volt,
                                 ohm)
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
    day, foot, grams, hour, inch, kg, km, m, meter, millimeter,
    minute, quart, s, second, speed_of_light, bit,
    byte, kibibyte, mebibyte, gibibyte, tebibyte, pebibyte, exbibyte,
    kilogram, gravitational_constant, electron_rest_mass)

from sympy.physics.units.definitions.dimension_definitions import (
    Dimension, charge, length, time, temperature, pressure,
    energy, mass
)
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises

k = PREFIXES["k"]


def test_str_repr():
    assert str(kg) == "kilogram"


def test_eq():
    # simple test
    assert 10*m == 10*m
    assert 10*m != 10*s


def test_convert_to():
    q = Quantity("q1")
    q.set_global_relative_scale_factor(S(5000), meter)

    assert q.convert_to(m) == 5000*m

    assert speed_of_light.convert_to(m / s) == 299792458 * m / s
    assert day.convert_to(s) == 86400*s

    # Wrong dimension to convert:
    assert q.convert_to(s) == q
    assert speed_of_light.convert_to(m) == speed_of_light

    expr = joule*second
    conv = convert_to(expr, joule)
    assert conv == joule*second


def test_Quantity_definition():
    q = Quantity("s10", abbrev="sabbr")
    q.set_global_relative_scale_factor(10, second)
    u = Quantity("u", abbrev="dam")
    u.set_global_relative_scale_factor(10, meter)
    km = Quantity("km")
    km.set_global_relative_scale_factor(kilo, meter)
    v = Quantity("u")
    v.set_global_relative_scale_factor(5*kilo, meter)

    assert q.scale_factor == 10
    assert q.dimension == time
    assert q.abbrev == Symbol("sabbr")

    assert u.dimension == length
    assert u.scale_factor == 10
    assert u.abbrev == Symbol("dam")

    assert km.scale_factor == 1000
    assert km.func(*km.args) == km
    assert km.func(*km.args).args == km.args

    assert v.dimension == length
    assert v.scale_factor == 5000


def test_abbrev():
    u = Quantity("u")
    u.set_global_relative_scale_factor(S.One, meter)

    assert u.name == Symbol("u")
    assert u.abbrev == Symbol("u")

    u = Quantity("u", abbrev="om")
    u.set_global_relative_scale_factor(S(2), meter)

    assert u.name == Symbol("u")
    assert u.abbrev == Symbol("om")
    assert u.scale_factor == 2
    assert isinstance(u.scale_factor, Number)

    u = Quantity("u", abbrev="ikm")
    u.set_global_relative_scale_factor(3*kilo, meter)

    assert u.abbrev == Symbol("ikm")
    assert u.scale_factor == 3000


def test_print():
    u = Quantity("unitname", abbrev="dam")
    assert repr(u) == "unitname"
    assert str(u) == "unitname"


def test_Quantity_eq():
    u = Quantity("u", abbrev="dam")
    v = Quantity("v1")
    assert u != v
    v = Quantity("v2", abbrev="ds")
    assert u != v
    v = Quantity("v3", abbrev="dm")
    assert u != v


def test_add_sub():
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    assert isinstance(u + v, Add)
    assert (u + v.convert_to(u)) == (1 + S.Half)*u
    assert isinstance(u - v, Add)
    assert (u - v.convert_to(u)) == S.Half*u


def test_quantity_abs():
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w3 = Quantity('v_w3')

    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)
    v_w3.set_global_relative_scale_factor(1, meter/second)

    expr = v_w3 - Abs(v_w1 - v_w2)

    assert SI.get_dimensional_expr(v_w1) == (length/time).name

    Dq = Dimension(SI.get_dimensional_expr(expr))

    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: 1,
        time: -1,
    }
    assert meter == sqrt(meter**2)


def test_check_unit_consistency():
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    def check_unit_consistency(expr):
        SI._collect_factor_and_dimension(expr)

    raises(ValueError, lambda: check_unit_consistency(u + w))
    raises(ValueError, lambda: check_unit_consistency(u - w))
    raises(ValueError, lambda: check_unit_consistency(u + 1))
    raises(ValueError, lambda: check_unit_consistency(u - 1))
    raises(ValueError, lambda: check_unit_consistency(1 - exp(u / w)))


def test_mul_div():
    u = Quantity("u")
    v = Quantity("v")
    t = Quantity("t")
    ut = Quantity("ut")
    v2 = Quantity("v")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    t.set_global_relative_scale_factor(S(2), second)
    ut.set_global_relative_scale_factor(S(20), meter*second)
    v2.set_global_relative_scale_factor(S(5), meter/second)

    assert 1 / u == u**(-1)
    assert u / 1 == u

    v1 = u / t
    v2 = v

    # Pow only supports structural equality:
    assert v1 != v2
    assert v1 == v2.convert_to(v1)

    # TODO: decide whether to allow such expression in the future
    # (requires somehow manipulating the core).
    # assert u / Quantity('l2', dimension=length, scale_factor=2) == 5

    assert u * 1 == u

    ut1 = u * t
    ut2 = ut

    # Mul only supports structural equality:
    assert ut1 != ut2
    assert ut1 == ut2.convert_to(ut1)

    # Mul only supports structural equality:
    lp1 = Quantity("lp1")
    lp1.set_global_relative_scale_factor(S(2), 1/meter)
    assert u * lp1 != 20

    assert u**0 == 1
    assert u**1 == u

    # TODO: Pow only support structural equality:
    u2 = Quantity("u2")
    u3 = Quantity("u3")
    u2.set_global_relative_scale_factor(S(100), meter**2)
    u3.set_global_relative_scale_factor(Rational(1, 10), 1/meter)

    assert u ** 2 != u2
    assert u ** -1 != u3

    assert u ** 2 == u2.convert_to(u)
    assert u ** -1 == u3.convert_to(u)


def test_units():
    assert convert_to((5*m/s * day) / km, 1) == 432
    assert convert_to(foot / meter, meter) == Rational(3048, 10000)
    # amu is a pure mass so mass/mass gives a number, not an amount (mol)
    # TODO: need better simplification routine:
    assert str(convert_to(grams/amu, grams).n(2)) == '6.0e+23'

    # Light from the sun needs about 8.3 minutes to reach earth
    t = (1*au / speed_of_light) / minute
    # TODO: need a better way to simplify expressions containing units:
    t = convert_to(convert_to(t, meter / minute), meter)
    assert t.simplify() == Rational(49865956897, 5995849160)

    # TODO: fix this, it should give `m` without `Abs`
    assert sqrt(m**2) == m
    assert (sqrt(m))**2 == m

    t = Symbol('t')
    assert integrate(t*m/s, (t, 1*s, 5*s)) == 12*m*s
    assert (t * m/s).integrate((t, 1*s, 5*s)) == 12*m*s


def test_issue_quart():
    assert convert_to(4 * quart / inch ** 3, meter) == 231
    assert convert_to(4 * quart / inch ** 3, millimeter) == 231

def test_electron_rest_mass():
    assert convert_to(electron_rest_mass, kilogram) == 9.1093837015e-31*kilogram
    assert convert_to(electron_rest_mass, grams) == 9.1093837015e-28*grams

def test_issue_5565():
    assert (m < s).is_Relational


def test_find_unit():
    assert find_unit('coulomb') == ['coulomb', 'coulombs', 'coulomb_constant']
    assert find_unit(coulomb) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(charge) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(inch) == [
        'm', 'au', 'cm', 'dm', 'ft', 'km', 'ly', 'mi', 'mm', 'nm', 'pm', 'um', 'yd',
        'nmi', 'feet', 'foot', 'inch', 'mile', 'yard', 'meter', 'miles', 'yards',
        'inches', 'meters', 'micron', 'microns', 'angstrom', 'angstroms', 'decimeter',
        'kilometer', 'lightyear', 'nanometer', 'picometer', 'centimeter', 'decimeters',
        'kilometers', 'lightyears', 'micrometer', 'millimeter', 'nanometers', 'picometers',
        'centimeters', 'micrometers', 'millimeters', 'nautical_mile', 'planck_length',
        'nautical_miles', 'astronomical_unit', 'astronomical_units']
    assert find_unit(inch**-1) == ['D', 'dioptre', 'optical_power']
    assert find_unit(length**-1) == ['D', 'dioptre', 'optical_power']
    assert find_unit(inch ** 2) == ['ha', 'hectare', 'planck_area']
    assert find_unit(inch ** 3) == [
        'L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter', 'quart', 'liters', 'quarts',
        'deciliter', 'centiliter', 'deciliters', 'milliliter',
        'centiliters', 'milliliters', 'planck_volume']
    assert find_unit('voltage') == ['V', 'v', 'volt', 'volts', 'planck_voltage']
    assert find_unit(grams) == ['g', 't', 'Da', 'kg', 'me', 'mg', 'ug', 'amu', 'mmu', 'amus',
                                'gram', 'mmus', 'grams', 'pound', 'tonne', 'dalton', 'pounds',
                                'kilogram', 'kilograms', 'microgram', 'milligram', 'metric_ton',
                                'micrograms', 'milligrams', 'planck_mass', 'milli_mass_unit', 'atomic_mass_unit',
                                'electron_rest_mass', 'atomic_mass_constant']


def test_Quantity_derivative():
    x = symbols("x")
    assert diff(x*meter, x) == meter
    assert diff(x**3*meter**2, x) == 3*x**2*meter**2
    assert diff(meter, meter) == 1
    assert diff(meter**2, meter) == 2*meter


def test_quantity_postprocessing():
    q1 = Quantity('q1')
    q2 = Quantity('q2')

    SI.set_quantity_dimension(q1, length*pressure**2*temperature/time)
    SI.set_quantity_dimension(q2, energy*pressure*temperature/(length**2*time))

    assert q1 + q2
    q = q1 + q2
    Dq = Dimension(SI.get_dimensional_expr(q))
    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: -1,
        mass: 2,
        temperature: 1,
        time: -5,
    }


def test_factor_and_dimension():
    assert (3000, Dimension(1)) == SI._collect_factor_and_dimension(3000)
    assert (1001, length) == SI._collect_factor_and_dimension(meter + km)
    assert (2, length/time) == SI._collect_factor_and_dimension(
        meter/second + 36*km/(10*hour))

    x, y = symbols('x y')
    assert (x + y/100, length) == SI._collect_factor_and_dimension(
        x*m + y*centimeter)

    cH = Quantity('cH')
    SI.set_quantity_dimension(cH, amount_of_substance/volume)

    pH = -log(cH)

    assert (1, volume/amount_of_substance) == SI._collect_factor_and_dimension(
        exp(pH))

    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')

    v_w1.set_global_relative_scale_factor(Rational(3, 2), meter/second)
    v_w2.set_global_relative_scale_factor(2, meter/second)

    expr = Abs(v_w1/2 - v_w2)
    assert (Rational(5, 4), length/time) == \
        SI._collect_factor_and_dimension(expr)

    expr = Rational(5, 2)*second/meter*v_w1 - 3000
    assert (-(2996 + Rational(1, 4)), Dimension(1)) == \
        SI._collect_factor_and_dimension(expr)

    expr = v_w1**(v_w2/v_w1)
    assert ((Rational(3, 2))**Rational(4, 3), (length/time)**Rational(4, 3)) == \
        SI._collect_factor_and_dimension(expr)


def test_dimensional_expr_of_derivative():
    l = Quantity('l')
    t = Quantity('t')
    t1 = Quantity('t1')
    l.set_global_relative_scale_factor(36, km)
    t.set_global_relative_scale_factor(1, hour)
    t1.set_global_relative_scale_factor(1, second)
    x = Symbol('x')
    y = Symbol('y')
    f = Function('f')
    dfdx = f(x, y).diff(x, y)
    dl_dt = dfdx.subs({f(x, y): l, x: t, y: t1})
    assert SI.get_dimensional_expr(dl_dt) ==\
        SI.get_dimensional_expr(l / t / t1) ==\
        Symbol("length")/Symbol("time")**2
    assert SI._collect_factor_and_dimension(dl_dt) ==\
        SI._collect_factor_and_dimension(l / t / t1) ==\
        (10, length/time**2)


def test_get_dimensional_expr_with_function():
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)

    assert SI.get_dimensional_expr(sin(v_w1)) == \
        sin(SI.get_dimensional_expr(v_w1))
    assert SI.get_dimensional_expr(sin(v_w1/v_w2)) == 1


def test_binary_information():
    assert convert_to(kibibyte, byte) == 1024*byte
    assert convert_to(mebibyte, byte) == 1024**2*byte
    assert convert_to(gibibyte, byte) == 1024**3*byte
    assert convert_to(tebibyte, byte) == 1024**4*byte
    assert convert_to(pebibyte, byte) == 1024**5*byte
    assert convert_to(exbibyte, byte) == 1024**6*byte

    assert kibibyte.convert_to(bit) == 8*1024*bit
    assert byte.convert_to(bit) == 8*bit

    a = 10*kibibyte*hour

    assert convert_to(a, byte) == 10240*byte*hour
    assert convert_to(a, minute) == 600*kibibyte*minute
    assert convert_to(a, [byte, minute]) == 614400*byte*minute


def test_conversion_with_2_nonstandard_dimensions():
    good_grade = Quantity("good_grade")
    kilo_good_grade = Quantity("kilo_good_grade")
    centi_good_grade = Quantity("centi_good_grade")

    kilo_good_grade.set_global_relative_scale_factor(1000, good_grade)
    centi_good_grade.set_global_relative_scale_factor(S.One/10**5, kilo_good_grade)

    charity_points = Quantity("charity_points")
    milli_charity_points = Quantity("milli_charity_points")
    missions = Quantity("missions")

    milli_charity_points.set_global_relative_scale_factor(S.One/1000, charity_points)
    missions.set_global_relative_scale_factor(251, charity_points)

    assert convert_to(
        kilo_good_grade*milli_charity_points*millimeter,
        [centi_good_grade, missions, centimeter]
    ) == S.One * 10**5 / (251*1000) / 10 * centi_good_grade*missions*centimeter


def test_eval_subs():
    energy, mass, force = symbols('energy mass force')
    expr1 = energy/mass
    units = {energy: kilogram*meter**2/second**2, mass: kilogram}
    assert expr1.subs(units) == meter**2/second**2
    expr2 = force/mass
    units = {force:gravitational_constant*kilogram**2/meter**2, mass:kilogram}
    assert expr2.subs(units) == gravitational_constant*kilogram/meter**2


def test_issue_14932():
    assert (log(inch) - log(2)).simplify() == log(inch/2)
    assert (log(inch) - log(foot)).simplify() == -log(12)
    p = symbols('p', positive=True)
    assert (log(inch) - log(p)).simplify() == log(inch/p)


def test_issue_14547():
    # the root issue is that an argument with dimensions should
    # not raise an error when the `arg - 1` calculation is
    # performed in the assumptions system
    from sympy.physics.units import foot, inch
    from sympy.core.relational import Eq
    assert log(foot).is_zero is None
    assert log(foot).is_positive is None
    assert log(foot).is_nonnegative is None
    assert log(foot).is_negative is None
    assert log(foot).is_algebraic is None
    assert log(foot).is_rational is None
    # doesn't raise error
    assert Eq(log(foot), log(inch)) is not None  # might be False or unevaluated

    x = Symbol('x')
    e = foot + x
    assert e.is_Add and set(e.args) == {foot, x}
    e = foot + 1
    assert e.is_Add and set(e.args) == {foot, 1}


def test_issue_22164():
    warnings.simplefilter("error")
    dm = Quantity("dm")
    SI.set_quantity_dimension(dm, length)
    SI.set_quantity_scale_factor(dm, 1)

    bad_exp = Quantity("bad_exp")
    SI.set_quantity_dimension(bad_exp, length)
    SI.set_quantity_scale_factor(bad_exp, 1)

    expr = dm ** bad_exp

    # deprecation warning is not expected here
    SI._collect_factor_and_dimension(expr)


def test_issue_22819():
    from sympy.physics.units import tonne, gram, Da
    from sympy.physics.units.systems.si import dimsys_SI
    assert tonne.convert_to(gram) == 1000000*gram
    assert dimsys_SI.get_dimensional_dependencies(area) == {length: 2}
    assert Da.scale_factor == 1.66053906660000e-24


def test_issue_20288():
    from sympy.core.numbers import E
    from sympy.physics.units import energy
    u = Quantity('u')
    v = Quantity('v')
    SI.set_quantity_dimension(u, energy)
    SI.set_quantity_dimension(v, energy)
    u.set_global_relative_scale_factor(1, joule)
    v.set_global_relative_scale_factor(1, joule)
    expr = 1 + exp(u**2/v**2)
    assert SI._collect_factor_and_dimension(expr) == (1 + E, Dimension(1))


def test_issue_24062():
    from sympy.core.numbers import E
    from sympy.physics.units import impedance, capacitance, time, ohm, farad, second

    R = Quantity('R')
    C = Quantity('C')
    T = Quantity('T')
    SI.set_quantity_dimension(R, impedance)
    SI.set_quantity_dimension(C, capacitance)
    SI.set_quantity_dimension(T, time)
    R.set_global_relative_scale_factor(1, ohm)
    C.set_global_relative_scale_factor(1, farad)
    T.set_global_relative_scale_factor(1, second)
    expr = T / (R * C)
    dim = SI._collect_factor_and_dimension(expr)[1]
    assert SI.get_dimension_system().is_dimensionless(dim)

    exp_expr = 1 + exp(expr)
    assert SI._collect_factor_and_dimension(exp_expr) == (1 + E, Dimension(1))

def test_issue_24211():
    from sympy.physics.units import time, velocity, acceleration, second, meter
    V1 = Quantity('V1')
    SI.set_quantity_dimension(V1, velocity)
    SI.set_quantity_scale_factor(V1, 1 * meter / second)
    A1 = Quantity('A1')
    SI.set_quantity_dimension(A1, acceleration)
    SI.set_quantity_scale_factor(A1, 1 * meter / second**2)
    T1 = Quantity('T1')
    SI.set_quantity_dimension(T1, time)
    SI.set_quantity_scale_factor(T1, 1 * second)

    expr = A1*T1 + V1
    # should not throw ValueError here
    SI._collect_factor_and_dimension(expr)


def test_prefixed_property():
    assert not meter.is_prefixed
    assert not joule.is_prefixed
    assert not day.is_prefixed
    assert not second.is_prefixed
    assert not volt.is_prefixed
    assert not ohm.is_prefixed
    assert centimeter.is_prefixed
    assert kilometer.is_prefixed
    assert kilogram.is_prefixed
    assert pebibyte.is_prefixed

def test_physics_constant():
    from sympy.physics.units import definitions

    for name in dir(definitions):
        quantity = getattr(definitions, name)
        if not isinstance(quantity, Quantity):
            continue
        if name.endswith('_constant'):
            assert isinstance(quantity, PhysicalConstant), f"{quantity} must be PhysicalConstant, but is {type(quantity)}"
            assert quantity.is_physical_constant, f"{name} is not marked as physics constant when it should be"

    for const in [gravitational_constant, molar_gas_constant, vacuum_permittivity, speed_of_light, elementary_charge]:
        assert isinstance(const, PhysicalConstant), f"{const} must be PhysicalConstant, but is {type(const)}"
        assert const.is_physical_constant, f"{const} is not marked as physics constant when it should be"

    assert not meter.is_physical_constant
    assert not joule.is_physical_constant
