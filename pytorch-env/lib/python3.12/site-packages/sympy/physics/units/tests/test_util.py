from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
    G, centimeter, coulomb, day, degree, gram, hbar, hour, inch, joule, kelvin,
    kilogram, kilometer, length, meter, mile, minute, newton, planck,
    planck_length, planck_mass, planck_temperature, planck_time, radians,
    second, speed_of_light, steradian, time, km)
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
from sympy.functions.elementary.miscellaneous import sqrt


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


L = length
T = time


def test_dim_simplify_add():
    # assert Add(L, L) == L
    assert L + L == L


def test_dim_simplify_mul():
    # assert Mul(L, T) == L*T
    assert L*T == L*T


def test_dim_simplify_pow():
    assert Pow(L, 2) == L**2


def test_dim_simplify_rec():
    # assert Mul(Add(L, L), T) == L*T
    assert (L + L) * T == L*T


def test_convert_to_quantities():
    assert convert_to(3, meter) == 3

    assert convert_to(mile, kilometer) == 25146*kilometer/15625
    assert convert_to(meter/second, speed_of_light) == speed_of_light/299792458
    assert convert_to(299792458*meter/second, speed_of_light) == speed_of_light
    assert convert_to(2*299792458*meter/second, speed_of_light) == 2*speed_of_light
    assert convert_to(speed_of_light, meter/second) == 299792458*meter/second
    assert convert_to(2*speed_of_light, meter/second) == 599584916*meter/second
    assert convert_to(day, second) == 86400*second
    assert convert_to(2*hour, minute) == 120*minute
    assert convert_to(mile, meter) == 201168*meter/125
    assert convert_to(mile/hour, kilometer/hour) == 25146*kilometer/(15625*hour)
    assert convert_to(3*newton, meter/second) == 3*newton
    assert convert_to(3*newton, kilogram*meter/second**2) == 3*meter*kilogram/second**2
    assert convert_to(kilometer + mile, meter) == 326168*meter/125
    assert convert_to(2*kilometer + 3*mile, meter) == 853504*meter/125
    assert convert_to(inch**2, meter**2) == 16129*meter**2/25000000
    assert convert_to(3*inch**2, meter) == 48387*meter**2/25000000
    assert convert_to(2*kilometer/hour + 3*mile/hour, meter/second) == 53344*meter/(28125*second)
    assert convert_to(2*kilometer/hour + 3*mile/hour, centimeter/second) == 213376*centimeter/(1125*second)
    assert convert_to(kilometer * (mile + kilometer), meter) == 2609344 * meter ** 2

    assert convert_to(steradian, coulomb) == steradian
    assert convert_to(radians, degree) == 180*degree/pi
    assert convert_to(radians, [meter, degree]) == 180*degree/pi
    assert convert_to(pi*radians, degree) == 180*degree
    assert convert_to(pi, degree) == 180*degree

    # https://github.com/sympy/sympy/issues/26263
    assert convert_to(sqrt(meter**2 + meter**2.0), meter) == sqrt(meter**2 + meter**2.0)
    assert convert_to((meter**2 + meter**2.0)**2, meter) == (meter**2 + meter**2.0)**2


def test_convert_to_tuples_of_quantities():
    from sympy.core.symbol import symbols

    alpha, beta = symbols('alpha beta')

    assert convert_to(speed_of_light, [meter, second]) == 299792458 * meter / second
    assert convert_to(speed_of_light, (meter, second)) == 299792458 * meter / second
    assert convert_to(speed_of_light, Tuple(meter, second)) == 299792458 * meter / second
    assert convert_to(joule, [meter, kilogram, second]) == kilogram*meter**2/second**2
    assert convert_to(joule, [centimeter, gram, second]) == 10000000*centimeter**2*gram/second**2
    assert convert_to(299792458*meter/second, [speed_of_light]) == speed_of_light
    assert convert_to(speed_of_light / 2, [meter, second, kilogram]) == meter/second*299792458 / 2
    # This doesn't make physically sense, but let's keep it as a conversion test:
    assert convert_to(2 * speed_of_light, [meter, second, kilogram]) == 2 * 299792458 * meter / second
    assert convert_to(G, [G, speed_of_light, planck]) == 1.0*G

    assert NS(convert_to(meter, [G, speed_of_light, hbar]), n=7) == '6.187142e+34*gravitational_constant**0.5000000*hbar**0.5000000/speed_of_light**1.500000'
    assert NS(convert_to(planck_mass, kilogram), n=7) == '2.176434e-8*kilogram'
    assert NS(convert_to(planck_length, meter), n=7) == '1.616255e-35*meter'
    assert NS(convert_to(planck_time, second), n=6) == '5.39125e-44*second'
    assert NS(convert_to(planck_temperature, kelvin), n=7) == '1.416784e+32*kelvin'
    assert NS(convert_to(convert_to(meter, [G, speed_of_light, planck]), meter), n=10) == '1.000000000*meter'

    # similar to https://github.com/sympy/sympy/issues/26263
    assert convert_to(sqrt(meter**2 + second**2.0), [meter, second]) == sqrt(meter**2 + second**2.0)
    assert convert_to((meter**2 + second**2.0)**2, [meter, second]) == (meter**2 + second**2.0)**2

    # similar to https://github.com/sympy/sympy/issues/21463
    assert convert_to(1/(beta*meter + meter), 1/meter) == 1/(beta*meter + meter)
    assert convert_to(1/(beta*meter + alpha*meter), 1/kilometer) == (1/(kilometer*beta/1000 + alpha*kilometer/1000))

def test_eval_simplify():
    from sympy.physics.units import cm, mm, km, m, K, kilo
    from sympy.core.symbol import symbols

    x, y = symbols('x y')

    assert (cm/mm).simplify() == 10
    assert (km/m).simplify() == 1000
    assert (km/cm).simplify() == 100000
    assert (10*x*K*km**2/m/cm).simplify() == 1000000000*x*kelvin
    assert (cm/km/m).simplify() == 1/(10000000*centimeter)

    assert (3*kilo*meter).simplify() == 3000*meter
    assert (4*kilo*meter/(2*kilometer)).simplify() == 2
    assert (4*kilometer**2/(kilo*meter)**2).simplify() == 4


def test_quantity_simplify():
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import kilo, foot
    from sympy.core.symbol import symbols

    x, y = symbols('x y')

    assert quantity_simplify(x*(8*kilo*newton*meter + y)) == x*(8000*meter*newton + y)
    assert quantity_simplify(foot*inch*(foot + inch)) == foot**2*(foot + foot/12)/12
    assert quantity_simplify(foot*inch*(foot*foot + inch*(foot + inch))) == foot**2*(foot**2 + foot/12*(foot + foot/12))/12
    assert quantity_simplify(2**(foot/inch*kilo/1000)*inch) == 4096*foot/12
    assert quantity_simplify(foot**2*inch + inch**2*foot) == 13*foot**3/144

def test_quantity_simplify_across_dimensions():
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import ampere, ohm, volt, joule, pascal, farad, second, watt, siemens, henry, tesla, weber, hour, newton

    assert quantity_simplify(ampere*ohm, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(6*ampere*ohm, across_dimensions=True, unit_system="SI") == 6*volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(volt/ohm, across_dimensions=True, unit_system="SI") == ampere
    assert quantity_simplify(joule/meter**3, across_dimensions=True, unit_system="SI") == pascal
    assert quantity_simplify(farad*ohm, across_dimensions=True, unit_system="SI") == second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt
    assert quantity_simplify(meter**3/second, across_dimensions=True, unit_system="SI") == meter**3/second
    assert quantity_simplify(joule/second, across_dimensions=True, unit_system="SI") == watt

    assert quantity_simplify(joule/coulomb, across_dimensions=True, unit_system="SI") == volt
    assert quantity_simplify(volt/ampere, across_dimensions=True, unit_system="SI") == ohm
    assert quantity_simplify(ampere/volt, across_dimensions=True, unit_system="SI") == siemens
    assert quantity_simplify(coulomb/volt, across_dimensions=True, unit_system="SI") == farad
    assert quantity_simplify(volt*second/ampere, across_dimensions=True, unit_system="SI") == henry
    assert quantity_simplify(volt*second/meter**2, across_dimensions=True, unit_system="SI") == tesla
    assert quantity_simplify(joule/ampere, across_dimensions=True, unit_system="SI") == weber

    assert quantity_simplify(5*kilometer/hour, across_dimensions=True, unit_system="SI") == 25*meter/(18*second)
    assert quantity_simplify(5*kilogram*meter/second**2, across_dimensions=True, unit_system="SI") == 5*newton

def test_check_dimensions():
    x = symbols('x')
    assert check_dimensions(inch + x) == inch + x
    assert check_dimensions(length + x) == length + x
    # after subs we get 2*length; check will clear the constant
    assert check_dimensions((length + x).subs(x, length)) == length
    assert check_dimensions(newton*meter + joule) == joule + meter*newton
    raises(ValueError, lambda: check_dimensions(inch + 1))
    raises(ValueError, lambda: check_dimensions(length + 1))
    raises(ValueError, lambda: check_dimensions(length + time))
    raises(ValueError, lambda: check_dimensions(meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + 3 * second))
    raises(ValueError, lambda: check_dimensions(1 / second + 1 / meter))
    raises(ValueError, lambda: check_dimensions(2 * meter*(mile + centimeter) + km))
