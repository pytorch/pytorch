from sympy.concrete.tests.test_sums_products import NS

from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import convert_to, coulomb_constant, elementary_charge, gravitational_constant, planck
from sympy.physics.units.definitions.unit_definitions import angstrom, statcoulomb, coulomb, second, gram, centimeter, erg, \
    newton, joule, dyne, speed_of_light, meter, farad, henry, statvolt, volt, ohm
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.cgs import cgs_gauss


def test_conversion_to_from_si():
    assert convert_to(statcoulomb, coulomb, cgs_gauss) == coulomb/2997924580
    assert convert_to(coulomb, statcoulomb, cgs_gauss) == 2997924580*statcoulomb
    assert convert_to(statcoulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == centimeter**(S(3)/2)*sqrt(gram)/second
    assert convert_to(coulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == 2997924580*centimeter**(S(3)/2)*sqrt(gram)/second

    # SI units have an additional base unit, no conversion in case of electromagnetism:
    assert convert_to(coulomb, statcoulomb, SI) == coulomb
    assert convert_to(statcoulomb, coulomb, SI) == statcoulomb

    # SI without electromagnetism:
    assert convert_to(erg, joule, SI) == joule/10**7
    assert convert_to(erg, joule, cgs_gauss) == joule/10**7
    assert convert_to(joule, erg, SI) == 10**7*erg
    assert convert_to(joule, erg, cgs_gauss) == 10**7*erg


    assert convert_to(dyne, newton, SI) == newton/10**5
    assert convert_to(dyne, newton, cgs_gauss) == newton/10**5
    assert convert_to(newton, dyne, SI) == 10**5*dyne
    assert convert_to(newton, dyne, cgs_gauss) == 10**5*dyne


def test_cgs_gauss_convert_constants():

    assert convert_to(speed_of_light, centimeter/second, cgs_gauss) == 29979245800*centimeter/second

    assert convert_to(coulomb_constant, 1, cgs_gauss) == 1
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, cgs_gauss) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, dyne*centimeter**2/statcoulomb**2, cgs_gauss) == centimeter**2*dyne/statcoulomb**2
    assert convert_to(coulomb_constant, 1, SI) == coulomb_constant
    assert NS(convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI)) == '8987551787.36818*meter**2*newton/coulomb**2'

    assert convert_to(elementary_charge, statcoulomb, cgs_gauss)
    assert convert_to(angstrom, centimeter, cgs_gauss) == 1*centimeter/10**8
    assert convert_to(gravitational_constant, dyne*centimeter**2/gram**2, cgs_gauss)
    assert NS(convert_to(planck, erg*second, cgs_gauss)) == '6.62607015e-27*erg*second'

    spc = 25000*second/(22468879468420441*centimeter)
    assert convert_to(ohm, second/centimeter, cgs_gauss) == spc
    assert convert_to(henry, second**2/centimeter, cgs_gauss) == spc*second
    assert convert_to(volt, statvolt, cgs_gauss) == 10**6*statvolt/299792458
    assert convert_to(farad, centimeter, cgs_gauss) == 299792458**2*centimeter/10**5
