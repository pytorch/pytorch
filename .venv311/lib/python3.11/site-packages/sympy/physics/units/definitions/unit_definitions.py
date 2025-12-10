from sympy.physics.units.definitions.dimension_definitions import current, temperature, amount_of_substance, \
    luminous_intensity, angle, charge, voltage, impedance, conductance, capacitance, inductance, magnetic_density, \
    magnetic_flux, information

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S as S_singleton
from sympy.physics.units.prefixes import kilo, mega, milli, micro, deci, centi, nano, pico, kibi, mebi, gibi, tebi, pebi, exbi
from sympy.physics.units.quantities import PhysicalConstant, Quantity

One = S_singleton.One

#### UNITS ####

# Dimensionless:
percent = percents = Quantity("percent", latex_repr=r"\%")
percent.set_global_relative_scale_factor(Rational(1, 100), One)

permille = Quantity("permille")
permille.set_global_relative_scale_factor(Rational(1, 1000), One)


# Angular units (dimensionless)
rad = radian = radians = Quantity("radian", abbrev="rad")
radian.set_global_dimension(angle)
deg = degree = degrees = Quantity("degree", abbrev="deg", latex_repr=r"^\circ")
degree.set_global_relative_scale_factor(pi/180, radian)
sr = steradian = steradians = Quantity("steradian", abbrev="sr")
mil = angular_mil = angular_mils = Quantity("angular_mil", abbrev="mil")

# Base units:
m = meter = meters = Quantity("meter", abbrev="m")

# gram; used to define its prefixed units
g = gram = grams = Quantity("gram", abbrev="g")

# NOTE: the `kilogram` has scale factor 1000. In SI, kg is a base unit, but
# nonetheless we are trying to be compatible with the `kilo` prefix. In a
# similar manner, people using CGS or gaussian units could argue that the
# `centimeter` rather than `meter` is the fundamental unit for length, but the
# scale factor of `centimeter` will be kept as 1/100 to be compatible with the
# `centi` prefix.  The current state of the code assumes SI unit dimensions, in
# the future this module will be modified in order to be unit system-neutral
# (that is, support all kinds of unit systems).
kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
kg.set_global_relative_scale_factor(kilo, gram)

s = second = seconds = Quantity("second", abbrev="s")
A = ampere = amperes = Quantity("ampere", abbrev='A')
ampere.set_global_dimension(current)
K = kelvin = kelvins = Quantity("kelvin", abbrev='K')
kelvin.set_global_dimension(temperature)
mol = mole = moles = Quantity("mole", abbrev="mol")
mole.set_global_dimension(amount_of_substance)
cd = candela = candelas = Quantity("candela", abbrev="cd")
candela.set_global_dimension(luminous_intensity)

# derived units
newton = newtons = N = Quantity("newton", abbrev="N")

kilonewton = kilonewtons = kN = Quantity("kilonewton", abbrev="kN")
kilonewton.set_global_relative_scale_factor(kilo, newton)

meganewton = meganewtons = MN = Quantity("meganewton", abbrev="MN")
meganewton.set_global_relative_scale_factor(mega, newton)

joule = joules = J = Quantity("joule", abbrev="J")
watt = watts = W = Quantity("watt", abbrev="W")
pascal = pascals = Pa = pa = Quantity("pascal", abbrev="Pa")
hertz = hz = Hz = Quantity("hertz", abbrev="Hz")

# CGS derived units:
dyne = Quantity("dyne")
dyne.set_global_relative_scale_factor(One/10**5, newton)
erg = Quantity("erg")
erg.set_global_relative_scale_factor(One/10**7, joule)

# MKSA extension to MKS: derived units
coulomb = coulombs = C = Quantity("coulomb", abbrev='C')
coulomb.set_global_dimension(charge)
volt = volts = v = V = Quantity("volt", abbrev='V')
volt.set_global_dimension(voltage)
ohm = ohms = Quantity("ohm", abbrev='ohm', latex_repr=r"\Omega")
ohm.set_global_dimension(impedance)
siemens = S = mho = mhos = Quantity("siemens", abbrev='S')
siemens.set_global_dimension(conductance)
farad = farads = F = Quantity("farad", abbrev='F')
farad.set_global_dimension(capacitance)
henry = henrys = H = Quantity("henry", abbrev='H')
henry.set_global_dimension(inductance)
tesla = teslas = T = Quantity("tesla", abbrev='T')
tesla.set_global_dimension(magnetic_density)
weber = webers = Wb = wb = Quantity("weber", abbrev='Wb')
weber.set_global_dimension(magnetic_flux)

# CGS units for electromagnetic quantities:
statampere = Quantity("statampere")
statcoulomb = statC = franklin = Quantity("statcoulomb", abbrev="statC")
statvolt = Quantity("statvolt")
gauss = Quantity("gauss")
maxwell = Quantity("maxwell")
debye = Quantity("debye")
oersted = Quantity("oersted")

# Other derived units:
optical_power = dioptre = diopter = D = Quantity("dioptre")
lux = lx = Quantity("lux", abbrev="lx")

# katal is the SI unit of catalytic activity
katal = kat = Quantity("katal", abbrev="kat")

# gray is the SI unit of absorbed dose
gray = Gy = Quantity("gray")

# becquerel is the SI unit of radioactivity
becquerel = Bq = Quantity("becquerel", abbrev="Bq")


# Common mass units

mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
mg.set_global_relative_scale_factor(milli, gram)

ug = microgram = micrograms = Quantity("microgram", abbrev="ug", latex_repr=r"\mu\text{g}")
ug.set_global_relative_scale_factor(micro, gram)

# Atomic mass constant
Da = dalton = amu = amus = atomic_mass_unit = atomic_mass_constant = PhysicalConstant("atomic_mass_constant")

t = metric_ton = tonne = Quantity("tonne", abbrev="t")
tonne.set_global_relative_scale_factor(mega, gram)

# Electron rest mass
me = electron_rest_mass = Quantity("electron_rest_mass", abbrev="me")


# Common length units

km = kilometer = kilometers = Quantity("kilometer", abbrev="km")
km.set_global_relative_scale_factor(kilo, meter)

dm = decimeter = decimeters = Quantity("decimeter", abbrev="dm")
dm.set_global_relative_scale_factor(deci, meter)

cm = centimeter = centimeters = Quantity("centimeter", abbrev="cm")
cm.set_global_relative_scale_factor(centi, meter)

mm = millimeter = millimeters = Quantity("millimeter", abbrev="mm")
mm.set_global_relative_scale_factor(milli, meter)

um = micrometer = micrometers = micron = microns = \
    Quantity("micrometer", abbrev="um", latex_repr=r'\mu\text{m}')
um.set_global_relative_scale_factor(micro, meter)

nm = nanometer = nanometers = Quantity("nanometer", abbrev="nm")
nm.set_global_relative_scale_factor(nano, meter)

pm = picometer = picometers = Quantity("picometer", abbrev="pm")
pm.set_global_relative_scale_factor(pico, meter)

ft = foot = feet = Quantity("foot", abbrev="ft")
ft.set_global_relative_scale_factor(Rational(3048, 10000), meter)

inch = inches = Quantity("inch")
inch.set_global_relative_scale_factor(Rational(1, 12), foot)

yd = yard = yards = Quantity("yard", abbrev="yd")
yd.set_global_relative_scale_factor(3, feet)

mi = mile = miles = Quantity("mile")
mi.set_global_relative_scale_factor(5280, feet)

nmi = nautical_mile = nautical_miles = Quantity("nautical_mile")
nmi.set_global_relative_scale_factor(6076, feet)

angstrom = angstroms = Quantity("angstrom", latex_repr=r'\r{A}')
angstrom.set_global_relative_scale_factor(Rational(1, 10**10), meter)


# Common volume and area units

ha = hectare = Quantity("hectare", abbrev="ha")

l = L = liter = liters = Quantity("liter", abbrev="l")

dl = dL = deciliter = deciliters = Quantity("deciliter", abbrev="dl")
dl.set_global_relative_scale_factor(Rational(1, 10), liter)

cl = cL = centiliter = centiliters = Quantity("centiliter", abbrev="cl")
cl.set_global_relative_scale_factor(Rational(1, 100), liter)

ml = mL = milliliter = milliliters = Quantity("milliliter", abbrev="ml")
ml.set_global_relative_scale_factor(Rational(1, 1000), liter)


# Common time units

ms = millisecond = milliseconds = Quantity("millisecond", abbrev="ms")
millisecond.set_global_relative_scale_factor(milli, second)

us = microsecond = microseconds = Quantity("microsecond", abbrev="us", latex_repr=r'\mu\text{s}')
microsecond.set_global_relative_scale_factor(micro, second)

ns = nanosecond = nanoseconds = Quantity("nanosecond", abbrev="ns")
nanosecond.set_global_relative_scale_factor(nano, second)

ps = picosecond = picoseconds = Quantity("picosecond", abbrev="ps")
picosecond.set_global_relative_scale_factor(pico, second)

minute = minutes = Quantity("minute")
minute.set_global_relative_scale_factor(60, second)

h = hour = hours = Quantity("hour")
hour.set_global_relative_scale_factor(60, minute)

day = days = Quantity("day")
day.set_global_relative_scale_factor(24, hour)

anomalistic_year = anomalistic_years = Quantity("anomalistic_year")
anomalistic_year.set_global_relative_scale_factor(365.259636, day)

sidereal_year = sidereal_years = Quantity("sidereal_year")
sidereal_year.set_global_relative_scale_factor(31558149.540, seconds)

tropical_year = tropical_years = Quantity("tropical_year")
tropical_year.set_global_relative_scale_factor(365.24219, day)

common_year = common_years = Quantity("common_year")
common_year.set_global_relative_scale_factor(365, day)

julian_year = julian_years = Quantity("julian_year")
julian_year.set_global_relative_scale_factor((365 + One/4), day)

draconic_year = draconic_years = Quantity("draconic_year")
draconic_year.set_global_relative_scale_factor(346.62, day)

gaussian_year = gaussian_years = Quantity("gaussian_year")
gaussian_year.set_global_relative_scale_factor(365.2568983, day)

full_moon_cycle = full_moon_cycles = Quantity("full_moon_cycle")
full_moon_cycle.set_global_relative_scale_factor(411.78443029, day)

year = years = tropical_year


#### CONSTANTS ####

# Newton constant
G = gravitational_constant = PhysicalConstant("gravitational_constant", abbrev="G")

# speed of light
c = speed_of_light = PhysicalConstant("speed_of_light", abbrev="c")

# elementary charge
elementary_charge = PhysicalConstant("elementary_charge", abbrev="e")

# Planck constant
planck = PhysicalConstant("planck", abbrev="h")

# Reduced Planck constant
hbar = PhysicalConstant("hbar", abbrev="hbar")

# Electronvolt
eV = electronvolt = electronvolts = PhysicalConstant("electronvolt", abbrev="eV")

# Avogadro number
avogadro_number = PhysicalConstant("avogadro_number")

# Avogadro constant
avogadro = avogadro_constant = PhysicalConstant("avogadro_constant")

# Boltzmann constant
boltzmann = boltzmann_constant = PhysicalConstant("boltzmann_constant")

# Stefan-Boltzmann constant
stefan = stefan_boltzmann_constant = PhysicalConstant("stefan_boltzmann_constant")

# Molar gas constant
R = molar_gas_constant = PhysicalConstant("molar_gas_constant", abbrev="R")

# Faraday constant
faraday_constant = PhysicalConstant("faraday_constant")

# Josephson constant
josephson_constant = PhysicalConstant("josephson_constant", abbrev="K_j")

# Von Klitzing constant
von_klitzing_constant = PhysicalConstant("von_klitzing_constant", abbrev="R_k")

# Acceleration due to gravity (on the Earth surface)
gee = gees = acceleration_due_to_gravity = PhysicalConstant("acceleration_due_to_gravity", abbrev="g")

# magnetic constant:
u0 = magnetic_constant = vacuum_permeability = PhysicalConstant("magnetic_constant")

# electric constat:
e0 = electric_constant = vacuum_permittivity = PhysicalConstant("vacuum_permittivity")

# vacuum impedance:
Z0 = vacuum_impedance = PhysicalConstant("vacuum_impedance", abbrev='Z_0', latex_repr=r'Z_{0}')

# Coulomb's constant:
coulomb_constant = coulombs_constant = electric_force_constant = \
    PhysicalConstant("coulomb_constant", abbrev="k_e")


atmosphere = atmospheres = atm = Quantity("atmosphere", abbrev="atm")

kPa = kilopascal = Quantity("kilopascal", abbrev="kPa")
kilopascal.set_global_relative_scale_factor(kilo, Pa)

bar = bars = Quantity("bar", abbrev="bar")

pound = pounds = Quantity("pound")  # exact

psi = Quantity("psi")

dHg0 = 13.5951  # approx value at 0 C
mmHg = torr = Quantity("mmHg")

atmosphere.set_global_relative_scale_factor(101325, pascal)
bar.set_global_relative_scale_factor(100, kPa)
pound.set_global_relative_scale_factor(Rational(45359237, 100000000), kg)

mmu = mmus = milli_mass_unit = Quantity("milli_mass_unit")

quart = quarts = Quantity("quart")


# Other convenient units and magnitudes

ly = lightyear = lightyears = Quantity("lightyear", abbrev="ly")

au = astronomical_unit = astronomical_units = Quantity("astronomical_unit", abbrev="AU")


# Fundamental Planck units:
planck_mass = Quantity("planck_mass", abbrev="m_P", latex_repr=r'm_\text{P}')

planck_time = Quantity("planck_time", abbrev="t_P", latex_repr=r't_\text{P}')

planck_temperature = Quantity("planck_temperature", abbrev="T_P",
                              latex_repr=r'T_\text{P}')

planck_length = Quantity("planck_length", abbrev="l_P", latex_repr=r'l_\text{P}')

planck_charge = Quantity("planck_charge", abbrev="q_P", latex_repr=r'q_\text{P}')


# Derived Planck units:
planck_area = Quantity("planck_area")

planck_volume = Quantity("planck_volume")

planck_momentum = Quantity("planck_momentum")

planck_energy = Quantity("planck_energy", abbrev="E_P", latex_repr=r'E_\text{P}')

planck_force = Quantity("planck_force", abbrev="F_P", latex_repr=r'F_\text{P}')

planck_power = Quantity("planck_power", abbrev="P_P", latex_repr=r'P_\text{P}')

planck_density = Quantity("planck_density", abbrev="rho_P", latex_repr=r'\rho_\text{P}')

planck_energy_density = Quantity("planck_energy_density", abbrev="rho^E_P")

planck_intensity = Quantity("planck_intensity", abbrev="I_P", latex_repr=r'I_\text{P}')

planck_angular_frequency = Quantity("planck_angular_frequency", abbrev="omega_P",
                                    latex_repr=r'\omega_\text{P}')

planck_pressure = Quantity("planck_pressure", abbrev="p_P", latex_repr=r'p_\text{P}')

planck_current = Quantity("planck_current", abbrev="I_P", latex_repr=r'I_\text{P}')

planck_voltage = Quantity("planck_voltage", abbrev="V_P", latex_repr=r'V_\text{P}')

planck_impedance = Quantity("planck_impedance", abbrev="Z_P", latex_repr=r'Z_\text{P}')

planck_acceleration = Quantity("planck_acceleration", abbrev="a_P",
                               latex_repr=r'a_\text{P}')


# Information theory units:
bit = bits = Quantity("bit")
bit.set_global_dimension(information)

byte = bytes = Quantity("byte")

kibibyte = kibibytes = Quantity("kibibyte")
mebibyte = mebibytes = Quantity("mebibyte")
gibibyte = gibibytes = Quantity("gibibyte")
tebibyte = tebibytes = Quantity("tebibyte")
pebibyte = pebibytes = Quantity("pebibyte")
exbibyte = exbibytes = Quantity("exbibyte")

byte.set_global_relative_scale_factor(8, bit)
kibibyte.set_global_relative_scale_factor(kibi, byte)
mebibyte.set_global_relative_scale_factor(mebi, byte)
gibibyte.set_global_relative_scale_factor(gibi, byte)
tebibyte.set_global_relative_scale_factor(tebi, byte)
pebibyte.set_global_relative_scale_factor(pebi, byte)
exbibyte.set_global_relative_scale_factor(exbi, byte)

# Older units for radioactivity
curie = Ci = Quantity("curie", abbrev="Ci")

rutherford = Rd = Quantity("rutherford", abbrev="Rd")
