"""
MKS unit system.

MKS stands for "meter, kilogram, second, ampere".
"""

from __future__ import annotations

from sympy.physics.units.definitions import Z0, ampere, coulomb, farad, henry, siemens, tesla, volt, weber, ohm
from sympy.physics.units.definitions.dimension_definitions import (
    capacitance, charge, conductance, current, impedance, inductance,
    magnetic_density, magnetic_flux, voltage)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.mks import MKS, dimsys_length_weight_time
from sympy.physics.units.quantities import Quantity

dims = (voltage, impedance, conductance, current, capacitance, inductance, charge,
        magnetic_density, magnetic_flux)

units = [ampere, volt, ohm, siemens, farad, henry, coulomb, tesla, weber]

all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend(units)

all_units.append(Z0)

dimsys_MKSA = dimsys_length_weight_time.extend([
    # Dimensional dependencies for base dimensions (MKSA not in MKS)
    current,
], new_dim_deps={
    # Dimensional dependencies for derived dimensions
    "voltage": {"mass": 1, "length": 2, "current": -1, "time": -3},
    "impedance": {"mass": 1, "length": 2, "current": -2, "time": -3},
    "conductance": {"mass": -1, "length": -2, "current": 2, "time": 3},
    "capacitance": {"mass": -1, "length": -2, "current": 2, "time": 4},
    "inductance": {"mass": 1, "length": 2, "current": -2, "time": -2},
    "charge": {"current": 1, "time": 1},
    "magnetic_density": {"mass": 1, "current": -1, "time": -2},
    "magnetic_flux": {"length": 2, "mass": 1, "current": -1, "time": -2},
})

MKSA = MKS.extend(base=(ampere,), units=all_units, name='MKSA', dimension_system=dimsys_MKSA, derived_units={
    magnetic_flux: weber,
    impedance: ohm,
    current: ampere,
    voltage: volt,
    inductance: henry,
    conductance: siemens,
    magnetic_density: tesla,
    charge: coulomb,
    capacitance: farad,
})
