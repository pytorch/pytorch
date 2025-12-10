from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.optics import Medium
from sympy.abc import epsilon, mu, n
from sympy.physics.units import speed_of_light, u0, e0, m, kg, s, A

from sympy.testing.pytest import raises

c = speed_of_light.convert_to(m/s)
e0 = e0.convert_to(A**2*s**4/(kg*m**3))
u0 = u0.convert_to(m*kg/(A**2*s**2))


def test_medium():
    m1 = Medium('m1')
    assert m1.intrinsic_impedance == sqrt(u0/e0)
    assert m1.speed == 1/sqrt(e0*u0)
    assert m1.refractive_index == c*sqrt(e0*u0)
    assert m1.permittivity == e0
    assert m1.permeability == u0
    m2 = Medium('m2', epsilon, mu)
    assert m2.intrinsic_impedance == sqrt(mu/epsilon)
    assert m2.speed == 1/sqrt(epsilon*mu)
    assert m2.refractive_index == c*sqrt(epsilon*mu)
    assert m2.permittivity == epsilon
    assert m2.permeability == mu
    # Increasing electric permittivity and magnetic permeability
    # by small amount from its value in vacuum.
    m3 = Medium('m3', 9.0*10**(-12)*s**4*A**2/(m**3*kg), 1.45*10**(-6)*kg*m/(A**2*s**2))
    assert m3.refractive_index > m1.refractive_index
    assert m3 != m1
    # Decreasing electric permittivity and magnetic permeability
    # by small amount from its value in vacuum.
    m4 = Medium('m4', 7.0*10**(-12)*s**4*A**2/(m**3*kg), 1.15*10**(-6)*kg*m/(A**2*s**2))
    assert m4.refractive_index < m1.refractive_index
    m5 = Medium('m5', permittivity=710*10**(-12)*s**4*A**2/(m**3*kg), n=1.33)
    assert abs(m5.intrinsic_impedance - 6.24845417765552*kg*m**2/(A**2*s**3)) \
                < 1e-12*kg*m**2/(A**2*s**3)
    assert abs(m5.speed - 225407863.157895*m/s) < 1e-6*m/s
    assert abs(m5.refractive_index - 1.33000000000000) < 1e-12
    assert abs(m5.permittivity - 7.1e-10*A**2*s**4/(kg*m**3)) \
                < 1e-20*A**2*s**4/(kg*m**3)
    assert abs(m5.permeability - 2.77206575232851e-8*kg*m/(A**2*s**2)) \
                < 1e-20*kg*m/(A**2*s**2)
    m6 = Medium('m6', None, mu, n)
    assert m6.permittivity == n**2/(c**2*mu)
    # test for equality of refractive indices
    assert Medium('m7').refractive_index == Medium('m8', e0, u0).refractive_index
    raises(ValueError, lambda:Medium('m9', e0, u0, 2))
