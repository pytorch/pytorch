from sympy.core.function import diff
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)

from sympy.abc import a, q, z


def test_mathieus():
    assert isinstance(mathieus(a, q, z), mathieus)
    assert mathieus(a, 0, z) == sin(sqrt(a)*z)
    assert conjugate(mathieus(a, q, z)) == mathieus(conjugate(a), conjugate(q), conjugate(z))
    assert diff(mathieus(a, q, z), z) == mathieusprime(a, q, z)

def test_mathieuc():
    assert isinstance(mathieuc(a, q, z), mathieuc)
    assert mathieuc(a, 0, z) == cos(sqrt(a)*z)
    assert diff(mathieuc(a, q, z), z) == mathieucprime(a, q, z)

def test_mathieusprime():
    assert isinstance(mathieusprime(a, q, z), mathieusprime)
    assert mathieusprime(a, 0, z) == sqrt(a)*cos(sqrt(a)*z)
    assert diff(mathieusprime(a, q, z), z) == (-a + 2*q*cos(2*z))*mathieus(a, q, z)

def test_mathieucprime():
    assert isinstance(mathieucprime(a, q, z), mathieucprime)
    assert mathieucprime(a, 0, z) == -sqrt(a)*sin(sqrt(a)*z)
    assert diff(mathieucprime(a, q, z), z) == (-a + 2*q*cos(2*z))*mathieuc(a, q, z)
