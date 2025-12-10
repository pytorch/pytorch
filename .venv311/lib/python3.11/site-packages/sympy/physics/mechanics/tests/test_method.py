from sympy.physics.mechanics.method import _Methods
from sympy.testing.pytest import raises

def test_method():
    raises(TypeError, lambda: _Methods())
