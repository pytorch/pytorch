from sympy.sets.ordinals import Ordinal, OmegaPower, ord0, omega
from sympy.testing.pytest import raises

def test_string_ordinals():
    assert str(omega) == 'w'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))) == 'w**5*3 + w**3*2'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))) == 'w**5*3 + 5'
    assert str(Ordinal(OmegaPower(1, 3), OmegaPower(0, 5))) == 'w*3 + 5'
    assert str(Ordinal(OmegaPower(omega + 1, 1), OmegaPower(3, 2))) == 'w**(w + 1) + w**3*2'

def test_addition_with_integers():
    assert 3 + Ordinal(OmegaPower(5, 3)) == Ordinal(OmegaPower(5, 3))
    assert Ordinal(OmegaPower(5, 3))+3 == Ordinal(OmegaPower(5, 3), OmegaPower(0, 3))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(0, 2))+3 == \
        Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))


def test_addition_with_ordinals():
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(3, 3)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 5))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(4, 2))
    assert Ordinal(OmegaPower(omega, 2), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(omega, 2), OmegaPower(4, 2))

def test_comparison():
    assert Ordinal(OmegaPower(5, 3)) > Ordinal(OmegaPower(4, 3), OmegaPower(2, 1))
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) < Ordinal(OmegaPower(5, 4))
    assert Ordinal(OmegaPower(5, 4)) < Ordinal(OmegaPower(5, 5), OmegaPower(4, 1))

    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))
    assert not Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == Ordinal(OmegaPower(5, 3))
    assert Ordinal(OmegaPower(omega, 3)) > Ordinal(OmegaPower(5, 3))

def test_multiplication_with_integers():
    w = omega
    assert 3*w == w
    assert w*9 == Ordinal(OmegaPower(1, 9))

def test_multiplication():
    w = omega
    assert w*(w + 1) == w*w + w
    assert (w + 1)*(w + 1) ==  w*w + w + 1
    assert w*1 == w
    assert 1*w == w
    assert w*ord0 == ord0
    assert ord0*w == ord0
    assert w**w == w * w**w
    assert (w**w)*w*w == w**(w + 2)

def test_exponentiation():
    w = omega
    assert w**2 == w*w
    assert w**3 == w*w*w
    assert w**(w + 1) == Ordinal(OmegaPower(omega + 1, 1))
    assert (w**w)*(w**w) == w**(w*2)

def test_comapre_not_instance():
    w = OmegaPower(omega + 1, 1)
    assert(not (w == None))
    assert(not (w < 5))
    raises(TypeError, lambda: w < 6.66)

def test_is_successort():
    w = Ordinal(OmegaPower(5, 1))
    assert not w.is_successor_ordinal
