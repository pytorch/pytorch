from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo

F, x, y, z = free_group("x, y, z")


def test_FreeGroup__init__():
    x, y, z = map(Symbol, "xyz")

    assert len(FreeGroup("x, y, z").generators) == 3
    assert len(FreeGroup(x).generators) == 1
    assert len(FreeGroup(("x", "y", "z"))) == 3
    assert len(FreeGroup((x, y, z)).generators) == 3


def test_free_group():
    G, a, b, c = free_group("a, b, c")
    assert F.generators == (x, y, z)
    assert x*z**2 in F
    assert x in F
    assert y*z**-1 in F
    assert (y*z)**0 in F
    assert a not in F
    assert a**0 not in F
    assert len(F) == 3
    assert str(F) == '<free group on the generators (x, y, z)>'
    assert not F == G
    assert F.order() is oo
    assert F.is_abelian == False
    assert F.center() == {F.identity}

    (e,) = free_group("")
    assert e.order() == 1
    assert e.generators == ()
    assert e.elements == {e.identity}
    assert e.is_abelian == True


def test_FreeGroup__hash__():
    assert hash(F)


def test_FreeGroup__eq__():
    assert free_group("x, y, z")[0] == free_group("x, y, z")[0]
    assert free_group("x, y, z")[0] is free_group("x, y, z")[0]

    assert free_group("x, y, z")[0] != free_group("a, x, y")[0]
    assert free_group("x, y, z")[0] is not free_group("a, x, y")[0]

    assert free_group("x, y")[0] != free_group("x, y, z")[0]
    assert free_group("x, y")[0] is not free_group("x, y, z")[0]

    assert free_group("x, y, z")[0] != free_group("x, y")[0]
    assert free_group("x, y, z")[0] is not free_group("x, y")[0]


def test_FreeGroup__getitem__():
    assert F[0:] == FreeGroup("x, y, z")
    assert F[1:] == FreeGroup("y, z")
    assert F[2:] == FreeGroup("z")


def test_FreeGroupElm__hash__():
    assert hash(x*y*z)


def test_FreeGroupElm_copy():
    f = x*y*z**3
    g = f.copy()
    h = x*y*z**7

    assert f == g
    assert f != h


def test_FreeGroupElm_inverse():
    assert x.inverse() == x**-1
    assert (x*y).inverse() == y**-1*x**-1
    assert (y*x*y**-1).inverse() == y*x**-1*y**-1
    assert (y**2*x**-1).inverse() == x*y**-2


def test_FreeGroupElm_type_error():
    raises(TypeError, lambda: 2/x)
    raises(TypeError, lambda: x**2 + y**2)
    raises(TypeError, lambda: x/2)


def test_FreeGroupElm_methods():
    assert (x**0).order() == 1
    assert (y**2).order() is oo
    assert (x**-1*y).commutator(x) == y**-1*x**-1*y*x
    assert len(x**2*y**-1) == 3
    assert len(x**-1*y**3*z) == 5


def test_FreeGroupElm_eliminate_word():
    w = x**5*y*x**2*y**-4*x
    assert w.eliminate_word( x, x**2 ) == x**10*y*x**4*y**-4*x**2
    w3 = x**2*y**3*x**-1*y
    assert w3.eliminate_word(x, x**2) == x**4*y**3*x**-2*y
    assert w3.eliminate_word(x, y) == y**5
    assert w3.eliminate_word(x, y**4) == y**8
    assert w3.eliminate_word(y, x**-1) == x**-3
    assert w3.eliminate_word(x, y*z) == y*z*y*z*y**3*z**-1
    assert (y**-3).eliminate_word(y, x**-1*z**-1) == z*x*z*x*z*x
    #assert w3.eliminate_word(x, y*x) == y*x*y*x**2*y*x*y*x*y*x*z**3
    #assert w3.eliminate_word(x, x*y) == x*y*x**2*y*x*y*x*y*x*y*z**3


def test_FreeGroupElm_array_form():
    assert (x*z).array_form == ((Symbol('x'), 1), (Symbol('z'), 1))
    assert (x**2*z*y*x**-2).array_form == \
        ((Symbol('x'), 2), (Symbol('z'), 1), (Symbol('y'), 1), (Symbol('x'), -2))
    assert (x**-2*y**-1).array_form == ((Symbol('x'), -2), (Symbol('y'), -1))


def test_FreeGroupElm_letter_form():
    assert (x**3).letter_form == (Symbol('x'), Symbol('x'), Symbol('x'))
    assert (x**2*z**-2*x).letter_form == \
        (Symbol('x'), Symbol('x'), -Symbol('z'), -Symbol('z'), Symbol('x'))


def test_FreeGroupElm_ext_rep():
    assert (x**2*z**-2*x).ext_rep == \
        (Symbol('x'), 2, Symbol('z'), -2, Symbol('x'), 1)
    assert (x**-2*y**-1).ext_rep == (Symbol('x'), -2, Symbol('y'), -1)
    assert (x*z).ext_rep == (Symbol('x'), 1, Symbol('z'), 1)


def test_FreeGroupElm__mul__pow__():
    x1 = x.group.dtype(((Symbol('x'), 1),))
    assert x**2 == x1*x

    assert (x**2*y*x**-2)**4 == x**2*y**4*x**-2
    assert (x**2)**2 == x**4
    assert (x**-1)**-1 == x
    assert (x**-1)**0 == F.identity
    assert (y**2)**-2 == y**-4

    assert x**2*x**-1 == x
    assert x**2*y**2*y**-1 == x**2*y
    assert x*x**-1 == F.identity

    assert x/x == F.identity
    assert x/x**2 == x**-1
    assert (x**2*y)/(x**2*y**-1) == x**2*y**2*x**-2
    assert (x**2*y)/(y**-1*x**2) == x**2*y*x**-2*y

    assert x*(x**-1*y*z*y**-1) == y*z*y**-1
    assert x**2*(x**-2*y**-1*z**2*y) == y**-1*z**2*y

    a = F.identity
    for n in range(10):
        assert a == x**n
        assert a**-1 == x**-n
        a *= x


def test_FreeGroupElm__len__():
    assert len(x**5*y*x**2*y**-4*x) == 13
    assert len(x**17) == 17
    assert len(y**0) == 0


def test_FreeGroupElm_comparison():
    assert not (x*y == y*x)
    assert x**0 == y**0

    assert x**2 < y**3
    assert not x**3 < y**2
    assert x*y < x**2*y
    assert x**2*y**2 < y**4
    assert not y**4 < y**-4
    assert not y**4 < x**-4
    assert y**-2 < y**2

    assert x**2 <= y**2
    assert x**2 <= x**2

    assert not y*z > z*y
    assert x > x**-1

    assert not x**2 >= y**2


def test_FreeGroupElm_syllables():
    w = x**5*y*x**2*y**-4*x
    assert w.number_syllables() == 5
    assert w.exponent_syllable(2) == 2
    assert w.generator_syllable(3) == Symbol('y')
    assert w.sub_syllables(1, 2) == y
    assert w.sub_syllables(3, 3) == F.identity


def test_FreeGroup_exponents():
    w1 = x**2*y**3
    assert w1.exponent_sum(x) == 2
    assert w1.exponent_sum(x**-1) == -2
    assert w1.generator_count(x) == 2

    w2 = x**2*y**4*x**-3
    assert w2.exponent_sum(x) == -1
    assert w2.generator_count(x) == 5


def test_FreeGroup_generators():
    assert (x**2*y**4*z**-1).contains_generators() == {x, y, z}
    assert (x**-1*y**3).contains_generators() == {x, y}


def test_FreeGroupElm_words():
    w = x**5*y*x**2*y**-4*x
    assert w.subword(2, 6) == x**3*y
    assert w.subword(3, 2) == F.identity
    assert w.subword(6, 10) == x**2*y**-2

    assert w.substituted_word(0, 7, y**-1) == y**-1*x*y**-4*x
    assert w.substituted_word(0, 7, y**2*x) == y**2*x**2*y**-4*x
