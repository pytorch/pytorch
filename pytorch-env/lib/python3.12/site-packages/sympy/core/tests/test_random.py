import random
from sympy.core.random import random as rand, seed, shuffle, _assumptions_shuffle
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.trigonometric import sin, acos
from sympy.abc import x


def test_random():
    random.seed(42)
    a = random.random()
    random.seed(42)
    Symbol('z').is_finite
    b = random.random()
    assert a == b

    got = set()
    for i in range(2):
        random.seed(28)
        m0, m1 = symbols('m_0 m_1', real=True)
        _ = acos(-m0/m1)
        got.add(random.uniform(0,1))
    assert len(got) == 1

    random.seed(10)
    y = 0
    for i in range(4):
        y += sin(random.uniform(-10,10) * x)
    random.seed(10)
    z = 0
    for i in range(4):
        z += sin(random.uniform(-10,10) * x)
    assert y == z


def test_seed():
    assert rand() < 1
    seed(1)
    a = rand()
    b = rand()
    seed(1)
    c = rand()
    d = rand()
    assert a == c
    if not c == d:
        assert a != b
    else:
        assert a == b

    abc = 'abc'
    first = list(abc)
    second = list(abc)
    third = list(abc)

    seed(123)
    shuffle(first)

    seed(123)
    shuffle(second)
    _assumptions_shuffle(third)

    assert first == second == third
