from sympy.core.singleton import S
from sympy.strategies.rl import (
    rm_id, glom, flatten, unpack, sort, distribute, subs, rebuild)
from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.abc import x


def test_rm_id():
    rmzeros = rm_id(lambda x: x == 0)
    assert rmzeros(Basic(S(0), S(1))) == Basic(S(1))
    assert rmzeros(Basic(S(0), S(0))) == Basic(S(0))
    assert rmzeros(Basic(S(2), S(1))) == Basic(S(2), S(1))


def test_glom():
    def key(x):
        return x.as_coeff_Mul()[1]

    def count(x):
        return x.as_coeff_Mul()[0]

    def newargs(cnt, arg):
        return cnt * arg

    rl = glom(key, count, newargs)

    result = rl(Add(x, -x, 3 * x, 2, 3, evaluate=False))
    expected = Add(3 * x, 5)
    assert set(result.args) == set(expected.args)


def test_flatten():
    assert flatten(Basic(S(1), S(2), Basic(S(3), S(4)))) == \
        Basic(S(1), S(2), S(3), S(4))


def test_unpack():
    assert unpack(Basic(S(2))) == 2
    assert unpack(Basic(S(2), S(3))) == Basic(S(2), S(3))


def test_sort():
    assert sort(str)(Basic(S(3), S(1), S(2))) == Basic(S(1), S(2), S(3))


def test_distribute():
    class T1(Basic):
        pass

    class T2(Basic):
        pass

    distribute_t12 = distribute(T1, T2)
    assert distribute_t12(T1(S(1), S(2), T2(S(3), S(4)), S(5))) == \
        T2(T1(S(1), S(2), S(3), S(5)), T1(S(1), S(2), S(4), S(5)))
    assert distribute_t12(T1(S(1), S(2), S(3))) == T1(S(1), S(2), S(3))


def test_distribute_add_mul():
    x, y = symbols('x, y')
    expr = Mul(2, Add(x, y), evaluate=False)
    expected = Add(Mul(2, x), Mul(2, y))
    distribute_mul = distribute(Mul, Add)
    assert distribute_mul(expr) == expected


def test_subs():
    rl = subs(1, 2)
    assert rl(1) == 2
    assert rl(3) == 3


def test_rebuild():
    expr = Basic.__new__(Add, S(1), S(2))
    assert rebuild(expr) == 3
