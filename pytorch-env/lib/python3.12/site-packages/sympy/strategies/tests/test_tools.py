from sympy.strategies.tools import subs, typed
from sympy.strategies.rl import rm_id
from sympy.core.basic import Basic
from sympy.core.singleton import S


def test_subs():
    from sympy.core.symbol import symbols
    a, b, c, d, e, f = symbols('a,b,c,d,e,f')
    mapping = {a: d, d: a, Basic(e): Basic(f)}
    expr = Basic(a, Basic(b, c), Basic(d, Basic(e)))
    result = Basic(d, Basic(b, c), Basic(a, Basic(f)))
    assert subs(mapping)(expr) == result


def test_subs_empty():
    assert subs({})(Basic(S(1), S(2))) == Basic(S(1), S(2))


def test_typed():
    class A(Basic):
        pass

    class B(Basic):
        pass

    rmzeros = rm_id(lambda x: x == S(0))
    rmones = rm_id(lambda x: x == S(1))
    remove_something = typed({A: rmzeros, B: rmones})

    assert remove_something(A(S(0), S(1))) == A(S(1))
    assert remove_something(B(S(0), S(1))) == B(S(0))
