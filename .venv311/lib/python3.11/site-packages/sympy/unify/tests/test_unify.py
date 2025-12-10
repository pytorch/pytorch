from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core

a,b,c = 'a', 'b', 'c'
w,x,y,z = map(Variable, 'wxyz')

C = Compound

def is_associative(x):
    return isinstance(x, Compound) and (x.op in ('Add', 'Mul', 'CAdd', 'CMul'))
def is_commutative(x):
    return isinstance(x, Compound) and (x.op in ('CAdd', 'CMul'))


def unify(a, b, s={}):
    return core.unify(a, b, s=s, is_associative=is_associative,
                          is_commutative=is_commutative)

def test_basic():
    assert list(unify(a, x, {})) == [{x: a}]
    assert list(unify(a, x, {x: 10})) == []
    assert list(unify(1, x, {})) == [{x: 1}]
    assert list(unify(a, a, {})) == [{}]
    assert list(unify((w, x), (y, z), {})) == [{w: y, x: z}]
    assert list(unify(x, (a, b), {})) == [{x: (a, b)}]

    assert list(unify((a, b), (x, x), {})) == []
    assert list(unify((y, z), (x, x), {}))!= []
    assert list(unify((a, (b, c)), (a, (x, y)), {})) == [{x: b, y: c}]

def test_ops():
    assert list(unify(C('Add', (a,b,c)), C('Add', (a,x,y)), {})) == \
            [{x:b, y:c}]
    assert list(unify(C('Add', (C('Mul', (1,2)), b,c)), C('Add', (x,y,c)), {})) == \
            [{x: C('Mul', (1,2)), y:b}]

def test_associative():
    c1 = C('Add', (1,2,3))
    c2 = C('Add', (x,y))
    assert tuple(unify(c1, c2, {})) == ({x: 1, y: C('Add', (2, 3))},
                                         {x: C('Add', (1, 2)), y: 3})

def test_commutative():
    c1 = C('CAdd', (1,2,3))
    c2 = C('CAdd', (x,y))
    result = list(unify(c1, c2, {}))
    assert  {x: 1, y: C('CAdd', (2, 3))} in result
    assert ({x: 2, y: C('CAdd', (1, 3))} in result or
            {x: 2, y: C('CAdd', (3, 1))} in result)

def _test_combinations_assoc():
    assert set(allcombinations((1,2,3), (a,b), True)) == \
        {(((1, 2), (3,)), (a, b)), (((1,), (2, 3)), (a, b))}

def _test_combinations_comm():
    assert set(allcombinations((1,2,3), (a,b), None)) == \
        {(((1,), (2, 3)), ('a', 'b')), (((2,), (3, 1)), ('a', 'b')),
             (((3,), (1, 2)), ('a', 'b')), (((1, 2), (3,)), ('a', 'b')),
             (((2, 3), (1,)), ('a', 'b')), (((3, 1), (2,)), ('a', 'b'))}

def test_allcombinations():
    assert set(allcombinations((1,2), (1,2), 'commutative')) ==\
        {(((1,),(2,)), ((1,),(2,))), (((1,),(2,)), ((2,),(1,)))}


def test_commutativity():
    c1 = Compound('CAdd', (a, b))
    c2 = Compound('CAdd', (x, y))
    assert is_commutative(c1) and is_commutative(c2)
    assert len(list(unify(c1, c2, {}))) == 2


def test_CondVariable():
    expr = C('CAdd', (1, 2))
    x = Variable('x')
    y = CondVariable('y', lambda a: a % 2 == 0)
    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (x, y))
    assert list(unify(expr, pattern, {})) == \
            [{x: 1, y: 2}]

    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (z, y))

    assert list(unify(expr, pattern, {})) == []

def test_defaultdict():
    assert next(unify(Variable('x'), 'foo')) == {Variable('x'): 'foo'}
