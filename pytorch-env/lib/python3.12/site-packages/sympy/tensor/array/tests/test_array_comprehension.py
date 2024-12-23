from sympy.tensor.array.array_comprehension import ArrayComprehension, ArrayComprehensionMap
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.abc import i, j, k, l
from sympy.testing.pytest import raises
from sympy.matrices import Matrix


def test_array_comprehension():
    a = ArrayComprehension(i*j, (i, 1, 3), (j, 2, 4))
    b = ArrayComprehension(i, (i, 1, j+1))
    c = ArrayComprehension(i+j+k+l, (i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    d = ArrayComprehension(k, (i, 1, 5))
    e = ArrayComprehension(i, (j, k+1, k+5))
    assert a.doit().tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    assert a.shape == (3, 3)
    assert a.is_shape_numeric == True
    assert a.tolist() == [[2, 3, 4], [4, 6, 8], [6, 9, 12]]
    assert a.tomatrix() == Matrix([
                           [2, 3, 4],
                           [4, 6, 8],
                           [6, 9, 12]])
    assert len(a) == 9
    assert isinstance(b.doit(), ArrayComprehension)
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    assert b.subs(j, 3) == ArrayComprehension(i, (i, 1, 4))
    assert b.free_symbols == {j}
    assert b.shape == (j + 1,)
    assert b.rank() == 1
    assert b.is_shape_numeric == False
    assert c.free_symbols == set()
    assert c.function == i + j + k + l
    assert c.limits == ((i, 1, 2), (j, 1, 3), (k, 1, 4), (l, 1, 5))
    assert c.doit().tolist() == [[[[4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11]],
                                  [[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]]],
                                 [[[5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12]],
                                  [[6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13]],
                                  [[7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13], [10, 11, 12, 13, 14]]]]
    assert c.free_symbols == set()
    assert c.variables == [i, j, k, l]
    assert c.bound_symbols == [i, j, k, l]
    assert d.doit().tolist() == [k, k, k, k, k]
    assert len(e) == 5
    raises(TypeError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, [1, 3, 2])))
    raises(ValueError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, 1)))
    raises(ValueError, lambda: ArrayComprehension(i*j, (i, 1, 3), (j, 2, j+1)))
    raises(ValueError, lambda: len(ArrayComprehension(i*j, (i, 1, 3), (j, 2, j+4))))
    raises(TypeError, lambda: ArrayComprehension(i*j, (i, 0, i + 1.5), (j, 0, 2)))
    raises(ValueError, lambda: b.tolist())
    raises(ValueError, lambda: b.tomatrix())
    raises(ValueError, lambda: c.tomatrix())

def test_arraycomprehensionmap():
    a = ArrayComprehensionMap(lambda i: i+1, (i, 1, 5))
    assert a.doit().tolist() == [2, 3, 4, 5, 6]
    assert a.shape == (5,)
    assert a.is_shape_numeric
    assert a.tolist() == [2, 3, 4, 5, 6]
    assert len(a) == 5
    assert isinstance(a.doit(), ImmutableDenseNDimArray)
    expr = ArrayComprehensionMap(lambda i: i+1, (i, 1, k))
    assert expr.doit() == expr
    assert expr.subs(k, 4) == ArrayComprehensionMap(lambda i: i+1, (i, 1, 4))
    assert expr.subs(k, 4).doit() == ImmutableDenseNDimArray([2, 3, 4, 5])
    b = ArrayComprehensionMap(lambda i: i+1, (i, 1, 2), (i, 1, 3), (i, 1, 4), (i, 1, 5))
    assert b.doit().tolist() == [[[[2, 3, 4, 5, 6], [3, 5, 7, 9, 11], [4, 7, 10, 13, 16], [5, 9, 13, 17, 21]],
                                  [[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[4, 7, 10, 13, 16], [7, 13, 19, 25, 31], [10, 19, 28, 37, 46], [13, 25, 37, 49, 61]]],
                                 [[[3, 5, 7, 9, 11], [5, 9, 13, 17, 21], [7, 13, 19, 25, 31], [9, 17, 25, 33, 41]],
                                  [[5, 9, 13, 17, 21], [9, 17, 25, 33, 41], [13, 25, 37, 49, 61], [17, 33, 49, 65, 81]],
                                  [[7, 13, 19, 25, 31], [13, 25, 37, 49, 61], [19, 37, 55, 73, 91], [25, 49, 73, 97, 121]]]]

    # tests about lambda expression
    assert ArrayComprehensionMap(lambda: 3, (i, 1, 5)).doit().tolist() == [3, 3, 3, 3, 3]
    assert ArrayComprehensionMap(lambda i: i+1, (i, 1, 5)).doit().tolist() == [2, 3, 4, 5, 6]
    raises(ValueError, lambda: ArrayComprehensionMap(i*j, (i, 1, 3), (j, 2, 4)))
    a = ArrayComprehensionMap(lambda i, j: i+j, (i, 1, 5))
    raises(ValueError, lambda: a.doit())
