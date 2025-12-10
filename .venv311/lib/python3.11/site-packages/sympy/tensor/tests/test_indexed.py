from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable

# import test:
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
    StrictGreaterThan, LessThan)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed


def test_Idx_construction():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i) != Idx(i, 1)
    assert Idx(i, a) == Idx(i, (0, a - 1))
    assert Idx(i, oo) == Idx(i, (0, oo))

    x = symbols('x', integer=False)
    raises(TypeError, lambda: Idx(x))
    raises(TypeError, lambda: Idx(0.5))
    raises(TypeError, lambda: Idx(i, x))
    raises(TypeError, lambda: Idx(i, 0.5))
    raises(TypeError, lambda: Idx(i, (x, 5)))
    raises(TypeError, lambda: Idx(i, (2, x)))
    raises(TypeError, lambda: Idx(i, (2, 3.5)))


def test_Idx_properties():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i).is_integer
    assert Idx(i).name == 'i'
    assert Idx(i + 2).name == 'i + 2'
    assert Idx('foo').name == 'foo'


def test_Idx_bounds():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i).lower is None
    assert Idx(i).upper is None
    assert Idx(i, a).lower == 0
    assert Idx(i, a).upper == a - 1
    assert Idx(i, 5).lower == 0
    assert Idx(i, 5).upper == 4
    assert Idx(i, oo).lower == 0
    assert Idx(i, oo).upper is oo
    assert Idx(i, (a, b)).lower == a
    assert Idx(i, (a, b)).upper == b
    assert Idx(i, (1, 5)).lower == 1
    assert Idx(i, (1, 5)).upper == 5
    assert Idx(i, (-oo, oo)).lower is -oo
    assert Idx(i, (-oo, oo)).upper is oo


def test_Idx_fixed_bounds():
    i, a, b, x = symbols('i a b x', integer=True)
    assert Idx(x).lower is None
    assert Idx(x).upper is None
    assert Idx(x, a).lower == 0
    assert Idx(x, a).upper == a - 1
    assert Idx(x, 5).lower == 0
    assert Idx(x, 5).upper == 4
    assert Idx(x, oo).lower == 0
    assert Idx(x, oo).upper is oo
    assert Idx(x, (a, b)).lower == a
    assert Idx(x, (a, b)).upper == b
    assert Idx(x, (1, 5)).lower == 1
    assert Idx(x, (1, 5)).upper == 5
    assert Idx(x, (-oo, oo)).lower is -oo
    assert Idx(x, (-oo, oo)).upper is oo


def test_Idx_inequalities():
    i14 = Idx("i14", (1, 4))
    i79 = Idx("i79", (7, 9))
    i46 = Idx("i46", (4, 6))
    i35 = Idx("i35", (3, 5))

    assert i14 <= 5
    assert i14 < 5
    assert not (i14 >= 5)
    assert not (i14 > 5)

    assert 5 >= i14
    assert 5 > i14
    assert not (5 <= i14)
    assert not (5 < i14)

    assert LessThan(i14, 5)
    assert StrictLessThan(i14, 5)
    assert not GreaterThan(i14, 5)
    assert not StrictGreaterThan(i14, 5)

    assert i14 <= 4
    assert isinstance(i14 < 4, StrictLessThan)
    assert isinstance(i14 >= 4, GreaterThan)
    assert not (i14 > 4)

    assert isinstance(i14 <= 1, LessThan)
    assert not (i14 < 1)
    assert i14 >= 1
    assert isinstance(i14 > 1, StrictGreaterThan)

    assert not (i14 <= 0)
    assert not (i14 < 0)
    assert i14 >= 0
    assert i14 > 0

    from sympy.abc import x

    assert isinstance(i14 < x, StrictLessThan)
    assert isinstance(i14 > x, StrictGreaterThan)
    assert isinstance(i14 <= x, LessThan)
    assert isinstance(i14 >= x, GreaterThan)

    assert i14 < i79
    assert i14 <= i79
    assert not (i14 > i79)
    assert not (i14 >= i79)

    assert i14 <= i46
    assert isinstance(i14 < i46, StrictLessThan)
    assert isinstance(i14 >= i46, GreaterThan)
    assert not (i14 > i46)

    assert isinstance(i14 < i35, StrictLessThan)
    assert isinstance(i14 > i35, StrictGreaterThan)
    assert isinstance(i14 <= i35, LessThan)
    assert isinstance(i14 >= i35, GreaterThan)

    iNone1 = Idx("iNone1")
    iNone2 = Idx("iNone2")

    assert isinstance(iNone1 < iNone2, StrictLessThan)
    assert isinstance(iNone1 > iNone2, StrictGreaterThan)
    assert isinstance(iNone1 <= iNone2, LessThan)
    assert isinstance(iNone1 >= iNone2, GreaterThan)


def test_Idx_inequalities_current_fails():
    i14 = Idx("i14", (1, 4))

    assert S(5) >= i14
    assert S(5) > i14
    assert not (S(5) <= i14)
    assert not (S(5) < i14)


def test_Idx_func_args():
    i, a, b = symbols('i a b', integer=True)
    ii = Idx(i)
    assert ii.func(*ii.args) == ii
    ii = Idx(i, a)
    assert ii.func(*ii.args) == ii
    ii = Idx(i, (a, b))
    assert ii.func(*ii.args) == ii


def test_Idx_subs():
    i, a, b = symbols('i a b', integer=True)
    assert Idx(i, a).subs(a, b) == Idx(i, b)
    assert Idx(i, a).subs(i, b) == Idx(b, a)

    assert Idx(i).subs(i, 2) == Idx(2)
    assert Idx(i, a).subs(a, 2) == Idx(i, 2)
    assert Idx(i, (a, b)).subs(i, 2) == Idx(2, (a, b))


def test_IndexedBase_sugar():
    i, j = symbols('i j', integer=True)
    a = symbols('a')
    A1 = Indexed(a, i, j)
    A2 = IndexedBase(a)
    assert A1 == A2[i, j]
    assert A1 == A2[(i, j)]
    assert A1 == A2[[i, j]]
    assert A1 == A2[Tuple(i, j)]
    assert all(a.is_Integer for a in A2[1, 0].args[1:])


def test_IndexedBase_subs():
    i = symbols('i', integer=True)
    a, b = symbols('a b')
    A = IndexedBase(a)
    B = IndexedBase(b)
    assert A[i] == B[i].subs(b, a)
    C = {1: 2}
    assert C[1] == A[1].subs(A, C)


def test_IndexedBase_shape():
    i, j, m, n = symbols('i j m n', integer=True)
    a = IndexedBase('a', shape=(m, m))
    b = IndexedBase('a', shape=(m, n))
    assert b.shape == Tuple(m, n)
    assert a[i, j] != b[i, j]
    assert a[i, j] == b[i, j].subs(n, m)
    assert b.func(*b.args) == b
    assert b[i, j].func(*b[i, j].args) == b[i, j]
    raises(IndexException, lambda: b[i])
    raises(IndexException, lambda: b[i, i, j])
    F = IndexedBase("F", shape=m)
    assert F.shape == Tuple(m)
    assert F[i].subs(i, j) == F[j]
    raises(IndexException, lambda: F[i, j])


def test_IndexedBase_assumptions():
    i = Symbol('i', integer=True)
    a = Symbol('a')
    A = IndexedBase(a, positive=True)
    for c in (A, A[i]):
        assert c.is_real
        assert c.is_complex
        assert not c.is_imaginary
        assert c.is_nonnegative
        assert c.is_nonzero
        assert c.is_commutative
        assert log(exp(c)) == c

    assert A != IndexedBase(a)
    assert A == IndexedBase(a, positive=True, real=True)
    assert A[i] != Indexed(a, i)


def test_IndexedBase_assumptions_inheritance():
    I = Symbol('I', integer=True)
    I_inherit = IndexedBase(I)
    I_explicit = IndexedBase('I', integer=True)

    assert I_inherit.is_integer
    assert I_explicit.is_integer
    assert I_inherit.label.is_integer
    assert I_explicit.label.is_integer
    assert I_inherit == I_explicit


def test_issue_17652():
    """Regression test issue #17652.

    IndexedBase.label should not upcast subclasses of Symbol
    """
    class SubClass(Symbol):
        pass

    x = SubClass('X')
    assert type(x) == SubClass
    base = IndexedBase(x)
    assert type(x) == SubClass
    assert type(base.label) == SubClass


def test_Indexed_constructor():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, j)
    assert A == Indexed(Symbol('A'), i, j)
    assert A == Indexed(IndexedBase('A'), i, j)
    raises(TypeError, lambda: Indexed(A, i, j))
    raises(IndexException, lambda: Indexed("A"))
    assert A.free_symbols == {A, A.base.label, i, j}


def test_Indexed_func_args():
    i, j = symbols('i j', integer=True)
    a = symbols('a')
    A = Indexed(a, i, j)
    assert A == A.func(*A.args)


def test_Indexed_subs():
    i, j, k = symbols('i j k', integer=True)
    a, b = symbols('a b')
    A = IndexedBase(a)
    B = IndexedBase(b)
    assert A[i, j] == B[i, j].subs(b, a)
    assert A[i, j] == A[i, k].subs(k, j)


def test_Indexed_properties():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, j)
    assert A.name == 'A[i, j]'
    assert A.rank == 2
    assert A.indices == (i, j)
    assert A.base == IndexedBase('A')
    assert A.ranges == [None, None]
    raises(IndexException, lambda: A.shape)

    n, m = symbols('n m', integer=True)
    assert Indexed('A', Idx(
        i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed('A', Idx(i, m), Idx(j, n)).shape == Tuple(m, n)
    raises(IndexException, lambda: Indexed("A", Idx(i, m), Idx(j)).shape)


def test_Indexed_shape_precedence():
    i, j = symbols('i j', integer=True)
    o, p = symbols('o p', integer=True)
    n, m = symbols('n m', integer=True)
    a = IndexedBase('a', shape=(o, p))
    assert a.shape == Tuple(o, p)
    assert Indexed(
        a, Idx(i, m), Idx(j, n)).ranges == [Tuple(0, m - 1), Tuple(0, n - 1)]
    assert Indexed(a, Idx(i, m), Idx(j, n)).shape == Tuple(o, p)
    assert Indexed(
        a, Idx(i, m), Idx(j)).ranges == [Tuple(0, m - 1), (None, None)]
    assert Indexed(a, Idx(i, m), Idx(j)).shape == Tuple(o, p)


def test_complex_indices():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, i + j)
    assert A.rank == 2
    assert A.indices == (i, i + j)


def test_not_interable():
    i, j = symbols('i j', integer=True)
    A = Indexed('A', i, i + j)
    assert not iterable(A)


def test_Indexed_coeff():
    N = Symbol('N', integer=True)
    len_y = N
    i = Idx('i', len_y-1)
    y = IndexedBase('y', shape=(len_y,))
    a = (1/y[i+1]*y[i]).coeff(y[i])
    b = (y[i]/y[i+1]).coeff(y[i])
    assert a == b


def test_differentiation():
    from sympy.functions.special.tensor_functions import KroneckerDelta
    i, j, k, l = symbols('i j k l', cls=Idx)
    a = symbols('a')
    m, n = symbols("m, n", integer=True, finite=True)
    assert m.is_real
    h, L = symbols('h L', cls=IndexedBase)
    hi, hj = h[i], h[j]

    expr = hi
    assert expr.diff(hj) == KroneckerDelta(i, j)
    assert expr.diff(hi) == KroneckerDelta(i, i)

    expr = S(2) * hi
    assert expr.diff(hj) == S(2) * KroneckerDelta(i, j)
    assert expr.diff(hi) == S(2) * KroneckerDelta(i, i)
    assert expr.diff(a) is S.Zero

    assert Sum(expr, (i, -oo, oo)).diff(hj) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr.diff(hj), (i, -oo, oo)) == Sum(2*KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hj).doit() == 2

    assert Sum(expr.diff(hi), (i, -oo, oo)).doit() == Sum(2, (i, -oo, oo)).doit()
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() is oo

    expr = a * hj * hj / S(2)
    assert expr.diff(hi) == a * h[j] * KroneckerDelta(i, j)
    assert expr.diff(a) == hj * hj / S(2)
    assert expr.diff(a, 2) is S.Zero

    assert Sum(expr, (i, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))
    assert Sum(expr.diff(hi), (i, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() == a*h[j]

    assert Sum(expr, (j, -oo, oo)).diff(hi) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))
    assert Sum(expr.diff(hi), (j, -oo, oo)) == Sum(a*KroneckerDelta(i, j)*h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(hi).doit() == a*h[i]

    expr = a * sin(hj * hj)
    assert expr.diff(hi) == 2*a*cos(hj * hj) * hj * KroneckerDelta(i, j)
    assert expr.diff(hj) == 2*a*cos(hj * hj) * hj

    expr = a * L[i, j] * h[j]
    assert expr.diff(hi) == a*L[i, j]*KroneckerDelta(i, j)
    assert expr.diff(hj) == a*L[i, j]
    assert expr.diff(L[i, j]) == a*h[j]
    assert expr.diff(L[k, l]) == a*KroneckerDelta(i, k)*KroneckerDelta(j, l)*h[j]
    assert expr.diff(L[i, l]) == a*KroneckerDelta(j, l)*h[j]

    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]) == Sum(a * KroneckerDelta(i, k) * KroneckerDelta(j, l) * h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]).doit() == a * KroneckerDelta(i, k) * h[l]

    assert h[m].diff(h[m]) == 1
    assert h[m].diff(h[n]) == KroneckerDelta(m, n)
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]) == Sum(a*KroneckerDelta(m, n), (m, -oo, oo))
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[n]).doit() == a
    assert Sum(a*h[m], (n, -oo, oo)).diff(h[n]) == Sum(a*KroneckerDelta(m, n), (n, -oo, oo))
    assert Sum(a*h[m], (m, -oo, oo)).diff(h[m]).doit() == oo*a


def test_indexed_series():
    A = IndexedBase("A")
    i = symbols("i", integer=True)
    assert sin(A[i]).series(A[i]) == A[i] - A[i]**3/6 + A[i]**5/120 + Order(A[i]**6, A[i])


def test_indexed_is_constant():
    A = IndexedBase("A")
    i, j, k = symbols("i,j,k")
    assert not A[i].is_constant()
    assert A[i].is_constant(j)
    assert not A[1+2*i, k].is_constant()
    assert not A[1+2*i, k].is_constant(i)
    assert A[1+2*i, k].is_constant(j)
    assert not A[1+2*i, k].is_constant(k)


def test_issue_12533():
    d = IndexedBase('d')
    assert IndexedBase(range(5)) == Range(0, 5, 1)
    assert d[0].subs(Symbol("d"), range(5)) == 0
    assert d[0].subs(d, range(5)) == 0
    assert d[1].subs(d, range(5)) == 1
    assert Indexed(Range(5), 2) == 2


def test_issue_12780():
    n = symbols("n")
    i = Idx("i", (0, n))
    raises(TypeError, lambda: i.subs(n, 1.5))


def test_issue_18604():
    m = symbols("m")
    assert Idx("i", m).name == 'i'
    assert Idx("i", m).lower == 0
    assert Idx("i", m).upper == m - 1
    m = symbols("m", real=False)
    raises(TypeError, lambda: Idx("i", m))

def test_Subs_with_Indexed():
    A = IndexedBase("A")
    i, j, k = symbols("i,j,k")
    x, y, z = symbols("x,y,z")
    f = Function("f")

    assert Subs(A[i], A[i], A[j]).diff(A[j]) == 1
    assert Subs(A[i], A[i], x).diff(A[i]) == 0
    assert Subs(A[i], A[i], x).diff(A[j]) == 0
    assert Subs(A[i], A[i], x).diff(x) == 1
    assert Subs(A[i], A[i], x).diff(y) == 0
    assert Subs(A[i], A[i], A[j]).diff(A[k]) == KroneckerDelta(j, k)
    assert Subs(x, x, A[i]).diff(A[j]) == KroneckerDelta(i, j)
    assert Subs(f(A[i]), A[i], x).diff(A[j]) == 0
    assert Subs(f(A[i]), A[i], A[k]).diff(A[j]) == Derivative(f(A[k]), A[k])*KroneckerDelta(j, k)
    assert Subs(x, x, A[i]**2).diff(A[j]) == 2*KroneckerDelta(i, j)*A[i]
    assert Subs(A[i], A[i], A[j]**2).diff(A[k]) == 2*KroneckerDelta(j, k)*A[j]

    assert Subs(A[i]*x, x, A[i]).diff(A[i]) == 2*A[i]
    assert Subs(A[i]*x, x, A[i]).diff(A[j]) == 2*A[i]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[j]).diff(A[i]) == A[j] + A[i]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[j]).diff(A[j]) == A[i] + A[j]*KroneckerDelta(i, j)
    assert Subs(A[i]*x, x, A[i]).diff(A[k]) == 2*A[i]*KroneckerDelta(i, k)
    assert Subs(A[i]*x, x, A[j]).diff(A[k]) == KroneckerDelta(i, k)*A[j] + KroneckerDelta(j, k)*A[i]

    assert Subs(A[i]*x, A[i], x).diff(A[i]) == 0
    assert Subs(A[i]*x, A[i], x).diff(A[j]) == 0
    assert Subs(A[i]*x, A[j], x).diff(A[i]) == x
    assert Subs(A[i]*x, A[j], x).diff(A[j]) == x*KroneckerDelta(i, j)
    assert Subs(A[i]*x, A[i], x).diff(A[k]) == 0
    assert Subs(A[i]*x, A[j], x).diff(A[k]) == x*KroneckerDelta(i, k)


def test_complicated_derivative_with_Indexed():
    x, y = symbols("x,y", cls=IndexedBase)
    sigma = symbols("sigma")
    i, j, k = symbols("i,j,k")
    m0,m1,m2,m3,m4,m5 = symbols("m0:6")
    f = Function("f")

    expr = f((x[i] - y[i])**2/sigma)
    _xi_1 = symbols("xi_1", cls=Dummy)
    assert expr.diff(x[m0]).dummy_eq(
        (x[i] - y[i])*KroneckerDelta(i, m0)*\
        2*Subs(
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
        )/sigma
    )
    assert expr.diff(x[m0]).diff(x[m1]).dummy_eq(
        2*KroneckerDelta(i, m0)*\
        KroneckerDelta(i, m1)*Subs(
            Derivative(f(_xi_1), _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
         )/sigma + \
        4*(x[i] - y[i])**2*KroneckerDelta(i, m0)*KroneckerDelta(i, m1)*\
        Subs(
            Derivative(f(_xi_1), _xi_1, _xi_1),
            (_xi_1,),
            ((x[i] - y[i])**2/sigma,)
        )/sigma**2
    )


def test_IndexedBase_commutative():
    t = IndexedBase('t', commutative=False)
    u = IndexedBase('u', commutative=False)
    v = IndexedBase('v')
    assert t[0]*v[0] == v[0]*t[0]
    assert t[0]*u[0] != u[0]*t[0]
