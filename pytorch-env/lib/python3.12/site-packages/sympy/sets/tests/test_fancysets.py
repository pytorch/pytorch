
from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
                                  ComplexRegion)
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
                             Intersection, ProductSet, SetKind)
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod

import itertools


def test_naturals():
    N = S.Naturals
    assert 5 in N
    assert -5 not in N
    assert 5.5 not in N
    ni = iter(N)
    a, b, c, d = next(ni), next(ni), next(ni), next(ni)
    assert (a, b, c, d) == (1, 2, 3, 4)
    assert isinstance(a, Basic)

    assert N.intersect(Interval(-5, 5)) == Range(1, 6)
    assert N.intersect(Interval(-5, 5, True, True)) == Range(1, 5)

    assert N.boundary == N
    assert N.is_open == False
    assert N.is_closed == True

    assert N.inf == 1
    assert N.sup is oo
    assert not N.contains(oo)
    for s in (S.Naturals0, S.Naturals):
        assert s.intersection(S.Reals) is s
        assert s.is_subset(S.Reals)

    assert N.as_relational(x) == And(Eq(floor(x), x), x >= 1, x < oo)


def test_naturals0():
    N = S.Naturals0
    assert 0 in N
    assert -1 not in N
    assert next(iter(N)) == 0
    assert not N.contains(oo)
    assert N.contains(sin(x)) == Contains(sin(x), N)


def test_integers():
    Z = S.Integers
    assert 5 in Z
    assert -5 in Z
    assert 5.5 not in Z
    assert not Z.contains(oo)
    assert not Z.contains(-oo)

    zi = iter(Z)
    a, b, c, d = next(zi), next(zi), next(zi), next(zi)
    assert (a, b, c, d) == (0, 1, -1, 2)
    assert isinstance(a, Basic)

    assert Z.intersect(Interval(-5, 5)) == Range(-5, 6)
    assert Z.intersect(Interval(-5, 5, True, True)) == Range(-4, 5)
    assert Z.intersect(Interval(5, S.Infinity)) == Range(5, S.Infinity)
    assert Z.intersect(Interval.Lopen(5, S.Infinity)) == Range(6, S.Infinity)

    assert Z.inf is -oo
    assert Z.sup is oo

    assert Z.boundary == Z
    assert Z.is_open == False
    assert Z.is_closed == True

    assert Z.as_relational(x) == And(Eq(floor(x), x), -oo < x, x < oo)


def test_ImageSet():
    raises(ValueError, lambda: ImageSet(x, S.Integers))
    assert ImageSet(Lambda(x, 1), S.Integers) == FiniteSet(1)
    assert ImageSet(Lambda(x, y), S.Integers) == {y}
    assert ImageSet(Lambda(x, 1), S.EmptySet) == S.EmptySet
    empty = Intersection(FiniteSet(log(2)/pi), S.Integers)
    assert unchanged(ImageSet, Lambda(x, 1), empty)  # issue #17471
    squares = ImageSet(Lambda(x, x**2), S.Naturals)
    assert 4 in squares
    assert 5 not in squares
    assert FiniteSet(*range(10)).intersect(squares) == FiniteSet(1, 4, 9)

    assert 16 not in squares.intersect(Interval(0, 10))

    si = iter(squares)
    a, b, c, d = next(si), next(si), next(si), next(si)
    assert (a, b, c, d) == (1, 4, 9, 16)

    harmonics = ImageSet(Lambda(x, 1/x), S.Naturals)
    assert Rational(1, 5) in harmonics
    assert Rational(.25) in harmonics
    assert harmonics.contains(.25) == Contains(
        0.25, ImageSet(Lambda(x, 1/x), S.Naturals), evaluate=False)
    assert Rational(.3) not in harmonics
    assert (1, 2) not in harmonics

    assert harmonics.is_iterable

    assert imageset(x, -x, Interval(0, 1)) == Interval(-1, 0)

    assert ImageSet(Lambda(x, x**2), Interval(0, 2)).doit() == Interval(0, 4)
    assert ImageSet(Lambda((x, y), 2*x), {4}, {3}).doit() == FiniteSet(8)
    assert (ImageSet(Lambda((x, y), x+y), {1, 2, 3}, {10, 20, 30}).doit() ==
                FiniteSet(11, 12, 13, 21, 22, 23, 31, 32, 33))

    c = Interval(1, 3) * Interval(1, 3)
    assert Tuple(2, 6) in ImageSet(Lambda(((x, y),), (x, 2*y)), c)
    assert Tuple(2, S.Half) in ImageSet(Lambda(((x, y),), (x, 1/y)), c)
    assert Tuple(2, -2) not in ImageSet(Lambda(((x, y),), (x, y**2)), c)
    assert Tuple(2, -2) in ImageSet(Lambda(((x, y),), (x, -2)), c)
    c3 = ProductSet(Interval(3, 7), Interval(8, 11), Interval(5, 9))
    assert Tuple(8, 3, 9) in ImageSet(Lambda(((t, y, x),), (y, t, x)), c3)
    assert Tuple(Rational(1, 8), 3, 9) in ImageSet(Lambda(((t, y, x),), (1/y, t, x)), c3)
    assert 2/pi not in ImageSet(Lambda(((x, y),), 2/x), c)
    assert 2/S(100) not in ImageSet(Lambda(((x, y),), 2/x), c)
    assert Rational(2, 3) in ImageSet(Lambda(((x, y),), 2/x), c)

    S1 = imageset(lambda x, y: x + y, S.Integers, S.Naturals)
    assert S1.base_pset == ProductSet(S.Integers, S.Naturals)
    assert S1.base_sets == (S.Integers, S.Naturals)

    # Passing a set instead of a FiniteSet shouldn't raise
    assert unchanged(ImageSet, Lambda(x, x**2), {1, 2, 3})

    S2 = ImageSet(Lambda(((x, y),), x+y), {(1, 2), (3, 4)})
    assert 3 in S2.doit()
    # FIXME: This doesn't yet work:
    #assert 3 in S2
    assert S2._contains(3) is None

    raises(TypeError, lambda: ImageSet(Lambda(x, x**2), 1))


def test_image_is_ImageSet():
    assert isinstance(imageset(x, sqrt(sin(x)), Range(5)), ImageSet)


def test_halfcircle():
    r, th = symbols('r, theta', real=True)
    L = Lambda(((r, th),), (r*cos(th), r*sin(th)))
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))

    assert (1, 0) in halfcircle
    assert (0, -1) not in halfcircle
    assert (0, 0) in halfcircle
    assert halfcircle._contains((r, 0)) is None
    assert not halfcircle.is_iterable


@XFAIL
def test_halfcircle_fail():
    r, th = symbols('r, theta', real=True)
    L = Lambda(((r, th),), (r*cos(th), r*sin(th)))
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))
    assert (r, 2*pi) not in halfcircle


def test_ImageSet_iterator_not_injective():
    L = Lambda(x, x - x % 2)  # produces 0, 2, 2, 4, 4, 6, 6, ...
    evens = ImageSet(L, S.Naturals)
    i = iter(evens)
    # No repeats here
    assert (next(i), next(i), next(i), next(i)) == (0, 2, 4, 6)


def test_inf_Range_len():
    raises(ValueError, lambda: len(Range(0, oo, 2)))
    assert Range(0, oo, 2).size is S.Infinity
    assert Range(0, -oo, -2).size is S.Infinity
    assert Range(oo, 0, -2).size is S.Infinity
    assert Range(-oo, 0, 2).size is S.Infinity


def test_Range_set():
    empty = Range(0)

    assert Range(5) == Range(0, 5) == Range(0, 5, 1)

    r = Range(10, 20, 2)
    assert 12 in r
    assert 8 not in r
    assert 11 not in r
    assert 30 not in r

    assert list(Range(0, 5)) == list(range(5))
    assert list(Range(5, 0, -1)) == list(range(5, 0, -1))


    assert Range(5, 15).sup == 14
    assert Range(5, 15).inf == 5
    assert Range(15, 5, -1).sup == 15
    assert Range(15, 5, -1).inf == 6
    assert Range(10, 67, 10).sup == 60
    assert Range(60, 7, -10).inf == 10

    assert len(Range(10, 38, 10)) == 3

    assert Range(0, 0, 5) == empty
    assert Range(oo, oo, 1) == empty
    assert Range(oo, 1, 1) == empty
    assert Range(-oo, 1, -1) == empty
    assert Range(1, oo, -1) == empty
    assert Range(1, -oo, 1) == empty
    assert Range(1, -4, oo) == empty
    ip = symbols('ip', positive=True)
    assert Range(0, ip, -1) == empty
    assert Range(0, -ip, 1) == empty
    assert Range(1, -4, -oo) == Range(1, 2)
    assert Range(1, 4, oo) == Range(1, 2)
    assert Range(-oo, oo).size == oo
    assert Range(oo, -oo, -1).size == oo
    raises(ValueError, lambda: Range(-oo, oo, 2))
    raises(ValueError, lambda: Range(x, pi, y))
    raises(ValueError, lambda: Range(x, y, 0))

    assert 5 in Range(0, oo, 5)
    assert -5 in Range(-oo, 0, 5)
    assert oo not in Range(0, oo)
    ni = symbols('ni', integer=False)
    assert ni not in Range(oo)
    u = symbols('u', integer=None)
    assert Range(oo).contains(u) is not False
    inf = symbols('inf', infinite=True)
    assert inf not in Range(-oo, oo)
    raises(ValueError, lambda: Range(0, oo, 2)[-1])
    raises(ValueError, lambda: Range(0, -oo, -2)[-1])
    assert Range(-oo, 1, 1)[-1] is S.Zero
    assert Range(oo, 1, -1)[-1] == 2
    assert inf not in Range(oo)
    assert Range(1, 10, 1)[-1] == 9
    assert all(i.is_Integer for i in Range(0, -1, 1))
    it = iter(Range(-oo, 0, 2))
    raises(TypeError, lambda: next(it))

    assert empty.intersect(S.Integers) == empty
    assert Range(-1, 10, 1).intersect(S.Complexes) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Reals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Rationals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Integers) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals) == Range(1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals0) == Range(0, 10, 1)

    # test slicing
    assert Range(1, 10, 1)[5] == 6
    assert Range(1, 12, 2)[5] == 11
    assert Range(1, 10, 1)[-1] == 9
    assert Range(1, 10, 3)[-1] == 7
    raises(ValueError, lambda: Range(oo,0,-1)[1:3:0])
    raises(ValueError, lambda: Range(oo,0,-1)[:1])
    raises(ValueError, lambda: Range(1, oo)[-2])
    raises(ValueError, lambda: Range(-oo, 1)[2])
    raises(IndexError, lambda: Range(10)[-20])
    raises(IndexError, lambda: Range(10)[20])
    raises(ValueError, lambda: Range(2, -oo, -2)[2:2:0])
    assert Range(2, -oo, -2)[2:2:2] == empty
    assert Range(2, -oo, -2)[:2:2] == Range(2, -2, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:2])
    assert Range(-oo, 4, 2)[::-2] == Range(2, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[::2])
    assert Range(oo, 2, -2)[::] == Range(oo, 2, -2)
    assert Range(-oo, 4, 2)[:-2:-2] == Range(2, 0, -4)
    assert Range(-oo, 4, 2)[:-2:2] == Range(-oo, 0, 4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:-2])
    assert Range(-oo, 4, 2)[-2::-2] == Range(0, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[-2:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[0::2])
    assert Range(oo, 2, -2)[0::] == Range(oo, 2, -2)
    raises(ValueError, lambda: Range(-oo, 4, 2)[0:-2:2])
    assert Range(oo, 2, -2)[0:-2:] == Range(oo, 6, -2)
    raises(ValueError, lambda: Range(oo, 2, -2)[0:2:])
    raises(ValueError, lambda: Range(-oo, 4, 2)[2::-1])
    assert Range(-oo, 4, 2)[-2::2] == Range(0, 4, 4)
    assert Range(oo, 0, -2)[-10:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[0])
    raises(ValueError, lambda: Range(oo, 0, -2)[-10:10:2])
    raises(ValueError, lambda: Range(oo, 0, -2)[0::-2])
    assert Range(oo, 0, -2)[0:-4:-2] == empty
    assert Range(oo, 0, -2)[:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[:1:-1])

    # test empty Range
    assert Range(x, x, y) == empty
    assert empty.reversed == empty
    assert 0 not in empty
    assert list(empty) == []
    assert len(empty) == 0
    assert empty.size is S.Zero
    assert empty.intersect(FiniteSet(0)) is S.EmptySet
    assert bool(empty) is False
    raises(IndexError, lambda: empty[0])
    assert empty[:0] == empty
    raises(NotImplementedError, lambda: empty.inf)
    raises(NotImplementedError, lambda: empty.sup)
    assert empty.as_relational(x) is S.false

    AB = [None] + list(range(12))
    for R in [
            Range(1, 10),
            Range(1, 10, 2),
        ]:
        r = list(R)
        for a, b, c in itertools.product(AB, AB, [-3, -1, None, 1, 3]):
            for reverse in range(2):
                r = list(reversed(r))
                R = R.reversed
                result = list(R[a:b:c])
                ans = r[a:b:c]
                txt = ('\n%s[%s:%s:%s] = %s -> %s' % (
                R, a, b, c, result, ans))
                check = ans == result
                assert check, txt

    assert Range(1, 10, 1).boundary == Range(1, 10, 1)

    for r in (Range(1, 10, 2), Range(1, oo, 2)):
        rev = r.reversed
        assert r.inf == rev.inf and r.sup == rev.sup
        assert r.step == -rev.step

    builtin_range = range

    raises(TypeError, lambda: Range(builtin_range(1)))
    assert S(builtin_range(10)) == Range(10)
    assert S(builtin_range(1000000000000)) == Range(1000000000000)

    # test Range.as_relational
    assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3)  & Eq(Mod(x, 1), 0)
    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo)  & Eq(Mod(x + 1, -2), 0)


def test_Range_symbolic():
    # symbolic Range
    xr = Range(x, x + 4, 5)
    sr = Range(x, y, t)
    i = Symbol('i', integer=True)
    ip = Symbol('i', integer=True, positive=True)
    ipr = Range(ip)
    inr = Range(0, -ip, -1)
    ir = Range(i, i + 19, 2)
    ir2 = Range(i, i*8, 3*i)
    i = Symbol('i', integer=True)
    inf = symbols('inf', infinite=True)
    raises(ValueError, lambda: Range(inf))
    raises(ValueError, lambda: Range(inf, 0, -1))
    raises(ValueError, lambda: Range(inf, inf, 1))
    raises(ValueError, lambda: Range(1, 1, inf))
    # args
    assert xr.args == (x, x + 5, 5)
    assert sr.args == (x, y, t)
    assert ir.args == (i, i + 20, 2)
    assert ir2.args == (i, 10*i, 3*i)
    # reversed
    raises(ValueError, lambda: xr.reversed)
    raises(ValueError, lambda: sr.reversed)
    assert ipr.reversed.args == (ip - 1, -1, -1)
    assert inr.reversed.args == (-ip + 1, 1, 1)
    assert ir.reversed.args == (i + 18, i - 2, -2)
    assert ir2.reversed.args == (7*i, -2*i, -3*i)
    # contains
    assert inf not in sr
    assert inf not in ir
    assert 0 in ipr
    assert 0 in inr
    raises(TypeError, lambda: 1 in ipr)
    raises(TypeError, lambda: -1 in inr)
    assert .1 not in sr
    assert .1 not in ir
    assert i + 1 not in ir
    assert i + 2 in ir
    raises(TypeError, lambda: x in xr)  # XXX is this what contains is supposed to do?
    raises(TypeError, lambda: 1 in sr)  # XXX is this what contains is supposed to do?
    # iter
    raises(ValueError, lambda: next(iter(xr)))
    raises(ValueError, lambda: next(iter(sr)))
    assert next(iter(ir)) == i
    assert next(iter(ir2)) == i
    assert sr.intersect(S.Integers) == sr
    assert sr.intersect(FiniteSet(x)) == Intersection({x}, sr)
    raises(ValueError, lambda: sr[:2])
    raises(ValueError, lambda: xr[0])
    raises(ValueError, lambda: sr[0])
    # len
    assert len(ir) == ir.size == 10
    assert len(ir2) == ir2.size == 3
    raises(ValueError, lambda: len(xr))
    raises(ValueError, lambda: xr.size)
    raises(ValueError, lambda: len(sr))
    raises(ValueError, lambda: sr.size)
    # bool
    assert bool(Range(0)) == False
    assert bool(xr)
    assert bool(ir)
    assert bool(ipr)
    assert bool(inr)
    raises(ValueError, lambda: bool(sr))
    raises(ValueError, lambda: bool(ir2))
    # inf
    raises(ValueError, lambda: xr.inf)
    raises(ValueError, lambda: sr.inf)
    assert ipr.inf == 0
    assert inr.inf == -ip + 1
    assert ir.inf == i
    raises(ValueError, lambda: ir2.inf)
    # sup
    raises(ValueError, lambda: xr.sup)
    raises(ValueError, lambda: sr.sup)
    assert ipr.sup == ip - 1
    assert inr.sup == 0
    assert ir.inf == i
    raises(ValueError, lambda: ir2.sup)
    # getitem
    raises(ValueError, lambda: xr[0])
    raises(ValueError, lambda: sr[0])
    raises(ValueError, lambda: sr[-1])
    raises(ValueError, lambda: sr[:2])
    assert ir[:2] == Range(i, i + 4, 2)
    assert ir[0] == i
    assert ir[-2] == i + 16
    assert ir[-1] == i + 18
    assert ir2[:2] == Range(i, 7*i, 3*i)
    assert ir2[0] == i
    assert ir2[-2] == 4*i
    assert ir2[-1] == 7*i
    raises(ValueError, lambda: Range(i)[-1])
    assert ipr[0] == ipr.inf == 0
    assert ipr[-1] == ipr.sup == ip - 1
    assert inr[0] == inr.sup == 0
    assert inr[-1] == inr.inf == -ip + 1
    raises(ValueError, lambda: ipr[-2])
    assert ir.inf == i
    assert ir.sup == i + 18
    raises(ValueError, lambda: Range(i).inf)
    # as_relational
    assert ir.as_relational(x) == ((x >= i) & (x <= i + 18) &
        Eq(Mod(-i + x, 2), 0))
    assert ir2.as_relational(x) == Eq(
        Mod(-i + x, 3*i), 0) & (((x >= i) & (x <= 7*i) & (3*i >= 1)) |
        ((x <= i) & (x >= 7*i) & (3*i <= -1)))
    assert Range(i, i + 1).as_relational(x) == Eq(x, i)
    assert sr.as_relational(z) == Eq(
        Mod(t, 1), 0) & Eq(Mod(x, 1), 0) & Eq(Mod(-x + z, t), 0
        ) & (((z >= x) & (z <= -t + y) & (t >= 1)) |
        ((z <= x) & (z >= -t + y) & (t <= -1)))
    assert xr.as_relational(z) == Eq(z, x) & Eq(Mod(x, 1), 0)
    # symbols can clash if user wants (but it must be integer)
    assert xr.as_relational(x) == Eq(Mod(x, 1), 0)
    # contains() for symbolic values (issue #18146)
    e = Symbol('e', integer=True, even=True)
    o = Symbol('o', integer=True, odd=True)
    assert Range(5).contains(i) == And(i >= 0, i <= 4)
    assert Range(1).contains(i) == Eq(i, 0)
    assert Range(-oo, 5, 1).contains(i) == (i <= 4)
    assert Range(-oo, oo).contains(i) == True
    assert Range(0, 8, 2).contains(i) == Contains(i, Range(0, 8, 2))
    assert Range(0, 8, 2).contains(e) == And(e >= 0, e <= 6)
    assert Range(0, 8, 2).contains(2*i) == And(2*i >= 0, 2*i <= 6)
    assert Range(0, 8, 2).contains(o) == False
    assert Range(1, 9, 2).contains(e) == False
    assert Range(1, 9, 2).contains(o) == And(o >= 1, o <= 7)
    assert Range(8, 0, -2).contains(o) == False
    assert Range(9, 1, -2).contains(o) == And(o >= 3, o <= 9)
    assert Range(-oo, 8, 2).contains(i) == Contains(i, Range(-oo, 8, 2))


def test_range_range_intersection():
    for a, b, r in [
            (Range(0), Range(1), S.EmptySet),
            (Range(3), Range(4, oo), S.EmptySet),
            (Range(3), Range(-3, -1), S.EmptySet),
            (Range(1, 3), Range(0, 3), Range(1, 3)),
            (Range(1, 3), Range(1, 4), Range(1, 3)),
            (Range(1, oo, 2), Range(2, oo, 2), S.EmptySet),
            (Range(0, oo, 2), Range(oo), Range(0, oo, 2)),
            (Range(0, oo, 2), Range(100), Range(0, 100, 2)),
            (Range(2, oo, 2), Range(oo), Range(2, oo, 2)),
            (Range(0, oo, 2), Range(5, 6), S.EmptySet),
            (Range(2, 80, 1), Range(55, 71, 4), Range(55, 71, 4)),
            (Range(0, 6, 3), Range(-oo, 5, 3), S.EmptySet),
            (Range(0, oo, 2), Range(5, oo, 3), Range(8, oo, 6)),
            (Range(4, 6, 2), Range(2, 16, 7), S.EmptySet),]:
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r
        a, b = b, a
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r


def test_range_interval_intersection():
    p = symbols('p', positive=True)
    assert isinstance(Range(3).intersect(Interval(p, p + 2)), Intersection)
    assert Range(4).intersect(Interval(0, 3)) == Range(4)
    assert Range(4).intersect(Interval(-oo, oo)) == Range(4)
    assert Range(4).intersect(Interval(1, oo)) == Range(1, 4)
    assert Range(4).intersect(Interval(1.1, oo)) == Range(2, 4)
    assert Range(4).intersect(Interval(0.1, 3)) == Range(1, 4)
    assert Range(4).intersect(Interval(0.1, 3.1)) == Range(1, 4)
    assert Range(4).intersect(Interval.open(0, 3)) == Range(1, 3)
    assert Range(4).intersect(Interval.open(0.1, 0.5)) is S.EmptySet
    assert Interval(-1, 5).intersect(S.Complexes) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Reals) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Integers) == Range(-1, 6)
    assert Interval(-1, 5).intersect(S.Naturals) == Range(1, 6)
    assert Interval(-1, 5).intersect(S.Naturals0) == Range(0, 6)

    # Null Range intersections
    assert Range(0).intersect(Interval(0.2, 0.8)) is S.EmptySet
    assert Range(0).intersect(Interval(-oo, oo)) is S.EmptySet


def test_range_is_finite_set():
    assert Range(-100, 100).is_finite_set is True
    assert Range(2, oo).is_finite_set is False
    assert Range(-oo, 50).is_finite_set is False
    assert Range(-oo, oo).is_finite_set is False
    assert Range(oo, -oo).is_finite_set is True
    assert Range(0, 0).is_finite_set is True
    assert Range(oo, oo).is_finite_set is True
    assert Range(-oo, -oo).is_finite_set is True
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    assert Range(n, n + 49).is_finite_set is True
    assert Range(n, 0).is_finite_set is True
    assert Range(-3, n + 7).is_finite_set is True
    assert Range(n, m).is_finite_set is True
    assert Range(n + m, m - n).is_finite_set is True
    assert Range(n, n + m + n).is_finite_set is True
    assert Range(n, oo).is_finite_set is False
    assert Range(-oo, n).is_finite_set is False
    assert Range(n, -oo).is_finite_set is True
    assert Range(oo, n).is_finite_set is True


def test_Range_is_iterable():
    assert Range(-100, 100).is_iterable is True
    assert Range(2, oo).is_iterable is False
    assert Range(-oo, 50).is_iterable is False
    assert Range(-oo, oo).is_iterable is False
    assert Range(oo, -oo).is_iterable is True
    assert Range(0, 0).is_iterable is True
    assert Range(oo, oo).is_iterable is True
    assert Range(-oo, -oo).is_iterable is True
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    p = Symbol('p', integer=True, positive=True)
    assert Range(n, n + 49).is_iterable is True
    assert Range(n, 0).is_iterable is False
    assert Range(-3, n + 7).is_iterable is False
    assert Range(-3, p + 7).is_iterable is False # Should work with better __iter__
    assert Range(n, m).is_iterable is False
    assert Range(n + m, m - n).is_iterable is False
    assert Range(n, n + m + n).is_iterable is False
    assert Range(n, oo).is_iterable is False
    assert Range(-oo, n).is_iterable is False
    x = Symbol('x')
    assert Range(x, x + 49).is_iterable is False
    assert Range(x, 0).is_iterable is False
    assert Range(-3, x + 7).is_iterable is False
    assert Range(x, m).is_iterable is False
    assert Range(x + m, m - x).is_iterable is False
    assert Range(x, x + m + x).is_iterable is False
    assert Range(x, oo).is_iterable is False
    assert Range(-oo, x).is_iterable is False


def test_Integers_eval_imageset():
    ans = ImageSet(Lambda(x, 2*x + Rational(3, 7)), S.Integers)
    im = imageset(Lambda(x, -2*x + Rational(3, 7)), S.Integers)
    assert im == ans
    im = imageset(Lambda(x, -2*x - Rational(11, 7)), S.Integers)
    assert im == ans
    y = Symbol('y')
    L = imageset(x, 2*x + y, S.Integers)
    assert y + 4 in L
    a, b, c = 0.092, 0.433, 0.341
    assert a in imageset(x, a + c*x, S.Integers)
    assert b in imageset(x, b + c*x, S.Integers)

    _x = symbols('x', negative=True)
    eq = _x**2 - _x + 1
    assert imageset(_x, eq, S.Integers).lamda.expr == _x**2 + _x + 1
    eq = 3*_x - 1
    assert imageset(_x, eq, S.Integers).lamda.expr == 3*_x + 2

    assert imageset(x, (x, 1/x), S.Integers) == \
        ImageSet(Lambda(x, (x, 1/x)), S.Integers)


def test_Range_eval_imageset():
    a, b, c = symbols('a b c')
    assert imageset(x, a*(x + b) + c, Range(3)) == \
        imageset(x, a*x + a*b + c, Range(3))
    eq = (x + 1)**2
    assert imageset(x, eq, Range(3)).lamda.expr == eq
    eq = a*(x + b) + c
    r = Range(3, -3, -2)
    imset = imageset(x, eq, r)
    assert imset.lamda.expr != eq
    assert list(imset) == [eq.subs(x, i).expand() for i in list(r)]


def test_fun():
    assert (FiniteSet(*ImageSet(Lambda(x, sin(pi*x/4)),
        Range(-10, 11))) == FiniteSet(-1, -sqrt(2)/2, 0, sqrt(2)/2, 1))


def test_Range_is_empty():
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)

    assert Range(0).is_empty
    assert not Range(1).is_empty
    assert Range(1, 0).is_empty
    assert not Range(-1, 0).is_empty
    assert Range(i).is_empty is None
    assert Range(n).is_empty
    assert Range(p).is_empty is False
    assert Range(n, 0).is_empty is False
    assert Range(n, p).is_empty is False
    assert Range(p, n).is_empty
    assert Range(n, -1).is_empty is None
    assert Range(p, n, -1).is_empty is False


def test_Reals():
    assert 5 in S.Reals
    assert S.Pi in S.Reals
    assert -sqrt(2) in S.Reals
    assert (2, 5) not in S.Reals
    assert sqrt(-1) not in S.Reals
    assert S.Reals == Interval(-oo, oo)
    assert S.Reals != Interval(0, oo)
    assert S.Reals.is_subset(Interval(-oo, oo))
    assert S.Reals.intersect(Range(-oo, oo)) == Range(-oo, oo)
    assert S.ComplexInfinity not in S.Reals
    assert S.NaN not in S.Reals
    assert x + S.ComplexInfinity not in S.Reals


def test_Complex():
    assert 5 in S.Complexes
    assert 5 + 4*I in S.Complexes
    assert S.Pi in S.Complexes
    assert -sqrt(2) in S.Complexes
    assert -I in S.Complexes
    assert sqrt(-1) in S.Complexes
    assert S.Complexes.intersect(S.Reals) == S.Reals
    assert S.Complexes.union(S.Reals) == S.Complexes
    assert S.Complexes == ComplexRegion(S.Reals*S.Reals)
    assert (S.Complexes == ComplexRegion(Interval(1, 2)*Interval(3, 4))) == False
    assert str(S.Complexes) == "Complexes"
    assert repr(S.Complexes) == "Complexes"


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def test_intersections():
    assert S.Integers.intersect(S.Reals) == S.Integers
    assert 5 in S.Integers.intersect(S.Reals)
    assert 5 in S.Integers.intersect(S.Reals)
    assert -5 not in S.Naturals.intersect(S.Reals)
    assert 5.5 not in S.Integers.intersect(S.Reals)
    assert 5 in S.Integers.intersect(Interval(3, oo))
    assert -5 in S.Integers.intersect(Interval(-oo, 3))
    assert all(x.is_Integer
            for x in take(10, S.Integers.intersect(Interval(3, oo)) ))


def test_infinitely_indexed_set_1():
    from sympy.abc import n, m
    assert imageset(Lambda(n, n), S.Integers) == imageset(Lambda(m, m), S.Integers)

    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(m, 2*m + 1), S.Integers)) is S.EmptySet

    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(n, 2*n + 1), S.Integers)) is S.EmptySet

    assert imageset(Lambda(m, 2*m), S.Integers).intersect(
                imageset(Lambda(n, 3*n), S.Integers)).dummy_eq(
            ImageSet(Lambda(t, 6*t), S.Integers))

    assert imageset(x, x/2 + Rational(1, 3), S.Integers).intersect(S.Integers) is S.EmptySet
    assert imageset(x, x/2 + S.Half, S.Integers).intersect(S.Integers) is S.Integers

    # https://github.com/sympy/sympy/issues/17355
    S53 = ImageSet(Lambda(n, 5*n + 3), S.Integers)
    assert S53.intersect(S.Integers) == S53


def test_infinitely_indexed_set_2():
    from sympy.abc import n
    a = Symbol('a', integer=True)
    assert imageset(Lambda(n, n), S.Integers) == \
        imageset(Lambda(n, n + a), S.Integers)
    assert imageset(Lambda(n, n + pi), S.Integers) == \
        imageset(Lambda(n, n + a + pi), S.Integers)
    assert imageset(Lambda(n, n), S.Integers) == \
        imageset(Lambda(n, -n + a), S.Integers)
    assert imageset(Lambda(n, -6*n), S.Integers) == \
        ImageSet(Lambda(n, 6*n), S.Integers)
    assert imageset(Lambda(n, 2*n + pi), S.Integers) == \
        ImageSet(Lambda(n, 2*n + pi - 2), S.Integers)


def test_imageset_intersect_real():
    from sympy.abc import n
    assert imageset(Lambda(n, n + (n - 1)*(n + 1)*I), S.Integers).intersect(S.Reals) == FiniteSet(-1, 1)
    im = (n - 1)*(n + S.Half)
    assert imageset(Lambda(n, n + im*I), S.Integers
        ).intersect(S.Reals) == FiniteSet(1)
    assert imageset(Lambda(n, n + im*(n + 1)*I), S.Naturals0
        ).intersect(S.Reals) == FiniteSet(1)
    assert imageset(Lambda(n, n/2 + im.expand()*I), S.Integers
        ).intersect(S.Reals) == ImageSet(Lambda(x, x/2), ConditionSet(
        n, Eq(n**2 - n/2 - S(1)/2, 0), S.Integers))
    assert imageset(Lambda(n, n/(1/n - 1) + im*(n + 1)*I), S.Integers
        ).intersect(S.Reals) == FiniteSet(S.Half)
    assert imageset(Lambda(n, n/(n - 6) +
        (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(
        S.Reals) == FiniteSet(-1)
    assert imageset(Lambda(n, n/(n**2 - 9) +
        (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(
        S.Reals) is S.EmptySet
    s = ImageSet(
        Lambda(n, -I*(I*(2*pi*n - pi/4) + log(Abs(sqrt(-I))))),
        S.Integers)
    # s is unevaluated, but after intersection the result
    # should be canonical
    assert s.intersect(S.Reals) == imageset(
        Lambda(n, 2*n*pi - pi/4), S.Integers) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


def test_imageset_intersect_interval():
    from sympy.abc import n
    f1 = ImageSet(Lambda(n, n*pi), S.Integers)
    f2 = ImageSet(Lambda(n, 2*n), Interval(0, pi))
    f3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    # complex expressions
    f4 = ImageSet(Lambda(n, n*I*pi), S.Integers)
    f5 = ImageSet(Lambda(n, 2*I*n*pi + pi/2), S.Integers)
    # non-linear expressions
    f6 = ImageSet(Lambda(n, log(n)), S.Integers)
    f7 = ImageSet(Lambda(n, n**2), S.Integers)
    f8 = ImageSet(Lambda(n, Abs(n)), S.Integers)
    f9 = ImageSet(Lambda(n, exp(n)), S.Naturals0)

    assert f1.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f1.intersect(Interval(0, 2*pi, False, True)) == FiniteSet(0, pi)
    assert f2.intersect(Interval(1, 2)) == Interval(1, 2)
    assert f3.intersect(Interval(-1, 1)) == S.EmptySet
    assert f3.intersect(Interval(-5, 5)) == FiniteSet(pi*Rational(-3, 2), pi/2)
    assert f4.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f4.intersect(Interval(1, 2)) == S.EmptySet
    assert f5.intersect(Interval(0, 1)) == S.EmptySet
    assert f6.intersect(Interval(0, 1)) == FiniteSet(S.Zero, log(2))
    assert f7.intersect(Interval(0, 10)) == Intersection(f7, Interval(0, 10))
    assert f8.intersect(Interval(0, 2)) == Intersection(f8, Interval(0, 2))
    assert f9.intersect(Interval(1, 2)) == Intersection(f9, Interval(1, 2))


def test_imageset_intersect_diophantine():
    from sympy.abc import m, n
    # Check that same lambda variable for both ImageSets is handled correctly
    img1 = ImageSet(Lambda(n, 2*n + 1), S.Integers)
    img2 = ImageSet(Lambda(n, 4*n + 1), S.Integers)
    assert img1.intersect(img2) == img2
    # Empty solution set returned by diophantine:
    assert ImageSet(Lambda(n, 2*n), S.Integers).intersect(
            ImageSet(Lambda(n, 2*n + 1), S.Integers)) == S.EmptySet
    # Check intersection with S.Integers:
    assert ImageSet(Lambda(n, 9/n + 20*n/3), S.Integers).intersect(
            S.Integers) == FiniteSet(-61, -23, 23, 61)
    # Single solution (2, 3) for diophantine solution:
    assert ImageSet(Lambda(n, (n - 2)**2), S.Integers).intersect(
            ImageSet(Lambda(n, -(n - 3)**2), S.Integers)) == FiniteSet(0)
    # Single parametric solution for diophantine solution:
    assert ImageSet(Lambda(n, n**2 + 5), S.Integers).intersect(
            ImageSet(Lambda(m, 2*m), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 4*n**2 + 4*n + 6), S.Integers))
    # 4 non-parametric solution couples for dioph. equation:
    assert ImageSet(Lambda(n, n**2 - 9), S.Integers).intersect(
            ImageSet(Lambda(m, -m**2), S.Integers)) == FiniteSet(-9, 0)
    # Double parametric solution for diophantine solution:
    assert ImageSet(Lambda(m, m**2 + 40), S.Integers).intersect(
            ImageSet(Lambda(n, 41*n), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(m, m**2 + 40), S.Integers),
            ImageSet(Lambda(n, 41*n), S.Integers)))
    # Check that diophantine returns *all* (8) solutions (permute=True)
    assert ImageSet(Lambda(n, n**4 - 2**4), S.Integers).intersect(
            ImageSet(Lambda(m, -m**4 + 3**4), S.Integers)) == FiniteSet(0, 65)
    assert ImageSet(Lambda(n, pi/12 + n*5*pi/12), S.Integers).intersect(
            ImageSet(Lambda(n, 7*pi/12 + n*11*pi/12), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 55*pi*n/12 + 17*pi/4), S.Integers))
    # TypeError raised by diophantine (#18081)
    assert ImageSet(Lambda(n, n*log(2)), S.Integers).intersection(
        S.Integers).dummy_eq(Intersection(ImageSet(
        Lambda(n, n*log(2)), S.Integers), S.Integers))
    # NotImplementedError raised by diophantine (no solver for cubic_thue)
    assert ImageSet(Lambda(n, n**3 + 1), S.Integers).intersect(
            ImageSet(Lambda(n, n**3), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(n, n**3 + 1), S.Integers),
            ImageSet(Lambda(n, n**3), S.Integers)))


def test_infinitely_indexed_set_3():
    from sympy.abc import n, m
    assert imageset(Lambda(m, 2*pi*m), S.Integers).intersect(
            imageset(Lambda(n, 3*pi*n), S.Integers)).dummy_eq(
        ImageSet(Lambda(t, 6*pi*t), S.Integers))
    assert imageset(Lambda(n, 2*n + 1), S.Integers) == \
        imageset(Lambda(n, 2*n - 1), S.Integers)
    assert imageset(Lambda(n, 3*n + 2), S.Integers) == \
        imageset(Lambda(n, 3*n - 1), S.Integers)


def test_ImageSet_simplification():
    from sympy.abc import n, m
    assert imageset(Lambda(n, n), S.Integers) == S.Integers
    assert imageset(Lambda(n, sin(n)),
                    imageset(Lambda(m, tan(m)), S.Integers)) == \
            imageset(Lambda(m, sin(tan(m))), S.Integers)
    assert imageset(n, 1 + 2*n, S.Naturals) == Range(3, oo, 2)
    assert imageset(n, 1 + 2*n, S.Naturals0) == Range(1, oo, 2)
    assert imageset(n, 1 - 2*n, S.Naturals) == Range(-1, -oo, -2)


def test_ImageSet_contains():
    assert (2, S.Half) in imageset(x, (x, 1/x), S.Integers)
    assert imageset(x, x + I*3, S.Integers).intersection(S.Reals) is S.EmptySet
    i = Dummy(integer=True)
    q = imageset(x, x + I*y, S.Integers).intersection(S.Reals)
    assert q.subs(y, I*i).intersection(S.Integers) is S.Integers
    q = imageset(x, x + I*y/x, S.Integers).intersection(S.Reals)
    assert q.subs(y, 0) is S.Integers
    assert q.subs(y, I*i*x).intersection(S.Integers) is S.Integers
    z = cos(1)**2 + sin(1)**2 - 1
    q = imageset(x, x + I*z, S.Integers).intersection(S.Reals)
    assert q is not S.EmptySet


def test_ComplexRegion_contains():
    r = Symbol('r', real=True)
    # contains in ComplexRegion
    a = Interval(2, 3)
    b = Interval(4, 6)
    c = Interval(7, 9)
    c1 = ComplexRegion(a*b)
    c2 = ComplexRegion(Union(a*b, c*a))
    assert 2.5 + 4.5*I in c1
    assert 2 + 4*I in c1
    assert 3 + 4*I in c1
    assert 8 + 2.5*I in c2
    assert 2.5 + 6.1*I not in c1
    assert 4.5 + 3.2*I not in c1
    assert c1.contains(x) == Contains(x, c1, evaluate=False)
    assert c1.contains(r) == False
    assert c2.contains(x) == Contains(x, c2, evaluate=False)
    assert c2.contains(r) == False

    r1 = Interval(0, 1)
    theta1 = Interval(0, 2*S.Pi)
    c3 = ComplexRegion(r1*theta1, polar=True)
    assert (0.5 + I*6/10) in c3
    assert (S.Half + I*6/10) in c3
    assert (S.Half + .6*I) in c3
    assert (0.5 + .6*I) in c3
    assert I in c3
    assert 1 in c3
    assert 0 in c3
    assert 1 + I not in c3
    assert 1 - I not in c3
    assert c3.contains(x) == Contains(x, c3, evaluate=False)
    assert c3.contains(r + 2*I) == Contains(
        r + 2*I, c3, evaluate=False)  # is in fact False
    assert c3.contains(1/(1 + r**2)) == Contains(
        1/(1 + r**2), c3, evaluate=False)  # is in fact True

    r2 = Interval(0, 3)
    theta2 = Interval(pi, 2*pi, left_open=True)
    c4 = ComplexRegion(r2*theta2, polar=True)
    assert c4.contains(0) == True
    assert c4.contains(2 + I) == False
    assert c4.contains(-2 + I) == False
    assert c4.contains(-2 - I) == True
    assert c4.contains(2 - I) == True
    assert c4.contains(-2) == False
    assert c4.contains(2) == True
    assert c4.contains(x) == Contains(x, c4, evaluate=False)
    assert c4.contains(3/(1 + r**2)) == Contains(
        3/(1 + r**2), c4, evaluate=False)  # is in fact True

    raises(ValueError, lambda: ComplexRegion(r1*theta1, polar=2))


def test_symbolic_Range():
    n = Symbol('n')
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    raises(ValueError, lambda: Range(n, n+1)[0])
    raises(ValueError, lambda: Range(n).size)

    n = Symbol('n', integer=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n, n+1)[0] == n
    raises(ValueError, lambda: Range(n).size)
    assert Range(n, n+1).size == 1

    n = Symbol('n', integer=True, nonnegative=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n+1)[0] == 0
    assert Range(n, n+1)[0] == n
    assert Range(n).size == n
    assert Range(n+1).size == n+1
    assert Range(n, n+1).size == 1

    n = Symbol('n', integer=True, positive=True)
    assert Range(n)[0] == 0
    assert Range(n, n+1)[0] == n
    assert Range(n).size == n
    assert Range(n, n+1).size == 1

    m = Symbol('m', integer=True, positive=True)

    assert Range(n, n+m)[0] == n
    assert Range(n, n+m).size == m
    assert Range(n, n+1).size == 1
    assert Range(n, n+m, 2).size == floor(m/2)

    m = Symbol('m', integer=True, positive=True, even=True)
    assert Range(n, n+m, 2).size == m/2


def test_issue_18400():
    n = Symbol('n', integer=True)
    raises(ValueError, lambda: imageset(lambda x: x*2, Range(n)))

    n = Symbol('n', integer=True, positive=True)
    # No exception
    assert imageset(lambda x: x*2, Range(n)) == imageset(lambda x: x*2, Range(n))


def test_ComplexRegion_intersect():
    # Polar form
    X_axis = ComplexRegion(Interval(0, oo)*FiniteSet(0, S.Pi), polar=True)

    unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    upper_half_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    lower_half_disk = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)
    right_half_disk = ComplexRegion(Interval(0, oo)*Interval(-S.Pi/2, S.Pi/2), polar=True)
    first_quad_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi/2), polar=True)

    assert upper_half_disk.intersect(unit_disk) == upper_half_unit_disk
    assert right_half_disk.intersect(first_quad_disk) == first_quad_disk
    assert upper_half_disk.intersect(right_half_disk) == first_quad_disk
    assert upper_half_disk.intersect(lower_half_disk) == X_axis

    c1 = ComplexRegion(Interval(0, 4)*Interval(0, 2*S.Pi), polar=True)
    assert c1.intersect(Interval(1, 5)) == Interval(1, 4)
    assert c1.intersect(Interval(4, 9)) == FiniteSet(4)
    assert c1.intersect(Interval(5, 12)) is S.EmptySet

    # Rectangular form
    X_axis = ComplexRegion(Interval(-oo, oo)*FiniteSet(0))

    unit_square = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    upper_half_unit_square = ComplexRegion(Interval(-1, 1)*Interval(0, 1))
    upper_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(0, oo))
    lower_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(-oo, 0))
    right_half_plane = ComplexRegion(Interval(0, oo)*Interval(-oo, oo))
    first_quad_plane = ComplexRegion(Interval(0, oo)*Interval(0, oo))

    assert upper_half_plane.intersect(unit_square) == upper_half_unit_square
    assert right_half_plane.intersect(first_quad_plane) == first_quad_plane
    assert upper_half_plane.intersect(right_half_plane) == first_quad_plane
    assert upper_half_plane.intersect(lower_half_plane) == X_axis

    c1 = ComplexRegion(Interval(-5, 5)*Interval(-10, 10))
    assert c1.intersect(Interval(2, 7)) == Interval(2, 5)
    assert c1.intersect(Interval(5, 7)) == FiniteSet(5)
    assert c1.intersect(Interval(6, 9)) is S.EmptySet

    # unevaluated object
    C1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    C2 = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    assert C1.intersect(C2) == Intersection(C1, C2, evaluate=False)


def test_ComplexRegion_union():
    # Polar form
    c1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    c2 = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    c3 = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    c4 = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)

    p1 = Union(Interval(0, 1)*Interval(0, 2*S.Pi), Interval(0, 1)*Interval(0, S.Pi))
    p2 = Union(Interval(0, oo)*Interval(0, S.Pi), Interval(0, oo)*Interval(S.Pi, 2*S.Pi))

    assert c1.union(c2) == ComplexRegion(p1, polar=True)
    assert c3.union(c4) == ComplexRegion(p2, polar=True)

    # Rectangular form
    c5 = ComplexRegion(Interval(2, 5)*Interval(6, 9))
    c6 = ComplexRegion(Interval(4, 6)*Interval(10, 12))
    c7 = ComplexRegion(Interval(0, 10)*Interval(-10, 0))
    c8 = ComplexRegion(Interval(12, 16)*Interval(14, 20))

    p3 = Union(Interval(2, 5)*Interval(6, 9), Interval(4, 6)*Interval(10, 12))
    p4 = Union(Interval(0, 10)*Interval(-10, 0), Interval(12, 16)*Interval(14, 20))

    assert c5.union(c6) == ComplexRegion(p3)
    assert c7.union(c8) == ComplexRegion(p4)

    assert c1.union(Interval(2, 4)) == Union(c1, Interval(2, 4), evaluate=False)
    assert c5.union(Interval(2, 4)) == Union(c5, ComplexRegion.from_real(Interval(2, 4)))


def test_ComplexRegion_from_real():
    c1 = ComplexRegion(Interval(0, 1) * Interval(0, 2 * S.Pi), polar=True)

    raises(ValueError, lambda: c1.from_real(c1))
    assert c1.from_real(Interval(-1, 1)) == ComplexRegion(Interval(-1, 1) * FiniteSet(0), False)


def test_ComplexRegion_measure():
    a, b = Interval(2, 5), Interval(4, 8)
    theta1, theta2 = Interval(0, 2*S.Pi), Interval(0, S.Pi)
    c1 = ComplexRegion(a*b)
    c2 = ComplexRegion(Union(a*theta1, b*theta2), polar=True)

    assert c1.measure == 12
    assert c2.measure == 9*pi


def test_normalize_theta_set():
    # Interval
    assert normalize_theta_set(Interval(pi, 2*pi)) == \
        Union(FiniteSet(0), Interval.Ropen(pi, 2*pi))
    assert normalize_theta_set(Interval(pi*Rational(9, 2), 5*pi)) == Interval(pi/2, pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), pi/2)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval.open(pi*Rational(-3, 2), pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval.open(pi*Rational(-7, 2), pi*Rational(-3, 2))) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-4*pi, 3*pi)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), -pi/2)) == Interval(pi/2, pi*Rational(3, 2))
    assert normalize_theta_set(Interval.open(0, 2*pi)) == Interval.open(0, 2*pi)
    assert normalize_theta_set(Interval.Ropen(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.Lopen(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(4*pi, pi*Rational(9, 2))) == Interval.open(0, pi/2)
    assert normalize_theta_set(Interval.Lopen(4*pi, pi*Rational(9, 2))) == Interval.Lopen(0, pi/2)
    assert normalize_theta_set(Interval.Ropen(4*pi, pi*Rational(9, 2))) == Interval.Ropen(0, pi/2)
    assert normalize_theta_set(Interval.open(3*pi, 5*pi)) == \
        Union(Interval.Ropen(0, pi), Interval.open(pi, 2*pi))

    # FiniteSet
    assert normalize_theta_set(FiniteSet(0, pi, 3*pi)) == FiniteSet(0, pi)
    assert normalize_theta_set(FiniteSet(0, pi/2, pi, 2*pi)) == FiniteSet(0, pi/2, pi)
    assert normalize_theta_set(FiniteSet(0, -pi/2, -pi, -2*pi)) == FiniteSet(0, pi, pi*Rational(3, 2))
    assert normalize_theta_set(FiniteSet(pi*Rational(-3, 2), pi/2)) == \
        FiniteSet(pi/2)
    assert normalize_theta_set(FiniteSet(2*pi)) == FiniteSet(0)

    # Unions
    assert normalize_theta_set(Union(Interval(0, pi/3), Interval(pi/2, pi))) == \
        Union(Interval(0, pi/3), Interval(pi/2, pi))
    assert normalize_theta_set(Union(Interval(0, pi), Interval(2*pi, pi*Rational(7, 3)))) == \
        Interval(0, pi)

    # ValueError for non-real sets
    raises(ValueError, lambda: normalize_theta_set(S.Complexes))

    # NotImplementedError for subset of reals
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(0, 1)))

    # NotImplementedError without pi as coefficient
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(1, 2*pi)))
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(2*pi, 10)))
    raises(NotImplementedError, lambda: normalize_theta_set(FiniteSet(0, 3, 3*pi)))


def test_ComplexRegion_FiniteSet():
    x, y, z, a, b, c = symbols('x y z a b c')

    # Issue #9669
    assert ComplexRegion(FiniteSet(a, b, c)*FiniteSet(x, y, z)) == \
        FiniteSet(a + I*x, a + I*y, a + I*z, b + I*x, b + I*y,
                  b + I*z, c + I*x, c + I*y, c + I*z)
    assert ComplexRegion(FiniteSet(2)*FiniteSet(3)) == FiniteSet(2 + 3*I)


def test_union_RealSubSet():
    assert (S.Complexes).union(Interval(1, 2)) == S.Complexes
    assert (S.Complexes).union(S.Integers) == S.Complexes


def test_SetKind_fancySet():
    G = lambda *args: ImageSet(Lambda(x, x ** 2), *args)
    assert G(Interval(1, 4)).kind is SetKind(NumberKind)
    assert G(FiniteSet(1, 4)).kind is SetKind(NumberKind)
    assert S.Rationals.kind is SetKind(NumberKind)
    assert S.Naturals.kind is SetKind(NumberKind)
    assert S.Integers.kind is SetKind(NumberKind)
    assert Range(3).kind is SetKind(NumberKind)
    a = Interval(2, 3)
    b = Interval(4, 6)
    c1 = ComplexRegion(a*b)
    assert c1.kind is SetKind(TupleKind(NumberKind, NumberKind))


def test_issue_9980():
    c1 = ComplexRegion(Interval(1, 2)*Interval(2, 3))
    c2 = ComplexRegion(Interval(1, 5)*Interval(1, 3))
    R = Union(c1, c2)
    assert simplify(R) == ComplexRegion(Union(Interval(1, 2)*Interval(2, 3), \
                                    Interval(1, 5)*Interval(1, 3)), False)
    assert c1.func(*c1.args) == c1
    assert R.func(*R.args) == R


def test_issue_11732():
    interval12 = Interval(1, 2)
    finiteset1234 = FiniteSet(1, 2, 3, 4)
    pointComplex = Tuple(1, 5)

    assert (interval12 in S.Naturals) == False
    assert (interval12 in S.Naturals0) == False
    assert (interval12 in S.Integers) == False
    assert (interval12 in S.Complexes) == False

    assert (finiteset1234 in S.Naturals) == False
    assert (finiteset1234 in S.Naturals0) == False
    assert (finiteset1234 in S.Integers) == False
    assert (finiteset1234 in S.Complexes) == False

    assert (pointComplex in S.Naturals) == False
    assert (pointComplex in S.Naturals0) == False
    assert (pointComplex in S.Integers) == False
    assert (pointComplex in S.Complexes) == True


def test_issue_11730():
    unit = Interval(0, 1)
    square = ComplexRegion(unit ** 2)

    assert Union(S.Complexes, FiniteSet(oo)) != S.Complexes
    assert Union(S.Complexes, FiniteSet(eye(4))) != S.Complexes
    assert Union(unit, square) == square
    assert Intersection(S.Reals, square) == unit


def test_issue_11938():
    unit = Interval(0, 1)
    ival = Interval(1, 2)
    cr1 = ComplexRegion(ival * unit)

    assert Intersection(cr1, S.Reals) == ival
    assert Intersection(cr1, unit) == FiniteSet(1)

    arg1 = Interval(0, S.Pi)
    arg2 = FiniteSet(S.Pi)
    arg3 = Interval(S.Pi / 4, 3 * S.Pi / 4)
    cp1 = ComplexRegion(unit * arg1, polar=True)
    cp2 = ComplexRegion(unit * arg2, polar=True)
    cp3 = ComplexRegion(unit * arg3, polar=True)

    assert Intersection(cp1, S.Reals) == Interval(-1, 1)
    assert Intersection(cp2, S.Reals) == Interval(-1, 0)
    assert Intersection(cp3, S.Reals) == FiniteSet(0)


def test_issue_11914():
    a, b = Interval(0, 1), Interval(0, pi)
    c, d = Interval(2, 3), Interval(pi, 3 * pi / 2)
    cp1 = ComplexRegion(a * b, polar=True)
    cp2 = ComplexRegion(c * d, polar=True)

    assert -3 in cp1.union(cp2)
    assert -3 in cp2.union(cp1)
    assert -5 not in cp1.union(cp2)


def test_issue_9543():
    assert ImageSet(Lambda(x, x**2), S.Naturals).is_subset(S.Reals)


def test_issue_16871():
    assert ImageSet(Lambda(x, x), FiniteSet(1)) == {1}
    assert ImageSet(Lambda(x, x - 3), S.Integers
        ).intersection(S.Integers) is S.Integers


@XFAIL
def test_issue_16871b():
    assert ImageSet(Lambda(x, x - 3), S.Integers).is_subset(S.Integers)


def test_issue_18050():
    assert imageset(Lambda(x, I*x + 1), S.Integers
        ) == ImageSet(Lambda(x, I*x + 1), S.Integers)
    assert imageset(Lambda(x, 3*I*x + 4 + 8*I), S.Integers
        ) == ImageSet(Lambda(x, 3*I*x + 4 + 2*I), S.Integers)
    # no 'Mod' for next 2 tests:
    assert imageset(Lambda(x, 2*x + 3*I), S.Integers
        ) == ImageSet(Lambda(x, 2*x + 3*I), S.Integers)
    r = Symbol('r', positive=True)
    assert imageset(Lambda(x, r*x + 10), S.Integers
        ) == ImageSet(Lambda(x, r*x + 10), S.Integers)
    # reduce real part:
    assert imageset(Lambda(x, 3*x + 8 + 5*I), S.Integers
        ) == ImageSet(Lambda(x, 3*x + 2 + 5*I), S.Integers)


def test_Rationals():
    assert S.Integers.is_subset(S.Rationals)
    assert S.Naturals.is_subset(S.Rationals)
    assert S.Naturals0.is_subset(S.Rationals)
    assert S.Rationals.is_subset(S.Reals)
    assert S.Rationals.inf is -oo
    assert S.Rationals.sup is oo
    it = iter(S.Rationals)
    assert [next(it) for i in range(12)] == [
        0, 1, -1, S.Half, 2, Rational(-1, 2), -2,
        Rational(1, 3), 3, Rational(-1, 3), -3, Rational(2, 3)]
    assert Basic() not in S.Rationals
    assert S.Half in S.Rationals
    assert S.Rationals.contains(0.5) == Contains(
        0.5, S.Rationals, evaluate=False)
    assert 2 in S.Rationals
    r = symbols('r', rational=True)
    assert r in S.Rationals
    raises(TypeError, lambda: x in S.Rationals)
    # issue #18134:
    assert S.Rationals.boundary == S.Reals
    assert S.Rationals.closure == S.Reals
    assert S.Rationals.is_open == False
    assert S.Rationals.is_closed == False


def test_NZQRC_unions():
    # check that all trivial number set unions are simplified:
    nbrsets = (S.Naturals, S.Naturals0, S.Integers, S.Rationals,
        S.Reals, S.Complexes)
    unions = (Union(a, b) for a in nbrsets for b in nbrsets)
    assert all(u.is_Union is False for u in unions)


def test_imageset_intersection():
    n = Dummy()
    s = ImageSet(Lambda(n, -I*(I*(2*pi*n - pi/4) +
        log(Abs(sqrt(-I))))), S.Integers)
    assert s.intersect(S.Reals) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


def test_issue_17858():
    assert 1 in Range(-oo, oo)
    assert 0 in Range(oo, -oo, -1)
    assert oo not in Range(-oo, oo)
    assert -oo not in Range(-oo, oo)

def test_issue_17859():
    r = Range(-oo,oo)
    raises(ValueError,lambda: r[::2])
    raises(ValueError, lambda: r[::-2])
    r = Range(oo,-oo,-1)
    raises(ValueError,lambda: r[::2])
    raises(ValueError, lambda: r[::-2])
