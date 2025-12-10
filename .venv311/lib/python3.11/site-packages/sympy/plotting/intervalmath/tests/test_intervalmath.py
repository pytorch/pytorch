from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises


def test_interval():
    assert (interval(1, 1) == interval(1, 1, is_valid=True)) == (True, True)
    assert (interval(1, 1) == interval(1, 1, is_valid=False)) == (True, False)
    assert (interval(1, 1) == interval(1, 1, is_valid=None)) == (True, None)
    assert (interval(1, 1.5) == interval(1, 2)) == (None, True)
    assert (interval(0, 1) == interval(2, 3)) == (False, True)
    assert (interval(0, 1) == interval(1, 2)) == (None, True)
    assert (interval(1, 2) != interval(1, 2)) == (False, True)
    assert (interval(1, 3) != interval(2, 3)) == (None, True)
    assert (interval(1, 3) != interval(-5, -3)) == (True, True)
    assert (
        interval(1, 3, is_valid=False) != interval(-5, -3)) == (True, False)
    assert (interval(1, 3, is_valid=None) != interval(-5, 3)) == (None, None)
    assert (interval(4, 4) != 4) == (False, True)
    assert (interval(1, 1) == 1) == (True, True)
    assert (interval(1, 3, is_valid=False) == interval(1, 3)) == (True, False)
    assert (interval(1, 3, is_valid=None) == interval(1, 3)) == (True, None)
    inter = interval(-5, 5)
    assert (interval(inter) == interval(-5, 5)) == (True, True)
    assert inter.width == 10
    assert 0 in inter
    assert -5 in inter
    assert 5 in inter
    assert interval(0, 3) in inter
    assert interval(-6, 2) not in inter
    assert -5.05 not in inter
    assert 5.3 not in inter
    interb = interval(-float('inf'), float('inf'))
    assert 0 in inter
    assert inter in interb
    assert interval(0, float('inf')) in interb
    assert interval(-float('inf'), 5) in interb
    assert interval(-1e50, 1e50) in interb
    assert (
        -interval(-1, -2, is_valid=False) == interval(1, 2)) == (True, False)
    raises(ValueError, lambda: interval(1, 2, 3))


def test_interval_add():
    assert (interval(1, 2) + interval(2, 3) == interval(3, 5)) == (True, True)
    assert (1 + interval(1, 2) == interval(2, 3)) == (True, True)
    assert (interval(1, 2) + 1 == interval(2, 3)) == (True, True)
    compare = (1 + interval(0, float('inf')) == interval(1, float('inf')))
    assert compare == (True, True)
    a = 1 + interval(2, 5, is_valid=False)
    assert a.is_valid is False
    a = 1 + interval(2, 5, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + interval(3, 5, is_valid=None)
    assert a.is_valid is False
    a = interval(3, 5) + interval(-1, 1, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + 1
    assert a.is_valid is False


def test_interval_sub():
    assert (interval(1, 2) - interval(1, 5) == interval(-4, 1)) == (True, True)
    assert (interval(1, 2) - 1 == interval(0, 1)) == (True, True)
    assert (1 - interval(1, 2) == interval(-1, 0)) == (True, True)
    a = 1 - interval(1, 2, is_valid=False)
    assert a.is_valid is False
    a = interval(1, 4, is_valid=None) - 1
    assert a.is_valid is None
    a = interval(1, 3, is_valid=False) - interval(1, 3)
    assert a.is_valid is False
    a = interval(1, 3, is_valid=None) - interval(1, 3)
    assert a.is_valid is None


def test_interval_inequality():
    assert (interval(1, 2) < interval(3, 4)) == (True, True)
    assert (interval(1, 2) < interval(2, 4)) == (None, True)
    assert (interval(1, 2) < interval(-2, 0)) == (False, True)
    assert (interval(1, 2) <= interval(2, 4)) == (True, True)
    assert (interval(1, 2) <= interval(1.5, 6)) == (None, True)
    assert (interval(2, 3) <= interval(1, 2)) == (None, True)
    assert (interval(2, 3) <= interval(1, 1.5)) == (False, True)
    assert (
        interval(1, 2, is_valid=False) <= interval(-2, 0)) == (False, False)
    assert (interval(1, 2, is_valid=None) <= interval(-2, 0)) == (False, None)
    assert (interval(1, 2) <= 1.5) == (None, True)
    assert (interval(1, 2) <= 3) == (True, True)
    assert (interval(1, 2) <= 0) == (False, True)
    assert (interval(5, 8) > interval(2, 3)) == (True, True)
    assert (interval(2, 5) > interval(1, 3)) == (None, True)
    assert (interval(2, 3) > interval(3.1, 5)) == (False, True)

    assert (interval(-1, 1) == 0) == (None, True)
    assert (interval(-1, 1) == 2) == (False, True)
    assert (interval(-1, 1) != 0) == (None, True)
    assert (interval(-1, 1) != 2) == (True, True)

    assert (interval(3, 5) > 2) == (True, True)
    assert (interval(3, 5) < 2) == (False, True)
    assert (interval(1, 5) < 2) == (None, True)
    assert (interval(1, 5) > 2) == (None, True)
    assert (interval(0, 1) > 2) == (False, True)
    assert (interval(1, 2) >= interval(0, 1)) == (True, True)
    assert (interval(1, 2) >= interval(0, 1.5)) == (None, True)
    assert (interval(1, 2) >= interval(3, 4)) == (False, True)
    assert (interval(1, 2) >= 0) == (True, True)
    assert (interval(1, 2) >= 1.2) == (None, True)
    assert (interval(1, 2) >= 3) == (False, True)
    assert (2 > interval(0, 1)) == (True, True)
    a = interval(-1, 1, is_valid=False) < interval(2, 5, is_valid=None)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) < interval(2, 5, is_valid=False)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) < interval(2, 5, is_valid=None)
    assert a == (True, None)
    a = interval(-1, 1, is_valid=False) > interval(-5, -2, is_valid=None)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) > interval(-5, -2, is_valid=False)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) > interval(-5, -2, is_valid=None)
    assert a == (True, None)


def test_interval_mul():
    assert (
        interval(1, 5) * interval(2, 10) == interval(2, 50)) == (True, True)
    a = interval(-1, 1) * interval(2, 10) == interval(-10, 10)
    assert a == (True, True)

    a = interval(-1, 1) * interval(-5, 3) == interval(-5, 5)
    assert a == (True, True)

    assert (interval(1, 3) * 2 == interval(2, 6)) == (True, True)
    assert (3 * interval(-1, 2) == interval(-3, 6)) == (True, True)

    a = 3 * interval(1, 2, is_valid=False)
    assert a.is_valid is False

    a = 3 * interval(1, 2, is_valid=None)
    assert a.is_valid is None

    a = interval(1, 5, is_valid=False) * interval(1, 2, is_valid=None)
    assert a.is_valid is False


def test_interval_div():
    div = interval(1, 2, is_valid=False) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=False)

    div = interval(1, 2, is_valid=None) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=None)

    div = 3 / interval(1, 2, is_valid=None)
    assert div == interval(-float('inf'), float('inf'), is_valid=None)
    a = interval(1, 2) / 0
    assert a.is_valid is False
    a = interval(0.5, 1) / interval(-1, 0)
    assert a.is_valid is None
    a = interval(0, 1) / interval(0, 1)
    assert a.is_valid is None

    a = interval(-1, 1) / interval(-1, 1)
    assert a.is_valid is None

    a = interval(-1, 2) / interval(0.5, 1) == interval(-2.0, 4.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(0.5, 1) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(0.5, 1) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(0.5, 1) == interval(-1.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(0.5, 1) == interval(0.5, 2.0)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(0.5, 1) == interval(0.5, 8.0)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(0.5, 1) == interval(-2.0, -0.5)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(0.5, 1) == interval(-8.0, -0.5)
    assert a == (True, True)
    a = interval(-1, 2) / interval(-2, -0.5) == interval(-4.0, 2.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(-2, -0.5) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(-2, -0.5) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(-2, -0.5) == interval(0.125, 1.0)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(-2, -0.5) == interval(-2.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(-2, -0.5) == interval(-8.0, -0.25)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(-2, -0.5) == interval(0.25, 2.0)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(-2, -0.5) == interval(0.25, 8.0)
    assert a == (True, True)
    a = interval(-5, 5, is_valid=False) / 2
    assert a.is_valid is False

def test_hashable():
    '''
    test that interval objects are hashable.
    this is required in order to be able to put them into the cache, which
    appears to be necessary for plotting in py3k. For details, see:

    https://github.com/sympy/sympy/pull/2101
    https://github.com/sympy/sympy/issues/6533
    '''
    hash(interval(1, 1))
    hash(interval(1, 1, is_valid=True))
    hash(interval(-4, -0.5))
    hash(interval(-2, -0.5))
    hash(interval(0.25, 8.0))
