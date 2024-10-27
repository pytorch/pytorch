from sympy.core.symbol import Symbol
from sympy.plotting.intervalmath import interval
from sympy.plotting.intervalmath.interval_membership import intervalMembership
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.testing.pytest import raises


def test_creation():
    assert intervalMembership(True, True)
    raises(TypeError, lambda: intervalMembership(True))
    raises(TypeError, lambda: intervalMembership(True, True, True))


def test_getitem():
    a = intervalMembership(True, False)
    assert a[0] is True
    assert a[1] is False
    raises(IndexError, lambda: a[2])


def test_str():
    a = intervalMembership(True, False)
    assert str(a) == 'intervalMembership(True, False)'
    assert repr(a) == 'intervalMembership(True, False)'


def test_equivalence():
    a = intervalMembership(True, True)
    b = intervalMembership(True, False)
    assert (a == b) is False
    assert (a != b) is True

    a = intervalMembership(True, False)
    b = intervalMembership(True, False)
    assert (a == b) is True
    assert (a != b) is False


def test_not():
    x = Symbol('x')

    r1 = x > -1
    r2 = x <= -1

    i = interval

    f1 = experimental_lambdify((x,), r1)
    f2 = experimental_lambdify((x,), r2)

    tt = i(-0.1, 0.1, is_valid=True)
    tn = i(-0.1, 0.1, is_valid=None)
    tf = i(-0.1, 0.1, is_valid=False)

    assert f1(tt) == ~f2(tt)
    assert f1(tn) == ~f2(tn)
    assert f1(tf) == ~f2(tf)

    nt = i(0.9, 1.1, is_valid=True)
    nn = i(0.9, 1.1, is_valid=None)
    nf = i(0.9, 1.1, is_valid=False)

    assert f1(nt) == ~f2(nt)
    assert f1(nn) == ~f2(nn)
    assert f1(nf) == ~f2(nf)

    ft = i(1.9, 2.1, is_valid=True)
    fn = i(1.9, 2.1, is_valid=None)
    ff = i(1.9, 2.1, is_valid=False)

    assert f1(ft) == ~f2(ft)
    assert f1(fn) == ~f2(fn)
    assert f1(ff) == ~f2(ff)


def test_boolean():
    # There can be 9*9 test cases in full mapping of the cartesian product.
    # But we only consider 3*3 cases for simplicity.
    s = [
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]

    # Reduced tests for 'And'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(None, None),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] & s[j] == next(a1_iter)

    # Reduced tests for 'Or'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(None, False),
        intervalMembership(True, False),
        intervalMembership(None, False),
        intervalMembership(None, None),
        intervalMembership(True, None),
        intervalMembership(True, False),
        intervalMembership(True, None),
        intervalMembership(True, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] | s[j] == next(a1_iter)

    # Reduced tests for 'Xor'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(None, False),
        intervalMembership(True, False),
        intervalMembership(None, False),
        intervalMembership(None, None),
        intervalMembership(None, None),
        intervalMembership(True, False),
        intervalMembership(None, None),
        intervalMembership(False, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] ^ s[j] == next(a1_iter)

    # Reduced tests for 'Not'
    a1 = [
        intervalMembership(True, False),
        intervalMembership(None, None),
        intervalMembership(False, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        assert ~s[i] == next(a1_iter)


def test_boolean_errors():
    a = intervalMembership(True, True)
    raises(ValueError, lambda: a & 1)
    raises(ValueError, lambda: a | 1)
    raises(ValueError, lambda: a ^ 1)
