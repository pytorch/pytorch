from sympy.holonomic.recurrence import RecurrenceOperators, RecurrenceOperator
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ


def test_RecurrenceOperator():
    n = symbols('n', integer=True)
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    assert Sn*n == (n + 1)*Sn
    assert Sn*n**2 == (n**2+1+2*n)*Sn
    assert Sn**2*n**2 == (n**2 + 4*n + 4)*Sn**2
    p = (Sn**3*n**2 + Sn*n)**2
    q = (n**2 + 3*n + 2)*Sn**2 + (2*n**3 + 19*n**2 + 57*n + 52)*Sn**4 + (n**4 + 18*n**3 + \
        117*n**2 + 324*n + 324)*Sn**6
    assert p == q


def test_RecurrenceOperatorEqPoly():
    n = symbols('n', integer=True)
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    rr = RecurrenceOperator([n**2, 0, 0], R)
    rr2 = RecurrenceOperator([n**2, 1, n], R)
    assert not rr == rr2

    # polynomial comparison issue, see https://github.com/sympy/sympy/pull/15799
    # should work once that is solved
    # d = rr.listofpoly[0]
    # assert rr == d

    d2 = rr2.listofpoly[0]
    assert not rr2 == d2


def test_RecurrenceOperatorPow():
    n = symbols('n', integer=True)
    R, _ = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    rr = RecurrenceOperator([n**2, 0, 0], R)
    a = RecurrenceOperator([R.base.one], R)
    for m in range(10):
        assert a == rr**m
        a *= rr
