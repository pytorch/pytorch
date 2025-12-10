from mpmath import *

def test_interval_identity():
    iv.dps = 15
    assert mpi(2) == mpi(2, 2)
    assert mpi(2) != mpi(-2, 2)
    assert not (mpi(2) != mpi(2, 2))
    assert mpi(-1, 1) == mpi(-1, 1)
    assert str(mpi('0.1')) == "[0.099999999999999991673, 0.10000000000000000555]"
    assert repr(mpi('0.1')) == "mpi('0.099999999999999992', '0.10000000000000001')"
    u = mpi(-1, 3)
    assert -1 in u
    assert 2 in u
    assert 3 in u
    assert -1.1 not in u
    assert 3.1 not in u
    assert mpi(-1, 3) in u
    assert mpi(0, 1) in u
    assert mpi(-1.1, 2) not in u
    assert mpi(2.5, 3.1) not in u
    w = mpi(-inf, inf)
    assert mpi(-5, 5) in w
    assert mpi(2, inf) in w
    assert mpi(0, 2) in mpi(0, 10)
    assert not (3 in mpi(-inf, 0))

def test_interval_hash():
    assert hash(mpi(3)) == hash(3)
    assert hash(mpi(3.25)) == hash(3.25)
    assert hash(mpi(3,4)) == hash(mpi(3,4))
    assert hash(iv.mpc(3)) == hash(3)
    assert hash(iv.mpc(3,4)) == hash(3+4j)
    assert hash(iv.mpc((1,3),(2,4))) == hash(iv.mpc((1,3),(2,4)))

def test_interval_arithmetic():
    iv.dps = 15
    assert mpi(2) + mpi(3,4) == mpi(5,6)
    assert mpi(1, 2)**2 == mpi(1, 4)
    assert mpi(1) + mpi(0, 1e-50) == mpi(1, mpf('1.0000000000000002'))
    x = 1 / (1 / mpi(3))
    assert x.a < 3 < x.b
    x = mpi(2) ** mpi(0.5)
    iv.dps += 5
    sq = iv.sqrt(2)
    iv.dps -= 5
    assert x.a < sq < x.b
    assert mpi(1) / mpi(1, inf)
    assert mpi(2, 3) / inf == mpi(0, 0)
    assert mpi(0) / inf == 0
    assert mpi(0) / 0 == mpi(-inf, inf)
    assert mpi(inf) / 0 == mpi(-inf, inf)
    assert mpi(0) * inf == mpi(-inf, inf)
    assert 1 / mpi(2, inf) == mpi(0, 0.5)
    assert str((mpi(50, 50) * mpi(-10, -10)) / 3) == \
        '[-166.66666666666668561, -166.66666666666665719]'
    assert mpi(0, 4) ** 3 == mpi(0, 64)
    assert mpi(2,4).mid == 3
    iv.dps = 30
    a = mpi(iv.pi)
    iv.dps = 15
    b = +a
    assert b.a < a.a
    assert b.b > a.b
    a = mpi(iv.pi)
    assert a == +a
    assert abs(mpi(-1,2)) == mpi(0,2)
    assert abs(mpi(0.5,2)) == mpi(0.5,2)
    assert abs(mpi(-3,2)) == mpi(0,3)
    assert abs(mpi(-3,-0.5)) == mpi(0.5,3)
    assert mpi(0) * mpi(2,3) == mpi(0)
    assert mpi(2,3) * mpi(0) == mpi(0)
    assert mpi(1,3).delta == 2
    assert mpi(1,2) - mpi(3,4) == mpi(-3,-1)
    assert mpi(-inf,0) - mpi(0,inf) == mpi(-inf,0)
    assert mpi(-inf,0) - mpi(-inf,inf) == mpi(-inf,inf)
    assert mpi(0,inf) - mpi(-inf,1) == mpi(-1,inf)

def test_interval_mul():
    assert mpi(-1, 0) * inf == mpi(-inf, 0)
    assert mpi(-1, 0) * -inf == mpi(0, inf)
    assert mpi(0, 1) * inf == mpi(0, inf)
    assert mpi(0, 1) * mpi(0, inf) == mpi(0, inf)
    assert mpi(-1, 1) * inf == mpi(-inf, inf)
    assert mpi(-1, 1) * mpi(0, inf) == mpi(-inf, inf)
    assert mpi(-1, 1) * mpi(-inf, inf) == mpi(-inf, inf)
    assert mpi(-inf, 0) * mpi(0, 1) == mpi(-inf, 0)
    assert mpi(-inf, 0) * mpi(0, 0) * mpi(-inf, 0)
    assert mpi(-inf, 0) * mpi(-inf, inf) == mpi(-inf, inf)
    assert mpi(-5,0)*mpi(-32,28) == mpi(-140,160)
    assert mpi(2,3) * mpi(-1,2) == mpi(-3,6)
    # Should be undefined?
    assert mpi(inf, inf) * 0 == mpi(-inf, inf)
    assert mpi(-inf, -inf) * 0 == mpi(-inf, inf)
    assert mpi(0) * mpi(-inf,2) == mpi(-inf,inf)
    assert mpi(0) * mpi(-2,inf) == mpi(-inf,inf)
    assert mpi(-2,inf) * mpi(0) == mpi(-inf,inf)
    assert mpi(-inf,2) * mpi(0) == mpi(-inf,inf)

def test_interval_pow():
    assert mpi(3)**2 == mpi(9, 9)
    assert mpi(-3)**2 == mpi(9, 9)
    assert mpi(-3, 1)**2 == mpi(0, 9)
    assert mpi(-3, -1)**2 == mpi(1, 9)
    assert mpi(-3, -1)**3 == mpi(-27, -1)
    assert mpi(-3, 1)**3 == mpi(-27, 1)
    assert mpi(-2, 3)**2 == mpi(0, 9)
    assert mpi(-3, 2)**2 == mpi(0, 9)
    assert mpi(4) ** -1 == mpi(0.25, 0.25)
    assert mpi(-4) ** -1 == mpi(-0.25, -0.25)
    assert mpi(4) ** -2 == mpi(0.0625, 0.0625)
    assert mpi(-4) ** -2 == mpi(0.0625, 0.0625)
    assert mpi(0, 1) ** inf == mpi(0, 1)
    assert mpi(0, 1) ** -inf == mpi(1, inf)
    assert mpi(0, inf) ** inf == mpi(0, inf)
    assert mpi(0, inf) ** -inf == mpi(0, inf)
    assert mpi(1, inf) ** inf == mpi(1, inf)
    assert mpi(1, inf) ** -inf == mpi(0, 1)
    assert mpi(2, 3) ** 1 == mpi(2, 3)
    assert mpi(2, 3) ** 0 == 1
    assert mpi(1,3) ** mpi(2) == mpi(1,9)

def test_interval_sqrt():
    assert mpi(4) ** 0.5 == mpi(2)

def test_interval_div():
    assert mpi(0.5, 1) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(0, 1) / mpi(0, 1) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(2, 2) == mpi(inf, inf)
    assert mpi(0, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(0, inf) / mpi(2, 2) == mpi(0, inf)
    assert mpi(2, inf) / mpi(2, 2) == mpi(1, inf)
    assert mpi(2, inf) / mpi(2, inf) == mpi(0, inf)
    assert mpi(-4, 8) / mpi(1, inf) == mpi(-4, 8)
    assert mpi(-4, 8) / mpi(0.5, inf) == mpi(-8, 16)
    assert mpi(-inf, 8) / mpi(0.5, inf) == mpi(-inf, 16)
    assert mpi(-inf, inf) / mpi(0.5, inf) == mpi(-inf, inf)
    assert mpi(8, inf) / mpi(0.5, inf) == mpi(0, inf)
    assert mpi(-8, inf) / mpi(0.5, inf) == mpi(-16, inf)
    assert mpi(-4, 8) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(0, 8) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(0, 0) / mpi(inf, inf) == mpi(0, 0)
    assert mpi(-inf, 0) / mpi(inf, inf) == mpi(-inf, 0)
    assert mpi(-inf, 8) / mpi(inf, inf) == mpi(-inf, 0)
    assert mpi(-inf, inf) / mpi(inf, inf) == mpi(-inf, inf)
    assert mpi(-8, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(0, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(8, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(inf, inf) / mpi(inf, inf) == mpi(0, inf)
    assert mpi(-1, 2) / mpi(0, 1) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(0, 1) == mpi(0.0, +inf)
    assert mpi(-1, 0) / mpi(0, 1) == mpi(-inf, 0.0)
    assert mpi(-0.5, -0.25) / mpi(0, 1) == mpi(-inf, -0.25)
    assert mpi(0.5, 1) / mpi(0, 1) == mpi(0.5, +inf)
    assert mpi(0.5, 4) / mpi(0, 1) == mpi(0.5, +inf)
    assert mpi(-1, -0.5) / mpi(0, 1) == mpi(-inf, -0.5)
    assert mpi(-4, -0.5) / mpi(0, 1) == mpi(-inf, -0.5)
    assert mpi(-1, 2) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, 0) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-0.5, -0.25) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0.5, 1) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(0.5, 4) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, -0.5) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-4, -0.5) / mpi(-2, 0.5) == mpi(-inf, +inf)
    assert mpi(-1, 2) / mpi(-1, 0) == mpi(-inf, +inf)
    assert mpi(0, 1) / mpi(-1, 0) == mpi(-inf, 0.0)
    assert mpi(-1, 0) / mpi(-1, 0) == mpi(0.0, +inf)
    assert mpi(-0.5, -0.25) / mpi(-1, 0) == mpi(0.25, +inf)
    assert mpi(0.5, 1) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(0.5, 4) / mpi(-1, 0) == mpi(-inf, -0.5)
    assert mpi(-1, -0.5) / mpi(-1, 0) == mpi(0.5, +inf)
    assert mpi(-4, -0.5) / mpi(-1, 0) == mpi(0.5, +inf)
    assert mpi(-1, 2) / mpi(0.5, 1) == mpi(-2.0, 4.0)
    assert mpi(0, 1) / mpi(0.5, 1) == mpi(0.0, 2.0)
    assert mpi(-1, 0) / mpi(0.5, 1) == mpi(-2.0, 0.0)
    assert mpi(-0.5, -0.25) / mpi(0.5, 1) == mpi(-1.0, -0.25)
    assert mpi(0.5, 1) / mpi(0.5, 1) == mpi(0.5, 2.0)
    assert mpi(0.5, 4) / mpi(0.5, 1) == mpi(0.5, 8.0)
    assert mpi(-1, -0.5) / mpi(0.5, 1) == mpi(-2.0, -0.5)
    assert mpi(-4, -0.5) / mpi(0.5, 1) == mpi(-8.0, -0.5)
    assert mpi(-1, 2) / mpi(-2, -0.5) == mpi(-4.0, 2.0)
    assert mpi(0, 1) / mpi(-2, -0.5) == mpi(-2.0, 0.0)
    assert mpi(-1, 0) / mpi(-2, -0.5) == mpi(0.0, 2.0)
    assert mpi(-0.5, -0.25) / mpi(-2, -0.5) == mpi(0.125, 1.0)
    assert mpi(0.5, 1) / mpi(-2, -0.5) == mpi(-2.0, -0.25)
    assert mpi(0.5, 4) / mpi(-2, -0.5) == mpi(-8.0, -0.25)
    assert mpi(-1, -0.5) / mpi(-2, -0.5) == mpi(0.25, 2.0)
    assert mpi(-4, -0.5) / mpi(-2, -0.5) == mpi(0.25, 8.0)
    # Should be undefined?
    assert mpi(0, 0) / mpi(0, 0) == mpi(-inf, inf)
    assert mpi(0, 0) / mpi(0, 1) == mpi(-inf, inf)

def test_interval_cos_sin():
    iv.dps = 15
    cos = iv.cos
    sin = iv.sin
    tan = iv.tan
    pi = iv.pi
    # Around 0
    assert cos(mpi(0)) == 1
    assert sin(mpi(0)) == 0
    assert cos(mpi(0,1)) == mpi(0.54030230586813965399, 1.0)
    assert sin(mpi(0,1)) == mpi(0, 0.8414709848078966159)
    assert cos(mpi(1,2)) == mpi(-0.4161468365471424069, 0.54030230586813976501)
    assert sin(mpi(1,2)) == mpi(0.84147098480789650488, 1.0)
    assert sin(mpi(1,2.5)) == mpi(0.59847214410395643824, 1.0)
    assert cos(mpi(-1, 1)) == mpi(0.54030230586813965399, 1.0)
    assert cos(mpi(-1, 0.5)) == mpi(0.54030230586813965399, 1.0)
    assert cos(mpi(-1, 1.5)) == mpi(0.070737201667702906405, 1.0)
    assert sin(mpi(-1,1)) == mpi(-0.8414709848078966159, 0.8414709848078966159)
    assert sin(mpi(-1,0.5)) == mpi(-0.8414709848078966159, 0.47942553860420300538)
    assert mpi(-0.8414709848078966159, 1.00000000000000002e-100) in sin(mpi(-1,1e-100))
    assert mpi(-2.00000000000000004e-100, 1.00000000000000002e-100) in sin(mpi(-2e-100,1e-100))
    # Same interval
    assert cos(mpi(2, 2.5))
    assert cos(mpi(3.5, 4)) == mpi(-0.93645668729079634129, -0.65364362086361182946)
    assert cos(mpi(5, 5.5)) == mpi(0.28366218546322624627, 0.70866977429126010168)
    assert mpi(0.59847214410395654927, 0.90929742682568170942) in sin(mpi(2, 2.5))
    assert sin(mpi(3.5, 4)) == mpi(-0.75680249530792831347, -0.35078322768961983646)
    assert sin(mpi(5, 5.5)) == mpi(-0.95892427466313856499, -0.70554032557039181306)
    # Higher roots
    iv.dps = 55
    w = 4*10**50 + mpi(0.5)
    for p in [15, 40, 80]:
        iv.dps = p
        assert 0 in sin(4*mpi(pi))
        assert 0 in sin(4*10**50*mpi(pi))
        assert 0 in cos((4+0.5)*mpi(pi))
        assert 0 in cos(w*mpi(pi))
        assert 1 in cos(4*mpi(pi))
        assert 1 in cos(4*10**50*mpi(pi))
    iv.dps = 15
    assert cos(mpi(2,inf)) == mpi(-1,1)
    assert sin(mpi(2,inf)) == mpi(-1,1)
    assert cos(mpi(-inf,2)) == mpi(-1,1)
    assert sin(mpi(-inf,2)) == mpi(-1,1)
    u = tan(mpi(0.5,1))
    assert mpf(u.a).ae(mp.tan(0.5))
    assert mpf(u.b).ae(mp.tan(1))
    v = iv.cot(mpi(0.5,1))
    assert mpf(v.a).ae(mp.cot(1))
    assert mpf(v.b).ae(mp.cot(0.5))
    # Sanity check of evaluation at n*pi and (n+1/2)*pi
    for n in range(-5,7,2):
        x = iv.cos(n*iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
        x = iv.sin((n+0.5)*iv.pi)
        assert -1 in x
        assert x >= -1
        assert x != -1
    for n in range(-6,8,2):
        x = iv.cos(n*iv.pi)
        assert 1 in x
        assert x <= 1
        if n:
            assert x != 1
        x = iv.sin((n+0.5)*iv.pi)
        assert 1 in x
        assert x <= 1
        assert x != 1
    for n in range(-6,7):
        x = iv.cos((n+0.5)*iv.pi)
        assert x.a < 0 < x.b
        x = iv.sin(n*iv.pi)
        if n:
            assert x.a < 0 < x.b

def test_interval_complex():
    # TODO: many more tests
    iv.dps = 15
    mp.dps = 15
    assert iv.mpc(2,3) == 2+3j
    assert iv.mpc(2,3) != 2+4j
    assert iv.mpc(2,3) != 1+3j
    assert 1+3j in iv.mpc([1,2],[3,4])
    assert 2+5j not in iv.mpc([1,2],[3,4])
    assert iv.mpc(1,2) + 1j == 1+3j
    assert iv.mpc([1,2],[2,3]) + 2+3j == iv.mpc([3,4],[5,6])
    assert iv.mpc([2,4],[4,8]) / 2 == iv.mpc([1,2],[2,4])
    assert iv.mpc([1,2],[2,4]) * 2j == iv.mpc([-8,-4],[2,4])
    assert iv.mpc([2,4],[4,8]) / 2j == iv.mpc([2,4],[-2,-1])
    assert iv.exp(2+3j).ae(mp.exp(2+3j))
    assert iv.log(2+3j).ae(mp.log(2+3j))
    assert (iv.mpc(2,3) ** iv.mpc(0.5,2)).ae(mp.mpc(2,3) ** mp.mpc(0.5,2))
    assert 1j in (iv.mpf(-1) ** 0.5)
    assert 1j in (iv.mpc(-1) ** 0.5)
    assert abs(iv.mpc(0)) == 0
    assert abs(iv.mpc(inf)) == inf
    assert abs(iv.mpc(3,4)) == 5
    assert abs(iv.mpc(4)) == 4
    assert abs(iv.mpc(0,4)) == 4
    assert abs(iv.mpc(0,[2,3])) == iv.mpf([2,3])
    assert abs(iv.mpc(0,[-3,2])) == iv.mpf([0,3])
    assert abs(iv.mpc([3,5],[4,12])) == iv.mpf([5,13])
    assert abs(iv.mpc([3,5],[-4,12])) == iv.mpf([3,13])
    assert iv.mpc(2,3) ** 0 == 1
    assert iv.mpc(2,3) ** 1 == (2+3j)
    assert iv.mpc(2,3) ** 2 == (2+3j)**2
    assert iv.mpc(2,3) ** 3 == (2+3j)**3
    assert iv.mpc(2,3) ** 4 == (2+3j)**4
    assert iv.mpc(2,3) ** 5 == (2+3j)**5
    assert iv.mpc(2,2) ** (-1) == (2+2j) ** (-1)
    assert iv.mpc(2,2) ** (-2) == (2+2j) ** (-2)
    assert iv.cos(2).ae(mp.cos(2))
    assert iv.sin(2).ae(mp.sin(2))
    assert iv.cos(2+3j).ae(mp.cos(2+3j))
    assert iv.sin(2+3j).ae(mp.sin(2+3j))

def test_interval_complex_arg():
    mp.dps = 15
    iv.dps = 15
    assert iv.arg(3) == 0
    assert iv.arg(0) == 0
    assert iv.arg([0,3]) == 0
    assert iv.arg(-3).ae(pi)
    assert iv.arg(2+3j).ae(iv.arg(2+3j))
    z = iv.mpc([-2,-1],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-1+4j))
    assert t.b.ae(mp.arg(-2+3j))
    z = iv.mpc([-2,1],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1+3j))
    assert t.b.ae(mp.arg(-2+3j))
    z = iv.mpc([1,2],[3,4])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(2+3j))
    assert t.b.ae(mp.arg(1+4j))
    z = iv.mpc([1,2],[-2,3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1-2j))
    assert t.b.ae(mp.arg(1+3j))
    z = iv.mpc([1,2],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(1-4j))
    assert t.b.ae(mp.arg(2-3j))
    z = iv.mpc([-1,2],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-1-3j))
    assert t.b.ae(mp.arg(2-3j))
    z = iv.mpc([-2,-1],[-4,-3])
    t = iv.arg(z)
    assert t.a.ae(mp.arg(-2-3j))
    assert t.b.ae(mp.arg(-1-4j))
    z = iv.mpc([-2,-1],[-3,3])
    t = iv.arg(z)
    assert t.a.ae(-mp.pi)
    assert t.b.ae(mp.pi)
    z = iv.mpc([-2,2],[-3,3])
    t = iv.arg(z)
    assert t.a.ae(-mp.pi)
    assert t.b.ae(mp.pi)

def test_interval_ae():
    iv.dps = 15
    x = iv.mpf([1,2])
    assert x.ae(1) is None
    assert x.ae(1.5) is None
    assert x.ae(2) is None
    assert x.ae(2.01) is False
    assert x.ae(0.99) is False
    x = iv.mpf(3.5)
    assert x.ae(3.5) is True
    assert x.ae(3.5+1e-15) is True
    assert x.ae(3.5-1e-15) is True
    assert x.ae(3.501) is False
    assert x.ae(3.499) is False
    assert x.ae(iv.mpf([3.5,3.501])) is None
    assert x.ae(iv.mpf([3.5,4.5+1e-15])) is None

def test_interval_nstr():
    iv.dps = n = 30
    x = mpi(1, 2)
    # FIXME: error_dps should not be necessary
    assert iv.nstr(x, n, mode='plusminus', error_dps=6) == '1.5 +- 0.5'
    assert iv.nstr(x, n, mode='plusminus', use_spaces=False, error_dps=6) == '1.5+-0.5'
    assert iv.nstr(x, n, mode='percent') == '1.5 (33.33%)'
    assert iv.nstr(x, n, mode='brackets', use_spaces=False) == '[1.0,2.0]'
    assert iv.nstr(x, n, mode='brackets' , brackets=('<', '>')) == '<1.0, 2.0>'
    x = mpi('5.2582327113062393041', '5.2582327113062749951')
    assert iv.nstr(x, n, mode='diff') == '5.2582327113062[393041, 749951]'
    assert iv.nstr(iv.cos(mpi(1)), n, mode='diff', use_spaces=False) == '0.54030230586813971740093660744[2955,3053]'
    assert iv.nstr(mpi('1e123', '1e129'), n, mode='diff') == '[1.0e+123, 1.0e+129]'
    exp = iv.exp
    assert iv.nstr(iv.exp(mpi('5000.1')), n, mode='diff') == '3.2797365856787867069110487[0926, 1191]e+2171'
    iv.dps = 15

def test_mpi_from_str():
    iv.dps = 15
    assert iv.convert('1.5 +- 0.5') == mpi(mpf('1.0'), mpf('2.0'))
    assert mpi(1, 2) in iv.convert('1.5 (33.33333333333333333333333333333%)')
    assert iv.convert('[1, 2]') == mpi(1, 2)
    assert iv.convert('1[2, 3]') == mpi(12, 13)
    assert iv.convert('1.[23,46]e-8') == mpi('1.23e-8', '1.46e-8')
    assert iv.convert('12[3.4,5.9]e4') == mpi('123.4e+4', '125.9e4')

def test_interval_gamma():
    mp.dps = 15
    iv.dps = 15
    # TODO: need many more tests
    assert iv.rgamma(0) == 0
    assert iv.fac(0) == 1
    assert iv.fac(1) == 1
    assert iv.fac(2) == 2
    assert iv.fac(3) == 6
    assert iv.gamma(0) == [-inf,inf]
    assert iv.gamma(1) == 1
    assert iv.gamma(2) == 1
    assert iv.gamma(3) == 2
    assert -3.5449077018110320546 in iv.gamma(-0.5)
    assert iv.loggamma(1) == 0
    assert iv.loggamma(2) == 0
    assert 0.69314718055994530942 in iv.loggamma(3)
    # Test tight log-gamma endpoints based on monotonicity
    xs = [iv.mpc([2,3],[1,4]),
          iv.mpc([2,3],[-4,-1]),
          iv.mpc([2,3],[-1,4]),
          iv.mpc([2,3],[-4,1]),
          iv.mpc([2,3],[-4,4]),
          iv.mpc([-3,-2],[2,4]),
          iv.mpc([-3,-2],[-4,-2])]
    for x in xs:
        ys = [mp.loggamma(mp.mpc(x.a,x.c)),
              mp.loggamma(mp.mpc(x.b,x.c)),
              mp.loggamma(mp.mpc(x.a,x.d)),
              mp.loggamma(mp.mpc(x.b,x.d))]
        if 0 in x.imag:
            ys += [mp.loggamma(x.a), mp.loggamma(x.b)]
        min_real = min([y.real for y in ys])
        max_real = max([y.real for y in ys])
        min_imag = min([y.imag for y in ys])
        max_imag = max([y.imag for y in ys])
        z = iv.loggamma(x)
        assert z.a.ae(min_real)
        assert z.b.ae(max_real)
        assert z.c.ae(min_imag)
        assert z.d.ae(max_imag)

def test_interval_conversions():
    mp.dps = 15
    iv.dps = 15
    for a, b in ((-0.0, 0), (0.0, 0.5), (1.0, 1), \
                 ('-inf', 20.5), ('-inf', float(sqrt(2)))):
        r = mpi(a, b)
        assert int(r.b) == int(b)
        assert float(r.a) == float(a)
        assert float(r.b) == float(b)
        assert complex(r.a) == complex(a)
        assert complex(r.b) == complex(b)
