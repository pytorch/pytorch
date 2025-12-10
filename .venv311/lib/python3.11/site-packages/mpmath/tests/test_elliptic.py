"""
Limited tests of the elliptic functions module.  A full suite of
extensive testing can be found in elliptic_torture_tests.py

Author of the first version: M.T. Taschuk

References:

[1] Abramowitz & Stegun. 'Handbook of Mathematical Functions, 9th Ed.',
    (Dover duplicate of 1972 edition)
[2] Whittaker 'A Course of Modern Analysis, 4th Ed.', 1946,
    Cambridge University Press

"""

import mpmath
import random
import pytest

from mpmath import *

def mpc_ae(a, b, eps=eps):
    res = True
    res = res and a.real.ae(b.real, eps)
    res = res and a.imag.ae(b.imag, eps)
    return res

zero = mpf(0)
one = mpf(1)

jsn = ellipfun('sn')
jcn = ellipfun('cn')
jdn = ellipfun('dn')

calculate_nome = lambda k: qfrom(k=k)

def test_ellipfun():
    mp.dps = 15
    assert ellipfun('ss', 0, 0) == 1
    assert ellipfun('cc', 0, 0) == 1
    assert ellipfun('dd', 0, 0) == 1
    assert ellipfun('nn', 0, 0) == 1
    assert ellipfun('sn', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cn', 0.25, 0).ae(cos(0.25))
    assert ellipfun('dn', 0.25, 0).ae(1)
    assert ellipfun('ns', 0.25, 0).ae(csc(0.25))
    assert ellipfun('nc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('nd', 0.25, 0).ae(1)
    assert ellipfun('sc', 0.25, 0).ae(tan(0.25))
    assert ellipfun('sd', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cd', 0.25, 0).ae(cos(0.25))
    assert ellipfun('cs', 0.25, 0).ae(cot(0.25))
    assert ellipfun('dc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('ds', 0.25, 0).ae(csc(0.25))
    assert ellipfun('sn', 0.25, 1).ae(tanh(0.25))
    assert ellipfun('cn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('dn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('ns', 0.25, 1).ae(coth(0.25))
    assert ellipfun('nc', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('nd', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('sc', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('sd', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('cd', 0.25, 1).ae(1)
    assert ellipfun('cs', 0.25, 1).ae(csch(0.25))
    assert ellipfun('dc', 0.25, 1).ae(1)
    assert ellipfun('ds', 0.25, 1).ae(csch(0.25))
    assert ellipfun('sn', 0.25, 0.5).ae(0.24615967096986145833)
    assert ellipfun('cn', 0.25, 0.5).ae(0.96922928989378439337)
    assert ellipfun('dn', 0.25, 0.5).ae(0.98473484156599474563)
    assert ellipfun('ns', 0.25, 0.5).ae(4.0624038700573130369)
    assert ellipfun('nc', 0.25, 0.5).ae(1.0317476065024692949)
    assert ellipfun('nd', 0.25, 0.5).ae(1.0155017958029488665)
    assert ellipfun('sc', 0.25, 0.5).ae(0.25397465134058993408)
    assert ellipfun('sd', 0.25, 0.5).ae(0.24997558792415733063)
    assert ellipfun('cd', 0.25, 0.5).ae(0.98425408443195497052)
    assert ellipfun('cs', 0.25, 0.5).ae(3.9374008182374110826)
    assert ellipfun('dc', 0.25, 0.5).ae(1.0159978158253033913)
    assert ellipfun('ds', 0.25, 0.5).ae(4.0003906313579720593)




def test_calculate_nome():
    mp.dps = 100

    q = calculate_nome(zero)
    assert(q == zero)

    mp.dps = 25
    # used Mathematica's EllipticNomeQ[m]
    math1 = [(mpf(1)/10, mpf('0.006584651553858370274473060')),
             (mpf(2)/10, mpf('0.01394285727531826872146409')),
             (mpf(3)/10, mpf('0.02227743615715350822901627')),
             (mpf(4)/10, mpf('0.03188334731336317755064299')),
             (mpf(5)/10, mpf('0.04321391826377224977441774')),
             (mpf(6)/10, mpf('0.05702025781460967637754953')),
             (mpf(7)/10, mpf('0.07468994353717944761143751')),
             (mpf(8)/10, mpf('0.09927369733882489703607378')),
             (mpf(9)/10, mpf('0.1401731269542615524091055')),
             (mpf(9)/10, mpf('0.1401731269542615524091055'))]

    for i in math1:
        m = i[0]
        q = calculate_nome(sqrt(m))
        assert q.ae(i[1])

    mp.dps = 15

def test_jtheta():
    mp.dps = 25

    z = q = zero
    for n in range(1,5):
        value = jtheta(n, z, q)
        assert(value == (n-1)//2)

    for q in [one, mpf(2)]:
        for n in range(1,5):
            pytest.raises(ValueError, lambda: jtheta(n, z, q))

    z = one/10
    q = one/11

    # Mathematical N[EllipticTheta[1, 1/10, 1/11], 25]
    res = mpf('0.1069552990104042681962096')
    result = jtheta(1, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[2, 1/10, 1/11], 25]
    res = mpf('1.101385760258855791140606')
    result = jtheta(2, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[3, 1/10, 1/11], 25]
    res = mpf('1.178319743354331061795905')
    result = jtheta(3, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[4, 1/10, 1/11], 25]
    res = mpf('0.8219318954665153577314573')
    result = jtheta(4, z, q)
    assert(result.ae(res))

    # test for sin zeros for jtheta(1, z, q)
    # test for cos zeros for jtheta(2, z, q)
    z1 = pi
    z2 = pi/2
    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)
        result = jtheta(1, z1, q)
        assert(result.ae(0))
        result = jtheta(2, z2, q)
        assert(result.ae(0))
    mp.dps = 15


def test_jtheta_issue_79():
    # near the circle of covergence |q| = 1 the convergence slows
    # down; for |q| > Q_LIM the theta functions raise ValueError
    mp.dps = 30
    mp.dps += 30
    q = mpf(6)/10 - one/10**6 - mpf(8)/10 * j
    mp.dps -= 30
    # Mathematica run first
    # N[EllipticTheta[3, 1, 6/10 - 10^-6 - 8/10*I], 2000]
    # then it works:
    # N[EllipticTheta[3, 1, 6/10 - 10^-6 - 8/10*I], 30]
    res = mpf('32.0031009628901652627099524264') + \
          mpf('16.6153027998236087899308935624') * j
    result = jtheta(3, 1, q)
    # check that for abs(q) > Q_LIM a ValueError exception is raised
    mp.dps += 30
    q = mpf(6)/10 - one/10**7 - mpf(8)/10 * j
    mp.dps -= 30
    pytest.raises(ValueError, lambda: jtheta(3, 1, q))

    # bug reported in issue 79
    mp.dps = 100
    z = (1+j)/3
    q = mpf(368983957219251)/10**15 + mpf(636363636363636)/10**15 * j
    # Mathematica N[EllipticTheta[1, z, q], 35]
    res = mpf('2.4439389177990737589761828991467471') + \
          mpf('0.5446453005688226915290954851851490') *j
    mp.dps = 30
    result = jtheta(1, z, q)
    assert(result.ae(res))
    mp.dps = 80
    z = 3 + 4*j
    q = 0.5 + 0.5*j
    r1 = jtheta(1, z, q)
    mp.dps = 15
    r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 80
    z = 3 + j
    q1 = exp(j*3)
    # longer test
    # for n in range(1, 6)
    for n in range(1, 2):
        mp.dps = 80
        q = q1*(1 - mpf(1)/10**n)
        r1 = jtheta(1, z, q)
        mp.dps = 15
        r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 15
    # issue 79 about high derivatives
    assert jtheta(3, 4.5, 0.25, 9).ae(1359.04892680683)
    assert jtheta(3, 4.5, 0.25, 50).ae(-6.14832772630905e+33)
    mp.dps = 50
    r = jtheta(3, 4.5, 0.25, 9)
    assert r.ae('1359.048926806828939547859396600218966947753213803')
    r = jtheta(3, 4.5, 0.25, 50)
    assert r.ae('-6148327726309051673317975084654262.4119215720343656')

def test_jtheta_identities():
    """
    Tests the some of the jacobi identidies found in Abramowitz,
    Sec. 16.28, Pg. 576. The identities are tested to 1 part in 10^98.
    """
    mp.dps = 110
    eps1 = ldexp(eps, 30)

    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)

        zstring = str(10*random.random())
        z = mpf(zstring)
        # Abramowitz 16.28.1
        # v_1(z, q)**2 * v_4(0, q)**2 =   v_3(z, q)**2 * v_2(0, q)**2
        #                               - v_2(z, q)**2 * v_3(0, q)**2
        term1 = (jtheta(1, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(3, z, q)**2) * (jtheta(2, zero, q)**2)
        term3 = (jtheta(2, z, q)**2) * (jtheta(3, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        zstring = str(100*random.random())
        z = mpf(zstring)
        # Abramowitz 16.28.2
        # v_2(z, q)**2 * v_4(0, q)**2 =   v_4(z, q)**2 * v_2(0, q)**2
        #                               - v_1(z, q)**2 * v_3(0, q)**2
        term1 = (jtheta(2, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(4, z, q)**2) * (jtheta(2, zero, q)**2)
        term3 = (jtheta(1, z, q)**2) * (jtheta(3, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.3
        # v_3(z, q)**2 * v_4(0, q)**2 =   v_4(z, q)**2 * v_3(0, q)**2
        #                               - v_1(z, q)**2 * v_2(0, q)**2
        term1 = (jtheta(3, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(4, z, q)**2) * (jtheta(3, zero, q)**2)
        term3 = (jtheta(1, z, q)**2) * (jtheta(2, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.4
        # v_4(z, q)**2 * v_4(0, q)**2 =   v_3(z, q)**2 * v_3(0, q)**2
        #                               - v_2(z, q)**2 * v_2(0, q)**2
        term1 = (jtheta(4, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(3, z, q)**2) * (jtheta(3, zero, q)**2)
        term3 = (jtheta(2, z, q)**2) * (jtheta(2, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.5
        # v_2(0, q)**4 + v_4(0, q)**4 == v_3(0, q)**4
        term1 = (jtheta(2, zero, q))**4
        term2 = (jtheta(4, zero, q))**4
        term3 = (jtheta(3, zero, q))**4
        equality = term1 + term2 - term3
        assert(equality.ae(0, eps1))
    mp.dps = 15

def test_jtheta_complex():
    mp.dps = 30
    z = mpf(1)/4 + j/8
    q = mpf(1)/3 + j/7
    # Mathematica N[EllipticTheta[1, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.31618034835986160705729105731678285') + \
          mpf('0.07542013825835103435142515194358975') * j
    r = jtheta(1, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[2, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('1.6530986428239765928634711417951828') + \
          mpf('0.2015344864707197230526742145361455') * j
    r = jtheta(2, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[3, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('1.6520564411784228184326012700348340') + \
          mpf('0.1998129119671271328684690067401823') * j
    r = jtheta(3, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[4, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.37619082382228348252047624089973824') - \
          mpf('0.15623022130983652972686227200681074') * j
    r = jtheta(4, z, q)
    assert(mpc_ae(r, res))

    # check some theta function identities
    mp.dos = 100
    z = mpf(1)/4 + j/8
    q = mpf(1)/3 + j/7
    mp.dps += 10
    a = [0,0, jtheta(2, 0, q), jtheta(3, 0, q), jtheta(4, 0, q)]
    t = [0, jtheta(1, z, q), jtheta(2, z, q), jtheta(3, z, q), jtheta(4, z, q)]
    r = [(t[2]*a[4])**2 - (t[4]*a[2])**2 + (t[1] *a[3])**2,
        (t[3]*a[4])**2 - (t[4]*a[3])**2 + (t[1] *a[2])**2,
        (t[1]*a[4])**2 - (t[3]*a[2])**2 + (t[2] *a[3])**2,
        (t[4]*a[4])**2 - (t[3]*a[3])**2 + (t[2] *a[2])**2,
        a[2]**4 + a[4]**4 - a[3]**4]
    mp.dps -= 10
    for x in r:
        assert(mpc_ae(x, mpc(0)))
    mp.dps = 15

def test_djtheta():
    mp.dps = 30

    z = one/7 + j/3
    q = one/8 + j/5
    # Mathematica N[EllipticThetaPrime[1, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('1.5555195883277196036090928995803201') - \
          mpf('0.02439761276895463494054149673076275') * j
    result = jtheta(1, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[2, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('0.19825296689470982332701283509685662') - \
          mpf('0.46038135182282106983251742935250009') * j
    result = jtheta(2, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[3, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('0.36492498415476212680896699407390026') - \
          mpf('0.57743812698666990209897034525640369') * j
    result = jtheta(3, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[4, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('-0.38936892528126996010818803742007352') + \
          mpf('0.66549886179739128256269617407313625') * j
    result = jtheta(4, z, q, 1)
    assert(mpc_ae(result, res))

    for i in range(10):
        q = (one*random.random() + j*random.random())/2
        # identity in Wittaker, Watson &21.41
        a = jtheta(1, 0, q, 1)
        b = jtheta(2, 0, q)*jtheta(3, 0, q)*jtheta(4, 0, q)
        assert(a.ae(b))

    # test higher derivatives
    mp.dps = 20
    for q,z in [(one/3, one/5), (one/3 + j/8, one/5),
        (one/3, one/5 + j/8), (one/3 + j/7, one/5 + j/8)]:
        for n in [1, 2, 3, 4]:
            r = jtheta(n, z, q, 2)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=2)
            assert r.ae(r1)
            r = jtheta(n, z, q, 3)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=3)
            assert r.ae(r1)

    # identity in Wittaker, Watson &21.41
    q = one/3
    z = zero
    a = [0]*5
    a[1] = jtheta(1, z, q, 3)/jtheta(1, z, q, 1)
    for n in [2,3,4]:
        a[n] = jtheta(n, z, q, 2)/jtheta(n, z, q)
    equality = a[2] + a[3] + a[4] - a[1]
    assert(equality.ae(0))
    mp.dps = 15

def test_jsn():
    """
    Test some special cases of the sn(z, q) function.
    """
    mp.dps = 100

    # trival case
    result = jsn(zero, zero)
    assert(result == zero)

    # Abramowitz Table 16.5
    #
    # sn(0, m) = 0

    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)

        equality = jsn(zero, q)
        assert(equality.ae(0))

    # Abramowitz Table 16.6.1
    #
    # sn(z, 0) = sin(z), m == 0
    #
    # sn(z, 1) = tanh(z), m == 1
    #
    # It would be nice to test these, but I find that they run
    # in to numerical trouble.  I'm currently treating as a boundary
    # case for sn function.

    mp.dps = 25
    arg = one/10
    #N[JacobiSN[1/10, 2^-100], 25]
    res = mpf('0.09983341664682815230681420')
    m = ldexp(one, -100)
    result = jsn(arg, m)
    assert(result.ae(res))

    # N[JacobiSN[1/10, 1/10], 25]
    res = mpf('0.09981686718599080096451168')
    result = jsn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15

def test_jcn():
    """
    Test some special cases of the cn(z, q) function.
    """
    mp.dps = 100

    # Abramowitz Table 16.5
    # cn(0, q) = 1
    qstring = str(random.random())
    q = mpf(qstring)
    cn = jcn(zero, q)
    assert(cn.ae(one))

    # Abramowitz Table 16.6.2
    #
    # cn(u, 0) = cos(u), m == 0
    #
    # cn(u, 1) = sech(z), m == 1
    #
    # It would be nice to test these, but I find that they run
    # in to numerical trouble.  I'm currently treating as a boundary
    # case for cn function.

    mp.dps = 25
    arg = one/10
    m = ldexp(one, -100)
    #N[JacobiCN[1/10, 2^-100], 25]
    res = mpf('0.9950041652780257660955620')
    result = jcn(arg, m)
    assert(result.ae(res))

    # N[JacobiCN[1/10, 1/10], 25]
    res = mpf('0.9950058256237368748520459')
    result = jcn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15

def test_jdn():
    """
    Test some special cases of the dn(z, q) function.
    """
    mp.dps = 100

    # Abramowitz Table 16.5
    # dn(0, q) = 1
    mstring = str(random.random())
    m = mpf(mstring)

    dn = jdn(zero, m)
    assert(dn.ae(one))

    mp.dps = 25
    # N[JacobiDN[1/10, 1/10], 25]
    res = mpf('0.9995017055025556219713297')
    arg = one/10
    result = jdn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15


def test_sn_cn_dn_identities():
    """
    Tests the some of the jacobi elliptic function identities found
    on Mathworld. Haven't found in Abramowitz.
    """
    mp.dps = 100
    N = 5
    for i in range(N):
        qstring = str(random.random())
        q = mpf(qstring)
        zstring = str(100*random.random())
        z = mpf(zstring)

        # MathWorld
        # sn(z, q)**2 + cn(z, q)**2 == 1
        term1 = jsn(z, q)**2
        term2 = jcn(z, q)**2
        equality = one - term1 - term2
        assert(equality.ae(0))

    # MathWorld
    # k**2 * sn(z, m)**2 + dn(z, m)**2 == 1
    for i in range(N):
        mstring = str(random.random())
        m = mpf(qstring)
        k = m.sqrt()
        zstring = str(10*random.random())
        z = mpf(zstring)
        term1 = k**2 * jsn(z, m)**2
        term2 = jdn(z, m)**2
        equality = one - term1 - term2
        assert(equality.ae(0))


    for i in range(N):
        mstring = str(random.random())
        m = mpf(mstring)
        k = m.sqrt()
        zstring = str(random.random())
        z = mpf(zstring)

        # MathWorld
        # k**2 * cn(z, m)**2 + (1 - k**2) = dn(z, m)**2
        term1 = k**2 * jcn(z, m)**2
        term2 = 1 - k**2
        term3 = jdn(z, m)**2
        equality = term3 - term1 - term2
        assert(equality.ae(0))

        K = ellipk(k**2)
        # Abramowitz Table 16.5
        # sn(K, m) = 1; K is K(k), first complete elliptic integral
        r = jsn(K, m)
        assert(r.ae(one))

        # Abramowitz Table 16.5
        # cn(K, q) = 0; K is K(k), first complete elliptic integral
        equality = jcn(K, m)
        assert(equality.ae(0))

        # Abramowitz Table 16.6.3
        # dn(z, 0) = 1, m == 0
        z = m
        value = jdn(z, zero)
        assert(value.ae(one))

    mp.dps = 15

def test_sn_cn_dn_complex():
    mp.dps = 30
    # N[JacobiSN[1/4 + I/8, 1/3 + I/7], 35] in Mathematica
    res = mpf('0.2495674401066275492326652143537') + \
          mpf('0.12017344422863833381301051702823') * j
    u = mpf(1)/4 + j/8
    m = mpf(1)/3 + j/7
    r = jsn(u, m)
    assert(mpc_ae(r, res))

    #N[JacobiCN[1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.9762691700944007312693721148331') - \
          mpf('0.0307203994181623243583169154824')*j
    r = jcn(u, m)
    #assert r.real.ae(res.real)
    #assert r.imag.ae(res.imag)
    assert(mpc_ae(r, res))

    #N[JacobiDN[1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.99639490163039577560547478589753039') - \
          mpf('0.01346296520008176393432491077244994')*j
    r = jdn(u, m)
    assert(mpc_ae(r, res))
    mp.dps = 15

def test_elliptic_integrals():
    # Test cases from Carlson's paper
    mp.dps = 15
    assert elliprd(0,2,1).ae(1.7972103521033883112)
    assert elliprd(2,3,4).ae(0.16510527294261053349)
    assert elliprd(j,-j,2).ae(0.65933854154219768919)
    assert elliprd(0,j,-j).ae(1.2708196271909686299 + 2.7811120159520578777j)
    assert elliprd(0,j-1,j).ae(-1.8577235439239060056 - 0.96193450888838559989j)
    assert elliprd(-2-j,-j,-1+j).ae(1.8249027393703805305 - 1.2218475784827035855j)
    # extra test cases
    assert elliprg(0,0,0) == 0
    assert elliprg(0,0,16).ae(2)
    assert elliprg(0,16,0).ae(2)
    assert elliprg(16,0,0).ae(2)
    assert elliprg(1,4,0).ae(1.2110560275684595248036)
    assert elliprg(1,0,4).ae(1.2110560275684595248036)
    assert elliprg(0,4,1).ae(1.2110560275684595248036)
    # should be symmetric -- fixes a bug present in the paper
    x,y,z = 1,1j,-1+1j
    assert elliprg(x,y,z).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(x,z,y).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(y,x,z).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(y,z,x).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(z,x,y).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(z,y,x).ae(0.64139146875812627545 + 0.58085463774808290907j)

    for n in [5, 15, 30, 60, 100]:
        mp.dps = n
        assert elliprf(1,2,0).ae('1.3110287771460599052324197949455597068413774757158115814084108519003952935352071251151477664807145467230678763')
        assert elliprf(0.5,1,0).ae('1.854074677301371918433850347195260046217598823521766905585928045056021776838119978357271861650371897277771871')
        assert elliprf(j,-j,0).ae('1.854074677301371918433850347195260046217598823521766905585928045056021776838119978357271861650371897277771871')
        assert elliprf(j-1,j,0).ae(mpc('0.79612586584233913293056938229563057846592264089185680214929401744498956943287031832657642790719940442165621412',
            '-1.2138566698364959864300942567386038975419875860741507618279563735753073152507112254567291141460317931258599889'))
        assert elliprf(2,3,4).ae('0.58408284167715170669284916892566789240351359699303216166309375305508295130412919665541330837704050454472379308')
        assert elliprf(j,-j,2).ae('1.0441445654064360931078658361850779139591660747973017593275012615517220315993723776182276555339288363064476126')
        assert elliprf(j-1,j,1-j).ae(mpc('0.93912050218619371196624617169781141161485651998254431830645241993282941057500174238125105410055253623847335313',
            '-0.53296252018635269264859303449447908970360344322834582313172115220559316331271520508208025270300138589669326136'))
        assert elliprc(0,0.25).ae(+pi)
        assert elliprc(2.25,2).ae(+ln2)
        assert elliprc(0,j).ae(mpc('1.1107207345395915617539702475151734246536554223439225557713489017391086982748684776438317336911913093408525532',
            '-1.1107207345395915617539702475151734246536554223439225557713489017391086982748684776438317336911913093408525532'))
        assert elliprc(-j,j).ae(mpc('1.2260849569072198222319655083097718755633725139745941606203839524036426936825652935738621522906572884239069297',
            '-0.34471136988767679699935618332997956653521218571295874986708834375026550946053920574015526038040124556716711353'))
        assert elliprc(0.25,-2).ae(ln2/3)
        assert elliprc(j,-1).ae(mpc('0.77778596920447389875196055840799837589537035343923012237628610795937014001905822029050288316217145443865649819',
            '0.1983248499342877364755170948292130095921681309577950696116251029742793455964385947473103628983664877025779304'))
        assert elliprj(0,1,2,3).ae('0.77688623778582332014190282640545501102298064276022952731669118325952563819813258230708177398475643634103990878')
        assert elliprj(2,3,4,5).ae('0.14297579667156753833233879421985774801466647854232626336218889885463800128817976132826443904216546421431528308')
        assert elliprj(2,3,4,-1+j).ae(mpc('0.13613945827770535203521374457913768360237593025944342652613569368333226052158214183059386307242563164036672709',
            '-0.38207561624427164249600936454845112611060375760094156571007648297226090050927156176977091273224510621553615189'))
        assert elliprj(j,-j,0,2).ae('1.6490011662710884518243257224860232300246792717163891216346170272567376981346412066066050103935109581019055806')
        assert elliprj(-1+j,-1-j,1,2).ae('0.94148358841220238083044612133767270187474673547917988681610772381758628963408843935027667916713866133196845063')
        assert elliprj(j,-j,0,1-j).ae(mpc('1.8260115229009316249372594065790946657011067182850435297162034335356430755397401849070610280860044610878657501',
            '1.2290661908643471500163617732957042849283739403009556715926326841959667290840290081010472716420690899886276961'))
        assert elliprj(-1+j,-1-j,1,-3+j).ae(mpc('-0.61127970812028172123588152373622636829986597243716610650831553882054127570542477508023027578037045504958619422',
            '-1.0684038390006807880182112972232562745485871763154040245065581157751693730095703406209466903752930797510491155'))
        assert elliprj(-1+j,-2-j,-j,-1+j).ae(mpc('1.8249027393703805304622013339009022294368078659619988943515764258335975852685224202567854526307030593012768954',
            '-1.2218475784827035854568450371590419833166777535029296025352291308244564398645467465067845461070602841312456831'))

        assert elliprg(0,16,16).ae(+pi)
        assert elliprg(2,3,4).ae('1.7255030280692277601061148835701141842692457170470456590515892070736643637303053506944907685301315299153040991')
        assert elliprg(0,j,-j).ae('0.42360654239698954330324956174109581824072295516347109253028968632986700241706737986160014699730561497106114281')
        assert elliprg(j-1,j,0).ae(mpc('0.44660591677018372656731970402124510811555212083508861036067729944477855594654762496407405328607219895053798354',
            '0.70768352357515390073102719507612395221369717586839400605901402910893345301718731499237159587077682267374159282'))
        assert elliprg(-j,j-1,j).ae(mpc('0.36023392184473309033675652092928695596803358846377334894215349632203382573844427952830064383286995172598964266',
            '0.40348623401722113740956336997761033878615232917480045914551915169013722542827052849476969199578321834819903921'))
        assert elliprg(0, mpf('0.0796'), 4).ae('1.0284758090288040009838871385180217366569777284430590125081211090574701293154645750017813190805144572673802094')
    mp.dps = 15

    # more test cases for the branch of ellippi / elliprj
    assert elliprj(-1-0.5j, -10-6j, -10-3j, -5+10j).ae(0.128470516743927699 + 0.102175950778504625j, abs_eps=1e-8)
    assert elliprj(1.987, 4.463 - 1.614j, 0, -3.965).ae(-0.341575118513811305 - 0.394703757004268486j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037+0.632j, 1.654, -0.9609).ae(-1.14735199581485639 - 0.134450158867472264j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037-0.632j, 1.654, -0.9609).ae(1.758765901861727 - 0.161002343366626892j, abs_eps=1e-5)
    assert elliprj(0.3068, -4.037+0.0632j, 1.654, -0.9609).ae(-1.17157627949475577 - 0.069182614173988811j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037+0.00632j, 1.654, -0.9609).ae(-1.17337595670549633 - 0.0623069224526925j, abs_eps=1e-8)

    # these require accurate integration
    assert elliprj(0.3068, -4.037-0.0632j, 1.654, -0.9609).ae(1.77940452391261626 + 0.0388711305592447234j)
    assert elliprj(0.3068, -4.037-0.00632j, 1.654, -0.9609).ae(1.77806722756403055 + 0.0592749824572262329j)
    # issue #571
    assert ellippi(2.1 + 0.94j, 2.3 + 0.98j, 2.5 + 0.01j).ae(-0.40652414240811963438 + 2.1547659461404749309j)

    assert ellippi(2.0-1.0j, 2.0+1.0j).ae(1.8578723151271115 - 1.18642180609983531j)
    assert ellippi(2.0-0.5j, 0.5+1.0j).ae(0.936761970766645807 - 1.61876787838890786j)
    assert ellippi(2.0, 1.0+1.0j).ae(0.999881420735506708 - 2.4139272867045391j)
    assert ellippi(2.0+1.0j, 2.0-1.0j).ae(1.8578723151271115 + 1.18642180609983531j)
    assert ellippi(2.0+1.0j, 2.0).ae(2.78474654927885845 + 2.02204728966993314j)

def test_issue_238():
    assert isnan(qfrom(m=nan))
