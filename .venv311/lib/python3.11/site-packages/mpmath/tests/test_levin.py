#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

# Attention:
#   These tests run with 15-20 decimal digits precision. For higher precision the
#   working precision must be raised.

def test_levin_0():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "levin", variant = "u")
        S, s, n = [], 0, 1
        while 1:
            s += mp.one / (n * n)
            n += 1
            S.append(s)
            v, e = L.update_psum(S)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.pi ** 2 / 6)
    assert err < eps
    w = mp.nsum(lambda n: 1/(n * n), [1, mp.inf], method = "levin", levin_variant = "u")
    err = abs(v - w)
    assert err < eps

def test_levin_1():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "levin", variant = "v")
        A, n = [], 1
        while 1:
            s = mp.mpf(n) ** (2 + 3j)
            n += 1
            A.append(s)
            v, e = L.update(A)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.zeta(-2-3j))
    assert err < eps
    w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method = "levin", levin_variant = "v")
    err = abs(v - w)
    assert err < eps

def test_levin_2():
    # [2] A. Sidi - "Pratical Extrapolation Methods" p.373
    mp.dps = 17
    z=mp.mpf(10)
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "sidi", variant = "t")
        n = 0
        while 1:
            s = (-1)**n * mp.fac(n) * z ** (-n)
            v, e = L.step(s)
            n += 1
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    exact = mp.quad(lambda x: mp.exp(-x)/(1+x/z),[0,mp.inf])
    # there is also a symbolic expression for the integral:
    #   exact = z * mp.exp(z) * mp.expint(1,z)
    err = abs(v - exact)
    assert err < eps
    w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method = "sidi", levin_variant = "t")
    assert err < eps

def test_levin_3():
    mp.dps = 17
    z=mp.mpf(2)
    eps = mp.mpf(mp.eps)
    with mp.extraprec(7*mp.prec):  # we need copious amount of precision to sum this highly divergent series
        L = mp.levin(method = "levin", variant = "t")
        n, s = 0, 0
        while 1:
            s += (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n))
            n += 1
            v, e = L.step_psum(s)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.8 * mp.log(eps))
    exact = mp.quad(lambda x: mp.exp( -x * x / 2 - z * x ** 4), [0,mp.inf]) * 2 / mp.sqrt(2 * mp.pi)
    # there is also a symbolic expression for the integral:
    #   exact = mp.exp(mp.one / (32 * z)) * mp.besselk(mp.one / 4, mp.one / (32 * z)) / (4 * mp.sqrt(z * mp.pi))
    err = abs(v - exact)
    assert err < eps
    w = mp.nsum(lambda n: (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n)), [0, mp.inf], method = "levin", levin_variant = "t", workprec = 8*mp.prec, steps = [2] + [1 for x in xrange(1000)])
    err = abs(v - w)
    assert err < eps

def test_levin_nsum():
    mp.dps = 17

    with mp.extraprec(mp.prec):
        z = mp.mpf(10) ** (-10)
        a = mp.nsum(lambda n: n**(-(1+z)), [1, mp.inf], method = "l") - 1 / z
        assert abs(a - mp.euler) < 1e-10

    eps = mp.exp(0.8 * mp.log(mp.eps))

    a = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "sidi")
    assert abs(a - mp.log(2)) < eps

    z = 2 + 1j
    f = lambda n: mp.rf(2 / mp.mpf(3), n) * mp.rf(4 / mp.mpf(3), n) * z**n / (mp.rf(1 / mp.mpf(3), n) * mp.fac(n))
    v = mp.nsum(f, [0, mp.inf], method = "levin", steps = [10 for x in xrange(1000)])
    exact = mp.hyp2f1(2 / mp.mpf(3), 4 / mp.mpf(3), 1 / mp.mpf(3), z)
    assert abs(exact - v) < eps

def test_cohen_alt_0():
    mp.dps = 17
    AC = mp.cohen_alt()
    S, s, n = [], 0, 1
    while 1:
        s += -((-1) ** n) * mp.one / (n * n)
        n += 1
        S.append(s)
        v, e = AC.update_psum(S)
        if e < mp.eps:
            break
        if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(mp.eps))
    err = abs(v - mp.pi ** 2 / 12)
    assert err < eps

def test_cohen_alt_1():
    mp.dps = 17
    A = []
    AC = mp.cohen_alt()
    n = 1
    while 1:
        A.append( mp.loggamma(1 + mp.one / (2 * n - 1)))
        A.append(-mp.loggamma(1 + mp.one / (2 * n)))
        n += 1
        v, e = AC.update(A)
        if e < mp.eps:
            break
        if n > 1000: raise RuntimeError("iteration limit exceeded")
    v = mp.exp(v)
    err = abs(v - 1.06215090557106)
    assert err < 1e-12
