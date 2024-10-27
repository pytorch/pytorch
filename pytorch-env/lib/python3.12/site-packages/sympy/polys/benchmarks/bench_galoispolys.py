"""Benchmarks for polynomials over Galois fields. """


from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime


def gathen_poly(n, p, K):
    return gf_from_dict({n: K.one, 1: K.one, 0: K.one}, p, K)


def shoup_poly(n, p, K):
    f = [K.one] * (n + 1)
    for i in range(1, n + 1):
        f[i] = (f[i - 1]**2 + K.one) % p
    return f


def genprime(n, K):
    return K(nextprime(int((2**n * pi).evalf())))

p_10 = genprime(10, ZZ)
f_10 = gathen_poly(10, p_10, ZZ)

p_20 = genprime(20, ZZ)
f_20 = gathen_poly(20, p_20, ZZ)


def timeit_gathen_poly_f10_zassenhaus():
    gf_factor_sqf(f_10, p_10, ZZ, method='zassenhaus')


def timeit_gathen_poly_f10_shoup():
    gf_factor_sqf(f_10, p_10, ZZ, method='shoup')


def timeit_gathen_poly_f20_zassenhaus():
    gf_factor_sqf(f_20, p_20, ZZ, method='zassenhaus')


def timeit_gathen_poly_f20_shoup():
    gf_factor_sqf(f_20, p_20, ZZ, method='shoup')

P_08 = genprime(8, ZZ)
F_10 = shoup_poly(10, P_08, ZZ)

P_18 = genprime(18, ZZ)
F_20 = shoup_poly(20, P_18, ZZ)


def timeit_shoup_poly_F10_zassenhaus():
    gf_factor_sqf(F_10, P_08, ZZ, method='zassenhaus')


def timeit_shoup_poly_F10_shoup():
    gf_factor_sqf(F_10, P_08, ZZ, method='shoup')


def timeit_shoup_poly_F20_zassenhaus():
    gf_factor_sqf(F_20, P_18, ZZ, method='zassenhaus')


def timeit_shoup_poly_F20_shoup():
    gf_factor_sqf(F_20, P_18, ZZ, method='shoup')
