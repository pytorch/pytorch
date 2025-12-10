from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
    modgcd_univariate,
    modgcd_bivariate,
    _chinese_remainder_reconstruction_multivariate,
    modgcd_multivariate,
    _to_ZZ_poly,
    _to_ANP_poly,
    func_field_modgcd,
    _func_field_modgcd_m)
from sympy.functions.elementary.miscellaneous import sqrt


def test_modgcd_univariate_integers():
    R, x = ring("x", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_univariate(f, g) == (0, 0, 0)

    f, g = R.zero, x
    assert modgcd_univariate(f, g) == (x, 0, 1)
    assert modgcd_univariate(g, f) == (x, 1, 0)

    f, g = R.zero, -x
    assert modgcd_univariate(f, g) == (x, 0, -1)
    assert modgcd_univariate(g, f) == (x, -1, 0)

    f, g = 2*x, R(2)
    assert modgcd_univariate(f, g) == (2, x, 1)

    f, g = 2*x + 2, 6*x**2 - 6
    assert modgcd_univariate(f, g) == (2*x + 2, 1, 3*x - 3)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert modgcd_univariate(f, g) == (h, cff, cfg)

    f = - 352518131239247345597970242177235495263669787845475025293906825864749649589178600387510272*x**49 \
        + 46818041807522713962450042363465092040687472354933295397472942006618953623327997952*x**42 \
        + 378182690892293941192071663536490788434899030680411695933646320291525827756032*x**35 \
        + 112806468807371824947796775491032386836656074179286744191026149539708928*x**28 \
        - 12278371209708240950316872681744825481125965781519138077173235712*x**21 \
        + 289127344604779611146960547954288113529690984687482920704*x**14 \
        + 19007977035740498977629742919480623972236450681*x**7 \
        + 311973482284542371301330321821976049

    g =   365431878023781158602430064717380211405897160759702125019136*x**21 \
        + 197599133478719444145775798221171663643171734081650688*x**14 \
        - 9504116979659010018253915765478924103928886144*x**7 \
        - 311973482284542371301330321821976049

    assert modgcd_univariate(f, f.diff(x))[0] == g

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert modgcd_univariate(f, g) == (h, cff, cfg)


def test_modgcd_bivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_bivariate(f, g) == (0, 0, 0)

    f, g = 2*x, R(2)
    assert modgcd_bivariate(f, g) == (2, x, 1)

    f, g = x + 2*y, x + y
    assert modgcd_bivariate(f, g) == (1, f, g)

    f, g = x**2 + 2*x*y + y**2, x**3 + y**3
    assert modgcd_bivariate(f, g) == (x + y, x + y, x**2 - x*y + y**2)

    f, g = x*y**2 + 2*x*y + x, x*y**3 + x
    assert modgcd_bivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)

    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1
    assert modgcd_bivariate(f, g) == (1, f, g)

    f = 2*x*y**2 + 4*x*y + 2*x + y**2 + 2*y + 1
    g = 2*x*y**3 + 2*x + y**3 + 1
    assert modgcd_bivariate(f, g) == (2*x*y + 2*x + y + 1, y + 1, y**2 - y + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert modgcd_bivariate(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert modgcd_bivariate(f, g) == (x + 1, 1, 2*x + 2)

    f = 2*x**2 + 4*x*y - 2*x - 4*y
    g = x**2 + x - 2
    assert modgcd_bivariate(f, g) == (x - 1, 2*x + 4*y, x + 2)

    f = 2*x**2 + 2*x*y - 3*x - 3*y
    g = 4*x*y - 2*x + 4*y**2 - 2*y
    assert modgcd_bivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)


def test_chinese_remainder():
    R, x, y = ring("x, y", ZZ)
    p, q = 3, 5

    hp = x**3*y - x**2 - 1
    hq = -x**3*y - 2*x*y**2 + 2

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)

    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq

    T, z = ring("z", R)
    p, q = 3, 7

    hp = (x*y + 1)*z**2 + x
    hq = (x**2 - 3*y)*z + 2

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)

    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq


def test_modgcd_multivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = R.zero, R.zero
    assert modgcd_multivariate(f, g) == (0, 0, 0)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert modgcd_multivariate(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert modgcd_multivariate(f, g) == (x + 1, 1, 2*x + 2)

    f = 2*x**2 + 2*x*y - 3*x - 3*y
    g = 4*x*y - 2*x + 4*y**2 - 2*y
    assert modgcd_multivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)

    f, g = x*y**2 + 2*x*y + x, x*y**3 + x
    assert modgcd_multivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)

    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1
    assert modgcd_multivariate(f, g) == (1, f, g)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert modgcd_multivariate(f, g) == (h, cff, cfg)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = x + y + z, -x - y - z - u
    assert modgcd_multivariate(f, g) == (1, f, g)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert modgcd_multivariate(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert modgcd_multivariate(f, g) == (h, cff, cfg)
    assert modgcd_multivariate(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g = x - y*z, x - y*z
    assert modgcd_multivariate(f, g) == (x - y*z, 1, 1)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, t = ring("x,y,z,t", ZZ)

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)

    assert H == h and H*cff == f and H*cfg == g


def test_to_ZZ_ANP_poly():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring("x", A)
    f = x*(sqrt(2) + 1)

    T, x_, z_ = ring("x_, z_", ZZ)
    f_ = x_*z_ + x_

    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f

    R, x, t, s = ring("x, t, s", A)
    f = x*t**2 + x*s + sqrt(2)

    D, t_, s_ = ring("t_, s_", ZZ)
    T, x_, z_ = ring("x_, z_", D)
    f_ = (t_**2 + s_)*x_ + z_

    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f


def test_modgcd_algebraic_field():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring("x", A)
    one = A.one

    f, g = 2*x, R(2)
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = 2*x, R(sqrt(2))
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = 2*x + 2, 6*x**2 - 6
    assert func_field_modgcd(f, g) == (x + 1, R(2), 6*x - 6)

    R, x, y = ring("x, y", A)

    f, g = x + sqrt(2)*y, x + y
    assert func_field_modgcd(f, g) == (one, f, g)

    f, g = x*y + sqrt(2)*y**2, R(sqrt(2))*y
    assert func_field_modgcd(f, g) == (y, x + sqrt(2)*y, R(sqrt(2)))

    f, g = x**2 + 2*sqrt(2)*x*y + 2*y**2, x + sqrt(2)*y
    assert func_field_modgcd(f, g) == (g, g, one)

    A = AlgebraicField(QQ, sqrt(2), sqrt(3))
    R, x, y, z = ring("x, y, z", A)

    h = x**2*y**7 + sqrt(6)/21*z
    f, g = h*(27*y**3 + 1), h*(y + x)
    assert func_field_modgcd(f, g) == (h, 27*y**3+1, y+x)

    h = x**13*y**3 + 1/2*x**10 + 1/sqrt(2)
    f, g = h*(x + 1), h*sqrt(2)/sqrt(3)
    assert func_field_modgcd(f, g) == (h, x + 1, R(sqrt(2)/sqrt(3)))

    A = AlgebraicField(QQ, sqrt(2)**(-1)*sqrt(3))
    R, x = ring("x", A)

    f, g = x + 1, x - 1
    assert func_field_modgcd(f, g) == (A.one, f, g)


# when func_field_modgcd supports function fields, this test can be changed
def test_modgcd_func_field():
    D, t = ring("t", ZZ)
    R, x, z = ring("x, z", D)

    minpoly = (z**2*t**2 + z**2*t - 1).drop(0)
    f, g = x + 1, x - 1

    assert _func_field_modgcd_m(f, g, minpoly) == R.one
