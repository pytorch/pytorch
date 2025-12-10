"""Tests for Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """

from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR

from sympy.polys.specialpolys import (
    f_polys,
    dmp_fateman_poly_F_1,
    dmp_fateman_poly_F_2,
    dmp_fateman_poly_F_3)

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()

def test_dup_gcdex():
    R, x = ring("x", QQ)

    f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    g = x**3 + x**2 - 4*x - 4

    s = -QQ(1,5)*x + QQ(3,5)
    t = QQ(1,5)*x**2 - QQ(6,5)*x + 2
    h = x + 1

    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)

    f = x**4 + 4*x**3 - x + 1
    g = x**3 - x + 1

    s, t, h = R.dup_gcdex(f, g)
    S, T, H = R.dup_gcdex(g, f)

    assert R.dup_add(R.dup_mul(s, f),
                     R.dup_mul(t, g)) == h
    assert R.dup_add(R.dup_mul(S, g),
                     R.dup_mul(T, f)) == H

    f = 2*x
    g = x**2 - 16

    s = QQ(1,32)*x
    t = -QQ(1,16)
    h = 1

    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)


def test_dup_invert():
    R, x = ring("x", QQ)
    assert R.dup_invert(2*x, x**2 - 16) == QQ(1,32)*x


def test_dup_euclidean_prs():
    R, x = ring("x", QQ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    assert R.dup_euclidean_prs(f, g) == [
        f,
        g,
        -QQ(5,9)*x**4 + QQ(1,9)*x**2 - QQ(1,3),
        -QQ(117,25)*x**2 - 9*x + QQ(441,25),
        QQ(233150,19773)*x - QQ(102500,6591),
        -QQ(1288744821,543589225)]


def test_dup_primitive_prs():
    R, x = ring("x", ZZ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    assert R.dup_primitive_prs(f, g) == [
        f,
        g,
        -5*x**4 + x**2 - 3,
        13*x**2 + 25*x - 49,
        4663*x - 6150,
        1]


def test_dup_subresultants():
    R, x = ring("x", ZZ)

    assert R.dup_resultant(0, 0) == 0

    assert R.dup_resultant(1, 0) == 0
    assert R.dup_resultant(0, 1) == 0

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    a = 15*x**4 - 3*x**2 + 9
    b = 65*x**2 + 125*x - 245
    c = 9326*x - 12300
    d = 260708

    assert R.dup_subresultants(f, g) == [f, g, a, b, c, d]
    assert R.dup_resultant(f, g) == R.dup_LC(d)

    f = x**2 - 2*x + 1
    g = x**2 - 1

    a = 2*x - 2

    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 0

    f = x**2 + 1
    g = x**2 - 1

    a = -2

    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 4

    f = x**2 - 1
    g = x**3 - x**2 + 2

    assert R.dup_resultant(f, g) == 0

    f = 3*x**3 - x
    g = 5*x**2 + 1

    assert R.dup_resultant(f, g) == 64

    f = x**2 - 2*x + 7
    g = x**3 - x + 5

    assert R.dup_resultant(f, g) == 265

    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 15*x**2 + 74*x - 120

    assert R.dup_resultant(f, g) == -8640

    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 10*x**2 + 29*x - 20

    assert R.dup_resultant(f, g) == 0

    f = x**3 - 1
    g = x**3 + 2*x**2 + 2*x - 1

    assert R.dup_resultant(f, g) == 16

    f = x**8 - 2
    g = x - 1

    assert R.dup_resultant(f, g) == -1


def test_dmp_subresultants():
    R, x, y = ring("x,y", ZZ)

    assert R.dmp_resultant(0, 0) == 0
    assert R.dmp_prs_resultant(0, 0)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 0) == 0
    assert R.dmp_qq_collins_resultant(0, 0) == 0

    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0

    assert R.dmp_resultant(0, 1) == 0
    assert R.dmp_prs_resultant(0, 1)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 1) == 0
    assert R.dmp_qq_collins_resultant(0, 1) == 0

    f = 3*x**2*y - y**3 - 4
    g = x**2 + x*y**3 - 9

    a = 3*x*y**4 + y**3 - 27*y + 4
    b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    r = R.dmp_LC(b)

    assert R.dmp_subresultants(f, g) == [f, g, a, b]

    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r

    f = -x**3 + 5
    g = 3*x**2*y + x**2

    a = 45*y**2 + 30*y + 5
    b = 675*y**3 + 675*y**2 + 225*y + 25

    r = R.dmp_LC(b)

    assert R.dmp_subresultants(f, g) == [f, g, a]
    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f = 6*x**2 - 3*x*y - 2*x*z + y*z
    g = x**2 - x*u - x*v + u*v

    r = y**2*z**2 - 3*y**2*z*u - 3*y**2*z*v + 9*y**2*u*v - 2*y*z**2*u \
      - 2*y*z**2*v + 6*y*z*u**2 + 12*y*z*u*v + 6*y*z*v**2 - 18*y*u**2*v \
      - 18*y*u*v**2 + 4*z**2*u*v - 12*z*u**2*v - 12*z*u*v**2 + 36*u**2*v**2

    assert R.dmp_zz_collins_resultant(f, g) == r.drop(x)

    R, x, y, z, u, v = ring("x,y,z,u,v", QQ)

    f = x**2 - QQ(1,2)*x*y - QQ(1,3)*x*z + QQ(1,6)*y*z
    g = x**2 - x*u - x*v + u*v

    r = QQ(1,36)*y**2*z**2 - QQ(1,12)*y**2*z*u - QQ(1,12)*y**2*z*v + QQ(1,4)*y**2*u*v \
      - QQ(1,18)*y*z**2*u - QQ(1,18)*y*z**2*v + QQ(1,6)*y*z*u**2 + QQ(1,3)*y*z*u*v \
      + QQ(1,6)*y*z*v**2 - QQ(1,2)*y*u**2*v - QQ(1,2)*y*u*v**2 + QQ(1,9)*z**2*u*v \
      - QQ(1,3)*z*u**2*v - QQ(1,3)*z*u*v**2 + u**2*v**2

    assert R.dmp_qq_collins_resultant(f, g) == r.drop(x)

    Rt, t = ring("t", ZZ)
    Rx, x = ring("x", Rt)

    f = x**6 - 5*x**4 + 5*x**2 + 4
    g = -6*t*x**5 + x**4 + 20*t*x**3 - 3*x**2 - 10*t*x + 6

    assert Rx.dup_resultant(f, g) == 2930944*t**6 + 2198208*t**4 + 549552*t**2 + 45796


def test_dup_discriminant():
    R, x = ring("x", ZZ)

    assert R.dup_discriminant(0) == 0
    assert R.dup_discriminant(x) == 1

    assert R.dup_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    assert R.dup_discriminant(5*x**5 + x**3 + 2) == 31252160
    assert R.dup_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    assert R.dup_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112


def test_dmp_discriminant():
    R, x = ring("x", ZZ)

    assert R.dmp_discriminant(0) == 0

    R, x, y = ring("x,y", ZZ)

    assert R.dmp_discriminant(0) == 0
    assert R.dmp_discriminant(y) == 0

    assert R.dmp_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    assert R.dmp_discriminant(5*x**5 + x**3 + 2) == 31252160
    assert R.dmp_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    assert R.dmp_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112

    assert R.dmp_discriminant(x**2*y + 2*y) == (-8*y**2).drop(x)
    assert R.dmp_discriminant(x*y**2 + 2*x) == 1

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_discriminant(x*y + z) == 1

    R, x, y, z, u = ring("x,y,z,u", ZZ)
    assert R.dmp_discriminant(x**2*y + x*z + u) == (-4*y*u + z**2).drop(x)

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)
    assert R.dmp_discriminant(x**3*y + x**2*z + x*u + v) == \
        (-27*y**2*v**2 + 18*y*z*u*v - 4*y*u**3 - 4*z**3*v + z**2*u**2).drop(x)


def test_dup_gcd():
    R, x = ring("x", ZZ)

    f, g = 0, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (0, 0, 0)

    f, g = 2, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, 0)

    f, g = -2, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, 0)

    f, g = 0, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 0, -1)

    f, g = 0, 2*x + 4
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2*x + 4, 0, 1)

    f, g = 2*x + 4, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2*x + 4, 1, 0)

    f, g = 2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, 1)

    f, g = -2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, 1)

    f, g = 2, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, -1)

    f, g = -2, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, -1)

    f, g = x**2 + 2*x + 1, 1
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 1)

    f, g = x**2 + 2*x + 1, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 2)

    f, g = 2*x**2 + 4*x + 2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, x**2 + 2*x + 1, 1)

    f, g = 2, 2*x**2 + 4*x + 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, x**2 + 2*x + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (x + 1, 1, 2*x + 2)

    f, g = x - 31, x
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, f, g)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    R, x = ring("x", QQ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert R.dup_qq_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_ff_prs_gcd(f, g) == (h, cff, cfg)

    R, x = ring("x", ZZ)

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

    assert R.dup_zz_heu_gcd(f, R.dup_diff(f, 1))[0] == g
    assert R.dup_rr_prs_gcd(f, R.dup_diff(f, 1))[0] == g

    R, x = ring("x", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    h = x + 1

    assert R.dup_qq_heu_gcd(f, g) == (h, g, QQ(1,2))
    assert R.dup_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    R, x = ring("x", ZZ)

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)


def test_dmp_gcd():
    R, x, y = ring("x,y", ZZ)

    f, g = 0, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (0, 0, 0)

    f, g = 2, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 0)

    f, g = -2, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 0)

    f, g = 0, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 0, -1)

    f, g = 0, 2*x + 4
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 0, 1)

    f, g = 2*x + 4, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 1, 0)

    f, g = 2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 1)

    f, g = -2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 1)

    f, g = 2, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, -1)

    f, g = -2, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, -1)

    f, g = x**2 + 2*x + 1, 1
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 1)

    f, g = x**2 + 2*x + 1, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 2)

    f, g = 2*x**2 + 4*x + 2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, x**2 + 2*x + 1, 1)

    f, g = 2, 2*x**2 + 4*x + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, x**2 + 2*x + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 1, 2*x + 2)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert R.dmp_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dmp_rr_prs_gcd(f, g) == (h, cff, cfg)

    assert R.dmp_zz_heu_gcd(g, f) == (h, cfg, cff)
    assert R.dmp_rr_prs_gcd(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(4, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(6, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(8, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_2(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(4, ZZ))
    H, cff, cfg = R.dmp_inner_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y = ring("x,y", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    h = x + 1

    assert R.dmp_qq_heu_gcd(f, g) == (h, g, QQ(1,2))
    assert R.dmp_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    R, x, y = ring("x,y", RR)

    f = 2.1*x*y**2 - 2.2*x*y + 2.1*x
    g = 1.0*x**3

    assert R.dmp_ff_prs_gcd(f, g) == \
        (1.0*x, 2.1*y**2 - 2.2*y + 2.1, 1.0*x**2)


def test_dup_lcm():
    R, x = ring("x", ZZ)

    assert R.dup_lcm(2, 6) == 6

    assert R.dup_lcm(2*x**3, 6*x) == 6*x**3
    assert R.dup_lcm(2*x**3, 3*x) == 6*x**3

    assert R.dup_lcm(x**2 + x, x) == x**2 + x
    assert R.dup_lcm(x**2 + x, 2*x) == 2*x**2 + 2*x
    assert R.dup_lcm(x**2 + 2*x, x) == x**2 + 2*x
    assert R.dup_lcm(2*x**2 + x, x) == 2*x**2 + x
    assert R.dup_lcm(2*x**2 + x, 2*x) == 4*x**2 + 2*x


def test_dmp_lcm():
    R, x, y = ring("x,y", ZZ)

    assert R.dmp_lcm(2, 6) == 6
    assert R.dmp_lcm(x, y) == x*y

    assert R.dmp_lcm(2*x**3, 6*x*y**2) == 6*x**3*y**2
    assert R.dmp_lcm(2*x**3, 3*x*y**2) == 6*x**3*y**2

    assert R.dmp_lcm(x**2*y, x*y**2) == x**2*y**2

    f = 2*x*y**5 - 3*x*y**4 - 2*x*y**3 + 3*x*y**2
    g = y**5 - 2*y**3 + y
    h = 2*x*y**7 - 3*x*y**6 - 4*x*y**5 + 6*x*y**4 + 2*x*y**3 - 3*x*y**2

    assert R.dmp_lcm(f, g) == h

    f = x**3 - 3*x**2*y - 9*x*y**2 - 5*y**3
    g = x**4 + 6*x**3*y + 12*x**2*y**2 + 10*x*y**3 + 3*y**4
    h = x**5 + x**4*y - 18*x**3*y**2 - 50*x**2*y**3 - 47*x*y**4 - 15*y**5

    assert R.dmp_lcm(f, g) == h


def test_dmp_content():
    R, x,y = ring("x,y", ZZ)

    assert R.dmp_content(-2) == 2

    f, g, F = 3*y**2 + 2*y + 1, 1, 0

    for i in range(0, 5):
        g *= f
        F += x**i*g

    assert R.dmp_content(F) == f.drop(x)

    R, x,y,z = ring("x,y,z", ZZ)

    assert R.dmp_content(f_4) == 1
    assert R.dmp_content(f_5) == 1

    R, x,y,z,t = ring("x,y,z,t", ZZ)
    assert R.dmp_content(f_6) == 1


def test_dmp_primitive():
    R, x,y = ring("x,y", ZZ)

    assert R.dmp_primitive(0) == (0, 0)
    assert R.dmp_primitive(1) == (1, 1)

    f, g, F = 3*y**2 + 2*y + 1, 1, 0

    for i in range(0, 5):
        g *= f
        F += x**i*g

    assert R.dmp_primitive(F) == (f.drop(x), F / f)

    R, x,y,z = ring("x,y,z", ZZ)

    cont, f = R.dmp_primitive(f_4)
    assert cont == 1 and f == f_4
    cont, f = R.dmp_primitive(f_5)
    assert cont == 1 and f == f_5

    R, x,y,z,t = ring("x,y,z,t", ZZ)

    cont, f = R.dmp_primitive(f_6)
    assert cont == 1 and f == f_6


def test_dup_cancel():
    R, x = ring("x", ZZ)

    f = 2*x**2 - 2
    g = x**2 - 2*x + 1

    p = 2*x + 2
    q = x - 1

    assert R.dup_cancel(f, g) == (p, q)
    assert R.dup_cancel(f, g, include=False) == (1, 1, p, q)

    f = -x - 2
    g = 3*x - 4

    F = x + 2
    G = -3*x + 4

    assert R.dup_cancel(f, g) == (f, g)
    assert R.dup_cancel(F, G) == (f, g)

    assert R.dup_cancel(0, 0) == (0, 0)
    assert R.dup_cancel(0, 0, include=False) == (1, 1, 0, 0)

    assert R.dup_cancel(x, 0) == (1, 0)
    assert R.dup_cancel(x, 0, include=False) == (1, 1, 1, 0)

    assert R.dup_cancel(0, x) == (0, 1)
    assert R.dup_cancel(0, x, include=False) == (1, 1, 0, 1)

    f = 0
    g = x
    one = 1

    assert R.dup_cancel(f, g, include=True) == (f, one)


def test_dmp_cancel():
    R, x, y = ring("x,y", ZZ)

    f = 2*x**2 - 2
    g = x**2 - 2*x + 1

    p = 2*x + 2
    q = x - 1

    assert R.dmp_cancel(f, g) == (p, q)
    assert R.dmp_cancel(f, g, include=False) == (1, 1, p, q)

    assert R.dmp_cancel(0, 0) == (0, 0)
    assert R.dmp_cancel(0, 0, include=False) == (1, 1, 0, 0)

    assert R.dmp_cancel(y, 0) == (1, 0)
    assert R.dmp_cancel(y, 0, include=False) == (1, 1, 1, 0)

    assert R.dmp_cancel(0, y) == (0, 1)
    assert R.dmp_cancel(0, y, include=False) == (1, 1, 0, 1)
