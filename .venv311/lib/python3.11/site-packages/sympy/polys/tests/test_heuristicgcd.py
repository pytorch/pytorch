from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from sympy.polys.heuristicgcd import heugcd


def test_heugcd_univariate_integers():
    R, x = ring("x", ZZ)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert heugcd(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert heugcd(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert heugcd(f, g) == (h, cff, cfg)

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

    # TODO: assert heugcd(f, f.diff(x))[0] == g

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert heugcd(f, g) == (h, cff, cfg)

def test_heugcd_multivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert heugcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert heugcd(f, g) == (x + 1, 1, 2*x + 2)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert heugcd(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert heugcd(f, g) == (h, cff, cfg)
    assert heugcd(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, t = ring("x,y,z,t", ZZ)

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g


def test_issue_10996():
    R, x, y, z = ring("x,y,z", ZZ)

    f = 12*x**6*y**7*z**3 - 3*x**4*y**9*z**3 + 12*x**3*y**5*z**4
    g = -48*x**7*y**8*z**3 + 12*x**5*y**10*z**3 - 48*x**5*y**7*z**2 + \
    36*x**4*y**7*z - 48*x**4*y**6*z**4 + 12*x**3*y**9*z**2 - 48*x**3*y**4 \
    - 9*x**2*y**9*z - 48*x**2*y**5*z**3 + 12*x*y**6 + 36*x*y**5*z**2 - 48*y**2*z

    H, cff, cfg = heugcd(f, g)

    assert H == 12*x**3*y**4 - 3*x*y**6 + 12*y**2*z
    assert H*cff == f and H*cfg == g


def test_issue_25793():
    R, x = ring("x", ZZ)
    f = x - 4851  # failure starts for values more than 4850
    g = f*(2*x + 1)
    H, cff, cfg = R.dup_zz_heu_gcd(f, g)
    assert H == f
    # needs a test for dmp, too, that fails in master before this change
