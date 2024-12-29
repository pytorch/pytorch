from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
    convolution, convolution_fft, convolution_ntt, convolution_fwht,
    convolution_subset, covering_product, intersecting_product,
    convolution_int)
from sympy.testing.pytest import raises
from sympy.abc import x, y

def test_convolution():
    # fft
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]
    c = [3, 5, 3, 7, 8]
    d = [1422, 6572, 3213, 5552]
    e = [-1, Rational(5, 3), Rational(7, 5)]

    assert convolution(a, b) == convolution_fft(a, b)
    assert convolution(a, b, dps=9) == convolution_fft(a, b, dps=9)
    assert convolution(a, d, dps=7) == convolution_fft(d, a, dps=7)
    assert convolution(a, d[1:], dps=3) == convolution_fft(d[1:], a, dps=3)

    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 19*2**10 + 1

    # ntt
    assert convolution(d, b, prime=q) == convolution_ntt(b, d, prime=q)
    assert convolution(c, b, prime=p) == convolution_ntt(b, c, prime=p)
    assert convolution(d, c, prime=p) == convolution_ntt(c, d, prime=p)
    raises(TypeError, lambda: convolution(b, d, dps=5, prime=q))
    raises(TypeError, lambda: convolution(b, d, dps=6, prime=q))

    # fwht
    assert convolution(a, b, dyadic=True) == convolution_fwht(a, b)
    assert convolution(a, b, dyadic=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(b, d, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, d, prime=p, dyadic=True))
    raises(TypeError, lambda: convolution(a, b, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, c, prime=p, dyadic=True))

    # subset
    assert convolution(a, b, subset=True) == convolution_subset(a, b) == \
            convolution(a, b, subset=True, dyadic=False) == \
                convolution(a, b, subset=True)
    assert convolution(a, b, subset=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(a, b, subset=True, dyadic=True))
    raises(TypeError, lambda: convolution(c, d, subset=True, dps=6))
    raises(TypeError, lambda: convolution(a, c, subset=True, prime=q))

    # integer
    assert convolution([0], [0]) == convolution_int([0], [0])
    assert convolution(b, c) == convolution_int(b, c)

    # rational
    assert convolution([Rational(1,2)], [Rational(1,2)]) == [Rational(1, 4)]
    assert convolution(b, e) == [-9, 10, Rational(239, 15), Rational(34, 3),
                                 Rational(32, 3), Rational(43, 5), Rational(113, 15),
                                 Rational(14, 5)]


def test_cyclic_convolution():
    # fft
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]

    assert convolution([1, 2, 3], [4, 5, 6], cycle=0) == \
            convolution([1, 2, 3], [4, 5, 6], cycle=5) == \
                convolution([1, 2, 3], [4, 5, 6])

    assert convolution([1, 2, 3], [4, 5, 6], cycle=3) == [31, 31, 28]

    a = [Rational(1, 3), Rational(7, 3), Rational(5, 9), Rational(2, 7), Rational(5, 8)]
    b = [Rational(3, 5), Rational(4, 7), Rational(7, 8), Rational(8, 9)]

    assert convolution(a, b, cycle=0) == \
            convolution(a, b, cycle=len(a) + len(b) - 1)

    assert convolution(a, b, cycle=4) == [Rational(87277, 26460), Rational(30521, 11340),
                            Rational(11125, 4032), Rational(3653, 1080)]

    assert convolution(a, b, cycle=6) == [Rational(20177, 20160), Rational(676, 315), Rational(47, 24),
                            Rational(3053, 1080), Rational(16397, 5292), Rational(2497, 2268)]

    assert convolution(a, b, cycle=9) == \
                convolution(a, b, cycle=0) + [S.Zero]

    # ntt
    a = [2313, 5323532, S(3232), 42142, 42242421]
    b = [S(33456), 56757, 45754, 432423]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=0) == \
            convolution(a, b, prime=19*2**10 + 1, cycle=8) == \
                convolution(a, b, prime=19*2**10 + 1)

    assert convolution(a, b, prime=19*2**10 + 1, cycle=5) == [96, 17146, 2664,
                                                                    15534, 3517]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=7) == [4643, 3458, 1260,
                                                        15534, 3517, 16314, 13688]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=9) == \
            convolution(a, b, prime=19*2**10 + 1) + [0]

    # fwht
    u, v, w, x, y = symbols('u v w x y')
    p, q, r, s, t = symbols('p q r s t')
    c = [u, v, w, x, y]
    d = [p, q, r, s, t]

    assert convolution(a, b, dyadic=True, cycle=3) == \
                        [2499522285783, 19861417974796, 4702176579021]

    assert convolution(a, b, dyadic=True, cycle=5) == [2718149225143,
            2114320852171, 20571217906407, 246166418903, 1413262436976]

    assert convolution(c, d, dyadic=True, cycle=4) == \
            [p*u + p*y + q*v + r*w + s*x + t*u + t*y,
             p*v + q*u + q*y + r*x + s*w + t*v,
             p*w + q*x + r*u + r*y + s*v + t*w,
             p*x + q*w + r*v + s*u + s*y + t*x]

    assert convolution(c, d, dyadic=True, cycle=6) == \
            [p*u + q*v + r*w + r*y + s*x + t*w + t*y,
             p*v + q*u + r*x + s*w + s*y + t*x,
             p*w + q*x + r*u + s*v,
             p*x + q*w + r*v + s*u,
             p*y + t*u,
             q*y + t*v]

    # subset
    assert convolution(a, b, subset=True, cycle=7) == [18266671799811,
                    178235365533, 213958794, 246166418903, 1413262436976,
                    2397553088697, 1932759730434]

    assert convolution(a[1:], b, subset=True, cycle=4) == \
            [178104086592, 302255835516, 244982785880, 3717819845434]

    assert convolution(a, b[:-1], subset=True, cycle=6) == [1932837114162,
            178235365533, 213958794, 245166224504, 1413262436976, 2397553088697]

    assert convolution(c, d, subset=True, cycle=3) == \
            [p*u + p*x + q*w + r*v + r*y + s*u + t*w,
             p*v + p*y + q*u + s*y + t*u + t*x,
             p*w + q*y + r*u + t*v]

    assert convolution(c, d, subset=True, cycle=5) == \
            [p*u + q*y + t*v,
             p*v + q*u + r*y + t*w,
             p*w + r*u + s*y + t*x,
             p*x + q*w + r*v + s*u,
             p*y + t*u]

    raises(ValueError, lambda: convolution([1, 2, 3], [4, 5, 6], cycle=-1))


def test_convolution_fft():
    assert all(convolution_fft([], x, dps=y) == [] for x in ([], [1]) for y in (None, 3))
    assert convolution_fft([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert convolution_fft([1], [5, 6, 7]) == [5, 6, 7]
    assert convolution_fft([1, 3], [5, 6, 7]) == [5, 21, 25, 21]

    assert convolution_fft([1 + 2*I], [2 + 3*I]) == [-4 + 7*I]
    assert convolution_fft([1 + 2*I, 3 + 4*I, 5 + 3*I/5], [Rational(2, 5) + 4*I/7]) == \
            [Rational(-26, 35) + I*48/35, Rational(-38, 35) + I*116/35, Rational(58, 35) + I*542/175]

    assert convolution_fft([Rational(3, 4), Rational(5, 6)], [Rational(7, 8), Rational(1, 3), Rational(2, 5)]) == \
                                    [Rational(21, 32), Rational(47, 48), Rational(26, 45), Rational(1, 3)]

    assert convolution_fft([Rational(1, 9), Rational(2, 3), Rational(3, 5)], [Rational(2, 5), Rational(3, 7), Rational(4, 9)]) == \
                                [Rational(2, 45), Rational(11, 35), Rational(8152, 14175), Rational(523, 945), Rational(4, 15)]

    assert convolution_fft([pi, E, sqrt(2)], [sqrt(3), 1/pi, 1/E]) == \
                    [sqrt(3)*pi, 1 + sqrt(3)*E, E/pi + pi*exp(-1) + sqrt(6),
                                            sqrt(2)/pi + 1, sqrt(2)*exp(-1)]

    assert convolution_fft([2321, 33123], [5321, 6321, 71323]) == \
                        [12350041, 190918524, 374911166, 2362431729]

    assert convolution_fft([312313, 31278232], [32139631, 319631]) == \
                        [10037624576503, 1005370659728895, 9997492572392]

    raises(TypeError, lambda: convolution_fft(x, y))
    raises(ValueError, lambda: convolution_fft([x, y], [y, x]))


def test_convolution_ntt():
    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 19*2**10 + 1
    r = 2*500000003 + 1 # only for sequences of length 1 or 2
    # s = 2*3*5*7 # composite modulus

    assert all(convolution_ntt([], x, prime=y) == [] for x in ([], [1]) for y in (p, q, r))
    assert convolution_ntt([2], [3], r) == [6]
    assert convolution_ntt([2, 3], [4], r) == [8, 12]

    assert convolution_ntt([32121, 42144, 4214, 4241], [32132, 3232, 87242], p) == [33867619,
                                    459741727, 79180879, 831885249, 381344700, 369993322]
    assert convolution_ntt([121913, 3171831, 31888131, 12], [17882, 21292, 29921, 312], q) == \
                                                [8158, 3065, 3682, 7090, 1239, 2232, 3744]

    assert convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], p) == \
                    convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], q)
    assert convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], p) == \
                    convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], q)

    raises(ValueError, lambda: convolution_ntt([2, 3], [4, 5], r))
    raises(ValueError, lambda: convolution_ntt([x, y], [y, x], q))
    raises(TypeError, lambda: convolution_ntt(x, y, p))


def test_convolution_fwht():
    assert convolution_fwht([], []) == []
    assert convolution_fwht([], [1]) == []
    assert convolution_fwht([1, 2, 3], [4, 5, 6]) == [32, 13, 18, 27]

    assert convolution_fwht([Rational(5, 7), Rational(6, 8), Rational(7, 3)], [2, 4, Rational(6, 7)]) == \
                                    [Rational(45, 7), Rational(61, 14), Rational(776, 147), Rational(419, 42)]

    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5), 4 + 5*I]
    b = [94, 51, 53, 45, 31, 27, 13]
    c = [3 + 4*I, 5 + 7*I, 3, Rational(7, 6), 8]

    assert convolution_fwht(a, b) == [53*sqrt(3) + 366 + 155*I,
                                    45*sqrt(3) + Rational(5848, 15) + 135*I,
                                    94*sqrt(3) + Rational(1257, 5) + 65*I,
                                    51*sqrt(3) + Rational(3974, 15),
                                    13*sqrt(3) + 452 + 470*I,
                                    Rational(4513, 15) + 255*I,
                                    31*sqrt(3) + Rational(1314, 5) + 265*I,
                                    27*sqrt(3) + Rational(3676, 15) + 225*I]

    assert convolution_fwht(b, c) == [Rational(1993, 2) + 733*I, Rational(6215, 6) + 862*I,
        Rational(1659, 2) + 527*I, Rational(1988, 3) + 551*I, 1019 + 313*I, Rational(3955, 6) + 325*I,
        Rational(1175, 2) + 52*I, Rational(3253, 6) + 91*I]

    assert convolution_fwht(a[3:], c) == [Rational(-54, 5) + I*293/5, -1 + I*204/5,
            Rational(133, 15) + I*35/6, Rational(409, 30) + 15*I, Rational(56, 5), 32 + 40*I, 0, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert convolution_fwht([u, v], [x, y]) == [u*x + v*y, u*y + v*x]

    assert convolution_fwht([u, v, w], [x, y]) == \
        [u*x + v*y, u*y + v*x, w*x, w*y]

    assert convolution_fwht([u, v, w], [x, y, z]) == \
        [u*x + v*y + w*z, u*y + v*x, u*z + w*x, v*z + w*y]

    raises(TypeError, lambda: convolution_fwht(x, y))
    raises(TypeError, lambda: convolution_fwht(x*y, u + v))


def test_convolution_subset():
    assert convolution_subset([], []) == []
    assert convolution_subset([], [Rational(1, 3)]) == []
    assert convolution_subset([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, Rational(5, 3), sqrt(3), 4 + 5*I]
    b = [64, 71, 55, 47, 33, 29, 15]
    c = [3 + I*2/3, 5 + 7*I, 7, Rational(7, 5), 9]

    assert convolution_subset(a, b) == [64, Rational(533, 3), 55 + 64*sqrt(3),
                                        71*sqrt(3) + Rational(1184, 3) + 320*I, 33, 84,
                                        15 + 33*sqrt(3), 29*sqrt(3) + 157 + 165*I]

    assert convolution_subset(b, c) == [192 + I*128/3, 533 + I*1486/3,
                                        613 + I*110/3, Rational(5013, 5) + I*1249/3,
                                        675 + 22*I, 891 + I*751/3,
                                        771 + 10*I, Rational(3736, 5) + 105*I]

    assert convolution_subset(a, c) == convolution_subset(c, a)
    assert convolution_subset(a[:2], b) == \
            [64, Rational(533, 3), 55, Rational(416, 3), 33, 84, 15, 25]

    assert convolution_subset(a[:2], c) == \
            [3 + I*2/3, 10 + I*73/9, 7, Rational(196, 15), 9, 15, 0, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert convolution_subset([u, v, w], [x, y]) == [u*x, u*y + v*x, w*x, w*y]
    assert convolution_subset([u, v, w, x], [y, z]) == \
                            [u*y, u*z + v*y, w*y, w*z + x*y]

    assert convolution_subset([u, v], [x, y, z]) == \
                    convolution_subset([x, y, z], [u, v])

    raises(TypeError, lambda: convolution_subset(x, z))
    raises(TypeError, lambda: convolution_subset(Rational(7, 3), u))


def test_covering_product():
    assert covering_product([], []) == []
    assert covering_product([], [Rational(1, 3)]) == []
    assert covering_product([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, Rational(5, 8), sqrt(7), 4 + 9*I]
    b = [66, 81, 95, 49, 37, 89, 17]
    c = [3 + I*2/3, 51 + 72*I, 7, Rational(7, 15), 91]

    assert covering_product(a, b) == [66, Rational(1383, 8), 95 + 161*sqrt(7),
                                        130*sqrt(7) + 1303 + 2619*I, 37,
                                        Rational(671, 4), 17 + 54*sqrt(7),
                                        89*sqrt(7) + Rational(4661, 8) + 1287*I]

    assert covering_product(b, c) == [198 + 44*I, 7740 + 10638*I,
                                        1412 + I*190/3, Rational(42684, 5) + I*31202/3,
                                        9484 + I*74/3, 22163 + I*27394/3,
                                        10621 + I*34/3, Rational(90236, 15) + 1224*I]

    assert covering_product(a, c) == covering_product(c, a)
    assert covering_product(b, c[:-1]) == [198 + 44*I, 7740 + 10638*I,
                                         1412 + I*190/3, Rational(42684, 5) + I*31202/3,
                                         111 + I*74/3, 6693 + I*27394/3,
                                         429 + I*34/3, Rational(23351, 15) + 1224*I]

    assert covering_product(a, c[:-1]) == [3 + I*2/3,
                            Rational(339, 4) + I*1409/12, 7 + 10*sqrt(7) + 2*sqrt(7)*I/3,
                            -403 + 772*sqrt(7)/15 + 72*sqrt(7)*I + I*12658/15]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert covering_product([u, v, w], [x, y]) == \
                            [u*x, u*y + v*x + v*y, w*x, w*y]

    assert covering_product([u, v, w, x], [y, z]) == \
                            [u*y, u*z + v*y + v*z, w*y, w*z + x*y + x*z]

    assert covering_product([u, v], [x, y, z]) == \
                    covering_product([x, y, z], [u, v])

    raises(TypeError, lambda: covering_product(x, z))
    raises(TypeError, lambda: covering_product(Rational(7, 3), u))


def test_intersecting_product():
    assert intersecting_product([], []) == []
    assert intersecting_product([], [Rational(1, 3)]) == []
    assert intersecting_product([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, sqrt(5), Rational(3, 8) + 5*I, 4 + 7*I]
    b = [67, 51, 65, 48, 36, 79, 27]
    c = [3 + I*2/5, 5 + 9*I, 7, Rational(7, 19), 13]

    assert intersecting_product(a, b) == [195*sqrt(5) + Rational(6979, 8) + 1886*I,
                                178*sqrt(5) + 520 + 910*I, Rational(841, 2) + 1344*I,
                                192 + 336*I, 0, 0, 0, 0]

    assert intersecting_product(b, c) == [Rational(128553, 19) + I*9521/5,
                Rational(17820, 19) + 1602*I, Rational(19264, 19), Rational(336, 19), 1846, 0, 0, 0]

    assert intersecting_product(a, c) == intersecting_product(c, a)
    assert intersecting_product(b[1:], c[:-1]) == [Rational(64788, 19) + I*8622/5,
                    Rational(12804, 19) + 1152*I, Rational(11508, 19), Rational(252, 19), 0, 0, 0, 0]

    assert intersecting_product(a, c[:-2]) == \
                    [Rational(-99, 5) + 10*sqrt(5) + 2*sqrt(5)*I/5 + I*3021/40,
                    -43 + 5*sqrt(5) + 9*sqrt(5)*I + 71*I, Rational(245, 8) + 84*I, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert intersecting_product([u, v, w], [x, y]) == \
                            [u*x + u*y + v*x + w*x + w*y, v*y, 0, 0]

    assert intersecting_product([u, v, w, x], [y, z]) == \
                        [u*y + u*z + v*y + w*y + w*z + x*y, v*z + x*z, 0, 0]

    assert intersecting_product([u, v], [x, y, z]) == \
                    intersecting_product([x, y, z], [u, v])

    raises(TypeError, lambda: intersecting_product(x, z))
    raises(TypeError, lambda: intersecting_product(u, Rational(8, 3)))


def test_convolution_int():
    assert convolution_int([1], [1]) == [1]
    assert convolution_int([1, 1], [0]) == [0]
    assert convolution_int([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert convolution_int([1], [5, 6, 7]) == [5, 6, 7]
    assert convolution_int([1, 3], [5, 6, 7]) == [5, 21, 25, 21]
    assert convolution_int([10, -5, 1, 3], [-5, 6, 7]) == [-50, 85, 35, -44, 25, 21]
    assert convolution_int([0, 1, 0, -1], [1, 0, -1, 0]) == [0, 1, 0, -2, 0, 1]
    assert convolution_int(
        [-341, -5, 1, 3, -71, -99, 43, 87],
        [5, 6, 7, 12, 345, 21, -78, -7, -89]
    ) == [-1705, -2071, -2412, -4106, -118035, -9774, 25998, 2981, 5509,
          -34317, 19228, 38870, 5485, 1724, -4436, -7743]
