import itertools
from sympy.core import GoldenRatio as phi
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.continued_fraction import \
    (continued_fraction_periodic as cf_p,
     continued_fraction_iterator as cf_i,
     continued_fraction_convergents as cf_c,
     continued_fraction_reduce as cf_r,
     continued_fraction as cf)
from sympy.testing.pytest import raises


def test_continued_fraction():
    assert cf_p(1, 1, 10, 0) == cf_p(1, 1, 0, 1)
    assert cf_p(1, -1, 10, 1) == cf_p(-1, 1, 10, -1)
    t = sqrt(2)
    assert cf((1 + t)*(1 - t)) == cf(-1)
    for n in [0, 2, Rational(2, 3), sqrt(2), 3*sqrt(2), 1 + 2*sqrt(3)/5,
            (2 - 3*sqrt(5))/7, 1 + sqrt(2), (-5 + sqrt(17))/4]:
        assert (cf_r(cf(n)) - n).expand() == 0
        assert (cf_r(cf(-n)) + n).expand() == 0
    raises(ValueError, lambda: cf(sqrt(2 + sqrt(3))))
    raises(ValueError, lambda: cf(sqrt(2) + sqrt(3)))
    raises(ValueError, lambda: cf(pi))
    raises(ValueError, lambda: cf(.1))

    raises(ValueError, lambda: cf_p(1, 0, 0))
    raises(ValueError, lambda: cf_p(1, 1, -1))
    assert cf_p(4, 3, 0) == [1, 3]
    assert cf_p(0, 3, 5) == [0, 1, [2, 1, 12, 1, 2, 2]]
    assert cf_p(1, 1, 0) == [1]
    assert cf_p(3, 4, 0) == [0, 1, 3]
    assert cf_p(4, 5, 0) == [0, 1, 4]
    assert cf_p(5, 6, 0) == [0, 1, 5]
    assert cf_p(11, 13, 0) == [0, 1, 5, 2]
    assert cf_p(16, 19, 0) == [0, 1, 5, 3]
    assert cf_p(27, 32, 0) == [0, 1, 5, 2, 2]
    assert cf_p(1, 2, 5) == [[1]]
    assert cf_p(0, 1, 2) == [1, [2]]
    assert cf_p(6, 7, 49) == [1, 1, 6]
    assert cf_p(3796, 1387, 0) == [2, 1, 2, 1, 4]
    assert cf_p(3245, 10000) == [0, 3, 12, 4, 13]
    assert cf_p(1932, 2568) == [0, 1, 3, 26, 2]
    assert cf_p(6589, 2569) == [2, 1, 1, 3, 2, 1, 3, 1, 23]

    def take(iterator, n=7):
        return list(itertools.islice(iterator, n))

    assert take(cf_i(phi)) == [1, 1, 1, 1, 1, 1, 1]
    assert take(cf_i(pi)) == [3, 7, 15, 1, 292, 1, 1]

    assert list(cf_i(Rational(17, 12))) == [1, 2, 2, 2]
    assert list(cf_i(Rational(-17, 12))) == [-2, 1, 1, 2, 2]

    assert list(cf_c([1, 6, 1, 8])) == [S.One, Rational(7, 6), Rational(8, 7), Rational(71, 62)]
    assert list(cf_c([2])) == [S(2)]
    assert list(cf_c([1, 1, 1, 1, 1, 1, 1])) == [S.One, S(2), Rational(3, 2), Rational(5, 3),
                                                 Rational(8, 5), Rational(13, 8), Rational(21, 13)]
    assert list(cf_c([1, 6, Rational(-1, 2), 4])) == [S.One, Rational(7, 6), Rational(5, 4), Rational(3, 2)]
    assert take(cf_c([[1]])) == [S.One, S(2), Rational(3, 2), Rational(5, 3), Rational(8, 5),
                                 Rational(13, 8), Rational(21, 13)]
    assert take(cf_c([1, [1, 2]])) == [S.One, S(2), Rational(5, 3), Rational(7, 4), Rational(19, 11),
                                    Rational(26, 15), Rational(71, 41)]

    cf_iter_e = (2 if i == 1 else i // 3 * 2 if i % 3 == 0 else 1 for i in itertools.count(1))
    assert take(cf_c(cf_iter_e)) == [S(2), S(3), Rational(8, 3), Rational(11, 4), Rational(19, 7),
                                     Rational(87, 32), Rational(106, 39)]

    assert cf_r([1, 6, 1, 8]) == Rational(71, 62)
    assert cf_r([3]) == S(3)
    assert cf_r([-1, 5, 1, 4]) == Rational(-24, 29)
    assert (cf_r([0, 1, 1, 7, [24, 8]]) - (sqrt(3) + 2)/7).expand() == 0
    assert cf_r([1, 5, 9]) == Rational(55, 46)
    assert (cf_r([[1]]) - (sqrt(5) + 1)/2).expand() == 0
    assert cf_r([-3, 1, 1, [2]]) == -1 - sqrt(2)
