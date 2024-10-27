from sympy.core.numbers import Rational
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.core import S, symbols
from sympy.testing.pytest import raises
from sympy.discrete.recurrences import linrec

def test_linrec():
    assert linrec(coeffs=[1, 1], init=[1, 1], n=20) == 10946
    assert linrec(coeffs=[1, 2, 3, 4, 5], init=[1, 1, 0, 2], n=10) == 1040
    assert linrec(coeffs=[0, 0, 11, 13], init=[23, 27], n=25) == 59628567384
    assert linrec(coeffs=[0, 0, 1, 1, 2], init=[1, 5, 3], n=15) == 165
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=70) == \
        56889923441670659718376223533331214868804815612050381493741233489928913241
    assert linrec(coeffs=[0]*55 + [1, 1, 2, 3], init=[0]*50 + [1, 2, 3], n=4000) == \
        702633573874937994980598979769135096432444135301118916539

    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**4)
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**5)

    assert all(linrec(coeffs=[1, 1], init=[0, 1], n=n) == fibonacci(n)
                                                    for n in range(95, 115))

    assert all(linrec(coeffs=[1, 1], init=[1, 1], n=n) == fibonacci(n + 1)
                                                    for n in range(595, 615))

    a = [S.Half, Rational(3, 4), Rational(5, 6), 7, Rational(8, 9), Rational(3, 5)]
    b = [1, 2, 8, Rational(5, 7), Rational(3, 7), Rational(2, 9), 6]
    x, y, z = symbols('x y z')

    assert linrec(coeffs=a[:5], init=b[:4], n=80) == \
        Rational(1726244235456268979436592226626304376013002142588105090705187189,
            1960143456748895967474334873705475211264)

    assert linrec(coeffs=a[:4], init=b[:4], n=50) == \
        Rational(368949940033050147080268092104304441, 504857282956046106624)

    assert linrec(coeffs=a[3:], init=b[:3], n=35) == \
        Rational(97409272177295731943657945116791049305244422833125109,
            814315512679031689453125)

    assert linrec(coeffs=[0]*60 + [Rational(2, 3), Rational(4, 5)], init=b, n=3000) == \
        Rational(26777668739896791448594650497024, 48084516708184142230517578125)

    raises(TypeError, lambda: linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4, 5], n=1))
    raises(TypeError, lambda: linrec(coeffs=a[:4], init=b[:5], n=10000))
    raises(ValueError, lambda: linrec(coeffs=a[:4], init=b[:4], n=-10000))
    raises(TypeError, lambda: linrec(x, b, n=10000))
    raises(TypeError, lambda: linrec(a, y, n=10000))

    assert linrec(coeffs=[x, y, z], init=[1, 1, 1], n=4) == \
        x**2  + x*y + x*z + y + z
    assert linrec(coeffs=[1, 2, 1], init=[x, y, z], n=20) == \
        269542*x + 664575*y + 578949*z
    assert linrec(coeffs=[0, 3, 1, 2], init=[x, y], n=30) == \
        58516436*x + 56372788*y
    assert linrec(coeffs=[0]*50 + [1, 2, 3], init=[x, y, z], n=1000) == \
        11477135884896*x + 25999077948732*y + 41975630244216*z
    assert linrec(coeffs=[], init=[1, 1], n=20) == 0
    assert linrec(coeffs=[x, y, z], init=[1, 2, 3], n=2) == 3
