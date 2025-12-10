from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises

i, j, k, l, m = symbols("i j k l m", integer=True, finite=True)
x, y = symbols("x y", commutative=False)


def test_deltaproduct_trivial():
    assert dp(x, (j, 1, 0)) == 1
    assert dp(x, (j, 1, 3)) == x**3
    assert dp(x + y, (j, 1, 3)) == (x + y)**3
    assert dp(x*y, (j, 1, 3)) == (x*y)**3
    assert dp(KD(i, j), (k, 1, 3)) == KD(i, j)
    assert dp(x*KD(i, j), (k, 1, 3)) == x**3*KD(i, j)
    assert dp(x*y*KD(i, j), (k, 1, 3)) == (x*y)**3*KD(i, j)


def test_deltaproduct_basic():
    assert dp(KD(i, j), (j, 1, 3)) == 0
    assert dp(KD(i, j), (j, 1, 1)) == KD(i, 1)
    assert dp(KD(i, j), (j, 2, 2)) == KD(i, 2)
    assert dp(KD(i, j), (j, 3, 3)) == KD(i, 3)
    assert dp(KD(i, j), (j, 1, k)) == KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(KD(i, j), (j, k, 3)) == KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(KD(i, j), (j, k, l)) == KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_mul_x_kd():
    assert dp(x*KD(i, j), (j, 1, 3)) == 0
    assert dp(x*KD(i, j), (j, 1, 1)) == x*KD(i, 1)
    assert dp(x*KD(i, j), (j, 2, 2)) == x*KD(i, 2)
    assert dp(x*KD(i, j), (j, 3, 3)) == x*KD(i, 3)
    assert dp(x*KD(i, j), (j, 1, k)) == x*KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(x*KD(i, j), (j, k, 3)) == x*KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(x*KD(i, j), (j, k, l)) == x*KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_mul_add_x_y_kd():
    assert dp((x + y)*KD(i, j), (j, 1, 3)) == 0
    assert dp((x + y)*KD(i, j), (j, 1, 1)) == (x + y)*KD(i, 1)
    assert dp((x + y)*KD(i, j), (j, 2, 2)) == (x + y)*KD(i, 2)
    assert dp((x + y)*KD(i, j), (j, 3, 3)) == (x + y)*KD(i, 3)
    assert dp((x + y)*KD(i, j), (j, 1, k)) == \
        (x + y)*KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp((x + y)*KD(i, j), (j, k, 3)) == \
        (x + y)*KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp((x + y)*KD(i, j), (j, k, l)) == \
        (x + y)*KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_add_kd_kd():
    assert dp(KD(i, k) + KD(j, k), (k, 1, 3)) == 0
    assert dp(KD(i, k) + KD(j, k), (k, 1, 1)) == KD(i, 1) + KD(j, 1)
    assert dp(KD(i, k) + KD(j, k), (k, 2, 2)) == KD(i, 2) + KD(j, 2)
    assert dp(KD(i, k) + KD(j, k), (k, 3, 3)) == KD(i, 3) + KD(j, 3)
    assert dp(KD(i, k) + KD(j, k), (k, 1, l)) == KD(l, 0) + \
        KD(i, 1)*KD(l, 1) + KD(j, 1)*KD(l, 1) + \
        KD(i, 1)*KD(j, 2)*KD(l, 2) + KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp(KD(i, k) + KD(j, k), (k, l, 3)) == KD(l, 4) + \
        KD(i, 3)*KD(l, 3) + KD(j, 3)*KD(l, 3) + \
        KD(i, 2)*KD(j, 3)*KD(l, 2) + KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp(KD(i, k) + KD(j, k), (k, l, m)) == KD(l, m + 1) + \
        KD(i, m)*KD(l, m) + KD(j, m)*KD(l, m) + \
        KD(i, m)*KD(j, m - 1)*KD(l, m - 1) + KD(i, m - 1)*KD(j, m)*KD(l, m - 1)


def test_deltaproduct_mul_x_add_kd_kd():
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == x*(KD(i, 1) + KD(j, 1))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == x*(KD(i, 2) + KD(j, 2))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == x*(KD(i, 3) + KD(j, 3))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        x*KD(i, 1)*KD(l, 1) + x*KD(j, 1)*KD(l, 1) + \
        x**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + x**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        x*KD(i, 3)*KD(l, 3) + x*KD(j, 3)*KD(l, 3) + \
        x**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + x**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        x*KD(i, m)*KD(l, m) + x*KD(j, m)*KD(l, m) + \
        x**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        x**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


def test_deltaproduct_mul_add_x_y_add_kd_kd():
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == \
        (x + y)*(KD(i, 1) + KD(j, 1))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == \
        (x + y)*(KD(i, 2) + KD(j, 2))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == \
        (x + y)*(KD(i, 3) + KD(j, 3))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        (x + y)*KD(i, 1)*KD(l, 1) + (x + y)*KD(j, 1)*KD(l, 1) + \
        (x + y)**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + \
        (x + y)**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        (x + y)*KD(i, 3)*KD(l, 3) + (x + y)*KD(j, 3)*KD(l, 3) + \
        (x + y)**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + \
        (x + y)**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        (x + y)*KD(i, m)*KD(l, m) + (x + y)*KD(j, m)*KD(l, m) + \
        (x + y)**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        (x + y)**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


def test_deltaproduct_add_mul_x_y_mul_x_kd():
    assert dp(x*y + x*KD(i, j), (j, 1, 3)) == (x*y)**3 + \
        x*(x*y)**2*KD(i, 1) + (x*y)*x*(x*y)*KD(i, 2) + (x*y)**2*x*KD(i, 3)
    assert dp(x*y + x*KD(i, j), (j, 1, 1)) == x*y + x*KD(i, 1)
    assert dp(x*y + x*KD(i, j), (j, 2, 2)) == x*y + x*KD(i, 2)
    assert dp(x*y + x*KD(i, j), (j, 3, 3)) == x*y + x*KD(i, 3)
    assert dp(x*y + x*KD(i, j), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            ((x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        )
    assert dp(x*y + x*KD(i, j), (j, k, 3)) == \
        (x*y)**(-k + 4) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )
    assert dp(x*y + x*KD(i, j), (j, k, l)) == \
        (x*y)**(-k + l + 1) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )


def test_deltaproduct_mul_x_add_y_kd():
    assert dp(x*(y + KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        x*(x*y)**2*KD(i, 1) + (x*y)*x*(x*y)*KD(i, 2) + (x*y)**2*x*KD(i, 3)
    assert dp(x*(y + KD(i, j)), (j, 1, 1)) == x*(y + KD(i, 1))
    assert dp(x*(y + KD(i, j)), (j, 2, 2)) == x*(y + KD(i, 2))
    assert dp(x*(y + KD(i, j)), (j, 3, 3)) == x*(y + KD(i, 3))
    assert dp(x*(y + KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            ((x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    assert dp(x*(y + KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    assert dp(x*(y + KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


def test_deltaproduct_mul_x_add_y_twokd():
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        2*x*(x*y)**2*KD(i, 1) + 2*x*y*x*x*y*KD(i, 2) + 2*(x*y)**2*x*KD(i, 3)
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 1)) == x*(y + 2*KD(i, 1))
    assert dp(x*(y + 2*KD(i, j)), (j, 2, 2)) == x*(y + 2*KD(i, 2))
    assert dp(x*(y + 2*KD(i, j)), (j, 3, 3)) == x*(y + 2*KD(i, 3))
    assert dp(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            (2*(x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    assert dp(x*(y + 2*KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    assert dp(x*(y + 2*KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


def test_deltaproduct_mul_add_x_y_add_y_kd():
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 3)) == ((x + y)*y)**3 + \
        (x + y)*((x + y)*y)**2*KD(i, 1) + \
        (x + y)*y*(x + y)**2*y*KD(i, 2) + \
        ((x + y)*y)**2*(x + y)*KD(i, 3)
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 1)) == (x + y)*(y + KD(i, 1))
    assert dp((x + y)*(y + KD(i, j)), (j, 2, 2)) == (x + y)*(y + KD(i, 2))
    assert dp((x + y)*(y + KD(i, j)), (j, 3, 3)) == (x + y)*(y + KD(i, 3))
    assert dp((x + y)*(y + KD(i, j)), (j, 1, k)) == \
        ((x + y)*y)**k + Piecewise(
            (((x + y)*y)**(-1)*((x + y)*y)**i*(x + y)*((x + y)*y
            )**k*((x + y)*y)**(-i), (i >= 1) & (i <= k)), (0, True))
    assert dp((x + y)*(y + KD(i, j)), (j, k, 3)) == (
        (x + y)*y)**4*((x + y)*y)**(-k) + Piecewise((((x + y)*y)**i*(
        (x + y)*y)**(-k)*(x + y)*((x + y)*y)**3*((x + y)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    assert dp((x + y)*(y + KD(i, j)), (j, k, l)) == \
        (x + y)*y*((x + y)*y)**l*((x + y)*y)**(-k) + Piecewise(
        (((x + y)*y)**i*((x + y)*y)**(-k)*(x + y)*((x + y)*y
        )**l*((x + y)*y)**(-i), (i >= k) & (i <= l)), (0, True))


def test_deltaproduct_mul_add_x_kd_add_y_kd():
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 3)) == \
        KD(i, 1)*(KD(i, k) + x)*((KD(i, k) + x)*y)**2 + \
        KD(i, 2)*(KD(i, k) + x)*y*(KD(i, k) + x)**2*y + \
        KD(i, 3)*((KD(i, k) + x)*y)**2*(KD(i, k) + x) + \
        ((KD(i, k) + x)*y)**3
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == \
        (x + KD(i, k))*(y + KD(i, 1))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == \
        (x + KD(i, k))*(y + KD(i, 2))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == \
        (x + KD(i, k))*(y + KD(i, 3))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == \
        ((KD(i, k) + x)*y)**k + Piecewise(
        (((KD(i, k) + x)*y)**(-1)*((KD(i, k) + x)*y)**i*(KD(i, k) + x
        )*((KD(i, k) + x)*y)**k*((KD(i, k) + x)*y)**(-i), (i >= 1
        ) & (i <= k)), (0, True))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == (
        (KD(i, k) + x)*y)**4*((KD(i, k) + x)*y)**(-k) + Piecewise(
        (((KD(i, k) + x)*y)**i*((KD(i, k) + x)*y)**(-k)*(KD(i, k)
        + x)*((KD(i, k) + x)*y)**3*((KD(i, k) + x)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == (
        KD(i, k) + x)*y*((KD(i, k) + x)*y)**l*((KD(i, k) + x)*y
        )**(-k) + Piecewise((((KD(i, k) + x)*y)**i*((KD(i, k) + x
        )*y)**(-k)*(KD(i, k) + x)*((KD(i, k) + x)*y)**l*((KD(i, k) + x
        )*y)**(-i), (i >= k) & (i <= l)), (0, True))


def test_deltasummation_trivial():
    assert ds(x, (j, 1, 0)) == 0
    assert ds(x, (j, 1, 3)) == 3*x
    assert ds(x + y, (j, 1, 3)) == 3*(x + y)
    assert ds(x*y, (j, 1, 3)) == 3*x*y
    assert ds(KD(i, j), (k, 1, 3)) == 3*KD(i, j)
    assert ds(x*KD(i, j), (k, 1, 3)) == 3*x*KD(i, j)
    assert ds(x*y*KD(i, j), (k, 1, 3)) == 3*x*y*KD(i, j)


def test_deltasummation_basic_numerical():
    n = symbols('n', integer=True, nonzero=True)
    assert ds(KD(n, 0), (n, 1, 3)) == 0

    # return unevaluated, until it gets implemented
    assert ds(KD(i**2, j**2), (j, -oo, oo)) == \
        Sum(KD(i**2, j**2), (j, -oo, oo))

    assert Piecewise((KD(i, k), And(1 <= i, i <= 3)), (0, True)) == \
        ds(KD(i, j)*KD(j, k), (j, 1, 3)) == \
        ds(KD(j, k)*KD(i, j), (j, 1, 3))

    assert ds(KD(i, k), (k, -oo, oo)) == 1
    assert ds(KD(i, k), (k, 0, oo)) == Piecewise((1, S.Zero <= i), (0, True))
    assert ds(KD(i, k), (k, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    assert ds(k*KD(i, j)*KD(j, k), (k, -oo, oo)) == j*KD(i, j)
    assert ds(j*KD(i, j), (j, -oo, oo)) == i
    assert ds(i*KD(i, j), (i, -oo, oo)) == j
    assert ds(x, (i, 1, 3)) == 3*x
    assert ds((i + j)*KD(i, j), (j, -oo, oo)) == 2*i


def test_deltasummation_basic_symbolic():
    assert ds(KD(i, j), (j, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    assert ds(KD(i, j), (j, 1, 1)) == Piecewise((1, Eq(i, 1)), (0, True))
    assert ds(KD(i, j), (j, 2, 2)) == Piecewise((1, Eq(i, 2)), (0, True))
    assert ds(KD(i, j), (j, 3, 3)) == Piecewise((1, Eq(i, 3)), (0, True))
    assert ds(KD(i, j), (j, 1, k)) == \
        Piecewise((1, And(1 <= i, i <= k)), (0, True))
    assert ds(KD(i, j), (j, k, 3)) == \
        Piecewise((1, And(k <= i, i <= 3)), (0, True))
    assert ds(KD(i, j), (j, k, l)) == \
        Piecewise((1, And(k <= i, i <= l)), (0, True))


def test_deltasummation_mul_x_kd():
    assert ds(x*KD(i, j), (j, 1, 3)) == \
        Piecewise((x, And(1 <= i, i <= 3)), (0, True))
    assert ds(x*KD(i, j), (j, 1, 1)) == Piecewise((x, Eq(i, 1)), (0, True))
    assert ds(x*KD(i, j), (j, 2, 2)) == Piecewise((x, Eq(i, 2)), (0, True))
    assert ds(x*KD(i, j), (j, 3, 3)) == Piecewise((x, Eq(i, 3)), (0, True))
    assert ds(x*KD(i, j), (j, 1, k)) == \
        Piecewise((x, And(1 <= i, i <= k)), (0, True))
    assert ds(x*KD(i, j), (j, k, 3)) == \
        Piecewise((x, And(k <= i, i <= 3)), (0, True))
    assert ds(x*KD(i, j), (j, k, l)) == \
        Piecewise((x, And(k <= i, i <= l)), (0, True))


def test_deltasummation_mul_add_x_y_kd():
    assert ds((x + y)*KD(i, j), (j, 1, 3)) == \
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 1, 1)) == \
        Piecewise((x + y, Eq(i, 1)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 2, 2)) == \
        Piecewise((x + y, Eq(i, 2)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 3, 3)) == \
        Piecewise((x + y, Eq(i, 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 1, k)) == \
        Piecewise((x + y, And(1 <= i, i <= k)), (0, True))
    assert ds((x + y)*KD(i, j), (j, k, 3)) == \
        Piecewise((x + y, And(k <= i, i <= 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, k, l)) == \
        Piecewise((x + y, And(k <= i, i <= l)), (0, True))


def test_deltasummation_add_kd_kd():
    assert ds(KD(i, k) + KD(j, k), (k, 1, 3)) == piecewise_fold(
        Piecewise((1, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        Piecewise((1, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        Piecewise((1, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 3, 3)) == piecewise_fold(
        Piecewise((1, Eq(i, 3)), (0, True)) +
        Piecewise((1, Eq(j, 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 1, l)) == piecewise_fold(
        Piecewise((1, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= l)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, l, 3)) == piecewise_fold(
        Piecewise((1, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(l <= j, j <= 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, l, m)) == piecewise_fold(
        Piecewise((1, And(l <= i, i <= m)), (0, True)) +
        Piecewise((1, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_add_mul_x_kd_kd():
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, 3)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 3, 3)) == piecewise_fold(
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((1, Eq(j, 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, l)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= l)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, l, 3)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(l <= j, j <= 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, l, m)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((1, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_mul_x_add_kd_kd():
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((x, Eq(j, 1)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((x, Eq(j, 2)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((x, Eq(j, 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= l)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(l <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_mul_add_x_y_add_kd_kd():
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 1)), (0, True)) +
        Piecewise((x + y, Eq(j, 1)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 2)), (0, True)) +
        Piecewise((x + y, Eq(j, 2)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 3)), (0, True)) +
        Piecewise((x + y, Eq(j, 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= l)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_add_mul_x_y_mul_x_kd():
    assert ds(x*y + x*KD(i, j), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_x_add_y_kd():
    assert ds(x*(y + KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_x_add_y_twokd():
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + 2*x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + 2*x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + 2*x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + 2*x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + 2*x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*x*y + 2*x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + 2*x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_add_x_y_add_y_kd():
    assert ds((x + y)*(y + KD(i, j)), (j, 1, 3)) == Piecewise(
        (3*(x + y)*y + x + y, And(1 <= i, i <= 3)), (3*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 1)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 2)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 3)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 1, k)) == Piecewise(
        (k*(x + y)*y + x + y, And(1 <= i, i <= k)), (k*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*(x + y)*y + x + y, And(k <= i, i <= 3)),
        ((4 - k)*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*(x + y)*y + x + y, And(k <= i, i <= l)),
        ((l - k + 1)*(x + y)*y, True))


def test_deltasummation_mul_add_x_kd_add_y_kd():
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(1 <= i, i <= 3)), (0, True)) +
        3*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 1)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 2)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 3)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(1 <= i, i <= k)), (0, True)) +
        k*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(k <= i, i <= 3)), (0, True)) +
        (4 - k)*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(k <= i, i <= l)), (0, True)) +
        (l - k + 1)*(KD(i, k) + x)*y)


def test_extract_delta():
    raises(ValueError, lambda: _extract_delta(KD(i, j) + KD(k, l), i))
