import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
    Pegasus, Anderson, Ridder, ANewton, Newton, MNewton, MDNewton

def test_findroot():
    # old tests, assuming secant
    mp.dps = 15
    assert findroot(lambda x: 4*x-3, mpf(5)).ae(0.75)
    assert findroot(sin, mpf(3)).ae(pi)
    assert findroot(sin, (mpf(3), mpf(3.14))).ae(pi)
    assert findroot(lambda x: x*x+1, mpc(2+2j)).ae(1j)
    # test all solvers with 1 starting point
    f = lambda x: cos(x)
    for solver in [Newton, Secant, MNewton, Muller, ANewton]:
        x = findroot(f, 2., solver=solver)
        assert abs(f(x)) < eps
    # test all solvers with interval of 2 points
    for solver in [Secant, Muller, Bisection, Illinois, Pegasus, Anderson,
                   Ridder]:
        x = findroot(f, (1., 2.), solver=solver)
        assert abs(f(x)) < eps
    # test types
    f = lambda x: (x - 2)**2

    assert isinstance(findroot(f, 1, tol=1e-10), mpf)
    assert isinstance(iv.findroot(f, 1., tol=1e-10), iv.mpf)
    assert isinstance(fp.findroot(f, 1, tol=1e-10), float)
    assert isinstance(fp.findroot(f, 1+0j, tol=1e-10), complex)

    # issue 401
    with pytest.raises(ValueError):
        with workprec(2):
            findroot(lambda x: x**2 - 4456178*x + 60372201703370,
                     mpc(real='5.278e+13', imag='-5.278e+13'))

    # issue 192
    with pytest.raises(ValueError):
        findroot(lambda x: -1, 0)

    # issue 387
    with pytest.raises(ValueError):
        findroot(lambda p: (1 - p)**30 - 1, 0.9)

def test_bisection():
    # issue 273
    assert findroot(lambda x: x**2-1,(0,2),solver='bisect') == 1

def test_mnewton():
    f = lambda x: polyval([1,3,3,1],x)
    x = findroot(f, -0.9, solver='mnewton')
    assert abs(f(x)) < eps

def test_anewton():
    f = lambda x: (x - 2)**100
    x = findroot(f, 1., solver=ANewton)
    assert abs(f(x)) < eps

def test_muller():
    f = lambda x: (2 + x)**3 + 2
    x = findroot(f, 1., solver=Muller)
    assert abs(f(x)) < eps

def test_multiplicity():
    for i in range(1, 5):
        assert multiplicity(lambda x: (x - 1)**i, 1) == i
    assert multiplicity(lambda x: x**2, 1) == 0

def test_multidimensional():
    def f(*x):
        return [3*x[0]**2-2*x[1]**2-1, x[0]**2-2*x[0]+x[1]**2+2*x[1]-8]
    assert mnorm(jacobian(f, (1,-2)) - matrix([[6,8],[0,-2]]),1) < 1.e-7
    for x, error in MDNewton(mp, f, (1,-2), verbose=0,
                             norm=lambda x: norm(x, inf)):
        pass
    assert norm(f(*x), 2) < 1e-14
    # The Chinese mathematician Zhu Shijie was the very first to solve this
    # nonlinear system 700 years ago
    f1 = lambda x, y: -x + 2*y
    f2 = lambda x, y: (x**2 + x*(y**2 - 2) - 4*y)  /  (x + 4)
    f3 = lambda x, y: sqrt(x**2 + y**2)
    def f(x, y):
        f1x = f1(x, y)
        return (f2(x, y) - f1x, f3(x, y) - f1x)
    x = findroot(f, (10, 10))
    assert [int(round(i)) for i in x] == [3, 4]

def test_trivial():
    assert findroot(lambda x: 0, 1) == 1
    assert findroot(lambda x: x, 0) == 0
    #assert findroot(lambda x, y: x + y, (1, -1)) == (1, -1)
