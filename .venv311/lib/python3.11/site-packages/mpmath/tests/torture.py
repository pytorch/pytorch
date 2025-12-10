"""
Torture tests for asymptotics and high precision evaluation of
special functions.

(Other torture tests may also be placed here.)

Running this file (gmpy recommended!) takes several CPU minutes.
With Python 2.6+, multiprocessing is used automatically to run tests
in parallel if many cores are available. (A single test may take between
a second and several minutes; possibly more.)

The idea:

* We evaluate functions at positive, negative, imaginary, 45- and 135-degree
  complex values with magnitudes between 10^-20 to 10^20, at precisions between
  5 and 150 digits (we can go even higher for fast functions).

* Comparing the result from two different precision levels provides
  a strong consistency check (particularly for functions that use
  different algorithms at different precision levels).

* That the computation finishes at all (without failure), within reasonable
  time, provides a check that evaluation works at all: that the code runs,
  that it doesn't get stuck in an infinite loop, and that it doesn't use
  some extremely slowly algorithm where it could use a faster one.

TODO:

* Speed up those functions that take long to finish!
* Generalize to test more cases; more options.
* Implement a timeout mechanism.
* Some functions are notably absent, including the following:
  * inverse trigonometric functions (some become inaccurate for complex arguments)
  * ci, si (not implemented properly for large complex arguments)
  * zeta functions (need to modify test not to try too large imaginary values)
  * and others...

"""


import sys, os
from timeit import default_timer as clock

if "-nogmpy" in sys.argv:
    sys.argv.remove('-nogmpy')
    os.environ['MPMATH_NOGMPY'] = 'Y'

filt = ''
if not sys.argv[-1].endswith(".py"):
    filt = sys.argv[-1]

from mpmath import *
from mpmath.libmp.backend import exec_

def test_asymp(f, maxdps=150, verbose=False, huge_range=False):
    dps = [5,15,25,50,90,150,500,1500,5000,10000]
    dps = [p for p in dps if p <= maxdps]
    def check(x,y,p,inpt):
        if abs(x-y)/abs(y) < workprec(20)(power)(10, -p+1):
            return
        print()
        print("Error!")
        print("Input:", inpt)
        print("dps =", p)
        print("Result 1:", x)
        print("Result 2:", y)
        print("Absolute error:", abs(x-y))
        print("Relative error:", abs(x-y)/abs(y))
        raise AssertionError
    exponents = range(-20,20)
    if huge_range:
        exponents += [-1000, -100, -50, 50, 100, 1000]
    for n in exponents:
        if verbose:
            sys.stdout.write(". ")
        mp.dps = 25
        xpos = mpf(10)**n / 1.1287
        xneg = -xpos
        ximag = xpos*j
        xcomplex1 = xpos*(1+j)
        xcomplex2 = xpos*(-1+j)
        for i in range(len(dps)):
            if verbose:
                print("Testing dps = %s" % dps[i])
            mp.dps = dps[i]
            new = f(xpos), f(xneg), f(ximag), f(xcomplex1), f(xcomplex2)
            if i != 0:
                p = dps[i-1]
                check(prev[0], new[0], p, xpos)
                check(prev[1], new[1], p, xneg)
                check(prev[2], new[2], p, ximag)
                check(prev[3], new[3], p, xcomplex1)
                check(prev[4], new[4], p, xcomplex2)
            prev = new
    if verbose:
        print()

a1, a2, a3, a4, a5 = 1.5, -2.25, 3.125, 4, 2

def test_bernoulli_huge():
    p, q = bernfrac(9000)
    assert p % 10**10 == 9636701091
    assert q == 4091851784687571609141381951327092757255270
    mp.dps = 15
    assert str(bernoulli(10**100)) == '-2.58183325604736e+987675256497386331227838638980680030172857347883537824464410652557820800494271520411283004120790908623'
    mp.dps = 50
    assert str(bernoulli(10**100)) == '-2.5818332560473632073252488656039475548106223822913e+987675256497386331227838638980680030172857347883537824464410652557820800494271520411283004120790908623'
    mp.dps = 15

cases = """\
test_bernoulli_huge()
test_asymp(lambda z: +pi, maxdps=10000)
test_asymp(lambda z: +e, maxdps=10000)
test_asymp(lambda z: +ln2, maxdps=10000)
test_asymp(lambda z: +ln10, maxdps=10000)
test_asymp(lambda z: +phi, maxdps=10000)
test_asymp(lambda z: +catalan, maxdps=5000)
test_asymp(lambda z: +euler, maxdps=5000)
test_asymp(lambda z: +glaisher, maxdps=1000)
test_asymp(lambda z: +khinchin, maxdps=1000)
test_asymp(lambda z: +twinprime, maxdps=150)
test_asymp(lambda z: stieltjes(2), maxdps=150)
test_asymp(lambda z: +mertens, maxdps=150)
test_asymp(lambda z: +apery, maxdps=5000)
test_asymp(sqrt, maxdps=10000, huge_range=True)
test_asymp(cbrt, maxdps=5000, huge_range=True)
test_asymp(lambda z: root(z,4), maxdps=5000, huge_range=True)
test_asymp(lambda z: root(z,-5), maxdps=5000, huge_range=True)
test_asymp(exp, maxdps=5000, huge_range=True)
test_asymp(expm1, maxdps=1500)
test_asymp(ln, maxdps=5000, huge_range=True)
test_asymp(cosh, maxdps=5000)
test_asymp(sinh, maxdps=5000)
test_asymp(tanh, maxdps=1500)
test_asymp(sin, maxdps=5000, huge_range=True)
test_asymp(cos, maxdps=5000, huge_range=True)
test_asymp(tan, maxdps=1500)
test_asymp(agm, maxdps=1500, huge_range=True)
test_asymp(ellipk, maxdps=1500)
test_asymp(ellipe, maxdps=1500)
test_asymp(lambertw, huge_range=True)
test_asymp(lambda z: lambertw(z,-1))
test_asymp(lambda z: lambertw(z,1))
test_asymp(lambda z: lambertw(z,4))
test_asymp(gamma)
test_asymp(loggamma)  # huge_range=True ?
test_asymp(ei)
test_asymp(e1)
test_asymp(li, huge_range=True)
test_asymp(ci)
test_asymp(si)
test_asymp(chi)
test_asymp(shi)
test_asymp(erf)
test_asymp(erfc)
test_asymp(erfi)
test_asymp(lambda z: besselj(2, z))
test_asymp(lambda z: bessely(2, z))
test_asymp(lambda z: besseli(2, z))
test_asymp(lambda z: besselk(2, z))
test_asymp(lambda z: besselj(-2.25, z))
test_asymp(lambda z: bessely(-2.25, z))
test_asymp(lambda z: besseli(-2.25, z))
test_asymp(lambda z: besselk(-2.25, z))
test_asymp(airyai)
test_asymp(airybi)
test_asymp(lambda z: hyp0f1(a1, z))
test_asymp(lambda z: hyp1f1(a1, a2, z))
test_asymp(lambda z: hyp1f2(a1, a2, a3, z))
test_asymp(lambda z: hyp2f0(a1, a2, z))
test_asymp(lambda z: hyperu(a1, a2, z))
test_asymp(lambda z: hyp2f1(a1, a2, a3, z))
test_asymp(lambda z: hyp2f2(a1, a2, a3, a4, z))
test_asymp(lambda z: hyp2f3(a1, a2, a3, a4, a5, z))
test_asymp(lambda z: coulombf(a1, a2, z))
test_asymp(lambda z: coulombg(a1, a2, z))
test_asymp(lambda z: polylog(2,z))
test_asymp(lambda z: polylog(3,z))
test_asymp(lambda z: polylog(-2,z))
test_asymp(lambda z: expint(4, z))
test_asymp(lambda z: expint(-4, z))
test_asymp(lambda z: expint(2.25, z))
test_asymp(lambda z: gammainc(2.5, z, 5))
test_asymp(lambda z: gammainc(2.5, 5, z))
test_asymp(lambda z: hermite(3, z))
test_asymp(lambda z: hermite(2.5, z))
test_asymp(lambda z: legendre(3, z))
test_asymp(lambda z: legendre(4, z))
test_asymp(lambda z: legendre(2.5, z))
test_asymp(lambda z: legenp(a1, a2, z))
test_asymp(lambda z: legenq(a1, a2, z), maxdps=90)   # abnormally slow
test_asymp(lambda z: jtheta(1, z, 0.5))
test_asymp(lambda z: jtheta(2, z, 0.5))
test_asymp(lambda z: jtheta(3, z, 0.5))
test_asymp(lambda z: jtheta(4, z, 0.5))
test_asymp(lambda z: jtheta(1, z, 0.5, 1))
test_asymp(lambda z: jtheta(2, z, 0.5, 1))
test_asymp(lambda z: jtheta(3, z, 0.5, 1))
test_asymp(lambda z: jtheta(4, z, 0.5, 1))
test_asymp(barnesg, maxdps=90)
"""

def testit(line):
    if filt in line:
        print(line)
        t1 = clock()
        exec_(line, globals(), locals())
        t2 = clock()
        elapsed = t2-t1
        print("Time:", elapsed, "for", line, "(OK)")

if __name__ == '__main__':
    try:
        from multiprocessing import Pool
        mapf = Pool(None).map
        print("Running tests with multiprocessing")
    except ImportError:
        print("Not using multiprocessing")
        mapf = map
    t1 = clock()
    tasks = cases.splitlines()
    mapf(testit, tasks)
    t2 = clock()
    print("Cumulative wall time:", t2-t1)
