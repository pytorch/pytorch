from mpmath import *

def test_sumem():
    mp.dps = 15
    assert sumem(lambda k: 1/k**2.5, [50, 100]).ae(0.0012524505324784962)
    assert sumem(lambda k: k**4 + 3*k + 1, [10, 100]).ae(2050333103)

def test_nsum():
    mp.dps = 15
    assert nsum(lambda x: x**2, [1, 3]) == 14
    assert nsum(lambda k: 1/factorial(k), [0, inf]).ae(e)
    assert nsum(lambda k: (-1)**(k+1) / k, [1, inf]).ae(log(2))
    assert nsum(lambda k: (-1)**(k+1) / k**2, [1, inf]).ae(pi**2 / 12)
    assert nsum(lambda k: (-1)**k / log(k), [2, inf]).ae(0.9242998972229388)
    assert nsum(lambda k: 1/k**2, [1, inf]).ae(pi**2 / 6)
    assert nsum(lambda k: 2**k/fac(k), [0, inf]).ae(exp(2))
    assert nsum(lambda k: 1/k**2, [4, inf], method='e').ae(0.2838229557371153)
    assert abs(fp.nsum(lambda k: 1/k**4, [1, fp.inf]) - 1.082323233711138) < 1e-5
    assert abs(fp.nsum(lambda k: 1/k**4, [1, fp.inf], method='e') - 1.082323233711138) < 1e-4

def test_nprod():
    mp.dps = 15
    assert nprod(lambda k: exp(1/k**2), [1,inf], method='r').ae(exp(pi**2/6))
    assert nprod(lambda x: x**2, [1, 3]) == 36

def test_fsum():
    mp.dps = 15
    assert fsum([]) == 0
    assert fsum([-4]) == -4
    assert fsum([2,3]) == 5
    assert fsum([1e-100,1]) == 1
    assert fsum([1,1e-100]) == 1
    assert fsum([1e100,1]) == 1e100
    assert fsum([1,1e100]) == 1e100
    assert fsum([1e-100,0]) == 1e-100
    assert fsum([1e-100,1e100,1e-100]) == 1e100
    assert fsum([2,1+1j,1]) == 4+1j
    assert fsum([2,inf,3]) == inf
    assert fsum([2,-1], absolute=1) == 3
    assert fsum([2,-1], squared=1) == 5
    assert fsum([1,1+j], squared=1) == 1+2j
    assert fsum([1,3+4j], absolute=1) == 6
    assert fsum([1,2+3j], absolute=1, squared=1) == 14
    assert isnan(fsum([inf,-inf]))
    assert fsum([inf,-inf], absolute=1) == inf
    assert fsum([inf,-inf], squared=1) == inf
    assert fsum([inf,-inf], absolute=1, squared=1) == inf
    assert iv.fsum([1,mpi(2,3)]) == mpi(3,4)

def test_fprod():
    mp.dps = 15
    assert fprod([]) == 1
    assert fprod([2,3]) == 6
