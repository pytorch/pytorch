from mpmath import *

def test_diff():
    mp.dps = 15
    assert diff(log, 2.0, n=0).ae(log(2))
    assert diff(cos, 1.0).ae(-sin(1))
    assert diff(abs, 0.0) == 0
    assert diff(abs, 0.0, direction=1) == 1
    assert diff(abs, 0.0, direction=-1) == -1
    assert diff(exp, 1.0).ae(e)
    assert diff(exp, 1.0, n=5).ae(e)
    assert diff(exp, 2.0, n=5, direction=3*j).ae(e**2)
    assert diff(lambda x: x**2, 3.0, method='quad').ae(6)
    assert diff(lambda x: 3+x**5, 3.0, n=2, method='quad').ae(540)
    assert diff(lambda x: 3+x**5, 3.0, n=2, method='step').ae(540)
    assert diffun(sin)(2).ae(cos(2))
    assert diffun(sin, n=2)(2).ae(-sin(2))

def test_diffs():
    mp.dps = 15
    assert [chop(d) for d in diffs(sin, 0, 1)] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 1, method='quad')] == [0, 1]
    assert [chop(d) for d in diffs(sin, 0, 2)] == [0, 1, 0]
    assert [chop(d) for d in diffs(sin, 0, 2, method='quad')] == [0, 1, 0]

def test_taylor():
    mp.dps = 15
    # Easy to test since the coefficients are exact in floating-point
    assert taylor(sqrt, 1, 4) == [1, 0.5, -0.125, 0.0625, -0.0390625]

def test_diff_partial():
    mp.dps = 15
    x,y,z = xyz = 2,3,7
    f = lambda x,y,z: 3*x**2 * (y+2)**3 * z**5
    assert diff(f, xyz, (0,0,0)).ae(25210500)
    assert diff(f, xyz, (0,0,1)).ae(18007500)
    assert diff(f, xyz, (0,0,2)).ae(10290000)
    assert diff(f, xyz, (0,1,0)).ae(15126300)
    assert diff(f, xyz, (0,1,1)).ae(10804500)
    assert diff(f, xyz, (0,1,2)).ae(6174000)
    assert diff(f, xyz, (0,2,0)).ae(6050520)
    assert diff(f, xyz, (0,2,1)).ae(4321800)
    assert diff(f, xyz, (0,2,2)).ae(2469600)
    assert diff(f, xyz, (1,0,0)).ae(25210500)
    assert diff(f, xyz, (1,0,1)).ae(18007500)
    assert diff(f, xyz, (1,0,2)).ae(10290000)
    assert diff(f, xyz, (1,1,0)).ae(15126300)
    assert diff(f, xyz, (1,1,1)).ae(10804500)
    assert diff(f, xyz, (1,1,2)).ae(6174000)
    assert diff(f, xyz, (1,2,0)).ae(6050520)
    assert diff(f, xyz, (1,2,1)).ae(4321800)
    assert diff(f, xyz, (1,2,2)).ae(2469600)
    assert diff(f, xyz, (2,0,0)).ae(12605250)
    assert diff(f, xyz, (2,0,1)).ae(9003750)
    assert diff(f, xyz, (2,0,2)).ae(5145000)
    assert diff(f, xyz, (2,1,0)).ae(7563150)
    assert diff(f, xyz, (2,1,1)).ae(5402250)
    assert diff(f, xyz, (2,1,2)).ae(3087000)
    assert diff(f, xyz, (2,2,0)).ae(3025260)
    assert diff(f, xyz, (2,2,1)).ae(2160900)
    assert diff(f, xyz, (2,2,2)).ae(1234800)
