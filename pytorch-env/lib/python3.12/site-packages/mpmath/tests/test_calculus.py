import pytest
from mpmath import *

def test_approximation():
    mp.dps = 15
    f = lambda x: cos(2-2*x)/x
    p, err = chebyfit(f, [2, 4], 8, error=True)
    assert err < 1e-5
    for i in range(10):
        x = 2 + i/5.
        assert abs(polyval(p, x) - f(x)) < err

def test_limits():
    mp.dps = 15
    assert limit(lambda x: (x-sin(x))/x**3, 0).ae(mpf(1)/6)
    assert limit(lambda n: (1+1/n)**n, inf).ae(e)

def test_polyval():
    assert polyval([], 3) == 0
    assert polyval([0], 3) == 0
    assert polyval([5], 3) == 5
    # 4x^3 - 2x + 5
    p = [4, 0, -2, 5]
    assert polyval(p,4) == 253
    assert polyval(p,4,derivative=True) == (253, 190)

def test_polyroots():
    p = polyroots([1,-4])
    assert p[0].ae(4)
    p, q = polyroots([1,2,3])
    assert p.ae(-1 - sqrt(2)*j)
    assert q.ae(-1 + sqrt(2)*j)
    #this is not a real test, it only tests a specific case
    assert polyroots([1]) == []
    pytest.raises(ValueError, lambda: polyroots([0]))

def test_polyroots_legendre():
    n = 64
    coeffs = [11975573020964041433067793888190275875, 0,
        -190100434726484311252477736051902332000, 0,
        1437919688271127330313741595496589239248, 0,
        -6897338342113537600691931230430793911840, 0,
        23556405536185284408974715545252277554280, 0,
        -60969520211303089058522793175947071316960, 0,
        124284021969194758465450309166353645376880, 0,
        -204721258548015217049921875719981284186016, 0,
        277415422258095841688223780704620656114900, 0,
        -313237834141273382807123548182995095192800, 0,
        297432255354328395601259515935229287637200, 0,
        -239057700565161140389797367947941296605600, 0,
        163356095386193445933028201431093219347160, 0,
        -95158890516229191805647495979277603503200, 0,
        47310254620162038075933656063247634556400, 0,
        -20071017111583894941305187420771723751200, 0,
        7255051932731034189479516844750603752850, 0,
        -2228176940331017311443863996901733412640, 0,
        579006552594977616773047095969088431600, 0,
        -126584428502545713788439446082310831200, 0,
        23112325428835593809686977515028663000, 0,
        -3491517141958743235617737161547844000, 0,
        431305058712550634988073414073557200, 0,
        -42927166660756742088912492757452000, 0,
        3378527005707706553294038781836500, 0,
        -205277590220215081719131470288800, 0,
        9330799555464321896324157740400, 0,
        -304114948474392713657972548576, 0,
        6695289961520387531608984680, 0,
        -91048139350447232095702560, 0,
        659769125727878493447120, 0,
        -1905929106580294155360, 0,
        916312070471295267]

    with mp.workdps(3):
        with pytest.raises(mp.NoConvergence):
            polyroots(coeffs, maxsteps=5, cleanup=True, error=False,
                      extraprec=n*10)

        roots = polyroots(coeffs, maxsteps=50, cleanup=True, error=False,
                    extraprec=n*10)
        roots = [str(r) for r in roots]
        assert roots == \
            ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961',
            '-0.946', '-0.93', '-0.911', '-0.889', '-0.866', '-0.841',
            '-0.813', '-0.784', '-0.753', '-0.72', '-0.685', '-0.649',
            '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402',
            '-0.357', '-0.311', '-0.265', '-0.217', '-0.17', '-0.121',
            '-0.073', '-0.0243', '0.0243', '0.073', '0.121', '0.17', '0.217',
            '0.265', '0.311', '0.357', '0.402', '0.446', '0.489', '0.531',
            '0.572', '0.611', '0.649', '0.685', '0.72', '0.753', '0.784',
            '0.813', '0.841', '0.866', '0.889', '0.911', '0.93', '0.946',
            '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']

def test_polyroots_legendre_init():
    extra_prec = 100
    coeffs = [11975573020964041433067793888190275875, 0,
        -190100434726484311252477736051902332000, 0,
        1437919688271127330313741595496589239248, 0,
        -6897338342113537600691931230430793911840, 0,
        23556405536185284408974715545252277554280, 0,
        -60969520211303089058522793175947071316960, 0,
        124284021969194758465450309166353645376880, 0,
        -204721258548015217049921875719981284186016, 0,
        277415422258095841688223780704620656114900, 0,
        -313237834141273382807123548182995095192800, 0,
        297432255354328395601259515935229287637200, 0,
        -239057700565161140389797367947941296605600, 0,
        163356095386193445933028201431093219347160, 0,
        -95158890516229191805647495979277603503200, 0,
        47310254620162038075933656063247634556400, 0,
        -20071017111583894941305187420771723751200, 0,
        7255051932731034189479516844750603752850, 0,
        -2228176940331017311443863996901733412640, 0,
        579006552594977616773047095969088431600, 0,
        -126584428502545713788439446082310831200, 0,
        23112325428835593809686977515028663000, 0,
        -3491517141958743235617737161547844000, 0,
        431305058712550634988073414073557200, 0,
        -42927166660756742088912492757452000, 0,
        3378527005707706553294038781836500, 0,
        -205277590220215081719131470288800, 0,
        9330799555464321896324157740400, 0,
        -304114948474392713657972548576, 0,
        6695289961520387531608984680, 0,
        -91048139350447232095702560, 0,
        659769125727878493447120, 0,
        -1905929106580294155360, 0,
        916312070471295267]

    roots_init =  matrix(['-0.999', '-0.996',  '-0.991', '-0.983', '-0.973',
                          '-0.961', '-0.946',  '-0.93',  '-0.911', '-0.889',
                          '-0.866', '-0.841',  '-0.813', '-0.784', '-0.753',
                          '-0.72',  '-0.685',  '-0.649', '-0.611', '-0.572',
                          '-0.531', '-0.489',  '-0.446', '-0.402', '-0.357',
                          '-0.311', '-0.265',  '-0.217', '-0.17',  '-0.121',
                          '-0.073', '-0.0243',  '0.0243', '0.073',  '0.121',
                          '0.17',    '0.217',   '0.265', ' 0.311',  '0.357',
                          '0.402',   '0.446',   '0.489',  '0.531',  '0.572',
                          '0.611',   '0.649',   '0.685',  '0.72',   '0.753',
                          '0.784',   '0.813',   '0.841',  '0.866',  '0.889',
                          '0.911',   '0.93',    '0.946',  '0.961',  '0.973',
                          '0.983',   '0.991',   '0.996',  '0.999',  '1.0'])
    with mp.workdps(2*mp.dps):
        roots_exact = polyroots(coeffs, maxsteps=50, cleanup=True, error=False,
                                extraprec=2*extra_prec)
    with pytest.raises(mp.NoConvergence):
        polyroots(coeffs, maxsteps=5, cleanup=True, error=False,
                  extraprec=extra_prec)
    roots,err = polyroots(coeffs, maxsteps=5, cleanup=True, error=True,
                          extraprec=extra_prec,roots_init=roots_init)
    assert max(matrix(roots_exact)-matrix(roots).apply(abs)) < err
    roots1,err1 = polyroots(coeffs, maxsteps=25, cleanup=True, error=True,
                            extraprec=extra_prec,roots_init=roots_init[:60])
    assert max(matrix(roots_exact)-matrix(roots1).apply(abs)) < err1

def test_pade():
    one = mpf(1)
    mp.dps = 20
    N = 10
    a = [one]
    k = 1
    for i in range(1, N+1):
        k *= i
        a.append(one/k)
    p, q = pade(a, N//2, N//2)
    for x in arange(0, 1, 0.1):
        r = polyval(p[::-1], x)/polyval(q[::-1], x)
        assert(r.ae(exp(x), 1.0e-10))
    mp.dps = 15

def test_fourier():
    mp.dps = 15
    c, s = fourier(lambda x: x+1, [-1, 2], 2)
    #plot([lambda x: x+1, lambda x: fourierval((c, s), [-1, 2], x)], [-1, 2])
    assert c[0].ae(1.5)
    assert c[1].ae(-3*sqrt(3)/(2*pi))
    assert c[2].ae(3*sqrt(3)/(4*pi))
    assert s[0] == 0
    assert s[1].ae(3/(2*pi))
    assert s[2].ae(3/(4*pi))
    assert fourierval((c, s), [-1, 2], 1).ae(1.9134966715663442)

def test_differint():
    mp.dps = 15
    assert differint(lambda t: t, 2, -0.5).ae(8*sqrt(2/pi)/3)

def test_invlap():
    mp.dps = 15
    t = 0.01
    fp = lambda p: 1/(p+1)**2
    ft = lambda t: t*exp(-t)
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)

    t = 0.01
    fp = lambda p: log(p)/p
    ft = lambda t: -euler-log(t)
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)
