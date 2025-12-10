from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int

def test_zeta_int_bug():
    assert mpf_zeta_int(0, 10) == from_float(-0.5)

def test_bernoulli():
    assert bernfrac(0) == (1,1)
    assert bernfrac(1) == (-1,2)
    assert bernfrac(2) == (1,6)
    assert bernfrac(3) == (0,1)
    assert bernfrac(4) == (-1,30)
    assert bernfrac(5) == (0,1)
    assert bernfrac(6) == (1,42)
    assert bernfrac(8) == (-1,30)
    assert bernfrac(10) == (5,66)
    assert bernfrac(12) == (-691,2730)
    assert bernfrac(18) == (43867,798)
    p, q = bernfrac(228)
    assert p % 10**10 == 164918161
    assert q == 625170
    p, q = bernfrac(1000)
    assert p % 10**10 == 7950421099
    assert q == 342999030
    mp.dps = 15
    assert bernoulli(0) == 1
    assert bernoulli(1) == -0.5
    assert bernoulli(2).ae(1./6)
    assert bernoulli(3) == 0
    assert bernoulli(4).ae(-1./30)
    assert bernoulli(5) == 0
    assert bernoulli(6).ae(1./42)
    assert str(bernoulli(10)) == '0.0757575757575758'
    assert str(bernoulli(234)) == '7.62772793964344e+267'
    assert str(bernoulli(10**5)) == '-5.82229431461335e+376755'
    assert str(bernoulli(10**8+2)) == '1.19570355039953e+676752584'

    mp.dps = 50
    assert str(bernoulli(10)) == '0.075757575757575757575757575757575757575757575757576'
    assert str(bernoulli(234)) == '7.6277279396434392486994969020496121553385863373331e+267'
    assert str(bernoulli(10**5)) == '-5.8222943146133508236497045360612887555320691004308e+376755'
    assert str(bernoulli(10**8+2)) == '1.1957035503995297272263047884604346914602088317782e+676752584'

    mp.dps = 1000
    assert bernoulli(10).ae(mpf(5)/66)

    mp.dps = 50000
    assert bernoulli(10).ae(mpf(5)/66)

    mp.dps = 15

def test_bernpoly_eulerpoly():
    mp.dps = 15
    assert bernpoly(0,-1).ae(1)
    assert bernpoly(0,0).ae(1)
    assert bernpoly(0,'1/2').ae(1)
    assert bernpoly(0,'3/4').ae(1)
    assert bernpoly(0,1).ae(1)
    assert bernpoly(0,2).ae(1)
    assert bernpoly(1,-1).ae('-3/2')
    assert bernpoly(1,0).ae('-1/2')
    assert bernpoly(1,'1/2').ae(0)
    assert bernpoly(1,'3/4').ae('1/4')
    assert bernpoly(1,1).ae('1/2')
    assert bernpoly(1,2).ae('3/2')
    assert bernpoly(2,-1).ae('13/6')
    assert bernpoly(2,0).ae('1/6')
    assert bernpoly(2,'1/2').ae('-1/12')
    assert bernpoly(2,'3/4').ae('-1/48')
    assert bernpoly(2,1).ae('1/6')
    assert bernpoly(2,2).ae('13/6')
    assert bernpoly(3,-1).ae(-3)
    assert bernpoly(3,0).ae(0)
    assert bernpoly(3,'1/2').ae(0)
    assert bernpoly(3,'3/4').ae('-3/64')
    assert bernpoly(3,1).ae(0)
    assert bernpoly(3,2).ae(3)
    assert bernpoly(4,-1).ae('119/30')
    assert bernpoly(4,0).ae('-1/30')
    assert bernpoly(4,'1/2').ae('7/240')
    assert bernpoly(4,'3/4').ae('7/3840')
    assert bernpoly(4,1).ae('-1/30')
    assert bernpoly(4,2).ae('119/30')
    assert bernpoly(5,-1).ae(-5)
    assert bernpoly(5,0).ae(0)
    assert bernpoly(5,'1/2').ae(0)
    assert bernpoly(5,'3/4').ae('25/1024')
    assert bernpoly(5,1).ae(0)
    assert bernpoly(5,2).ae(5)
    assert bernpoly(10,-1).ae('665/66')
    assert bernpoly(10,0).ae('5/66')
    assert bernpoly(10,'1/2').ae('-2555/33792')
    assert bernpoly(10,'3/4').ae('-2555/34603008')
    assert bernpoly(10,1).ae('5/66')
    assert bernpoly(10,2).ae('665/66')
    assert bernpoly(11,-1).ae(-11)
    assert bernpoly(11,0).ae(0)
    assert bernpoly(11,'1/2').ae(0)
    assert bernpoly(11,'3/4').ae('-555731/4194304')
    assert bernpoly(11,1).ae(0)
    assert bernpoly(11,2).ae(11)
    assert eulerpoly(0,-1).ae(1)
    assert eulerpoly(0,0).ae(1)
    assert eulerpoly(0,'1/2').ae(1)
    assert eulerpoly(0,'3/4').ae(1)
    assert eulerpoly(0,1).ae(1)
    assert eulerpoly(0,2).ae(1)
    assert eulerpoly(1,-1).ae('-3/2')
    assert eulerpoly(1,0).ae('-1/2')
    assert eulerpoly(1,'1/2').ae(0)
    assert eulerpoly(1,'3/4').ae('1/4')
    assert eulerpoly(1,1).ae('1/2')
    assert eulerpoly(1,2).ae('3/2')
    assert eulerpoly(2,-1).ae(2)
    assert eulerpoly(2,0).ae(0)
    assert eulerpoly(2,'1/2').ae('-1/4')
    assert eulerpoly(2,'3/4').ae('-3/16')
    assert eulerpoly(2,1).ae(0)
    assert eulerpoly(2,2).ae(2)
    assert eulerpoly(3,-1).ae('-9/4')
    assert eulerpoly(3,0).ae('1/4')
    assert eulerpoly(3,'1/2').ae(0)
    assert eulerpoly(3,'3/4').ae('-11/64')
    assert eulerpoly(3,1).ae('-1/4')
    assert eulerpoly(3,2).ae('9/4')
    assert eulerpoly(4,-1).ae(2)
    assert eulerpoly(4,0).ae(0)
    assert eulerpoly(4,'1/2').ae('5/16')
    assert eulerpoly(4,'3/4').ae('57/256')
    assert eulerpoly(4,1).ae(0)
    assert eulerpoly(4,2).ae(2)
    assert eulerpoly(5,-1).ae('-3/2')
    assert eulerpoly(5,0).ae('-1/2')
    assert eulerpoly(5,'1/2').ae(0)
    assert eulerpoly(5,'3/4').ae('361/1024')
    assert eulerpoly(5,1).ae('1/2')
    assert eulerpoly(5,2).ae('3/2')
    assert eulerpoly(10,-1).ae(2)
    assert eulerpoly(10,0).ae(0)
    assert eulerpoly(10,'1/2').ae('-50521/1024')
    assert eulerpoly(10,'3/4').ae('-36581523/1048576')
    assert eulerpoly(10,1).ae(0)
    assert eulerpoly(10,2).ae(2)
    assert eulerpoly(11,-1).ae('-699/4')
    assert eulerpoly(11,0).ae('691/4')
    assert eulerpoly(11,'1/2').ae(0)
    assert eulerpoly(11,'3/4').ae('-512343611/4194304')
    assert eulerpoly(11,1).ae('-691/4')
    assert eulerpoly(11,2).ae('699/4')
    # Potential accuracy issues
    assert bernpoly(10000,10000).ae('5.8196915936323387117e+39999')
    assert bernpoly(200,17.5).ae(3.8048418524583064909e244)
    assert eulerpoly(200,17.5).ae(-3.7309911582655785929e275)

def test_gamma():
    mp.dps = 15
    assert gamma(0.25).ae(3.6256099082219083119)
    assert gamma(0.0001).ae(9999.4228832316241908)
    assert gamma(300).ae('1.0201917073881354535e612')
    assert gamma(-0.5).ae(-3.5449077018110320546)
    assert gamma(-7.43).ae(0.00026524416464197007186)
    #assert gamma(Rational(1,2)) == gamma(0.5)
    #assert gamma(Rational(-7,3)).ae(gamma(mpf(-7)/3))
    assert gamma(1+1j).ae(0.49801566811835604271 - 0.15494982830181068512j)
    assert gamma(-1+0.01j).ae(-0.422733904013474115 + 99.985883082635367436j)
    assert gamma(20+30j).ae(-1453876687.5534810 + 1163777777.8031573j)
    # Should always give exact factorials when they can
    # be represented as mpfs under the current working precision
    fact = 1
    for i in range(1, 18):
        assert gamma(i) == fact
        fact *= i
    for dps in [170, 600]:
        fact = 1
        mp.dps = dps
        for i in range(1, 105):
            assert gamma(i) == fact
            fact *= i
    mp.dps = 100
    assert gamma(0.5).ae(sqrt(pi))
    mp.dps = 15
    assert factorial(0) == fac(0) == 1
    assert factorial(3) == 6
    assert isnan(gamma(nan))
    assert gamma(1100).ae('4.8579168073569433667e2866')
    assert rgamma(0) == 0
    assert rgamma(-1) == 0
    assert rgamma(2) == 1.0
    assert rgamma(3) == 0.5
    assert loggamma(2+8j).ae(-8.5205176753667636926 + 10.8569497125597429366j)
    assert loggamma('1e10000').ae('2.302485092994045684017991e10004')
    assert loggamma('1e10000j').ae(mpc('-1.570796326794896619231322e10000','2.302485092994045684017991e10004'))

def test_fac2():
    mp.dps = 15
    assert [fac2(n) for n in range(10)] == [1,1,2,3,8,15,48,105,384,945]
    assert fac2(-5).ae(1./3)
    assert fac2(-11).ae(-1./945)
    assert fac2(50).ae(5.20469842636666623e32)
    assert fac2(0.5+0.75j).ae(0.81546769394688069176-0.34901016085573266889j)
    assert fac2(inf) == inf
    assert isnan(fac2(-inf))

def test_gamma_quotients():
    mp.dps = 15
    h = 1e-8
    ep = 1e-4
    G = gamma
    assert gammaprod([-1],[-3,-4]) == 0
    assert gammaprod([-1,0],[-5]) == inf
    assert abs(gammaprod([-1],[-2]) - G(-1+h)/G(-2+h)) < 1e-4
    assert abs(gammaprod([-4,-3],[-2,0]) - G(-4+h)*G(-3+h)/G(-2+h)/G(0+h)) < 1e-4
    assert rf(3,0) == 1
    assert rf(2.5,1) == 2.5
    assert rf(-5,2) == 20
    assert rf(j,j).ae(gamma(2*j)/gamma(j))
    assert rf('-255.5815971722918','-0.5119253100282322').ae('-0.1952720278805729485')  # issue 421
    assert ff(-2,0) == 1
    assert ff(-2,1) == -2
    assert ff(4,3) == 24
    assert ff(3,4) == 0
    assert binomial(0,0) == 1
    assert binomial(1,0) == 1
    assert binomial(0,-1) == 0
    assert binomial(3,2) == 3
    assert binomial(5,2) == 10
    assert binomial(5,3) == 10
    assert binomial(5,5) == 1
    assert binomial(-1,0) == 1
    assert binomial(-2,-4) == 3
    assert binomial(4.5, 1.5) == 6.5625
    assert binomial(1100,1) == 1100
    assert binomial(1100,2) == 604450
    assert beta(1,1) == 1
    assert beta(0,0) == inf
    assert beta(3,0) == inf
    assert beta(-1,-1) == inf
    assert beta(1.5,1).ae(2/3.)
    assert beta(1.5,2.5).ae(pi/16)
    assert (10**15*beta(10,100)).ae(2.3455339739604649879)
    assert beta(inf,inf) == 0
    assert isnan(beta(-inf,inf))
    assert isnan(beta(-3,inf))
    assert isnan(beta(0,inf))
    assert beta(inf,0.5) == beta(0.5,inf) == 0
    assert beta(inf,-1.5) == inf
    assert beta(inf,-0.5) == -inf
    assert beta(1+2j,-1-j/2).ae(1.16396542451069943086+0.08511695947832914640j)
    assert beta(-0.5,0.5) == 0
    assert beta(-3,3).ae(-1/3.)
    assert beta('-255.5815971722918','-0.5119253100282322').ae('18.157330562703710339')  # issue 421

def test_zeta():
    mp.dps = 15
    assert zeta(2).ae(pi**2 / 6)
    assert zeta(2.0).ae(pi**2 / 6)
    assert zeta(mpc(2)).ae(pi**2 / 6)
    assert zeta(100).ae(1)
    assert zeta(0).ae(-0.5)
    assert zeta(0.5).ae(-1.46035450880958681)
    assert zeta(-1).ae(-mpf(1)/12)
    assert zeta(-2) == 0
    assert zeta(-3).ae(mpf(1)/120)
    assert zeta(-4) == 0
    assert zeta(-100) == 0
    assert isnan(zeta(nan))
    assert zeta(1e-30).ae(-0.5)
    assert zeta(-1e-30).ae(-0.5)
    # Zeros in the critical strip
    assert zeta(mpc(0.5, 14.1347251417346937904)).ae(0)
    assert zeta(mpc(0.5, 21.0220396387715549926)).ae(0)
    assert zeta(mpc(0.5, 25.0108575801456887632)).ae(0)
    assert zeta(mpc(1e-30,1e-40)).ae(-0.5)
    assert zeta(mpc(-1e-30,1e-40)).ae(-0.5)
    mp.dps = 50
    im = '236.5242296658162058024755079556629786895294952121891237'
    assert zeta(mpc(0.5, im)).ae(0, 1e-46)
    mp.dps = 15
    # Complex reflection formula
    assert (zeta(-60+3j) / 10**34).ae(8.6270183987866146+15.337398548226238j)
    # issue #358
    assert zeta(0,0.5) == 0
    assert zeta(0,0) == 0.5
    assert zeta(0,0.5,1).ae(-0.34657359027997265)
    # see issue #390
    assert zeta(-1.5,0.5j).ae(-0.13671400162512768475 + 0.11411333638426559139j)

def test_altzeta():
    mp.dps = 15
    assert altzeta(-2) == 0
    assert altzeta(-4) == 0
    assert altzeta(-100) == 0
    assert altzeta(0) == 0.5
    assert altzeta(-1) == 0.25
    assert altzeta(-3) == -0.125
    assert altzeta(-5) == 0.25
    assert altzeta(-21) == 1180529130.25
    assert altzeta(1).ae(log(2))
    assert altzeta(2).ae(pi**2/12)
    assert altzeta(10).ae(73*pi**10/6842880)
    assert altzeta(50) < 1
    assert altzeta(60, rounding='d') < 1
    assert altzeta(60, rounding='u') == 1
    assert altzeta(10000, rounding='d') < 1
    assert altzeta(10000, rounding='u') == 1
    assert altzeta(3+0j) == altzeta(3)
    s = 3+4j
    assert altzeta(s).ae((1-2**(1-s))*zeta(s))
    s = -3+4j
    assert altzeta(s).ae((1-2**(1-s))*zeta(s))
    assert altzeta(-100.5).ae(4.58595480083585913e+108)
    assert altzeta(1.3).ae(0.73821404216623045)
    assert altzeta(1e-30).ae(0.5)
    assert altzeta(-1e-30).ae(0.5)
    assert altzeta(mpc(1e-30,1e-40)).ae(0.5)
    assert altzeta(mpc(-1e-30,1e-40)).ae(0.5)

def test_zeta_huge():
    mp.dps = 15
    assert zeta(inf) == 1
    mp.dps = 50
    assert zeta(100).ae('1.0000000000000000000000000000007888609052210118073522')
    assert zeta(40*pi).ae('1.0000000000000000000000000000000000000148407238666182')
    mp.dps = 10000
    v = zeta(33000)
    mp.dps = 15
    assert str(v-1) == '1.02363019598118e-9934'
    assert zeta(pi*1000, rounding=round_up) > 1
    assert zeta(3000, rounding=round_up) > 1
    assert zeta(pi*1000) == 1
    assert zeta(3000) == 1

def test_zeta_negative():
    mp.dps = 150
    a = -pi*10**40
    mp.dps = 15
    assert str(zeta(a)) == '2.55880492708712e+1233536161668617575553892558646631323374078'
    mp.dps = 50
    assert str(zeta(a)) == '2.5588049270871154960875033337384432038436330847333e+1233536161668617575553892558646631323374078'
    mp.dps = 15

def test_polygamma():
    mp.dps = 15
    psi0 = lambda z: psi(0,z)
    psi1 = lambda z: psi(1,z)
    assert psi0(3) == psi(0,3) == digamma(3)
    #assert psi2(3) == psi(2,3) == tetragamma(3)
    #assert psi3(3) == psi(3,3) == pentagamma(3)
    assert psi0(pi).ae(0.97721330794200673)
    assert psi0(-pi).ae(7.8859523853854902)
    assert psi0(-pi+1).ae(7.5676424992016996)
    assert psi0(pi+j).ae(1.04224048313859376 + 0.35853686544063749j)
    assert psi0(-pi-j).ae(1.3404026194821986 - 2.8824392476809402j)
    assert findroot(psi0, 1).ae(1.4616321449683622)
    assert psi0(1e-10).ae(-10000000000.57722)
    assert psi0(1e-40).ae(-1.000000000000000e+40)
    assert psi0(1e-10+1e-10j).ae(-5000000000.577215 + 5000000000.000000j)
    assert psi0(1e-40+1e-40j).ae(-5.000000000000000e+39 + 5.000000000000000e+39j)
    assert psi0(inf) == inf
    assert psi1(inf) == 0
    assert psi(2,inf) == 0
    assert psi1(pi).ae(0.37424376965420049)
    assert psi1(-pi).ae(53.030438740085385)
    assert psi1(pi+j).ae(0.32935710377142464 - 0.12222163911221135j)
    assert psi1(-pi-j).ae(-0.30065008356019703 + 0.01149892486928227j)
    assert (10**6*psi(4,1+10*pi*j)).ae(-6.1491803479004446 - 0.3921316371664063j)
    assert psi0(1+10*pi*j).ae(3.4473994217222650 + 1.5548808324857071j)
    assert isnan(psi0(nan))
    assert isnan(psi0(-inf))
    assert psi0(-100.5).ae(4.615124601338064)
    assert psi0(3+0j).ae(psi0(3))
    assert psi0(-100+3j).ae(4.6106071768714086321+3.1117510556817394626j)
    assert isnan(psi(2,mpc(0,inf)))
    assert isnan(psi(2,mpc(0,nan)))
    assert isnan(psi(2,mpc(0,-inf)))
    assert isnan(psi(2,mpc(1,inf)))
    assert isnan(psi(2,mpc(1,nan)))
    assert isnan(psi(2,mpc(1,-inf)))
    assert isnan(psi(2,mpc(inf,inf)))
    assert isnan(psi(2,mpc(nan,nan)))
    assert isnan(psi(2,mpc(-inf,-inf)))
    mp.dps = 30
    # issue #534
    assert digamma(-0.75+1j).ae(mpc('0.46317279488182026118963809283042317', '2.4821070143037957102007677817351115'))
    mp.dps = 15

def test_polygamma_high_prec():
    mp.dps = 100
    assert str(psi(0,pi)) == "0.9772133079420067332920694864061823436408346099943256380095232865318105924777141317302075654362928734"
    assert str(psi(10,pi)) == "-12.98876181434889529310283769414222588307175962213707170773803550518307617769657562747174101900659238"

def test_polygamma_identities():
    mp.dps = 15
    psi0 = lambda z: psi(0,z)
    psi1 = lambda z: psi(1,z)
    psi2 = lambda z: psi(2,z)
    assert psi0(0.5).ae(-euler-2*log(2))
    assert psi0(1).ae(-euler)
    assert psi1(0.5).ae(0.5*pi**2)
    assert psi1(1).ae(pi**2/6)
    assert psi1(0.25).ae(pi**2 + 8*catalan)
    assert psi2(1).ae(-2*apery)
    mp.dps = 20
    u = -182*apery+4*sqrt(3)*pi**3
    mp.dps = 15
    assert psi(2,5/6.).ae(u)
    assert psi(3,0.5).ae(pi**4)

def test_foxtrot_identity():
    # A test of the complex digamma function.
    # See http://mathworld.wolfram.com/FoxTrotSeries.html and
    # http://mathworld.wolfram.com/DigammaFunction.html
    psi0 = lambda z: psi(0,z)
    mp.dps = 50
    a = (-1)**fraction(1,3)
    b = (-1)**fraction(2,3)
    x = -psi0(0.5*a) - psi0(-0.5*b) + psi0(0.5*(1+a)) + psi0(0.5*(1-b))
    y = 2*pi*sech(0.5*sqrt(3)*pi)
    assert x.ae(y)
    mp.dps = 15

def test_polygamma_high_order():
    mp.dps = 100
    assert str(psi(50, pi)) == "-1344100348958402765749252447726432491812.641985273160531055707095989227897753035823152397679626136483"
    assert str(psi(50, pi + 14*e)) == "-0.00000000000000000189793739550804321623512073101895801993019919886375952881053090844591920308111549337295143780341396"
    assert str(psi(50, pi + 14*e*j)) == ("(-0.0000000000000000522516941152169248975225472155683565752375889510631513244785"
        "9377385233700094871256507814151956624433 - 0.00000000000000001813157041407010184"
        "702414110218205348527862196327980417757665282244728963891298080199341480881811613j)")
    mp.dps = 15
    assert str(psi(50, pi)) == "-1.34410034895841e+39"
    assert str(psi(50, pi + 14*e)) == "-1.89793739550804e-18"
    assert str(psi(50, pi + 14*e*j)) == "(-5.2251694115217e-17 - 1.81315704140701e-17j)"

def test_harmonic():
    mp.dps = 15
    assert harmonic(0) == 0
    assert harmonic(1) == 1
    assert harmonic(2) == 1.5
    assert harmonic(3).ae(1. + 1./2 + 1./3)
    assert harmonic(10**10).ae(23.603066594891989701)
    assert harmonic(10**1000).ae(2303.162308658947)
    assert harmonic(0.5).ae(2-2*log(2))
    assert harmonic(inf) == inf
    assert harmonic(2+0j) == 1.5+0j
    assert harmonic(1+2j).ae(1.4918071802755104+0.92080728264223022j)

def test_gamma_huge_1():
    mp.dps = 500
    x = mpf(10**10) / 7
    mp.dps = 15
    assert str(gamma(x)) == "6.26075321389519e+12458010678"
    mp.dps = 50
    assert str(gamma(x)) == "6.2607532138951929201303779291707455874010420783933e+12458010678"
    mp.dps = 15

def test_gamma_huge_2():
    mp.dps = 500
    x = mpf(10**100) / 19
    mp.dps = 15
    assert str(gamma(x)) == (\
        "1.82341134776679e+5172997469323364168990133558175077136829182824042201886051511"
        "9656908623426021308685461258226190190661")
    mp.dps = 50
    assert str(gamma(x)) == (\
        "1.82341134776678875374414910350027596939980412984e+5172997469323364168990133558"
        "1750771368291828240422018860515119656908623426021308685461258226190190661")

def test_gamma_huge_3():
    mp.dps = 500
    x = 10**80 // 3 + 10**70*j / 7
    mp.dps = 15
    y = gamma(x)
    assert str(y.real) == (\
        "-6.82925203918106e+2636286142112569524501781477865238132302397236429627932441916"
        "056964386399485392600")
    assert str(y.imag) == (\
        "8.54647143678418e+26362861421125695245017814778652381323023972364296279324419160"
        "56964386399485392600")
    mp.dps = 50
    y = gamma(x)
    assert str(y.real) == (\
        "-6.8292520391810548460682736226799637356016538421817e+26362861421125695245017814"
        "77865238132302397236429627932441916056964386399485392600")
    assert str(y.imag) == (\
        "8.5464714367841748507479306948130687511711420234015e+263628614211256952450178147"
        "7865238132302397236429627932441916056964386399485392600")

def test_gamma_huge_4():
    x = 3200+11500j
    mp.dps = 15
    assert str(gamma(x)) == \
        "(8.95783268539713e+5164 - 1.94678798329735e+5164j)"
    mp.dps = 50
    assert str(gamma(x)) == (\
        "(8.9578326853971339570292952697675570822206567327092e+5164"
        " - 1.9467879832973509568895402139429643650329524144794e+51"
        "64j)")
    mp.dps = 15

def test_gamma_huge_5():
    mp.dps = 500
    x = 10**60 * j / 3
    mp.dps = 15
    y = gamma(x)
    assert str(y.real) == "-3.27753899634941e-227396058973640224580963937571892628368354580620654233316839"
    assert str(y.imag) == "-7.1519888950416e-227396058973640224580963937571892628368354580620654233316841"
    mp.dps = 50
    y = gamma(x)
    assert str(y.real) == (\
        "-3.2775389963494132168950056995974690946983219123935e-22739605897364022458096393"
        "7571892628368354580620654233316839")
    assert str(y.imag) == (\
        "-7.1519888950415979749736749222530209713136588885897e-22739605897364022458096393"
        "7571892628368354580620654233316841")
    mp.dps = 15

def test_gamma_huge_7():
    mp.dps = 100
    a = 3 + j/mpf(10)**1000
    mp.dps = 15
    y = gamma(a)
    assert str(y.real) == "2.0"
    # wrong
    #assert str(y.imag) == "2.16735365342606e-1000"
    assert str(y.imag) == "1.84556867019693e-1000"
    mp.dps = 50
    y = gamma(a)
    assert str(y.real) == "2.0"
    #assert str(y.imag) == "2.1673536534260596065418805612488708028522563689298e-1000"
    assert str(y.imag) ==  "1.8455686701969342787869758198351951379156813281202e-1000"

def test_stieltjes():
    mp.dps = 15
    assert stieltjes(0).ae(+euler)
    mp.dps = 25
    assert stieltjes(1).ae('-0.07281584548367672486058637587')
    assert stieltjes(2).ae('-0.009690363192872318484530386035')
    assert stieltjes(3).ae('0.002053834420303345866160046543')
    assert stieltjes(4).ae('0.002325370065467300057468170178')
    mp.dps = 15
    assert stieltjes(1).ae(-0.07281584548367672486058637587)
    assert stieltjes(2).ae(-0.009690363192872318484530386035)
    assert stieltjes(3).ae(0.002053834420303345866160046543)
    assert stieltjes(4).ae(0.0023253700654673000574681701775)

def test_barnesg():
    mp.dps = 15
    assert barnesg(0) == barnesg(-1) == 0
    assert [superfac(i) for i in range(8)] == [1, 1, 2, 12, 288, 34560, 24883200, 125411328000]
    assert str(superfac(1000)) == '3.24570818422368e+1177245'
    assert isnan(barnesg(nan))
    assert isnan(superfac(nan))
    assert isnan(hyperfac(nan))
    assert barnesg(inf) == inf
    assert superfac(inf) == inf
    assert hyperfac(inf) == inf
    assert isnan(superfac(-inf))
    assert barnesg(0.7).ae(0.8068722730141471)
    assert barnesg(2+3j).ae(-0.17810213864082169+0.04504542715447838j)
    assert [hyperfac(n) for n in range(7)] == [1, 1, 4, 108, 27648, 86400000, 4031078400000]
    assert [hyperfac(n) for n in range(0,-7,-1)] == [1,1,-1,-4,108,27648,-86400000]
    a = barnesg(-3+0j)
    assert a == 0 and isinstance(a, mpc)
    a = hyperfac(-3+0j)
    assert a == -4 and isinstance(a, mpc)

def test_polylog():
    mp.dps = 15
    zs = [mpmathify(z) for z in [0, 0.5, 0.99, 4, -0.5, -4, 1j, 3+4j]]
    for z in zs: assert polylog(1, z).ae(-log(1-z))
    for z in zs: assert polylog(0, z).ae(z/(1-z))
    for z in zs: assert polylog(-1, z).ae(z/(1-z)**2)
    for z in zs: assert polylog(-2, z).ae(z*(1+z)/(1-z)**3)
    for z in zs: assert polylog(-3, z).ae(z*(1+4*z+z**2)/(1-z)**4)
    assert polylog(3, 7).ae(5.3192579921456754382-5.9479244480803301023j)
    assert polylog(3, -7).ae(-4.5693548977219423182)
    assert polylog(2, 0.9).ae(1.2997147230049587252)
    assert polylog(2, -0.9).ae(-0.75216317921726162037)
    assert polylog(2, 0.9j).ae(-0.17177943786580149299+0.83598828572550503226j)
    assert polylog(2, 1.1).ae(1.9619991013055685931-0.2994257606855892575j)
    assert polylog(2, -1.1).ae(-0.89083809026228260587)
    assert polylog(2, 1.1*sqrt(j)).ae(0.58841571107611387722+1.09962542118827026011j)
    assert polylog(-2, 0.9).ae(1710)
    assert polylog(-2, -0.9).ae(-90/6859.)
    assert polylog(3, 0.9).ae(1.0496589501864398696)
    assert polylog(-3, 0.9).ae(48690)
    assert polylog(-3, -4).ae(-0.0064)
    assert polylog(0.5+j/3, 0.5+j/2).ae(0.31739144796565650535 + 0.99255390416556261437j)
    assert polylog(3+4j,1).ae(zeta(3+4j))
    assert polylog(3+4j,-1).ae(-altzeta(3+4j))
    # issue 390
    assert polylog(1.5, -48.910886523731889).ae(-6.272992229311817)
    assert polylog(1.5, 200).ae(-8.349608319033686529 - 8.159694826434266042j)
    assert polylog(-2+0j, -2).ae(mpf(1)/13.5)
    assert polylog(-2+0j, 1.25).ae(-180)

def test_bell_polyexp():
    mp.dps = 15
    # TODO: more tests for polyexp
    assert (polyexp(0,1e-10)*10**10).ae(1.00000000005)
    assert (polyexp(1,1e-10)*10**10).ae(1.0000000001)
    assert polyexp(5,3j).ae(-607.7044517476176454+519.962786482001476087j)
    assert polyexp(-1,3.5).ae(12.09537536175543444)
    # bell(0,x) = 1
    assert bell(0,0) == 1
    assert bell(0,1) == 1
    assert bell(0,2) == 1
    assert bell(0,inf) == 1
    assert bell(0,-inf) == 1
    assert isnan(bell(0,nan))
    # bell(1,x) = x
    assert bell(1,4) == 4
    assert bell(1,0) == 0
    assert bell(1,inf) == inf
    assert bell(1,-inf) == -inf
    assert isnan(bell(1,nan))
    # bell(2,x) = x*(1+x)
    assert bell(2,-1) == 0
    assert bell(2,0) == 0
    # large orders / arguments
    assert bell(10) == 115975
    assert bell(10,1) == 115975
    assert bell(10, -8) == 11054008
    assert bell(5,-50) == -253087550
    assert bell(50,-50).ae('3.4746902914629720259e74')
    mp.dps = 80
    assert bell(50,-50) == 347469029146297202586097646631767227177164818163463279814268368579055777450
    assert bell(40,50) == 5575520134721105844739265207408344706846955281965031698187656176321717550
    assert bell(74) == 5006908024247925379707076470957722220463116781409659160159536981161298714301202
    mp.dps = 15
    assert bell(10,20j) == 7504528595600+15649605360020j
    # continuity of the generalization
    assert bell(0.5,0).ae(sinc(pi*0.5))

def test_primezeta():
    mp.dps = 15
    assert primezeta(0.9).ae(1.8388316154446882243 + 3.1415926535897932385j)
    assert primezeta(4).ae(0.076993139764246844943)
    assert primezeta(1) == inf
    assert primezeta(inf) == 0
    assert isnan(primezeta(nan))

def test_rs_zeta():
    mp.dps = 15
    assert zeta(0.5+100000j).ae(1.0730320148577531321 + 5.7808485443635039843j)
    assert zeta(0.75+100000j).ae(1.837852337251873704 + 1.9988492668661145358j)
    assert zeta(0.5+1000000j, derivative=3).ae(1647.7744105852674733 - 1423.1270943036622097j)
    assert zeta(1+1000000j, derivative=3).ae(3.4085866124523582894 - 18.179184721525947301j)
    assert zeta(1+1000000j, derivative=1).ae(-0.10423479366985452134 - 0.74728992803359056244j)
    assert zeta(0.5-1000000j, derivative=1).ae(11.636804066002521459 + 17.127254072212996004j)
    # Additional sanity tests using fp arithmetic.
    # Some more high-precision tests are found in the docstrings
    def ae(x, y, tol=1e-6):
        return abs(x-y) < tol*abs(y)
    assert ae(fp.zeta(0.5-100000j), 1.0730320148577531321 - 5.7808485443635039843j)
    assert ae(fp.zeta(0.75-100000j), 1.837852337251873704 - 1.9988492668661145358j)
    assert ae(fp.zeta(0.5+1e6j), 0.076089069738227100006 + 2.8051021010192989554j)
    assert ae(fp.zeta(0.5+1e6j, derivative=1), 11.636804066002521459 - 17.127254072212996004j)
    assert ae(fp.zeta(1+1e6j), 0.94738726251047891048 + 0.59421999312091832833j)
    assert ae(fp.zeta(1+1e6j, derivative=1), -0.10423479366985452134 - 0.74728992803359056244j)
    assert ae(fp.zeta(0.5+100000j, derivative=1), 10.766962036817482375 - 30.92705282105996714j)
    assert ae(fp.zeta(0.5+100000j, derivative=2), -119.40515625740538429 + 217.14780631141830251j)
    assert ae(fp.zeta(0.5+100000j, derivative=3), 1129.7550282628460881 - 1685.4736895169690346j)
    assert ae(fp.zeta(0.5+100000j, derivative=4), -10407.160819314958615 + 13777.786698628045085j)
    assert ae(fp.zeta(0.75+100000j, derivative=1), -0.41742276699594321475 - 6.4453816275049955949j)
    assert ae(fp.zeta(0.75+100000j, derivative=2), -9.214314279161977266 + 35.07290795337967899j)
    assert ae(fp.zeta(0.75+100000j, derivative=3), 110.61331857820103469 - 236.87847130518129926j)
    assert ae(fp.zeta(0.75+100000j, derivative=4), -1054.334275898559401 + 1769.9177890161596383j)

def test_siegelz():
    mp.dps = 15
    assert siegelz(100000).ae(5.87959246868176504171)
    assert siegelz(100000, derivative=2).ae(-54.1172711010126452832)
    assert siegelz(100000, derivative=3).ae(-278.930831343966552538)
    assert siegelz(100000+j,derivative=1).ae(678.214511857070283307-379.742160779916375413j)



def test_zeta_near_1():
    # Test for a former bug in mpf_zeta and mpc_zeta
    mp.dps = 15
    s1 = fadd(1, '1e-10', exact=True)
    s2 = fadd(1, '-1e-10', exact=True)
    s3 = fadd(1, '1e-10j', exact=True)
    assert zeta(s1).ae(1.000000000057721566490881444e10)
    assert zeta(s2).ae(-9.99999999942278433510574872e9)
    z = zeta(s3)
    assert z.real.ae(0.57721566490153286060)
    assert z.imag.ae(-9.9999999999999999999927184e9)
    mp.dps = 30
    s1 = fadd(1, '1e-50', exact=True)
    s2 = fadd(1, '-1e-50', exact=True)
    s3 = fadd(1, '1e-50j', exact=True)
    assert zeta(s1).ae('1e50')
    assert zeta(s2).ae('-1e50')
    z = zeta(s3)
    assert z.real.ae('0.57721566490153286060651209008240243104215933593992')
    assert z.imag.ae('-1e50')
