from mpmath import *
from mpmath.libmp import ifac

import sys
if "-dps" in sys.argv:
    maxdps = int(sys.argv[sys.argv.index("-dps")+1])
else:
    maxdps = 1000

raise_ = "-raise" in sys.argv

errcount = 0

def check(name, func, z, y):
    global errcount
    try:
        x = func(z)
    except:
        errcount += 1
        if raise_:
            raise
        print()
        print(name)
        print("EXCEPTION")
        import traceback
        traceback.print_tb(sys.exc_info()[2])
        print()
        return
    xre = x.real
    xim = x.imag
    yre = y.real
    yim = y.imag
    tol = eps*8
    err = 0
    if abs(xre-yre) > abs(yre)*tol:
        err = 1
        print()
        print("Error! %s (re = %s, wanted %s, err=%s)" % (name, nstr(xre,10), nstr(yre,10), nstr(abs(xre-yre))))
        errcount += 1
        if raise_:
            raise SystemExit
    if abs(xim-yim) > abs(yim)*tol:
        err = 1
        print()
        print("Error! %s (im = %s, wanted %s, err=%s)" % (name, nstr(xim,10), nstr(yim,10), nstr(abs(xim-yim))))
        errcount += 1
        if raise_:
            raise SystemExit
    if not err:
        sys.stdout.write("%s ok; " % name)

def testcase(case):
    z, result = case
    print("Testing z =", z)
    mp.dps = 1010
    z = eval(z)
    mp.dps = maxdps + 50
    if result is None:
        gamma_val = gamma(z)
        loggamma_val = loggamma(z)
        factorial_val = factorial(z)
        rgamma_val = rgamma(z)
    else:
        loggamma_val = eval(result)
        gamma_val = exp(loggamma_val)
        factorial_val = z * gamma_val
        rgamma_val = 1/gamma_val
    for dps in [5, 10, 15, 25, 40, 60, 90, 120, 250, 600, 1000, 1800, 3600]:
        if dps > maxdps:
            break
        mp.dps = dps
        print("dps = %s" % dps)
        check("gamma", gamma, z, gamma_val)
        check("rgamma", rgamma, z, rgamma_val)
        check("loggamma", loggamma, z, loggamma_val)
        check("factorial", factorial, z, factorial_val)
        print()
        mp.dps = 15

testcases = []

# Basic values
for n in list(range(1,200)) + list(range(201,2000,17)):
    testcases.append(["%s" % n, None])
for n in range(-200,200):
    testcases.append(["%s+0.5" % n, None])
    testcases.append(["%s+0.37" % n, None])

testcases += [\
["(0.1+1j)", None],
["(-0.1+1j)", None],
["(0.1-1j)", None],
["(-0.1-1j)", None],
["10j", None],
["-10j", None],
["100j", None],
["10000j", None],
["-10000000j", None],
["(10**100)*j", None],
["125+(10**100)*j", None],
["-125+(10**100)*j", None],
["(10**10)*(1+j)", None],
["(10**10)*(-1+j)", None],
["(10**100)*(1+j)", None],
["(10**100)*(-1+j)", None],
["(1.5-1j)", None],
["(6+4j)", None],
["(4+1j)", None],
["(3.5+2j)", None],
["(1.5-1j)", None],
["(-6-4j)", None],
["(-2-3j)", None],
["(-2.5-2j)", None],
["(4+1j)", None],
["(3+3j)", None],
["(2-2j)", None],
["1", "0"],
["2", "0"],
["3", "log(2)"],
["4", "log(6)"],
["5", "log(24)"],
["0.5", "log(pi)/2"],
["1.5", "log(sqrt(pi)/2)"],
["2.5", "log(3*sqrt(pi)/4)"],
["mpf('0.37')", None],
["0.25", "log(sqrt(2*sqrt(2*pi**3)/agm(1,sqrt(2))))"],
["-0.4", None],
["mpf('-1.9')", None],
["mpf('12.8')", None],
["mpf('33.7')", None],
["mpf('95.2')", None],
["mpf('160.3')", None],
["mpf('2057.8')", None],
["25", "log(ifac(24))"],
["80", "log(ifac(79))"],
["500", "log(ifac(500-1))"],
["8000", "log(ifac(8000-1))"],
["8000.5", None],
["mpf('8000.1')", None],
["mpf('1.37e10')", None],
["mpf('1.37e10')*(1+j)", None],
["mpf('1.37e10')*(-1+j)", None],
["mpf('1.37e10')*(-1-j)", None],
["mpf('1.37e10')*(-1+j)", None],
["mpf('1.37e100')", None],
["mpf('1.37e100')*(1+j)", None],
["mpf('1.37e100')*(-1+j)", None],
["mpf('1.37e100')*(-1-j)", None],
["mpf('1.37e100')*(-1+j)", None],
["3+4j",
"mpc('"
"-1.7566267846037841105306041816232757851567066070613445016197619371316057169"
"4723618263960834804618463052988607348289672535780644470689771115236512106002"
"5970873471563240537307638968509556191696167970488390423963867031934333890838"
"8009531786948197210025029725361069435208930363494971027388382086721660805397"
"9163230643216054580167976201709951509519218635460317367338612500626714783631"
"7498317478048447525674016344322545858832610325861086336204591943822302971823"
"5161814175530618223688296232894588415495615809337292518431903058265147109853"
"1710568942184987827643886816200452860853873815413367529829631430146227470517"
"6579967222200868632179482214312673161276976117132204633283806161971389519137"
"1243359764435612951384238091232760634271570950240717650166551484551654327989"
"9360285030081716934130446150245110557038117075172576825490035434069388648124"
"6678152254554001586736120762641422590778766100376515737713938521275749049949"
"1284143906816424244705094759339932733567910991920631339597278805393743140853"
"391550313363278558195609260225928','"
"4.74266443803465792819488940755002274088830335171164611359052405215840070271"
"5906813009373171139767051863542508136875688550817670379002790304870822775498"
"2809996675877564504192565392367259119610438951593128982646945990372179860613"
"4294436498090428077839141927485901735557543641049637962003652638924845391650"
"9546290137755550107224907606529385248390667634297183361902055842228798984200"
"9591180450211798341715874477629099687609819466457990642030707080894518168924"
"6805549314043258530272479246115112769957368212585759640878745385160943755234"
"9398036774908108204370323896757543121853650025529763655312360354244898913463"
"7115955702828838923393113618205074162812089732064414530813087483533203244056"
"0546577484241423134079056537777170351934430586103623577814746004431994179990"
"5318522939077992613855205801498201930221975721246498720895122345420698451980"
"0051215797310305885845964334761831751370672996984756815410977750799748813563"
"8784405288158432214886648743541773208808731479748217023665577802702269468013"
"673719173759245720489020315779001')"],
]

for z in [4, 14, 34, 64]:
    testcases.append(["(2+j)*%s/3" % z, None])
    testcases.append(["(-2+j)*%s/3" % z, None])
    testcases.append(["(1+2*j)*%s/3" % z, None])
    testcases.append(["(2-j)*%s/3" % z, None])
    testcases.append(["(20+j)*%s/3" % z, None])
    testcases.append(["(-20+j)*%s/3" % z, None])
    testcases.append(["(1+20*j)*%s/3" % z, None])
    testcases.append(["(20-j)*%s/3" % z, None])
    testcases.append(["(200+j)*%s/3" % z, None])
    testcases.append(["(-200+j)*%s/3" % z, None])
    testcases.append(["(1+200*j)*%s/3" % z, None])
    testcases.append(["(200-j)*%s/3" % z, None])

# Poles
for n in [0,1,2,3,4,25,-1,-2,-3,-4,-20,-21,-50,-51,-200,-201,-20000,-20001]:
    for t in ['1e-5', '1e-20', '1e-100', '1e-10000']:
        testcases.append(["fadd(%s,'%s',exact=True)" % (n, t), None])
        testcases.append(["fsub(%s,'%s',exact=True)" % (n, t), None])
        testcases.append(["fadd(%s,'%sj',exact=True)" % (n, t), None])
        testcases.append(["fsub(%s,'%sj',exact=True)" % (n, t), None])

if __name__ == "__main__":
    from timeit import default_timer as clock
    tot_time = 0.0
    for case in testcases:
        t1 = clock()
        testcase(case)
        t2 = clock()
        print("Test time:", t2-t1)
        print()
        tot_time += (t2-t1)
    print("Total time:", tot_time)
    print("Errors:", errcount)
