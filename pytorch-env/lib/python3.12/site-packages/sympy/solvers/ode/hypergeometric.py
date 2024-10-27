r'''
This module contains the implementation of the 2nd_hypergeometric hint for
dsolve. This is an incomplete implementation of the algorithm described in [1].
The algorithm solves 2nd order linear ODEs of the form

.. math:: y'' + A(x) y' + B(x) y = 0\text{,}

where `A` and `B` are rational functions. The algorithm should find any
solution of the form

.. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
Currently only the 2F1 case is implemented in SymPy but the other cases are
described in the paper and could be implemented in future (contributions
welcome!).

References
==========

.. [1] L. Chan, E.S. Cheb-Terrab, Non-Liouvillian solutions for second order
       linear ODEs, (2004).
       https://arxiv.org/abs/math-ph/0402063
'''

from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants


def match_2nd_hypergeometric(eq, func):
    x = func.args[0]
    df = func.diff(x)
    a3 = Wild('a3', exclude=[func, func.diff(x), func.diff(x, 2)])
    b3 = Wild('b3', exclude=[func, func.diff(x), func.diff(x, 2)])
    c3 = Wild('c3', exclude=[func, func.diff(x), func.diff(x, 2)])
    deq = a3*(func.diff(x, 2)) + b3*df + c3*func
    r = collect(eq,
        [func.diff(x, 2), func.diff(x), func]).match(deq)
    if r:
        if not all(val.is_polynomial() for val in r.values()):
            n, d = eq.as_numer_denom()
            eq = expand(n)
            r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)

    if r and r[a3]!=0:
        A = cancel(r[b3]/r[a3])
        B = cancel(r[c3]/r[a3])
        return [A, B]
    else:
        return []


def equivalence_hypergeometric(A, B, func):
    # This method for finding the equivalence is only for 2F1 type.
    # We can extend it for 1F1 and 0F1 type also.
    x = func.args[0]

    # making given equation in normal form
    I1 = factor(cancel(A.diff(x)/2 + A**2/4 - B))

    # computing shifted invariant(J1) of the equation
    J1 = factor(cancel(x**2*I1 + S(1)/4))
    num, dem = J1.as_numer_denom()
    num = powdenest(expand(num))
    dem = powdenest(expand(dem))
    # this function will compute the different powers of variable(x) in J1.
    # then it will help in finding value of k. k is power of x such that we can express
    # J1 = x**k * J0(x**k) then all the powers in J0 become integers.
    def _power_counting(num):
        _pow = {0}
        for val in num:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    _pow.add(val.as_base_exp()[1])
                elif val == x:
                    _pow.add(val.as_base_exp()[1])
                else:
                    _pow.update(_power_counting(val.args))
        return _pow

    pow_num = _power_counting((num, ))
    pow_dem = _power_counting((dem, ))
    pow_dem.update(pow_num)

    _pow = pow_dem
    k = gcd(_pow)

    # computing I0 of the given equation
    I0 = powdenest(simplify(factor(((J1/k**2) - S(1)/4)/((x**k)**2))), force=True)
    I0 = factor(cancel(powdenest(I0.subs(x, x**(S(1)/k)), force=True)))

    # Before this point I0, J1 might be functions of e.g. sqrt(x) but replacing
    # x with x**(1/k) should result in I0 being a rational function of x or
    # otherwise the hypergeometric solver cannot be used. Note that k can be a
    # non-integer rational such as 2/7.
    if not I0.is_rational_function(x):
        return None

    num, dem = I0.as_numer_denom()

    max_num_pow = max(_power_counting((num, )))
    dem_args = dem.args
    sing_point = []
    dem_pow = []
    # calculating singular point of I0.
    for arg in dem_args:
        if arg.has(x):
            if isinstance(arg, Pow):
                # (x-a)**n
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg.as_base_exp()[0], x).keys())[0])
            else:
                # (x-a) type
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg, x).keys())[0])

    dem_pow.sort()
    # checking if equivalence is exists or not.

    if equivalence(max_num_pow, dem_pow) == "2F1":
        return {'I0':I0, 'k':k, 'sing_point':sing_point, 'type':"2F1"}
    else:
        return None


def match_2nd_2F1_hypergeometric(I, k, sing_point, func):
    x = func.args[0]
    a = Wild("a")
    b = Wild("b")
    c = Wild("c")
    t = Wild("t")
    s = Wild("s")
    r = Wild("r")
    alpha = Wild("alpha")
    beta = Wild("beta")
    gamma = Wild("gamma")
    delta = Wild("delta")
    # I0 of the standerd 2F1 equation.
    I0 = ((a-b+1)*(a-b-1)*x**2 + 2*((1-a-b)*c + 2*a*b)*x + c*(c-2))/(4*x**2*(x-1)**2)
    if sing_point != [0, 1]:
        # If singular point is [0, 1] then we have standerd equation.
        eqs = []
        sing_eqs = [-beta/alpha, -delta/gamma, (delta-beta)/(alpha-gamma)]
        # making equations for the finding the mobius transformation
        for i in range(3):
            if i<len(sing_point):
                eqs.append(Eq(sing_eqs[i], sing_point[i]))
            else:
                eqs.append(Eq(1/sing_eqs[i], 0))
        # solving above equations for the mobius transformation
        _beta = -alpha*sing_point[0]
        _delta = -gamma*sing_point[1]
        _gamma = alpha
        if len(sing_point) == 3:
            _gamma = (_beta + sing_point[2]*alpha)/(sing_point[2] - sing_point[1])
        mob = (alpha*x + beta)/(gamma*x + delta)
        mob = mob.subs(beta, _beta)
        mob = mob.subs(delta, _delta)
        mob = mob.subs(gamma, _gamma)
        mob = cancel(mob)
        t = (beta - delta*x)/(gamma*x - alpha)
        t = cancel(((t.subs(beta, _beta)).subs(delta, _delta)).subs(gamma, _gamma))
    else:
        mob = x
        t = x

    # applying mobius transformation in I to make it into I0.
    I = I.subs(x, t)
    I = I*(t.diff(x))**2
    I = factor(I)
    dict_I = {x**2:0, x:0, 1:0}
    I0_num, I0_dem = I0.as_numer_denom()
    # collecting coeff of (x**2, x), of the standerd equation.
    # substituting (a-b) = s, (a+b) = r
    dict_I0 = {x**2:s**2 - 1, x:(2*(1-r)*c + (r+s)*(r-s)), 1:c*(c-2)}
    # collecting coeff of (x**2, x) from I0 of the given equation.
    dict_I.update(collect(expand(cancel(I*I0_dem)), [x**2, x], evaluate=False))
    eqs = []
    # We are comparing the coeff of powers of different x, for finding the values of
    # parameters of standerd equation.
    for key in [x**2, x, 1]:
        eqs.append(Eq(dict_I[key], dict_I0[key]))

    # We can have many possible roots for the equation.
    # I am selecting the root on the basis that when we have
    # standard equation eq = x*(x-1)*f(x).diff(x, 2) + ((a+b+1)*x-c)*f(x).diff(x) + a*b*f(x)
    # then root should be a, b, c.

    _c = 1 - factor(sqrt(1+eqs[2].lhs))
    if not _c.has(Symbol):
        _c = min(list(roots(eqs[2], c)))
    _s = factor(sqrt(eqs[0].lhs + 1))
    _r = _c - factor(sqrt(_c**2 + _s**2 + eqs[1].lhs - 2*_c))
    _a = (_r + _s)/2
    _b = (_r - _s)/2

    rn = {'a':simplify(_a), 'b':simplify(_b), 'c':simplify(_c), 'k':k, 'mobius':mob, 'type':"2F1"}
    return rn


def equivalence(max_num_pow, dem_pow):
    # this function is made for checking the equivalence with 2F1 type of equation.
    # max_num_pow is the value of maximum power of x in numerator
    # and dem_pow is list of powers of different factor of form (a*x b).
    # reference from table 1 in paper - "Non-Liouvillian solutions for second order
    # linear ODEs" by L. Chan, E.S. Cheb-Terrab.
    # We can extend it for 1F1 and 0F1 type also.

    if max_num_pow == 2:
        if dem_pow in [[2, 2], [2, 2, 2]]:
            return "2F1"
    elif max_num_pow == 1:
        if dem_pow in [[1, 2, 2], [2, 2, 2], [1, 2], [2, 2]]:
            return "2F1"
    elif max_num_pow == 0:
        if dem_pow in [[1, 1, 2], [2, 2], [1, 2, 2], [1, 1], [2], [1, 2], [2, 2]]:
            return "2F1"

    return None


def get_sol_2F1_hypergeometric(eq, func, match_object):
    x = func.args[0]
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.polys.polytools import factor
    C0, C1 = get_numbered_constants(eq, num=2)
    a = match_object['a']
    b = match_object['b']
    c = match_object['c']
    A = match_object['A']

    sol = None

    if c.is_integer == False:
        sol = C0*hyper([a, b], [c], x) + C1*hyper([a-c+1, b-c+1], [2-c], x)*x**(1-c)
    elif c == 1:
        y2 = Integral(exp(Integral((-(a+b+1)*x + c)/(x**2-x), x))/(hyperexpand(hyper([a, b], [c], x))**2), x)*hyper([a, b], [c], x)
        sol = C0*hyper([a, b], [c], x) + C1*y2
    elif (c-a-b).is_integer == False:
        sol = C0*hyper([a, b], [1+a+b-c], 1-x) + C1*hyper([c-a, c-b], [1+c-a-b], 1-x)*(1-x)**(c-a-b)

    if sol:
        # applying transformation in the solution
        subs = match_object['mobius']
        dtdx = simplify(1/(subs.diff(x)))
        _B = ((a + b + 1)*x - c).subs(x, subs)*dtdx
        _B = factor(_B + ((x**2 -x).subs(x, subs))*(dtdx.diff(x)*dtdx))
        _A = factor((x**2 - x).subs(x, subs)*(dtdx**2))
        e = exp(logcombine(Integral(cancel(_B/(2*_A)), x), force=True))
        sol = sol.subs(x, match_object['mobius'])
        sol = sol.subs(x, x**match_object['k'])
        e = e.subs(x, x**match_object['k'])

        if not A.is_zero:
            e1 = Integral(A/2, x)
            e1 = exp(logcombine(e1, force=True))
            sol = cancel((e/e1)*x**((-match_object['k']+1)/2))*sol
            sol = Eq(func, sol)
            return sol

        sol = cancel((e)*x**((-match_object['k']+1)/2))*sol
        sol = Eq(func, sol)
    return sol
