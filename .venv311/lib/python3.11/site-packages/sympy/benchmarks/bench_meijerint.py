# conceal the implicit import from the code quality tester
from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.bessel import besseli
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.integrals.transforms import (mellin_transform,
    inverse_fourier_transform, inverse_mellin_transform,
    laplace_transform, inverse_laplace_transform, fourier_transform)

LT = laplace_transform
FT = fourier_transform
MT = mellin_transform
IFT = inverse_fourier_transform
ILT = inverse_laplace_transform
IMT = inverse_mellin_transform

from sympy.abc import x, y
nu, beta, rho = symbols('nu beta rho')

apos, bpos, cpos, dpos, posk, p = symbols('a b c d k p', positive=True)
k = Symbol('k', real=True)
negk = Symbol('k', negative=True)

mu1, mu2 = symbols('mu1 mu2', real=True, nonzero=True, finite=True)
sigma1, sigma2 = symbols('sigma1 sigma2', real=True, nonzero=True,
                         finite=True, positive=True)
rate = Symbol('lambda', positive=True)


def normal(x, mu, sigma):
    return 1/sqrt(2*pi*sigma**2)*exp(-(x - mu)**2/2/sigma**2)


def exponential(x, rate):
    return rate*exp(-rate*x)
alpha, beta = symbols('alpha beta', positive=True)
betadist = x**(alpha - 1)*(1 + x)**(-alpha - beta)*gamma(alpha + beta) \
    /gamma(alpha)/gamma(beta)
kint = Symbol('k', integer=True, positive=True)
chi = 2**(1 - kint/2)*x**(kint - 1)*exp(-x**2/2)/gamma(kint/2)
chisquared = 2**(-k/2)/gamma(k/2)*x**(k/2 - 1)*exp(-x/2)
dagum = apos*p/x*(x/bpos)**(apos*p)/(1 + x**apos/bpos**apos)**(p + 1)
d1, d2 = symbols('d1 d2', positive=True)
f = sqrt(((d1*x)**d1 * d2**d2)/(d1*x + d2)**(d1 + d2))/x \
    /gamma(d1/2)/gamma(d2/2)*gamma((d1 + d2)/2)
nupos, sigmapos = symbols('nu sigma', positive=True)
rice = x/sigmapos**2*exp(-(x**2 + nupos**2)/2/sigmapos**2)*besseli(0, x*
                         nupos/sigmapos**2)
mu = Symbol('mu', real=True)
laplace = exp(-abs(x - mu)/bpos)/2/bpos

u = Symbol('u', polar=True)
tpos = Symbol('t', positive=True)


def E(expr):
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (x, 0, oo), (y, -oo, oo), meijerg=True)
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (y, -oo, oo), (x, 0, oo), meijerg=True)

bench = [
    'MT(x**nu*Heaviside(x - 1), x, s)',
    'MT(x**nu*Heaviside(1 - x), x, s)',
    'MT((1-x)**(beta - 1)*Heaviside(1-x), x, s)',
    'MT((x-1)**(beta - 1)*Heaviside(x-1), x, s)',
    'MT((1+x)**(-rho), x, s)',
    'MT(abs(1-x)**(-rho), x, s)',
    'MT((1-x)**(beta-1)*Heaviside(1-x) + a*(x-1)**(beta-1)*Heaviside(x-1), x, s)',
    'MT((x**a-b**a)/(x-b), x, s)',
    'MT((x**a-bpos**a)/(x-bpos), x, s)',
    'MT(exp(-x), x, s)',
    'MT(exp(-1/x), x, s)',
    'MT(log(x)**4*Heaviside(1-x), x, s)',
    'MT(log(x)**3*Heaviside(x-1), x, s)',
    'MT(log(x + 1), x, s)',
    'MT(log(1/x + 1), x, s)',
    'MT(log(abs(1 - x)), x, s)',
    'MT(log(abs(1 - 1/x)), x, s)',
    'MT(log(x)/(x+1), x, s)',
    'MT(log(x)**2/(x+1), x, s)',
    'MT(log(x)/(x+1)**2, x, s)',
    'MT(erf(sqrt(x)), x, s)',

    'MT(besselj(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2, x, s)',
    'MT(besselj(a, sqrt(x))*besselj(-a, sqrt(x)), x, s)',
    'MT(besselj(a - 1, sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*besselj(b, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2 + besselj(-a, sqrt(x))**2, x, s)',
    'MT(bessely(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(b, sqrt(x)), x, s)',
    'MT(bessely(a, sqrt(x))**2, x, s)',

    'MT(besselk(a, 2*sqrt(x)), x, s)',
    'MT(besselj(a, 2*sqrt(2*sqrt(x)))*besselk(a, 2*sqrt(2*sqrt(x))), x, s)',
    'MT(besseli(a, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(besseli(b, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(exp(-x/2)*besselk(a, x/2), x, s)',

    # later: ILT, IMT

    'LT((t-apos)**bpos*exp(-cpos*(t-apos))*Heaviside(t-apos), t, s)',
    'LT(t**apos, t, s)',
    'LT(Heaviside(t), t, s)',
    'LT(Heaviside(t - apos), t, s)',
    'LT(1 - exp(-apos*t), t, s)',
    'LT((exp(2*t)-1)*exp(-bpos - t)*Heaviside(t)/2, t, s, noconds=True)',
    'LT(exp(t), t, s)',
    'LT(exp(2*t), t, s)',
    'LT(exp(apos*t), t, s)',
    'LT(log(t/apos), t, s)',
    'LT(erf(t), t, s)',
    'LT(sin(apos*t), t, s)',
    'LT(cos(apos*t), t, s)',
    'LT(exp(-apos*t)*sin(bpos*t), t, s)',
    'LT(exp(-apos*t)*cos(bpos*t), t, s)',
    'LT(besselj(0, t), t, s, noconds=True)',
    'LT(besselj(1, t), t, s, noconds=True)',

    'FT(Heaviside(1 - abs(2*apos*x)), x, k)',
    'FT(Heaviside(1-abs(apos*x))*(1-abs(apos*x)), x, k)',
    'FT(exp(-apos*x)*Heaviside(x), x, k)',
    'IFT(1/(apos + 2*pi*I*x), x, posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, -posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, negk)',
    'FT(x*exp(-apos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x)*sin(bpos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x**2), x, k)',
    'IFT(sqrt(pi/apos)*exp(-(pi*k)**2/apos), k, x)',
    'FT(exp(-apos*abs(x)), x, k)',

    'integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**2*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**3*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x*y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate((x+y+1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate((x+y-1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                   (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(y**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(exponential(x, rate), (x, 0, oo), meijerg=True)',
    'integrate(x*exponential(x, rate), (x, 0, oo), meijerg=True)',
    'integrate(x**2*exponential(x, rate), (x, 0, oo), meijerg=True)',
    'E(1)',
    'E(x*y)',
    'E(x*y**2)',
    'E((x+y+1)**2)',
    'E(x+y+1)',
    'E((x+y-1)**2)',
    'integrate(betadist, (x, 0, oo), meijerg=True)',
    'integrate(x*betadist, (x, 0, oo), meijerg=True)',
    'integrate(x**2*betadist, (x, 0, oo), meijerg=True)',
    'integrate(chi, (x, 0, oo), meijerg=True)',
    'integrate(x*chi, (x, 0, oo), meijerg=True)',
    'integrate(x**2*chi, (x, 0, oo), meijerg=True)',
    'integrate(chisquared, (x, 0, oo), meijerg=True)',
    'integrate(x*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(x**2*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(((x-k)/sqrt(2*k))**3*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(dagum, (x, 0, oo), meijerg=True)',
    'integrate(x*dagum, (x, 0, oo), meijerg=True)',
    'integrate(x**2*dagum, (x, 0, oo), meijerg=True)',
    'integrate(f, (x, 0, oo), meijerg=True)',
    'integrate(x*f, (x, 0, oo), meijerg=True)',
    'integrate(x**2*f, (x, 0, oo), meijerg=True)',
    'integrate(rice, (x, 0, oo), meijerg=True)',
    'integrate(laplace, (x, -oo, oo), meijerg=True)',
    'integrate(x*laplace, (x, -oo, oo), meijerg=True)',
    'integrate(x**2*laplace, (x, -oo, oo), meijerg=True)',
    'integrate(log(x) * x**(k-1) * exp(-x) / gamma(k), (x, 0, oo))',

    'integrate(sin(z*x)*(x**2-1)**(-(y+S(1)/2)), (x, 1, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*exp(-x**2), (x, 0, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*besselk(0,x), (x, 0, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*exp(-x**2), (x, 0, oo), meijerg=True)',
    'integrate(besselj(a,x)*besselj(b,x)/x, (x,0,oo), meijerg=True)',

    'hyperexpand(meijerg((-s - a/2 + 1, -s + a/2 + 1), (-a/2 - S(1)/2, -s + a/2 + S(3)/2), (a/2, -a/2), (-a/2 - S(1)/2, -s + a/2 + S(3)/2), 1))',
    "gammasimp(S('2**(2*s)*(-pi*gamma(-a + 1)*gamma(a + 1)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 3/2)*gamma(a + s + 1)/(a*(a + s)) - gamma(-a - 1/2)*gamma(-a + 1)*gamma(a + 1)*gamma(a + 3/2)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a + s + 1)*gamma(a - s + 1)/(a*(-a + s)))*gamma(-2*s + 1)*gamma(s + 1)/(pi*s*gamma(-a - 1/2)*gamma(a + 3/2)*gamma(-s + 1)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 1)*gamma(a - s + 3/2))'))",

    'mellin_transform(E1(x), x, s)',
    'inverse_mellin_transform(gamma(s)/s, s, x, (0, oo))',
    'mellin_transform(expint(a, x), x, s)',
    'mellin_transform(Si(x), x, s)',
    'inverse_mellin_transform(-2**s*sqrt(pi)*gamma((s + 1)/2)/(2*s*gamma(-s/2 + 1)), s, x, (-1, 0))',
    'mellin_transform(Ci(sqrt(x)), x, s)',
    'inverse_mellin_transform(-4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S(1)/2)),s, u, (0, 1))',
    'laplace_transform(Ci(x), x, s)',
    'laplace_transform(expint(a, x), x, s)',
    'laplace_transform(expint(1, x), x, s)',
    'laplace_transform(expint(2, x), x, s)',
    'inverse_laplace_transform(-log(1 + s**2)/2/s, s, u)',
    'inverse_laplace_transform(log(s + 1)/s, s, x)',
    'inverse_laplace_transform((s - log(s + 1))/s**2, s, x)',
    'laplace_transform(Chi(x), x, s)',
    'laplace_transform(Shi(x), x, s)',

    'integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True, conds="none")',
    'integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True, conds="none")',
    'integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True,conds="none")',
    'integrate(-cos(x)/x, (x, tpos, oo), meijerg=True)',
    'integrate(-sin(x)/x, (x, tpos, oo), meijerg=True)',
    'integrate(sin(x)/x, (x, 0, z), meijerg=True)',
    'integrate(sinh(x)/x, (x, 0, z), meijerg=True)',
    'integrate(exp(-x)/x, x, meijerg=True)',
    'integrate(exp(-x)/x**2, x, meijerg=True)',
    'integrate(cos(u)/u, u, meijerg=True)',
    'integrate(cosh(u)/u, u, meijerg=True)',
    'integrate(expint(1, x), x, meijerg=True)',
    'integrate(expint(2, x), x, meijerg=True)',
    'integrate(Si(x), x, meijerg=True)',
    'integrate(Ci(u), u, meijerg=True)',
    'integrate(Shi(x), x, meijerg=True)',
    'integrate(Chi(u), u, meijerg=True)',
    'integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True)',
    'integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True)'
]

from time import time
from sympy.core.cache import clear_cache
import sys

timings = []

if __name__ == '__main__':
    for n, string in enumerate(bench):
        clear_cache()
        _t = time()
        exec(string)
        _t = time() - _t
        timings += [(_t, string)]
        sys.stdout.write('.')
        sys.stdout.flush()
        if n % (len(bench) // 10) == 0:
            sys.stdout.write('%s' % (10*n // len(bench)))
    print()

    timings.sort(key=lambda x: -x[0])

    for ti, string in timings:
        print('%.2fs %s' % (ti, string))
