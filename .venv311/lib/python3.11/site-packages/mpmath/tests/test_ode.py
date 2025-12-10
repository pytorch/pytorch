#from mpmath.calculus import ODE_step_euler, ODE_step_rk4, odeint, arange
from mpmath import odefun, cos, sin, mpf, sinc, mp

'''
solvers = [ODE_step_euler, ODE_step_rk4]

def test_ode1():
    """
    Let's solve:

    x'' + w**2 * x = 0

    i.e. x1 = x, x2 = x1':

    x1' =  x2
    x2' = -x1
    """
    def derivs((x1, x2), t):
        return x2, -x1

    for solver in solvers:
        t = arange(0, 3.1415926, 0.005)
        sol = odeint(derivs, (0., 1.), t, solver)
        x1 = [a[0] for a in sol]
        x2 = [a[1] for a in sol]
        # the result is x1 = sin(t), x2 = cos(t)
        # let's just check the end points for t = pi
        assert abs(x1[-1]) < 1e-2
        assert abs(x2[-1] - (-1)) < 1e-2

def test_ode2():
    """
    Let's solve:

    x' - x = 0

    i.e. x = exp(x)

    """
    def derivs((x), t):
        return x

    for solver in solvers:
        t = arange(0, 1, 1e-3)
        sol = odeint(derivs, (1.,), t, solver)
        x = [a[0] for a in sol]
        # the result is x = exp(t)
        # let's just check the end point for t = 1, i.e. x = e
        assert abs(x[-1] - 2.718281828) < 1e-2
'''

def test_odefun_rational():
    mp.dps = 15
    # A rational function
    f = lambda t: 1/(1+mpf(t)**2)
    g = odefun(lambda x, y: [-2*x*y[0]**2], 0, [f(0)])
    assert f(2).ae(g(2)[0])

def test_odefun_sinc_large():
    mp.dps = 15
    # Sinc function; test for large x
    f = sinc
    g = odefun(lambda x, y: [(cos(x)-y[0])/x], 1, [f(1)], tol=0.01, degree=5)
    assert abs(f(100) - g(100)[0])/f(100) < 0.01

def test_odefun_harmonic():
    mp.dps = 15
    # Harmonic oscillator
    f = odefun(lambda x, y: [-y[1], y[0]], 0, [1, 0])
    for x in [0, 1, 2.5, 8, 3.7]:    #  we go back to 3.7 to check caching
        c, s = f(x)
        assert c.ae(cos(x))
        assert s.ae(sin(x))
