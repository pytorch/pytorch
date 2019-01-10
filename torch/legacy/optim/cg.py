import math

INFINITY = float('inf')


def sqrt_nothrow(x):
    return math.sqrt(x) if x >= 0 else float('nan')


def cg(opfunc, x, config, state=None):
    """

    This cg implementation is a rewrite of minimize.m written by Carl
    E. Rasmussen. It is supposed to produce exactly same results (give
    or take numerical accuracy due to some changed order of
    operations). You can compare the result on rosenbrock with minimize.m.
    http://www.gatsby.ucl.ac.uk/~edward/code/minimize/example.html

        [x fx c] = minimize([0 0]', 'rosenbrock', -25)

    Note that we limit the number of function evaluations only, it seems much
    more important in practical use.

    ARGS:

    - `opfunc` : a function that takes a single input, the point of evaluation.
    - `x`      : the initial point
    - `state` : a table of parameters and temporary allocations.
    - `state['maxEval']`     : max number of function evaluations
    - `state['maxIter']`     : max number of iterations
    - `state['df0']` : if you pass torch.Tensor they will be used for temp storage
    - `state['df1']` : if you pass torch.Tensor they will be used for temp storage
    - `state['df2']` : if you pass torch.Tensor they will be used for temp storage
    - `state['df3']` : if you pass torch.Tensor they will be used for temp storage
    - `state['s']`   : if you pass torch.Tensor they will be used for temp storage
    - `state['x0']`  : if you pass torch.Tensor they will be used for temp storage

    RETURN:
    - `x*` : the new x vector, at the optimal point
    - `f`  : a table of all function values where
        `f[1]` is the value of the function before any optimization and
        `f[#f]` is the final fully optimized value, at x*

    (Koray Kavukcuoglu, 2012)
    """
    # parameters
    if config is None and state is None:
        raise ValueError("cg requires a dictionary to retain state between iterations")
    state = state if state is not None else config
    rho = config.get('rho', 0.01)
    sig = config.get('sig', 0.5)
    _int = config.get('int', 0.1)
    ext = config.get('ext', 3.0)
    maxIter = config.get('maxIter', 20)
    ratio = config.get('ratio', 100)
    maxEval = config.get('maxEval', maxIter * 1.25)
    red = 1

    i = 0
    ls_failed = 0
    fx = []

    # we need three points for the interpolation/extrapolation stuff
    z1, z2, z3 = 0, 0, 0
    d1, d2, d3 = 0, 0, 0
    f1, f2, f3 = 0, 0, 0

    df1 = state.get('df1', x.new())
    df2 = state.get('df2', x.new())
    df3 = state.get('df3', x.new())

    df1.resize_as_(x)
    df2.resize_as_(x)
    df3.resize_as_(x)

    # search direction
    s = state.get('s', x.new())
    s.resize_as_(x)

    # we need a temp storage for X
    x0 = state.get('x0', x.new())
    f0 = 0
    df0 = state.get('df0', x.new())
    x0.resize_as_(x)
    df0.resize_as_(x)

    # evaluate at initial point
    f1, tdf = opfunc(x)
    fx.append(f1)
    df1.copy_(tdf)
    i = i + 1

    # initial search direction
    s.copy_(df1).mul_(-1)

    d1 = -s.dot(s)         # slope
    z1 = red / (1 - d1)         # initial step

    while i < abs(maxEval):
        x0.copy_(x)
        f0 = f1
        df0.copy_(df1)

        x.add_(z1, s)
        f2, tdf = opfunc(x)
        df2.copy_(tdf)
        i = i + 1
        d2 = df2.dot(s)
        f3, d3, z3 = f1, d1, -z1   # init point 3 equal to point 1
        m = min(maxIter, maxEval - i)
        success = 0
        limit = -1

        while True:
            while (f2 > f1 + z1 * rho * d1 or d2 > -sig * d1) and m > 0:
                limit = z1
                if f2 > f1:
                    z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)
                else:
                    A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
                    z2 = (sqrt_nothrow(B * B - A * d2 * z3 * z3) - B) / A

                if z2 != z2 or z2 == INFINITY or z2 == -INFINITY:
                    z2 = z3 / 2

                z2 = max(min(z2, _int * z3), (1 - _int) * z3)
                z1 = z1 + z2
                x.add_(z2, s)
                f2, tdf = opfunc(x)
                df2.copy_(tdf)
                i = i + 1
                m = m - 1
                d2 = df2.dot(s)
                z3 = z3 - z2

            if f2 > f1 + z1 * rho * d1 or d2 > -sig * d1:
                break
            elif d2 > sig * d1:
                success = 1
                break
            elif m == 0:
                break

            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
            B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
            _denom = (B + sqrt_nothrow(B * B - A * d2 * z3 * z3))
            z2 = -d2 * z3 * z3 / _denom if _denom != 0 else float('nan')

            if z2 != z2 or z2 == INFINITY or z2 == -INFINITY or z2 < 0:
                if limit < -0.5:
                    z2 = z1 * (ext - 1)
                else:
                    z2 = (limit - z1) / 2
            elif (limit > -0.5) and (z2 + z1) > limit:
                z2 = (limit - z1) / 2
            elif limit < -0.5 and (z2 + z1) > z1 * ext:
                z2 = z1 * (ext - 1)
            elif z2 < -z3 * _int:
                z2 = -z3 * _int
            elif limit > -0.5 and z2 < (limit - z1) * (1 - _int):
                z2 = (limit - z1) * (1 - _int)

            f3 = f2
            d3 = d2
            z3 = -z2
            z1 = z1 + z2
            x.add_(z2, s)

            f2, tdf = opfunc(x)
            df2.copy_(tdf)
            i = i + 1
            m = m - 1
            d2 = df2.dot(s)

        if success == 1:
            f1 = f2
            fx.append(f1)
            ss = (df2.dot(df2) - df2.dot(df1)) / df1.dot(df1)
            s.mul_(ss)
            s.add_(-1, df2)
            tmp = df1.clone()
            df1.copy_(df2)
            df2.copy_(tmp)
            d2 = df1.dot(s)
            if d2 > 0:
                s.copy_(df1)
                s.mul_(-1)
                d2 = -s.dot(s)

            z1 = z1 * min(ratio, d1 / (d2 - 1e-320))
            d1 = d2
            ls_failed = 0
        else:
            x.copy_(x0)
            f1 = f0
            df1.copy_(df0)
            if ls_failed or i > maxEval:
                break

            tmp = df1.clone()
            df1.copy_(df2)
            df2.copy_(tmp)
            s.copy_(df1)
            s.mul_(-1)
            d1 = -s.dot(s)
            z1 = 1 / (1 - d1)
            ls_failed = 1

    state['df0'] = df0
    state['df1'] = df1
    state['df2'] = df2
    state['df3'] = df3
    state['x0'] = x0
    state['s'] = s
    return x, fx, i
