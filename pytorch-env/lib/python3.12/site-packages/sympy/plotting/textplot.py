from sympy.core.numbers import Float
from sympy.core.symbol import Dummy
from sympy.utilities.lambdify import lambdify

import math


def is_valid(x):
    """Check if a floating point number is valid"""
    if x is None:
        return False
    if isinstance(x, complex):
        return False
    return not math.isinf(x) and not math.isnan(x)


def rescale(y, W, H, mi, ma):
    """Rescale the given array `y` to fit into the integer values
    between `0` and `H-1` for the values between ``mi`` and ``ma``.
    """
    y_new = []

    norm = ma - mi
    offset = (ma + mi) / 2

    for x in range(W):
        if is_valid(y[x]):
            normalized = (y[x] - offset) / norm
            if not is_valid(normalized):
                y_new.append(None)
            else:
                rescaled = Float((normalized*H + H/2) * (H-1)/H).round()
                rescaled = int(rescaled)
                y_new.append(rescaled)
        else:
            y_new.append(None)
    return y_new


def linspace(start, stop, num):
    return [start + (stop - start) * x / (num-1) for x in range(num)]


def textplot_str(expr, a, b, W=55, H=21):
    """Generator for the lines of the plot"""
    free = expr.free_symbols
    if len(free) > 1:
        raise ValueError(
            "The expression must have a single variable. (Got {})"
            .format(free))
    x = free.pop() if free else Dummy()
    f = lambdify([x], expr)
    if isinstance(a, complex):
        if a.imag == 0:
            a = a.real
    if isinstance(b, complex):
        if b.imag == 0:
            b = b.real
    a = float(a)
    b = float(b)

    # Calculate function values
    x = linspace(a, b, W)
    y = []
    for val in x:
        try:
            y.append(f(val))
        # Not sure what exceptions to catch here or why...
        except (ValueError, TypeError, ZeroDivisionError):
            y.append(None)

    # Normalize height to screen space
    y_valid = list(filter(is_valid, y))
    if y_valid:
        ma = max(y_valid)
        mi = min(y_valid)
        if ma == mi:
            if ma:
                mi, ma = sorted([0, 2*ma])
            else:
                mi, ma = -1, 1
    else:
        mi, ma = -1, 1
    y_range = ma - mi
    precision = math.floor(math.log10(y_range)) - 1
    precision *= -1
    mi = round(mi, precision)
    ma = round(ma, precision)
    y = rescale(y, W, H, mi, ma)

    y_bins = linspace(mi, ma, H)

    # Draw plot
    margin = 7
    for h in range(H - 1, -1, -1):
        s = [' '] * W
        for i in range(W):
            if y[i] == h:
                if (i == 0 or y[i - 1] == h - 1) and (i == W - 1 or y[i + 1] == h + 1):
                    s[i] = '/'
                elif (i == 0 or y[i - 1] == h + 1) and (i == W - 1 or y[i + 1] == h - 1):
                    s[i] = '\\'
                else:
                    s[i] = '.'

        if h == 0:
            for i in range(W):
                s[i] = '_'

        # Print y values
        if h in (0, H//2, H - 1):
            prefix = ("%g" % y_bins[h]).rjust(margin)[:margin]
        else:
            prefix = " "*margin
        s = "".join(s)
        if h == H//2:
            s = s.replace(" ", "-")
        yield prefix + " |" + s

    # Print x values
    bottom = " " * (margin + 2)
    bottom += ("%g" % x[0]).ljust(W//2)
    if W % 2 == 1:
        bottom += ("%g" % x[W//2]).ljust(W//2)
    else:
        bottom += ("%g" % x[W//2]).ljust(W//2-1)
    bottom += "%g" % x[-1]
    yield bottom


def textplot(expr, a, b, W=55, H=21):
    r"""
    Print a crude ASCII art plot of the SymPy expression 'expr' (which
    should contain a single symbol, e.g. x or something else) over the
    interval [a, b].

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.plotting import textplot
    >>> t = Symbol('t')
    >>> textplot(sin(t)*t, 0, 15)
     14 |                                                  ...
        |                                                     .
        |                                                 .
        |                                                      .
        |                                                .
        |                            ...
        |                           /   .               .
        |                          /
        |                         /      .
        |                        .        .            .
    1.5 |----.......--------------------------------------------
        |....       \           .          .
        |            \         /                      .
        |             ..      /             .
        |               \    /                       .
        |                ....
        |                                    .
        |                                     .     .
        |
        |                                      .   .
    -11 |_______________________________________________________
         0                          7.5                        15
    """
    for line in textplot_str(expr, a, b, W, H):
        print(line)
