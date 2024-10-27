"""
Plotting (requires matplotlib)
"""

from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange

class VisualizationMethods(object):
    plot_ignore = (ValueError, ArithmeticError, ZeroDivisionError, NoConvergence)

def plot(ctx, f, xlim=[-5,5], ylim=None, points=200, file=None, dpi=None,
    singularities=[], axes=None):
    r"""
    Shows a simple 2D plot of a function `f(x)` or list of functions
    `[f_0(x), f_1(x), \ldots, f_n(x)]` over a given interval
    specified by *xlim*. Some examples::

        plot(lambda x: exp(x)*li(x), [1, 4])
        plot([cos, sin], [-4, 4])
        plot([fresnels, fresnelc], [-4, 4])
        plot([sqrt, cbrt], [-4, 4])
        plot(lambda t: zeta(0.5+t*j), [-20, 20])
        plot([floor, ceil, abs, sign], [-5, 5])

    Points where the function raises a numerical exception or
    returns an infinite value are removed from the graph.
    Singularities can also be excluded explicitly
    as follows (useful for removing erroneous vertical lines)::

        plot(cot, ylim=[-5, 5])   # bad
        plot(cot, ylim=[-5, 5], singularities=[-pi, 0, pi])  # good

    For parts where the function assumes complex values, the
    real part is plotted with dashes and the imaginary part
    is plotted with dots.

    .. note :: This function requires matplotlib (pylab).
    """
    if file:
        axes = None
    fig = None
    if not axes:
        import pylab
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    if not isinstance(f, (tuple, list)):
        f = [f]
    a, b = xlim
    colors = ['b', 'r', 'g', 'm', 'k']
    for n, func in enumerate(f):
        x = ctx.arange(a, b, (b-a)/float(points))
        segments = []
        segment = []
        in_complex = False
        for i in xrange(len(x)):
            try:
                if i != 0:
                    for sing in singularities:
                        if x[i-1] <= sing and x[i] >= sing:
                            raise ValueError
                v = func(x[i])
                if ctx.isnan(v) or abs(v) > 1e300:
                    raise ValueError
                if hasattr(v, "imag") and v.imag:
                    re = float(v.real)
                    im = float(v.imag)
                    if not in_complex:
                        in_complex = True
                        segments.append(segment)
                        segment = []
                    segment.append((float(x[i]), re, im))
                else:
                    if in_complex:
                        in_complex = False
                        segments.append(segment)
                        segment = []
                    if hasattr(v, "real"):
                        v = v.real
                    segment.append((float(x[i]), v))
            except ctx.plot_ignore:
                if segment:
                    segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for segment in segments:
            x = [s[0] for s in segment]
            y = [s[1] for s in segment]
            if not x:
                continue
            c = colors[n % len(colors)]
            if len(segment[0]) == 3:
                z = [s[2] for s in segment]
                axes.plot(x, y, '--'+c, linewidth=3)
                axes.plot(x, z, ':'+c, linewidth=3)
            else:
                axes.plot(x, y, c, linewidth=3)
    axes.set_xlim([float(_) for _ in xlim])
    if ylim:
        axes.set_ylim([float(_) for _ in ylim])
    axes.set_xlabel('x')
    axes.set_ylabel('f(x)')
    axes.grid(True)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def default_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    a = (float(ctx.arg(z)) + ctx.pi) / (2*ctx.pi)
    a = (a + 0.5) % 1.0
    b = 1.0 - float(1/(1.0+abs(z)**0.3))
    return hls_to_rgb(a, b, 0.8)

blue_orange_colors = [
  (-1.0,  (0.0, 0.0, 0.0)),
  (-0.95, (0.1, 0.2, 0.5)),   # dark blue
  (-0.5,  (0.0, 0.5, 1.0)),   # blueish
  (-0.05, (0.4, 0.8, 0.8)),   # cyanish
  ( 0.0,  (1.0, 1.0, 1.0)),
  ( 0.05, (1.0, 0.9, 0.3)),   # yellowish
  ( 0.5,  (0.9, 0.5, 0.0)),   # orangeish
  ( 0.95, (0.7, 0.1, 0.0)),   # redish
  ( 1.0,  (0.0, 0.0, 0.0)),
  ( 2.0,  (0.0, 0.0, 0.0)),
]

def phase_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    w = float(ctx.arg(z)) / pi
    w = max(min(w, 1.0), -1.0)
    for i in range(1,len(blue_orange_colors)):
        if blue_orange_colors[i][0] > w:
            a, (ra, ga, ba) = blue_orange_colors[i-1]
            b, (rb, gb, bb) = blue_orange_colors[i]
            s = (w-a) / (b-a)
            return ra+(rb-ra)*s, ga+(gb-ga)*s, ba+(bb-ba)*s

def cplot(ctx, f, re=[-5,5], im=[-5,5], points=2000, color=None,
    verbose=False, file=None, dpi=None, axes=None):
    """
    Plots the given complex-valued function *f* over a rectangular part
    of the complex plane specified by the pairs of intervals *re* and *im*.
    For example::

        cplot(lambda z: z, [-2, 2], [-10, 10])
        cplot(exp)
        cplot(zeta, [0, 1], [0, 50])

    By default, the complex argument (phase) is shown as color (hue) and
    the magnitude is show as brightness. You can also supply a
    custom color function (*color*). This function should take a
    complex number as input and return an RGB 3-tuple containing
    floats in the range 0.0-1.0.

    Alternatively, you can select a builtin color function by passing
    a string as *color*:

      * "default" - default color scheme
      * "phase" - a color scheme that only renders the phase of the function,
         with white for positive reals, black for negative reals, gold in the
         upper half plane, and blue in the lower half plane.

    To obtain a sharp image, the number of points may need to be
    increased to 100,000 or thereabout. Since evaluating the
    function that many times is likely to be slow, the 'verbose'
    option is useful to display progress.

    .. note :: This function requires matplotlib (pylab).
    """
    if color is None or color == "default":
        color = ctx.default_color_function
    if color == "phase":
        color = ctx.phase_color_function
    import pylab
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    rea, reb = re
    ima, imb = im
    dre = reb - rea
    dim = imb - ima
    M = int(ctx.sqrt(points*dre/dim)+1)
    N = int(ctx.sqrt(points*dim/dre)+1)
    x = pylab.linspace(rea, reb, M)
    y = pylab.linspace(ima, imb, N)
    # Note: we have to be careful to get the right rotation.
    # Test with these plots:
    #   cplot(lambda z: z if z.real < 0 else 0)
    #   cplot(lambda z: z if z.imag < 0 else 0)
    w = pylab.zeros((N, M, 3))
    for n in xrange(N):
        for m in xrange(M):
            z = ctx.mpc(x[m], y[n])
            try:
                v = color(f(z))
            except ctx.plot_ignore:
                v = (0.5, 0.5, 0.5)
            w[n,m] = v
        if verbose:
            print(str(n) + ' of ' + str(N))
    rea, reb, ima, imb = [float(_) for _ in [rea, reb, ima, imb]]
    axes.imshow(w, extent=(rea, reb, ima, imb), origin='lower')
    axes.set_xlabel('Re(z)')
    axes.set_ylabel('Im(z)')
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def splot(ctx, f, u=[-5,5], v=[-5,5], points=100, keep_aspect=True, \
          wireframe=False, file=None, dpi=None, axes=None):
    """
    Plots the surface defined by `f`.

    If `f` returns a single component, then this plots the surface
    defined by `z = f(x,y)` over the rectangular domain with
    `x = u` and `y = v`.

    If `f` returns three components, then this plots the parametric
    surface `x, y, z = f(u,v)` over the pairs of intervals `u` and `v`.

    For example, to plot a simple function::

        >>> from mpmath import *
        >>> f = lambda x, y: sin(x+y)*cos(y)
        >>> splot(f, [-pi,pi], [-pi,pi])    # doctest: +SKIP

    Plotting a donut::

        >>> r, R = 1, 2.5
        >>> f = lambda u, v: [r*cos(u), (R+r*sin(u))*cos(v), (R+r*sin(u))*sin(v)]
        >>> splot(f, [0, 2*pi], [0, 2*pi])    # doctest: +SKIP

    .. note :: This function requires matplotlib (pylab) 0.98.5.3 or higher.
    """
    import pylab
    import mpl_toolkits.mplot3d as mplot3d
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = mplot3d.axes3d.Axes3D(fig)
    ua, ub = u
    va, vb = v
    du = ub - ua
    dv = vb - va
    if not isinstance(points, (list, tuple)):
        points = [points, points]
    M, N = points
    u = pylab.linspace(ua, ub, M)
    v = pylab.linspace(va, vb, N)
    x, y, z = [pylab.zeros((M, N)) for i in xrange(3)]
    xab, yab, zab = [[0, 0] for i in xrange(3)]
    for n in xrange(N):
        for m in xrange(M):
            fdata = f(ctx.convert(u[m]), ctx.convert(v[n]))
            try:
                x[m,n], y[m,n], z[m,n] = fdata
            except TypeError:
                x[m,n], y[m,n], z[m,n] = u[m], v[n], fdata
            for c, cab in [(x[m,n], xab), (y[m,n], yab), (z[m,n], zab)]:
                if c < cab[0]:
                    cab[0] = c
                if c > cab[1]:
                    cab[1] = c
    if wireframe:
        axes.plot_wireframe(x, y, z, rstride=4, cstride=4)
    else:
        axes.plot_surface(x, y, z, rstride=4, cstride=4)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    if keep_aspect:
        dx, dy, dz = [cab[1] - cab[0] for cab in [xab, yab, zab]]
        maxd = max(dx, dy, dz)
        if dx < maxd:
            delta = maxd - dx
            axes.set_xlim3d(xab[0] - delta / 2.0, xab[1] + delta / 2.0)
        if dy < maxd:
            delta = maxd - dy
            axes.set_ylim3d(yab[0] - delta / 2.0, yab[1] + delta / 2.0)
        if dz < maxd:
            delta = maxd - dz
            axes.set_zlim3d(zab[0] - delta / 2.0, zab[1] + delta / 2.0)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()


VisualizationMethods.plot = plot
VisualizationMethods.default_color_function = default_color_function
VisualizationMethods.phase_color_function = phase_color_function
VisualizationMethods.cplot = cplot
VisualizationMethods.splot = splot
