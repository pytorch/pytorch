from .functions import defun_wrapped

@defun_wrapped
def squarew(ctx, t, amplitude=1, period=1):
    P = period
    A = amplitude
    return A*((-1)**ctx.floor(2*t/P))

@defun_wrapped
def trianglew(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period

    return 2*A*(0.5 - ctx.fabs(1 - 2*ctx.frac(t/P + 0.25)))

@defun_wrapped
def sawtoothw(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period
    return A*ctx.frac(t/P)

@defun_wrapped
def unit_triangle(ctx, t, amplitude=1):
    A = amplitude
    if t <= -1 or t >= 1:
        return ctx.zero
    return A*(-ctx.fabs(t) + 1)

@defun_wrapped
def sigmoid(ctx, t, amplitude=1):
    A = amplitude
    return A / (1 + ctx.exp(-t))
