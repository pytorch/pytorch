from mpmath import *
from random import seed, randint, random
import math

# Test compatibility with Python floats, which are
# IEEE doubles (53-bit)

N = 5000
seed(1)

# Choosing exponents between roughly -140, 140 ensures that
# the Python floats don't overflow or underflow
xs = [(random()-1) * 10**randint(-140, 140) for x in range(N)]
ys = [(random()-1) * 10**randint(-140, 140) for x in range(N)]

# include some equal values
ys[int(N*0.8):] = xs[int(N*0.8):]

# Detect whether Python is compiled to use 80-bit floating-point
# instructions, in which case the double compatibility test breaks
uses_x87 = -4.1974624032366689e+117 / -8.4657370748010221e-47 \
    == 4.9581771393902231e+163

def test_double_compatibility():
    mp.prec = 53
    for x, y in zip(xs, ys):
        mpx = mpf(x)
        mpy = mpf(y)
        assert mpf(x) == x
        assert (mpx < mpy) == (x < y)
        assert (mpx > mpy) == (x > y)
        assert (mpx == mpy) == (x == y)
        assert (mpx != mpy) == (x != y)
        assert (mpx <= mpy) == (x <= y)
        assert (mpx >= mpy) == (x >= y)
        assert mpx == mpx
        if uses_x87:
            mp.prec = 64
            a = mpx + mpy
            b = mpx * mpy
            c = mpx / mpy
            d = mpx % mpy
            mp.prec = 53
            assert +a == x + y
            assert +b == x * y
            assert +c == x / y
            assert +d == x % y
        else:
            assert mpx + mpy == x + y
            assert mpx * mpy == x * y
            assert mpx / mpy == x / y
            assert mpx % mpy == x % y
        assert abs(mpx) == abs(x)
        assert mpf(repr(x)) == x
        assert ceil(mpx) == math.ceil(x)
        assert floor(mpx) == math.floor(x)

def test_sqrt():
    # this fails quite often. it appers to be float
    # that rounds the wrong way, not mpf
    fail = 0
    mp.prec = 53
    for x in xs:
        x = abs(x)
        mp.prec = 100
        mp_high = mpf(x)**0.5
        mp.prec = 53
        mp_low = mpf(x)**0.5
        fp = x**0.5
        assert abs(mp_low-mp_high) <= abs(fp-mp_high)
        fail += mp_low != fp
    assert fail < N/10

def test_bugs():
    # particular bugs
    assert mpf(4.4408920985006262E-16) < mpf(1.7763568394002505E-15)
    assert mpf(-4.4408920985006262E-16) > mpf(-1.7763568394002505E-15)
