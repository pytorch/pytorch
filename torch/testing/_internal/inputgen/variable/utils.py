import math
import struct
import sys


def nextup(x):
    """Return the next floating-point value after x towards infinity."""

    # Check the Python version
    if sys.version_info >= (3, 9):
        # Use the built-in math.nextafter for Python 3.9 and later
        return math.nextafter(x, math.inf)

    if math.isnan(x) or (math.isinf(x) and x > 0):
        return x
    x = 0.0 if x == -0.0 else x
    n = struct.unpack("<q", struct.pack("<d", x))[0]
    n = n + 1 if n >= 0 else n - 1
    return struct.unpack("<d", struct.pack("<q", n))[0]


def nextdown(x):
    """Return the next floating-point value after x towards -infinity."""
    return -nextup(-x)
