import sys
import math

def nextafter(x: float, y: float) -> float:
    """Return the next floating-point value after x towards y."""

    # Check the Python version
    if sys.version_info >= (3, 9):
        # Use the built-in math.nextafter for Python 3.9 and later
        return math.nextafter(x, y)
    else:
        # Compute nextafter for Python versions before 3.9
        if math.isnan(x) or math.isnan(y):
            # If either x or y is NaN, return NaN
            return math.nan

        if x == y:
            # If x is equal to y, return y
            return y

        # Set up the direction based on whether y is greater or smaller than x
        direction = 1 if y > x else -1

        # Handle the case when x is 0
        if x == 0:
            return math.copysign(math.ulp(0.0), direction)

        # Compute the next floating point value
        m, e = math.frexp(x)
        if x > 0 and direction > 0 or x < 0 and direction < 0:
            m += math.ldexp(1.0, -53)  # Increment the significand
        else:
            m -= math.ldexp(1.0, -53)  # Decrement the significand

        # Handle the case when the result overflows or underflows
        if not (math.ldexp(m, e) == x or math.ldexp(m, e) == y):
            m, e = math.frexp(y)

        return math.ldexp(m, e)
