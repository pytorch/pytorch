from collections import defaultdict

from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int


def digits(n, b=10, digits=None):
    """
    Return a list of the digits of ``n`` in base ``b``. The first
    element in the list is ``b`` (or ``-b`` if ``n`` is negative).

    Examples
    ========

    >>> from sympy.ntheory.digits import digits
    >>> digits(35)
    [10, 3, 5]

    If the number is negative, the negative sign will be placed on the
    base (which is the first element in the returned list):

    >>> digits(-35)
    [-10, 3, 5]

    Bases other than 10 (and greater than 1) can be selected with ``b``:

    >>> digits(27, b=2)
    [2, 1, 1, 0, 1, 1]

    Use the ``digits`` keyword if a certain number of digits is desired:

    >>> digits(35, digits=4)
    [10, 0, 0, 3, 5]

    Parameters
    ==========

    n: integer
        The number whose digits are returned.

    b: integer
        The base in which digits are computed.

    digits: integer (or None for all digits)
        The number of digits to be returned (padded with zeros, if
        necessary).

    See Also
    ========
    sympy.core.intfunc.num_digits, count_digits
    """

    b = as_int(b)
    n = as_int(n)
    if b < 2:
        raise ValueError("b must be greater than 1")
    else:
        x, y = abs(n), []
        while x >= b:
            x, r = divmod(x, b)
            y.append(r)
        y.append(x)
        y.append(-b if n < 0 else b)
        y.reverse()
        ndig = len(y) - 1
        if digits is not None:
            if ndig > digits:
                raise ValueError(
                    "For %s, at least %s digits are needed." % (n, ndig))
            elif ndig < digits:
                y[1:1] = [0]*(digits - ndig)
        return y


def count_digits(n, b=10):
    """
    Return a dictionary whose keys are the digits of ``n`` in the
    given base, ``b``, with keys indicating the digits appearing in the
    number and values indicating how many times that digit appeared.

    Examples
    ========

    >>> from sympy.ntheory import count_digits

    >>> count_digits(1111339)
    {1: 4, 3: 2, 9: 1}

    The digits returned are always represented in base-10
    but the number itself can be entered in any format that is
    understood by Python; the base of the number can also be
    given if it is different than 10:

    >>> n = 0xFA; n
    250
    >>> count_digits(_)
    {0: 1, 2: 1, 5: 1}
    >>> count_digits(n, 16)
    {10: 1, 15: 1}

    The default dictionary will return a 0 for any digit that did
    not appear in the number. For example, which digits appear 7
    times in ``77!``:

    >>> from sympy import factorial
    >>> c77 = count_digits(factorial(77))
    >>> [i for i in range(10) if c77[i] == 7]
    [1, 3, 7, 9]

    See Also
    ========
    sympy.core.intfunc.num_digits, digits
    """
    rv = defaultdict(int, multiset(digits(n, b)).items())
    rv.pop(b) if b in rv else rv.pop(-b)  # b or -b is there
    return rv


def is_palindromic(n, b=10):
    """return True if ``n`` is the same when read from left to right
    or right to left in the given base, ``b``.

    Examples
    ========

    >>> from sympy.ntheory import is_palindromic

    >>> all(is_palindromic(i) for i in (-11, 1, 22, 121))
    True

    The second argument allows you to test numbers in other
    bases. For example, 88 is palindromic in base-10 but not
    in base-8:

    >>> is_palindromic(88, 8)
    False

    On the other hand, a number can be palindromic in base-8 but
    not in base-10:

    >>> 0o121, is_palindromic(0o121)
    (81, False)

    Or it might be palindromic in both bases:

    >>> oct(121), is_palindromic(121, 8) and is_palindromic(121)
    ('0o171', True)

    """
    return _palindromic(digits(n, b), 1)
