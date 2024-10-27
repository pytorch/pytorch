""" This module contains the Mathieu functions.
"""

from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos


class MathieuBase(Function):
    """
    Abstract base class for Mathieu functions.

    This class is meant to reduce code duplication.

    """

    unbranched = True

    def _eval_conjugate(self):
        a, q, z = self.args
        return self.func(a.conjugate(), q.conjugate(), z.conjugate())


class mathieus(MathieuBase):
    r"""
    The Mathieu Sine function $S(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieus
    >>> from sympy.abc import a, q, z

    >>> mathieus(a, q, z)
    mathieus(a, q, z)

    >>> mathieus(a, 0, z)
    sin(sqrt(a)*z)

    >>> diff(mathieus(a, q, z), z)
    mathieusprime(a, q, z)

    See Also
    ========

    mathieuc: Mathieu cosine function.
    mathieusprime: Derivative of Mathieu sine function.
    mathieucprime: Derivative of Mathieu cosine function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuS/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieusprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)


class mathieuc(MathieuBase):
    r"""
    The Mathieu Cosine function $C(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieuc
    >>> from sympy.abc import a, q, z

    >>> mathieuc(a, q, z)
    mathieuc(a, q, z)

    >>> mathieuc(a, 0, z)
    cos(sqrt(a)*z)

    >>> diff(mathieuc(a, q, z), z)
    mathieucprime(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieusprime: Derivative of Mathieu sine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuC/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieusprime(MathieuBase):
    r"""
    The derivative $S^{\prime}(a,q,z)$ of the Mathieu Sine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieusprime
    >>> from sympy.abc import a, q, z

    >>> mathieusprime(a, q, z)
    mathieusprime(a, q, z)

    >>> mathieusprime(a, 0, z)
    sqrt(a)*cos(sqrt(a)*z)

    >>> diff(mathieusprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieus(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuSPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieus(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sqrt(a)*cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieucprime(MathieuBase):
    r"""
    The derivative $C^{\prime}(a,q,z)$ of the Mathieu Cosine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieucprime
    >>> from sympy.abc import a, q, z

    >>> mathieucprime(a, q, z)
    mathieucprime(a, q, z)

    >>> mathieucprime(a, 0, z)
    -sqrt(a)*sin(sqrt(a)*z)

    >>> diff(mathieucprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuCPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieuc(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return -sqrt(a)*sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)
