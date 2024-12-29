def finite_diff(expression, variable, increment=1):
    """
    Takes as input a polynomial expression and the variable used to construct
    it and returns the difference between function's value when the input is
    incremented to 1 and the original function value. If you want an increment
    other than one supply it as a third argument.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.series.kauers import finite_diff
    >>> finite_diff(x**2, x)
    2*x + 1
    >>> finite_diff(y**3 + 2*y**2 + 3*y + 4, y)
    3*y**2 + 7*y + 6
    >>> finite_diff(x**2 + 3*x + 8, x, 2)
    4*x + 10
    >>> finite_diff(z**3 + 8*z, z, 3)
    9*z**2 + 27*z + 51
    """
    expression = expression.expand()
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    return expression2 - expression

def finite_diff_kauers(sum):
    """
    Takes as input a Sum instance and returns the difference between the sum
    with the upper index incremented by 1 and the original sum. For example,
    if S(n) is a sum, then finite_diff_kauers will return S(n + 1) - S(n).

    Examples
    ========

    >>> from sympy.series.kauers import finite_diff_kauers
    >>> from sympy import Sum
    >>> from sympy.abc import x, y, m, n, k
    >>> finite_diff_kauers(Sum(k, (k, 1, n)))
    n + 1
    >>> finite_diff_kauers(Sum(1/k, (k, 1, n)))
    1/(n + 1)
    >>> finite_diff_kauers(Sum((x*y**2), (x, 1, n), (y, 1, m)))
    (m + 1)**2*(n + 1)
    >>> finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n)))
    (m + 1)*(n + 1)
    """
    function = sum.function
    for l in sum.limits:
        function = function.subs(l[0], l[- 1] + 1)
    return function
