"""
The Schur number S(k) is the largest integer n for which the interval [1,n]
can be partitioned into k sum-free sets.(https://mathworld.wolfram.com/SchurNumber.html)
"""
import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer


class SchurNumber(Function):
    r"""
    This function creates a SchurNumber object
    which is evaluated for `k \le 5` otherwise only
    the lower bound information can be retrieved.

    Examples
    ========

    >>> from sympy.combinatorics.schur_number import SchurNumber

    Since S(3) = 13, hence the output is a number
    >>> SchurNumber(3)
    13

    We do not know the Schur number for values greater than 5, hence
    only the object is returned
    >>> SchurNumber(6)
    SchurNumber(6)

    Now, the lower bound information can be retrieved using lower_bound()
    method
    >>> SchurNumber(6).lower_bound()
    536

    """

    @classmethod
    def eval(cls, k):
        if k.is_Number:
            if k is S.Infinity:
                return S.Infinity
            if k.is_zero:
                return S.Zero
            if not k.is_integer or k.is_negative:
                raise ValueError("k should be a positive integer")
            first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
            if k <= 5:
                return Integer(first_known_schur_numbers[k])

    def lower_bound(self):
        f_ = self.args[0]
        # Improved lower bounds known for S(6) and S(7)
        if f_ == 6:
            return Integer(536)
        if f_ == 7:
            return Integer(1680)
        # For other cases, use general expression
        if f_.is_Integer:
            return 3*self.func(f_ - 1).lower_bound() - 1
        return (3**f_ - 1)/2


def _schur_subsets_number(n):

    if n is S.Infinity:
        raise ValueError("Input must be finite")
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    elif n <= 3:
        min_k = 1
    else:
        min_k = math.ceil(math.log(2*n + 1, 3))

    return Integer(min_k)


def schur_partition(n):
    """

    This function returns the partition in the minimum number of sum-free subsets
    according to the lower bound given by the Schur Number.

    Parameters
    ==========

    n: a number
        n is the upper limit of the range [1, n] for which we need to find and
        return the minimum number of free subsets according to the lower bound
        of schur number

    Returns
    =======

    List of lists
        List of the minimum number of sum-free subsets

    Notes
    =====

    It is possible for some n to make the partition into less
    subsets since the only known Schur numbers are:
    S(1) = 1, S(2) = 4, S(3) = 13, S(4) = 44.
    e.g for n = 44 the lower bound from the function above is 5 subsets but it has been proven
    that can be done with 4 subsets.

    Examples
    ========

    For n = 1, 2, 3 the answer is the set itself

    >>> from sympy.combinatorics.schur_number import schur_partition
    >>> schur_partition(2)
    [[1, 2]]

    For n > 3, the answer is the minimum number of sum-free subsets:

    >>> schur_partition(5)
    [[3, 2], [5], [1, 4]]

    >>> schur_partition(8)
    [[3, 2], [6, 5, 8], [1, 4, 7]]
    """

    if isinstance(n, Basic) and not n.is_Number:
        raise ValueError("Input value must be a number")

    number_of_subsets = _schur_subsets_number(n)
    if n == 1:
        sum_free_subsets = [[1]]
    elif n == 2:
        sum_free_subsets = [[1, 2]]
    elif n == 3:
        sum_free_subsets = [[1, 2, 3]]
    else:
        sum_free_subsets = [[1, 4], [2, 3]]

    while len(sum_free_subsets) < number_of_subsets:
        sum_free_subsets = _generate_next_list(sum_free_subsets, n)
        missed_elements = [3*k + 1 for k in range(len(sum_free_subsets), (n-1)//3 + 1)]
        sum_free_subsets[-1] += missed_elements

    return sum_free_subsets


def _generate_next_list(current_list, n):
    new_list = []

    for item in current_list:
        temp_1 = [number*3 for number in item if number*3 <= n]
        temp_2 = [number*3 - 1 for number in item if number*3 - 1 <= n]
        new_item = temp_1 + temp_2
        new_list.append(new_item)

    last_list = [3*k + 1 for k in range(len(current_list)+1) if 3*k + 1 <= n]
    new_list.append(last_list)
    current_list = new_list

    return current_list
