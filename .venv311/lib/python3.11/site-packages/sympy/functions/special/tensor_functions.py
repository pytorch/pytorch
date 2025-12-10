from math import prod

from sympy.core import S, Integer
from sympy.core.function import DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups

###############################################################################
###################### Kronecker Delta, Levi-Civita etc. ######################
###############################################################################


def Eijk(*args, **kwargs):
    """
    Represent the Levi-Civita symbol.

    This is a compatibility wrapper to ``LeviCivita()``.

    See Also
    ========

    LeviCivita

    """
    return LeviCivita(*args, **kwargs)


def eval_levicivita(*args):
    """Evaluate Levi-Civita symbol."""
    n = len(args)
    return prod(
        prod(args[j] - args[i] for j in range(i + 1, n))
        / factorial(i) for i in range(n))
    # converting factorial(i) to int is slightly faster


class LeviCivita(DefinedFunction):
    """
    Represent the Levi-Civita symbol.

    Explanation
    ===========

    For even permutations of indices it returns 1, for odd permutations -1, and
    for everything else (a repeated index) it returns 0.

    Thus it represents an alternating pseudotensor.

    Examples
    ========

    >>> from sympy import LeviCivita
    >>> from sympy.abc import i, j, k
    >>> LeviCivita(1, 2, 3)
    1
    >>> LeviCivita(1, 3, 2)
    -1
    >>> LeviCivita(1, 2, 2)
    0
    >>> LeviCivita(i, j, k)
    LeviCivita(i, j, k)
    >>> LeviCivita(i, j, i)
    0

    See Also
    ========

    Eijk

    """

    is_integer = True

    @classmethod
    def eval(cls, *args):
        if all(isinstance(a, (SYMPY_INTS, Integer)) for a in args):
            return eval_levicivita(*args)
        if has_dups(args):
            return S.Zero

    def doit(self, **hints):
        return eval_levicivita(*self.args)


class KroneckerDelta(DefinedFunction):
    """
    The discrete, or Kronecker, delta function.

    Explanation
    ===========

    A function that takes in two integers $i$ and $j$. It returns $0$ if $i$
    and $j$ are not equal, or it returns $1$ if $i$ and $j$ are equal.

    Examples
    ========

    An example with integer indices:

        >>> from sympy import KroneckerDelta
        >>> KroneckerDelta(1, 2)
        0
        >>> KroneckerDelta(3, 3)
        1

    Symbolic indices:

        >>> from sympy.abc import i, j, k
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

    Parameters
    ==========

    i : Number, Symbol
        The first index of the delta function.
    j : Number, Symbol
        The second index of the delta function.

    See Also
    ========

    eval
    DiracDelta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta

    """

    is_integer = True

    @classmethod
    def eval(cls, i, j, delta_range=None):
        """
        Evaluates the discrete delta function.

        Examples
        ========

        >>> from sympy import KroneckerDelta
        >>> from sympy.abc import i, j, k

        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

        # indirect doctest

        """

        if delta_range is not None:
            dinf, dsup = delta_range
            if (dinf - i > 0) == True:
                return S.Zero
            if (dinf - j > 0) == True:
                return S.Zero
            if (dsup - i < 0) == True:
                return S.Zero
            if (dsup - j < 0) == True:
                return S.Zero

        diff = i - j
        if diff.is_zero:
            return S.One
        elif fuzzy_not(diff.is_zero):
            return S.Zero

        if i.assumptions0.get("below_fermi") and \
                j.assumptions0.get("above_fermi"):
            return S.Zero
        if j.assumptions0.get("below_fermi") and \
                i.assumptions0.get("above_fermi"):
            return S.Zero
        # to make KroneckerDelta canonical
        # following lines will check if inputs are in order
        # if not, will return KroneckerDelta with correct order
        if default_sort_key(j) < default_sort_key(i):
            if delta_range:
                return cls(j, i, delta_range)
            else:
                return cls(j, i)

    @property
    def delta_range(self):
        if len(self.args) > 2:
            return self.args[2]

    def _eval_power(self, expt):
        if expt.is_positive:
            return self
        if expt.is_negative and expt is not S.NegativeOne:
            return 1/self

    @property
    def is_above_fermi(self):
        """
        True if Delta can be non-zero above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_above_fermi
        True
        >>> KroneckerDelta(p, i).is_above_fermi
        False
        >>> KroneckerDelta(p, q).is_above_fermi
        True

        See Also
        ========

        is_below_fermi, is_only_below_fermi, is_only_above_fermi

        """
        if self.args[0].assumptions0.get("below_fermi"):
            return False
        if self.args[1].assumptions0.get("below_fermi"):
            return False
        return True

    @property
    def is_below_fermi(self):
        """
        True if Delta can be non-zero below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_below_fermi
        False
        >>> KroneckerDelta(p, i).is_below_fermi
        True
        >>> KroneckerDelta(p, q).is_below_fermi
        True

        See Also
        ========

        is_above_fermi, is_only_above_fermi, is_only_below_fermi

        """
        if self.args[0].assumptions0.get("above_fermi"):
            return False
        if self.args[1].assumptions0.get("above_fermi"):
            return False
        return True

    @property
    def is_only_above_fermi(self):
        """
        True if Delta is restricted to above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_only_above_fermi
        True
        >>> KroneckerDelta(p, q).is_only_above_fermi
        False
        >>> KroneckerDelta(p, i).is_only_above_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_below_fermi

        """
        return ( self.args[0].assumptions0.get("above_fermi")
                or
                self.args[1].assumptions0.get("above_fermi")
                ) or False

    @property
    def is_only_below_fermi(self):
        """
        True if Delta is restricted to below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, i).is_only_below_fermi
        True
        >>> KroneckerDelta(p, q).is_only_below_fermi
        False
        >>> KroneckerDelta(p, a).is_only_below_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_above_fermi

        """
        return ( self.args[0].assumptions0.get("below_fermi")
                or
                self.args[1].assumptions0.get("below_fermi")
                ) or False

    @property
    def indices_contain_equal_information(self):
        """
        Returns True if indices are either both above or below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, q).indices_contain_equal_information
        True
        >>> KroneckerDelta(p, q+1).indices_contain_equal_information
        True
        >>> KroneckerDelta(i, p).indices_contain_equal_information
        False

        """
        if (self.args[0].assumptions0.get("below_fermi") and
                self.args[1].assumptions0.get("below_fermi")):
            return True
        if (self.args[0].assumptions0.get("above_fermi")
                and self.args[1].assumptions0.get("above_fermi")):
            return True

        # if both indices are general we are True, else false
        return self.is_below_fermi and self.is_above_fermi

    @property
    def preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        Explanation
        ===========

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, 'a' is preferred before
        'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).preferred_index
        i
        >>> KroneckerDelta(p, a).preferred_index
        a
        >>> KroneckerDelta(i, j).preferred_index
        i

        See Also
        ========

        killable_index

        """
        if self._get_preferred_index():
            return self.args[1]
        else:
            return self.args[0]

    @property
    def killable_index(self):
        """
        Returns the index which is preferred to substitute in the final
        expression.

        Explanation
        ===========

        The index to substitute is the index with less information regarding
        fermi level. If indices contain the same information, 'a' is preferred
        before 'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).killable_index
        p
        >>> KroneckerDelta(p, a).killable_index
        p
        >>> KroneckerDelta(i, j).killable_index
        j

        See Also
        ========

        preferred_index

        """
        if self._get_preferred_index():
            return self.args[0]
        else:
            return self.args[1]

    def _get_preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, index 0 is returned.

        """
        if not self.is_above_fermi:
            if self.args[0].assumptions0.get("below_fermi"):
                return 0
            else:
                return 1
        elif not self.is_below_fermi:
            if self.args[0].assumptions0.get("above_fermi"):
                return 0
            else:
                return 1
        else:
            return 0

    @property
    def indices(self):
        return self.args[0:2]

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        i, j = args
        return Piecewise((0, Ne(i, j)), (1, True))
