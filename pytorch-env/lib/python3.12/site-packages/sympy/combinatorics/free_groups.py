from __future__ import annotations

from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int


@public
def free_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> F, x, y, z = free_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> x**2*y**-1
    x**2*y**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group,) + tuple(_free_group.generators)

@public
def xfree_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1)))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import xfree_group
    >>> F, (x, y, z) = xfree_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> y**2*x**-2*z**-1
    y**2*x**-2*z**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group, _free_group.generators)

@public
def vfree_group(symbols):
    """Construct a free group and inject ``f_0, f_1, ..., f_(n-1)`` as symbols
    into the global namespace.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import vfree_group
    >>> vfree_group("x, y, z")
    <free group on the generators (x, y, z)>
    >>> x**2*y**-2*z # noqa: F821
    x**2*y**-2*z
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    pollute([sym.name for sym in _free_group.symbols], _free_group.generators)
    return _free_group


def _parse_symbols(symbols):
    if not symbols:
        return ()
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True)
    elif isinstance(symbols, (Expr, FreeGroupElement)):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols
    raise ValueError("The type of `symbols` must be one of the following: "
                     "a str, Symbol/Expr or a sequence of "
                     "one of these types")


##############################################################################
#                          FREE GROUP                                        #
##############################################################################

_free_group_cache: dict[int, FreeGroup] = {}

class FreeGroup(DefaultPrinting):
    """
    Free group with finite or infinite number of generators. Its input API
    is that of a str, Symbol/Expr or a sequence of one of
    these types (which may be empty)

    See Also
    ========

    sympy.polys.rings.PolyRing

    References
    ==========

    .. [1] https://www.gap-system.org/Manuals/doc/ref/chap37.html

    .. [2] https://en.wikipedia.org/wiki/Free_group

    """
    is_associative = True
    is_group = True
    is_FreeGroup = True
    is_PermutationGroup = False
    relators: list[Expr] = []

    def __new__(cls, symbols):
        symbols = tuple(_parse_symbols(symbols))
        rank = len(symbols)
        _hash = hash((cls.__name__, symbols, rank))
        obj = _free_group_cache.get(_hash)

        if obj is None:
            obj = object.__new__(cls)
            obj._hash = _hash
            obj._rank = rank
            # dtype method is used to create new instances of FreeGroupElement
            obj.dtype = type("FreeGroupElement", (FreeGroupElement,), {"group": obj})
            obj.symbols = symbols
            obj.generators = obj._generators()
            obj._gens_set = set(obj.generators)
            for symbol, generator in zip(obj.symbols, obj.generators):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if hasattr(obj, name):
                        setattr(obj, name, generator)

            _free_group_cache[_hash] = obj

        return obj

    def _generators(group):
        """Returns the generators of the FreeGroup.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> F.generators
        (x, y, z)

        """
        gens = []
        for sym in group.symbols:
            elm = ((sym, 1),)
            gens.append(group.dtype(elm))
        return tuple(gens)

    def clone(self, symbols=None):
        return self.__class__(symbols or self.symbols)

    def __contains__(self, i):
        """Return True if ``i`` is contained in FreeGroup."""
        if not isinstance(i, FreeGroupElement):
            return False
        group = i.group
        return self == group

    def __hash__(self):
        return self._hash

    def __len__(self):
        return self.rank

    def __str__(self):
        if self.rank > 30:
            str_form = "<free group with %s generators>" % self.rank
        else:
            str_form = "<free group on the generators "
            gens = self.generators
            str_form += str(gens) + ">"
        return str_form

    __repr__ = __str__

    def __getitem__(self, index):
        symbols = self.symbols[index]
        return self.clone(symbols=symbols)

    def __eq__(self, other):
        """No ``FreeGroup`` is equal to any "other" ``FreeGroup``.
        """
        return self is other

    def index(self, gen):
        """Return the index of the generator `gen` from ``(f_0, ..., f_(n-1))``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.index(y)
        1
        >>> F.index(x)
        0

        """
        if isinstance(gen, self.dtype):
            return self.generators.index(gen)
        else:
            raise ValueError("expected a generator of Free Group %s, got %s" % (self, gen))

    def order(self):
        """Return the order of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.order()
        oo

        >>> free_group("")[0].order()
        1

        """
        if self.rank == 0:
            return S.One
        else:
            return S.Infinity

    @property
    def elements(self):
        """
        Return the elements of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> (z,) = free_group("")
        >>> z.elements
        {<identity>}

        """
        if self.rank == 0:
            # A set containing Identity element of `FreeGroup` self is returned
            return {self.identity}
        else:
            raise ValueError("Group contains infinitely many elements"
                            ", hence cannot be represented")

    @property
    def rank(self):
        r"""
        In group theory, the `rank` of a group `G`, denoted `G.rank`,
        can refer to the smallest cardinality of a generating set
        for G, that is

        \operatorname{rank}(G)=\min\{ |X|: X\subseteq G, \left\langle X\right\rangle =G\}.

        """
        return self._rank

    @property
    def is_abelian(self):
        """Returns if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.is_abelian
        False

        """
        return self.rank in (0, 1)

    @property
    def identity(self):
        """Returns the identity element of free group."""
        return self.dtype()

    def contains(self, g):
        """Tests if Free Group element ``g`` belong to self, ``G``.

        In mathematical terms any linear combination of generators
        of a Free Group is contained in it.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.contains(x**3*y**2)
        True

        """
        if not isinstance(g, FreeGroupElement):
            return False
        elif self != g.group:
            return False
        else:
            return True

    def center(self):
        """Returns the center of the free group `self`."""
        return {self.identity}


############################################################################
#                          FreeGroupElement                                #
############################################################################


class FreeGroupElement(CantSympify, DefaultPrinting, tuple):
    """Used to create elements of FreeGroup. It cannot be used directly to
    create a free group element. It is called by the `dtype` method of the
    `FreeGroup` class.

    """
    is_assoc_word = True

    def new(self, init):
        return self.__class__(init)

    _hash = None

    def __hash__(self):
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.group, frozenset(tuple(self))))
        return _hash

    def copy(self):
        return self.new(self)

    @property
    def is_identity(self):
        if self.array_form == ():
            return True
        else:
            return False

    @property
    def array_form(self):
        """
        SymPy provides two different internal kinds of representation
        of associative words. The first one is called the `array_form`
        which is a tuple containing `tuples` as its elements, where the
        size of each tuple is two. At the first position the tuple
        contains the `symbol-generator`, while at the second position
        of tuple contains the exponent of that generator at the position.
        Since elements (i.e. words) do not commute, the indexing of tuple
        makes that property to stay.

        The structure in ``array_form`` of ``FreeGroupElement`` is of form:

        ``( ( symbol_of_gen, exponent ), ( , ), ... ( , ) )``

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> (x*z).array_form
        ((x, 1), (z, 1))
        >>> (x**2*z*y*x**2).array_form
        ((x, 2), (z, 1), (y, 1), (x, 2))

        See Also
        ========

        letter_repr

        """
        return tuple(self)

    @property
    def letter_form(self):
        """
        The letter representation of a ``FreeGroupElement`` is a tuple
        of generator symbols, with each entry corresponding to a group
        generator. Inverses of the generators are represented by
        negative generator symbols.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b, c, d = free_group("a b c d")
        >>> (a**3).letter_form
        (a, a, a)
        >>> (a**2*d**-2*a*b**-4).letter_form
        (a, a, -d, -d, a, -b, -b, -b, -b)
        >>> (a**-2*b**3*d).letter_form
        (-a, -a, b, b, b, d)

        See Also
        ========

        array_form

        """
        return tuple(flatten([(i,)*j if j > 0 else (-i,)*(-j)
                    for i, j in self.array_form]))

    def __getitem__(self, i):
        group = self.group
        r = self.letter_form[i]
        if r.is_Symbol:
            return group.dtype(((r, 1),))
        else:
            return group.dtype(((-r, -1),))

    def index(self, gen):
        if len(gen) != 1:
            raise ValueError()
        return (self.letter_form).index(gen.letter_form[0])

    @property
    def letter_form_elm(self):
        """
        """
        group = self.group
        r = self.letter_form
        return [group.dtype(((elm,1),)) if elm.is_Symbol \
                else group.dtype(((-elm,-1),)) for elm in r]

    @property
    def ext_rep(self):
        """This is called the External Representation of ``FreeGroupElement``
        """
        return tuple(flatten(self.array_form))

    def __contains__(self, gen):
        return gen.array_form[0][0] in tuple([r[0] for r in self.array_form])

    def __str__(self):
        if self.is_identity:
            return "<identity>"

        str_form = ""
        array_form = self.array_form
        for i in range(len(array_form)):
            if i == len(array_form) - 1:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0])
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1])
            else:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0]) + "*"
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1]) + "*"
        return str_form

    __repr__ = __str__

    def __pow__(self, n):
        n = as_int(n)
        result = self.group.identity
        if n == 0:
            return result
        if n < 0:
            n = -n
            x = self.inverse()
        else:
            x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __mul__(self, other):
        """Returns the product of elements belonging to the same ``FreeGroup``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x*y**2*y**-4
        x*y**-2
        >>> z*y**-2
        z*y**-2
        >>> x**2*y*y**-1*x**-2
        <identity>

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        r = list(self.array_form + other.array_form)
        zero_mul_simp(r, len(self.array_form) - 1)
        return group.dtype(tuple(r))

    def __truediv__(self, other):
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        return self*(other.inverse())

    def __rtruediv__(self, other):
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        return other*(self.inverse())

    def __add__(self, other):
        return NotImplemented

    def inverse(self):
        """
        Returns the inverse of a ``FreeGroupElement`` element

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x.inverse()
        x**-1
        >>> (x*y).inverse()
        y**-1*x**-1

        """
        group = self.group
        r = tuple([(i, -j) for i, j in self.array_form[::-1]])
        return group.dtype(r)

    def order(self):
        """Find the order of a ``FreeGroupElement``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> (x**2*y*y**-1*x**-2).order()
        1

        """
        if self.is_identity:
            return S.One
        else:
            return S.Infinity

    def commutator(self, other):
        """
        Return the commutator of `self` and `x`: ``~x*~self*x*self``

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise ValueError("commutator of only FreeGroupElement of the same "
                    "FreeGroup exists")
        else:
            return self.inverse()*other.inverse()*self*other

    def eliminate_words(self, words, _all=False, inverse=True):
        '''
        Replace each subword from the dictionary `words` by words[subword].
        If words is a list, replace the words by the identity.

        '''
        again = True
        new = self
        if isinstance(words, dict):
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, words[sub], _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        else:
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        return new

    def eliminate_word(self, gen, by=None, _all=False, inverse=True):
        """
        For an associative word `self`, a subword `gen`, and an associative
        word `by` (identity by default), return the associative word obtained by
        replacing each occurrence of `gen` in `self` by `by`. If `_all = True`,
        the occurrences of `gen` that may appear after the first substitution will
        also be replaced and so on until no occurrences are found. This might not
        always terminate (e.g. `(x).eliminate_word(x, x**2, _all=True)`).

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> w = x**5*y*x**2*y**-4*x
        >>> w.eliminate_word( x, x**2 )
        x**10*y*x**4*y**-4*x**2
        >>> w.eliminate_word( x, y**-1 )
        y**-11
        >>> w.eliminate_word(x**5)
        y*x**2*y**-4*x
        >>> w.eliminate_word(x*y, y)
        x**4*y*x**2*y**-4*x

        See Also
        ========
        substituted_word

        """
        if by is None:
            by = self.group.identity
        if self.is_independent(gen) or gen == by:
            return self
        if gen == self:
            return by
        if gen**-1 == by:
            _all = False
        word = self
        l = len(gen)

        try:
            i = word.subword_index(gen)
            k = 1
        except ValueError:
            if not inverse:
                return word
            try:
                i = word.subword_index(gen**-1)
                k = -1
            except ValueError:
                return word

        word = word.subword(0, i)*by**k*word.subword(i+l, len(word)).eliminate_word(gen, by)

        if _all:
            return word.eliminate_word(gen, by, _all=True, inverse=inverse)
        else:
            return word

    def __len__(self):
        """
        For an associative word `self`, returns the number of letters in it.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> len(w)
        13
        >>> len(a**17)
        17
        >>> len(w**0)
        0

        """
        return sum(abs(j) for (i, j) in self)

    def __eq__(self, other):
        """
        Two  associative words are equal if they are words over the
        same alphabet and if they are sequences of the same letters.
        This is equivalent to saying that the external representations
        of the words are equal.
        There is no "universal" empty word, every alphabet has its own
        empty word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")
        >>> f
        <free group on the generators (swapnil0, swapnil1)>
        >>> g, swap0, swap1 = free_group("swap0 swap1")
        >>> g
        <free group on the generators (swap0, swap1)>

        >>> swapnil0 == swapnil1
        False
        >>> swapnil0*swapnil1 == swapnil1/swapnil1*swapnil0*swapnil1
        True
        >>> swapnil0*swapnil1 == swapnil1*swapnil0
        False
        >>> swapnil1**0 == swap0**0
        False

        """
        group = self.group
        if not isinstance(other, group.dtype):
            return False
        return tuple.__eq__(self, other)

    def __lt__(self, other):
        """
        The  ordering  of  associative  words is defined by length and
        lexicography (this ordering is called short-lex ordering), that
        is, shorter words are smaller than longer words, and words of the
        same length are compared w.r.t. the lexicographical ordering induced
        by the ordering of generators. Generators  are  sorted  according
        to the order in which they were created. If the generators are
        invertible then each generator `g` is larger than its inverse `g^{-1}`,
        and `g^{-1}` is larger than every generator that is smaller than `g`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> b < a
        False
        >>> a < a.inverse()
        False

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                             "be compared")
        l = len(self)
        m = len(other)
        # implement lenlex order
        if l < m:
            return True
        elif l > m:
            return False
        for i in range(l):
            a = self[i].array_form[0]
            b = other[i].array_form[0]
            p = group.symbols.index(a[0])
            q = group.symbols.index(b[0])
            if p < q:
                return True
            elif p > q:
                return False
            elif a[1] < b[1]:
                return True
            elif a[1] > b[1]:
                return False
        return False

    def __le__(self, other):
        return (self == other or self < other)

    def __gt__(self, other):
        """

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> y**2 > x**2
        True
        >>> y*z > z*y
        False
        >>> x > x.inverse()
        True

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                             "be compared")
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def exponent_sum(self, gen):
        """
        For an associative word `self` and a generator or inverse of generator
        `gen`, ``exponent_sum`` returns the number of times `gen` appears in
        `self` minus the number of times its inverse appears in `self`. If
        neither `gen` nor its inverse occur in `self` then 0 is returned.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.exponent_sum(x)
        2
        >>> w.exponent_sum(x**-1)
        -2
        >>> w = x**2*y**4*x**-3
        >>> w.exponent_sum(x)
        -1

        See Also
        ========

        generator_count

        """
        if len(gen) != 1:
            raise ValueError("gen must be a generator or inverse of a generator")
        s = gen.array_form[0]
        return s[1]*sum(i[1] for i in self.array_form if i[0] == s[0])

    def generator_count(self, gen):
        """
        For an associative word `self` and a generator `gen`,
        ``generator_count`` returns the multiplicity of generator
        `gen` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.generator_count(x)
        2
        >>> w = x**2*y**4*x**-3
        >>> w.generator_count(x)
        5

        See Also
        ========

        exponent_sum

        """
        if len(gen) != 1 or gen.array_form[0][1] < 0:
            raise ValueError("gen must be a generator")
        s = gen.array_form[0]
        return s[1]*sum(abs(i[1]) for i in self.array_form if i[0] == s[0])

    def subword(self, from_i, to_j, strict=True):
        """
        For an associative word `self` and two positive integers `from_i` and
        `to_j`, `subword` returns the subword of `self` that begins at position
        `from_i` and ends at `to_j - 1`, indexing is done with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.subword(2, 6)
        a**3*b

        """
        group = self.group
        if not strict:
            from_i = max(from_i, 0)
            to_j = min(len(self), to_j)
        if from_i < 0 or to_j > len(self):
            raise ValueError("`from_i`, `to_j` must be positive and no greater than "
                    "the length of associative word")
        if to_j <= from_i:
            return group.identity
        else:
            letter_form = self.letter_form[from_i: to_j]
            array_form = letter_form_to_array_form(letter_form, group)
            return group.dtype(array_form)

    def subword_index(self, word, start = 0):
        '''
        Find the index of `word` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**2*b*a*b**3
        >>> w.subword_index(a*b*a*b)
        1

        '''
        l = len(word)
        self_lf = self.letter_form
        word_lf = word.letter_form
        index = None
        for i in range(start,len(self_lf)-l+1):
            if self_lf[i:i+l] == word_lf:
                index = i
                break
        if index is not None:
            return index
        else:
            raise ValueError("The given word is not a subword of self")

    def is_dependent(self, word):
        """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**4*y**-3).is_dependent(x**4*y**-2)
        True
        >>> (x**2*y**-1).is_dependent(x*y)
        False
        >>> (x*y**2*x*y**2).is_dependent(x*y**2)
        True
        >>> (x**12).is_dependent(x**-4)
        True

        See Also
        ========

        is_independent

        """
        try:
            return self.subword_index(word) is not None
        except ValueError:
            pass
        try:
            return self.subword_index(word**-1) is not None
        except ValueError:
            return False

    def is_independent(self, word):
        """

        See Also
        ========

        is_dependent

        """
        return not self.is_dependent(word)

    def contains_generators(self):
        """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> (x**2*y**-1).contains_generators()
        {x, y}
        >>> (x**3*z).contains_generators()
        {x, z}

        """
        group = self.group
        gens = {group.dtype(((syllable[0], 1),)) for syllable in self.array_form}
        return gens

    def cyclic_subword(self, from_i, to_j):
        group = self.group
        l = len(self)
        letter_form = self.letter_form
        period1 = int(from_i/l)
        if from_i >= l:
            from_i -= l*period1
            to_j -= l*period1
        diff = to_j - from_i
        word = letter_form[from_i: to_j]
        period2 = int(to_j/l) - 1
        word += letter_form*period2 + letter_form[:diff-l+from_i-l*period2]
        word = letter_form_to_array_form(word, group)
        return group.dtype(word)

    def cyclic_conjugates(self):
        """Returns a words which are cyclic to the word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x*y*x*y*x
        >>> w.cyclic_conjugates()
        {x*y*x**2*y, x**2*y*x*y, y*x*y*x**2, y*x**2*y*x, x*y*x*y*x}
        >>> s = x*y*x**2*y*x
        >>> s.cyclic_conjugates()
        {x**2*y*x**2*y, y*x**2*y*x**2, x*y*x**2*y*x}

        References
        ==========

        .. [1] https://planetmath.org/cyclicpermutation

        """
        return {self.cyclic_subword(i, i+len(self)) for i in range(len(self))}

    def is_cyclic_conjugate(self, w):
        """
        Checks whether words ``self``, ``w`` are cyclic conjugates.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w1 = x**2*y**5
        >>> w2 = x*y**5*x
        >>> w1.is_cyclic_conjugate(w2)
        True
        >>> w3 = x**-1*y**5*x**-1
        >>> w3.is_cyclic_conjugate(w2)
        False

        """
        l1 = len(self)
        l2 = len(w)
        if l1 != l2:
            return False
        w1 = self.identity_cyclic_reduction()
        w2 = w.identity_cyclic_reduction()
        letter1 = w1.letter_form
        letter2 = w2.letter_form
        str1 = ' '.join(map(str, letter1))
        str2 = ' '.join(map(str, letter2))
        if len(str1) != len(str2):
            return False

        return str1 in str2 + ' ' + str2

    def number_syllables(self):
        """Returns the number of syllables of the associative word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")
        >>> (swapnil1**3*swapnil0*swapnil1**-1).number_syllables()
        3

        """
        return len(self.array_form)

    def exponent_syllable(self, i):
        """
        Returns the exponent of the `i`-th syllable of the associative word
        `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.exponent_syllable( 2 )
        2

        """
        return self.array_form[i][1]

    def generator_syllable(self, i):
        """
        Returns the symbol of the generator that is involved in the
        i-th syllable of the associative word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.generator_syllable( 3 )
        b

        """
        return self.array_form[i][0]

    def sub_syllables(self, from_i, to_j):
        """
        `sub_syllables` returns the subword of the associative word `self` that
        consists of syllables from positions `from_to` to `to_j`, where
        `from_to` and `to_j` must be positive integers and indexing is done
        with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a, b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.sub_syllables(1, 2)
        b
        >>> w.sub_syllables(3, 3)
        <identity>

        """
        if not isinstance(from_i, int) or not isinstance(to_j, int):
            raise ValueError("both arguments should be integers")
        group = self.group
        if to_j <= from_i:
            return group.identity
        else:
            r = tuple(self.array_form[from_i: to_j])
            return group.dtype(r)

    def substituted_word(self, from_i, to_j, by):
        """
        Returns the associative word obtained by replacing the subword of
        `self` that begins at position `from_i` and ends at position `to_j - 1`
        by the associative word `by`. `from_i` and `to_j` must be positive
        integers, indexing is done with origin 0. In other words,
        `w.substituted_word(w, from_i, to_j, by)` is the product of the three
        words: `w.subword(0, from_i)`, `by`, and
        `w.subword(to_j len(w))`.

        See Also
        ========

        eliminate_word

        """
        lw = len(self)
        if from_i >= to_j or from_i > lw or to_j > lw:
            raise ValueError("values should be within bounds")

        # otherwise there are four possibilities

        # first if from=1 and to=lw then
        if from_i == 0 and to_j == lw:
            return by
        elif from_i == 0:  # second if from_i=1 (and to_j < lw) then
            return by*self.subword(to_j, lw)
        elif to_j == lw:   # third if to_j=1 (and from_i > 1) then
            return self.subword(0, from_i)*by
        else:              # finally
            return self.subword(0, from_i)*by*self.subword(to_j, lw)

    def is_cyclically_reduced(self):
        r"""Returns whether the word is cyclically reduced or not.
        A word is cyclically reduced if by forming the cycle of the
        word, the word is not reduced, i.e a word w = `a_1 ... a_n`
        is called cyclically reduced if `a_1 \ne a_n^{-1}`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**-1*x**-1).is_cyclically_reduced()
        False
        >>> (y*x**2*y**2).is_cyclically_reduced()
        True

        """
        if not self:
            return True
        return self[0] != self[-1]**-1

    def identity_cyclic_reduction(self):
        """Return a unique cyclically reduced version of the word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).identity_cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).identity_cyclic_reduction()
        x**2*y**-1

        References
        ==========

        .. [1] https://planetmath.org/cyclicallyreduced

        """
        word = self.copy()
        group = self.group
        while not word.is_cyclically_reduced():
            exp1 = word.exponent_syllable(0)
            exp2 = word.exponent_syllable(-1)
            r = exp1 + exp2
            if r == 0:
                rep = word.array_form[1: word.number_syllables() - 1]
            else:
                rep = ((word.generator_syllable(0), exp1 + exp2),) + \
                        word.array_form[1: word.number_syllables() - 1]
            word = group.dtype(rep)
        return word

    def cyclic_reduction(self, removed=False):
        """Return a cyclically reduced version of the word. Unlike
        `identity_cyclic_reduction`, this will not cyclically permute
        the reduced word - just remove the "unreduced" bits on either
        side of it. Compare the examples with those of
        `identity_cyclic_reduction`.

        When `removed` is `True`, return a tuple `(word, r)` where
        self `r` is such that before the reduction the word was either
        `r*word*r**-1`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction()
        y**-1*x**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction(removed=True)
        (y**-1*x**2, x**-3)

        """
        word = self.copy()
        g = self.group.identity
        while not word.is_cyclically_reduced():
            exp1 = abs(word.exponent_syllable(0))
            exp2 = abs(word.exponent_syllable(-1))
            exp = min(exp1, exp2)
            start = word[0]**abs(exp)
            end = word[-1]**abs(exp)
            word = start**-1*word*end**-1
            g = g*start
        if removed:
            return word, g
        return word

    def power_of(self, other):
        '''
        Check if `self == other**n` for some integer n.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> ((x*y)**2).power_of(x*y)
        True
        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)
        True

        '''
        if self.is_identity:
            return True

        l = len(other)
        if l == 1:
            # self has to be a power of one generator
            gens = self.contains_generators()
            s = other in gens or other**-1 in gens
            return len(gens) == 1 and s

        # if self is not cyclically reduced and it is a power of other,
        # other isn't cyclically reduced and the parts removed during
        # their reduction must be equal
        reduced, r1 = self.cyclic_reduction(removed=True)
        if not r1.is_identity:
            other, r2 = other.cyclic_reduction(removed=True)
            if r1 == r2:
                return reduced.power_of(other)
            return False

        if len(self) < l or len(self) % l:
            return False

        prefix = self.subword(0, l)
        if prefix == other or prefix**-1 == other:
            rest = self.subword(l, len(self))
            return rest.power_of(other)
        return False


def letter_form_to_array_form(array_form, group):
    """
    This method converts a list given with possible repetitions of elements in
    it. It returns a new list such that repetitions of consecutive elements is
    removed and replace with a tuple element of size two such that the first
    index contains `value` and the second index contains the number of
    consecutive repetitions of `value`.

    """
    a = list(array_form[:])
    new_array = []
    n = 1
    symbols = group.symbols
    for i in range(len(a)):
        if i == len(a) - 1:
            if a[i] == a[i - 1]:
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -n))
                else:
                    new_array.append((a[i], n))
            else:
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -1))
                else:
                    new_array.append((a[i], 1))
            return new_array
        elif a[i] == a[i + 1]:
            n += 1
        else:
            if (-a[i]) in symbols:
                new_array.append((-a[i], -n))
            else:
                new_array.append((a[i], n))
            n = 1


def zero_mul_simp(l, index):
    """Used to combine two reduced words."""
    while index >=0 and index < len(l) - 1 and l[index][0] == l[index + 1][0]:
        exp = l[index][1] + l[index + 1][1]
        base = l[index][0]
        l[index] = (base, exp)
        del l[index + 1]
        if l[index][1] == 0:
            del l[index]
            index -= 1
