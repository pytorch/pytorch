from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group


class PolycyclicGroup(DefaultPrinting):

    is_group = True
    is_solvable = True

    def __init__(self, pc_sequence, pc_series, relative_order, collector=None):
        """

        Parameters
        ==========

        pc_sequence : list
            A sequence of elements whose classes generate the cyclic factor
            groups of pc_series.
        pc_series : list
            A subnormal sequence of subgroups where each factor group is cyclic.
        relative_order : list
            The orders of factor groups of pc_series.
        collector : Collector
            By default, it is None. Collector class provides the
            polycyclic presentation with various other functionalities.

        """
        self.pcgs = pc_sequence
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.collector = Collector(self.pcgs, pc_series, relative_order) if not collector else collector

    def is_prime_order(self):
        return all(isprime(order) for order in self.relative_order)

    def length(self):
        return len(self.pcgs)


class Collector(DefaultPrinting):

    """
    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 8.1.3
    """

    def __init__(self, pcgs, pc_series, relative_order, free_group_=None, pc_presentation=None):
        """

        Most of the parameters for the Collector class are the same as for PolycyclicGroup.
        Others are described below.

        Parameters
        ==========

        free_group_ : tuple
            free_group_ provides the mapping of polycyclic generating
            sequence with the free group elements.
        pc_presentation : dict
            Provides the presentation of polycyclic groups with the
            help of power and conjugate relators.

        See Also
        ========

        PolycyclicGroup

        """
        self.pcgs = pcgs
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.free_group = free_group('x:{}'.format(len(pcgs)))[0] if not free_group_ else free_group_
        self.index = {s: i for i, s in enumerate(self.free_group.symbols)}
        self.pc_presentation = self.pc_relators()

    def minimal_uncollected_subword(self, word):
        r"""
        Returns the minimal uncollected subwords.

        Explanation
        ===========

        A word ``v`` defined on generators in ``X`` is a minimal
        uncollected subword of the word ``w`` if ``v`` is a subword
        of ``w`` and it has one of the following form

        * `v = {x_{i+1}}^{a_j}x_i`

        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`

        * `v = {x_i}^{a_j}`

        for `a_j` not in `\{1, \ldots, s-1\}`. Where, ``s`` is the power
        exponent of the corresponding generator.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> collector.minimal_uncollected_subword(word)
        ((x2, 2),)

        """
        # To handle the case word = <identity>
        if not word:
            return None

        array = word.array_form
        re = self.relative_order
        index = self.index

        for i in range(len(array)):
            s1, e1 = array[i]

            if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]]-1):
                return ((s1, e1), )

        for i in range(len(array)-1):
            s1, e1 = array[i]
            s2, e2 = array[i+1]

            if index[s1] > index[s2]:
                e = 1 if e2 > 0 else -1
                return ((s1, e1), (s2, e))

        return None

    def relations(self):
        """
        Separates the given relators of pc presentation in power and
        conjugate relations.

        Returns
        =======

        (power_rel, conj_rel)
            Separates pc presentation into power and conjugate relations.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> power_rel, conj_rel = collector.relations()
        >>> power_rel
        {x0**2: (), x1**3: ()}
        >>> conj_rel
        {x0**-1*x1*x0: x1**2}

        See Also
        ========

        pc_relators

        """
        power_relators = {}
        conjugate_relators = {}
        for key, value in self.pc_presentation.items():
            if len(key.array_form) == 1:
                power_relators[key] = value
            else:
                conjugate_relators[key] = value
        return power_relators, conjugate_relators

    def subword_index(self, word, w):
        """
        Returns the start and ending index of a given
        subword in a word.

        Parameters
        ==========

        word : FreeGroupElement
            word defined on free group elements for a
            polycyclic group.
        w : FreeGroupElement
            subword of a given word, whose starting and
            ending index to be computed.

        Returns
        =======

        (i, j)
            A tuple containing starting and ending index of ``w``
            in the given word.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> w = x2**2*x1
        >>> collector.subword_index(word, w)
        (0, 3)
        >>> w = x1**7
        >>> collector.subword_index(word, w)
        (2, 9)

        """
        low = -1
        high = -1
        for i in range(len(word)-len(w)+1):
            if word.subword(i, i+len(w)) == w:
                low = i
                high = i+len(w)
                break
        if low == high == -1:
            return -1, -1
        return low, high

    def map_relation(self, w):
        """
        Return a conjugate relation.

        Explanation
        ===========

        Given a word formed by two free group elements, the
        corresponding conjugate relation with those free
        group elements is formed and mapped with the collected
        word in the polycyclic presentation.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1 = free_group("x0, x1")
        >>> w = x1*x0
        >>> collector.map_relation(w)
        x1**2

        See Also
        ========

        pc_presentation

        """
        array = w.array_form
        s1 = array[0][0]
        s2 = array[1][0]
        key = ((s2, -1), (s1, 1), (s2, 1))
        key = self.free_group.dtype(key)
        return self.pc_presentation[key]


    def collected_word(self, word):
        r"""
        Return the collected form of a word.

        Explanation
        ===========

        A word ``w`` is called collected, if `w = {x_{i_1}}^{a_1} * \ldots *
        {x_{i_r}}^{a_r}` with `i_1 < i_2< \ldots < i_r` and `a_j` is in
        `\{1, \ldots, {s_j}-1\}`.

        Otherwise w is uncollected.

        Parameters
        ==========

        word : FreeGroupElement
            An uncollected word.

        Returns
        =======

        word
            A collected word of form `w = {x_{i_1}}^{a_1}, \ldots,
            {x_{i_r}}^{a_r}` with `i_1, i_2, \ldots, i_r` and `a_j \in
            \{1, \ldots, {s_j}-1\}`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1, x2, x3 = free_group("x0, x1, x2, x3")
        >>> word = x3*x2*x1*x0
        >>> collected_word = collector.collected_word(word)
        >>> free_to_perm = {}
        >>> free_group = collector.free_group
        >>> for sym, gen in zip(free_group.symbols, collector.pcgs):
        ...     free_to_perm[sym] = gen
        >>> G1 = PermutationGroup()
        >>> for w in word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G1 = PermutationGroup([perm] + G1.generators)
        >>> G2 = PermutationGroup()
        >>> for w in collected_word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G2 = PermutationGroup([perm] + G2.generators)

        The two are not identical, but they are equivalent:

        >>> G1.equals(G2), G1 == G2
        (True, False)

        See Also
        ========

        minimal_uncollected_subword

        """
        free_group = self.free_group
        while True:
            w = self.minimal_uncollected_subword(word)
            if not w:
                break

            low, high = self.subword_index(word, free_group.dtype(w))
            if low == -1:
                continue

            s1, e1 = w[0]
            if len(w) == 1:
                re = self.relative_order[self.index[s1]]
                q = e1 // re
                r = e1-q*re

                key = ((w[0][0], re), )
                key = free_group.dtype(key)
                if self.pc_presentation[key]:
                    presentation = self.pc_presentation[key].array_form
                    sym, exp = presentation[0]
                    word_ = ((w[0][0], r), (sym, q*exp))
                    word_ = free_group.dtype(word_)
                else:
                    if r != 0:
                        word_ = ((w[0][0], r), )
                        word_ = free_group.dtype(word_)
                    else:
                        word_ = None
                word = word.eliminate_word(free_group.dtype(w), word_)

            if len(w) == 2 and w[1][1] > 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

            elif len(w) == 2 and w[1][1] < 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2**-1*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

        return word


    def pc_relators(self):
        r"""
        Return the polycyclic presentation.

        Explanation
        ===========

        There are two types of relations used in polycyclic
        presentation.

        * Power relations : Power relators are of the form `x_i^{re_i}`,
          where `i \in \{0, \ldots, \mathrm{len(pcgs)}\}`, ``x`` represents polycyclic
          generator and ``re`` is the corresponding relative order.

        * Conjugate relations : Conjugate relators are of the form `x_j^-1x_ix_j`,
          where `j < i \in \{0, \ldots, \mathrm{len(pcgs)}\}`.

        Returns
        =======

        A dictionary with power and conjugate relations as key and
        their collected form as corresponding values.

        Notes
        =====

        Identity Permutation is mapped with empty ``()``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> S = SymmetricGroup(49).sylow_subgroup(7)
        >>> der = S.derived_series()
        >>> G = der[len(der)-2]
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> len(pcgs)
        6
        >>> free_group = collector.free_group
        >>> pc_resentation = collector.pc_presentation
        >>> free_to_perm = {}
        >>> for s, g in zip(free_group.symbols, pcgs):
        ...     free_to_perm[s] = g

        >>> for k, v in pc_resentation.items():
        ...     k_array = k.array_form
        ...     if v != ():
        ...        v_array = v.array_form
        ...     lhs = Permutation()
        ...     for gen in k_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         lhs = lhs*free_to_perm[s]**e
        ...     if v == ():
        ...         assert lhs.is_identity
        ...         continue
        ...     rhs = Permutation()
        ...     for gen in v_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         rhs = rhs*free_to_perm[s]**e
        ...     assert lhs == rhs

        """
        free_group = self.free_group
        rel_order = self.relative_order
        pc_relators = {}
        perm_to_free = {}
        pcgs = self.pcgs

        for gen, s in zip(pcgs, free_group.generators):
            perm_to_free[gen**-1] = s**-1
            perm_to_free[gen] = s

        pcgs = pcgs[::-1]
        series = self.pc_series[::-1]
        rel_order = rel_order[::-1]
        collected_gens = []

        for i, gen in enumerate(pcgs):
            re = rel_order[i]
            relation = perm_to_free[gen]**re
            G = series[i]

            l = G.generator_product(gen**re, original = True)
            l.reverse()

            word = free_group.identity
            for g in l:
                word = word*perm_to_free[g]

            word = self.collected_word(word)
            pc_relators[relation] = word if word else ()
            self.pc_presentation = pc_relators

            collected_gens.append(gen)
            if len(collected_gens) > 1:
                conj = collected_gens[len(collected_gens)-1]
                conjugator = perm_to_free[conj]

                for j in range(len(collected_gens)-1):
                    conjugated = perm_to_free[collected_gens[j]]

                    relation = conjugator**-1*conjugated*conjugator
                    gens = conj**-1*collected_gens[j]*conj

                    l = G.generator_product(gens, original = True)
                    l.reverse()
                    word = free_group.identity
                    for g in l:
                        word = word*perm_to_free[g]

                    word = self.collected_word(word)
                    pc_relators[relation] = word if word else ()
                    self.pc_presentation = pc_relators

        return pc_relators

    def exponent_vector(self, element):
        r"""
        Return the exponent vector of length equal to the
        length of polycyclic generating sequence.

        Explanation
        ===========

        For a given generator/element ``g`` of the polycyclic group,
        it can be represented as `g = {x_1}^{e_1}, \ldots, {x_n}^{e_n}`,
        where `x_i` represents polycyclic generators and ``n`` is
        the number of generators in the free_group equal to the length
        of pcgs.

        Parameters
        ==========

        element : Permutation
            Generator of a polycyclic group.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> collector.exponent_vector(G[0])
        [1, 0, 0, 0]
        >>> exp = collector.exponent_vector(G[1])
        >>> g = Permutation()
        >>> for i in range(len(exp)):
        ...     g = g*pcgs[i]**exp[i] if exp[i] else g
        >>> assert g == G[1]

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.4

        """
        free_group = self.free_group
        G = PermutationGroup()
        for g in self.pcgs:
            G = PermutationGroup([g] + G.generators)
        gens = G.generator_product(element, original = True)
        gens.reverse()

        perm_to_free = {}
        for sym, g in zip(free_group.generators, self.pcgs):
            perm_to_free[g**-1] = sym**-1
            perm_to_free[g] = sym
        w = free_group.identity
        for g in gens:
            w = w*perm_to_free[g]

        word = self.collected_word(w)

        index = self.index
        exp_vector = [0]*len(free_group)
        word = word.array_form
        for t in word:
            exp_vector[index[t[0]]] = t[1]
        return exp_vector

    def depth(self, element):
        r"""
        Return the depth of a given element.

        Explanation
        ===========

        The depth of a given element ``g`` is defined by
        `\mathrm{dep}[g] = i` if `e_1 = e_2 = \ldots = e_{i-1} = 0`
        and `e_i != 0`, where ``e`` represents the exponent-vector.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.depth(G[0])
        2
        >>> collector.depth(G[1])
        1

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.5

        """
        exp_vector = self.exponent_vector(element)
        return next((i+1 for i, x in enumerate(exp_vector) if x), len(self.pcgs)+1)

    def leading_exponent(self, element):
        r"""
        Return the leading non-zero exponent.

        Explanation
        ===========

        The leading exponent for a given element `g` is defined
        by `\mathrm{leading\_exponent}[g]` `= e_i`, if `\mathrm{depth}[g] = i`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.leading_exponent(G[1])
        1

        """
        exp_vector = self.exponent_vector(element)
        depth = self.depth(element)
        if depth != len(self.pcgs)+1:
            return exp_vector[depth-1]
        return None

    def _sift(self, z, g):
        h = g
        d = self.depth(h)
        while d < len(self.pcgs) and z[d-1] != 1:
            k = z[d-1]
            e = self.leading_exponent(h)*(self.leading_exponent(k))**-1
            e = e % self.relative_order[d-1]
            h = k**-e*h
            d = self.depth(h)
        return h

    def induced_pcgs(self, gens):
        """

        Parameters
        ==========

        gens : list
            A list of generators on which polycyclic subgroup
            is to be defined.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
        z = [1]*len(self.pcgs)
        G = gens
        while G:
            g = G.pop(0)
            h = self._sift(z, g)
            d = self.depth(h)
            if d < len(self.pcgs):
                for gen in z:
                    if gen != 1:
                        G.append(h**-1*gen**-1*h*gen)
                z[d-1] = h;
        z = [gen for gen in z if gen != 1]
        return z

    def constructive_membership_test(self, ipcgs, g):
        """
        Return the exponent vector for induced pcgs.
        """
        e = [0]*len(ipcgs)
        h = g
        d = self.depth(h)
        for i, gen in enumerate(ipcgs):
            while self.depth(gen) == d:
                f = self.leading_exponent(h)*self.leading_exponent(gen)
                f = f % self.relative_order[d-1]
                h = gen**(-f)*h
                e[i] = f
                d = self.depth(h)
        if h == 1:
            return e
        return False
