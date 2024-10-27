# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from bisect import bisect_right, insort
from collections import Counter

import attr

from hypothesis.errors import InvalidState
from hypothesis.internal.conjecture.dfa import DFA, cached
from hypothesis.internal.conjecture.junkdrawer import (
    IntList,
    NotFound,
    SelfOrganisingList,
    find_integer,
)

"""
This module contains an implementation of the L* algorithm
for learning a deterministic finite automaton based on an
unknown membership function and a series of examples of
strings that may or may not satisfy it.

The two relevant papers for understanding this are:

* Angluin, Dana. "Learning regular sets from queries and counterexamples."
  Information and computation 75.2 (1987): 87-106.
* Rivest, Ronald L., and Robert E. Schapire. "Inference of finite automata
  using homing sequences." Information and Computation 103.2 (1993): 299-347.
  Note that we only use the material from section 4.5 "Improving Angluin's L*
  algorithm" (page 318), and all of the rest of the material on homing
  sequences can be skipped.

The former explains the core algorithm, the latter a modification
we use (which we have further modified) which allows it to
be implemented more efficiently.

Although we continue to call this L*, we in fact depart heavily from it to the
point where honestly this is an entirely different algorithm and we should come
up with a better name.

We have several major departures from the papers:

1. We learn the automaton lazily as we traverse it. This is particularly
   valuable because if we make many corrections on the same string we only
   have to learn the transitions that correspond to the string we are
   correcting on.
2. We make use of our ``find_integer`` method rather than a binary search
   as proposed in the Rivest and Schapire paper, as we expect that
   usually most strings will be mispredicted near the beginning.
3. We try to learn a smaller alphabet of "interestingly distinct"
   values. e.g. if all bytes larger than two result in an invalid
   string, there is no point in distinguishing those bytes. In aid
   of this we learn a single canonicalisation table which maps integers
   to smaller integers that we currently think are equivalent, and learn
   their inequivalence where necessary. This may require more learning
   steps, as at each stage in the process we might learn either an
   inequivalent pair of integers or a new experiment, but it may greatly
   reduce the number of membership queries we have to make.


In addition, we have a totally different approach for mapping a string to its
canonical representative, which will be explained below inline. The general gist
is that our implementation is much more willing to make mistakes: It will often
create a DFA that is demonstrably wrong, based on information that it already
has, but where it is too expensive to discover that before it causes us to
make a mistake.

A note on performance: This code is not really fast enough for
us to ever want to run in production on large strings, and this
is somewhat intrinsic. We should only use it in testing or for
learning languages offline that we can record for later use.

"""


@attr.s(slots=True)
class DistinguishedState:
    """Relevant information for a state that we have witnessed as definitely
    distinct from ones we have previously seen so far."""

    # Index of this state in the learner's list of states
    index: int = attr.ib()

    # A string that witnesses this state (i.e. when starting from the origin
    # and following this string you will end up in this state).
    label: str = attr.ib()

    # A boolean as to whether this is an accepting state.
    accepting: bool = attr.ib()

    # A list of experiments that it is necessary to run to determine whether
    # a string is in this state. This is stored as a dict mapping experiments
    # to their expected result. A string is only considered to lead to this
    # state if ``all(learner.member(s + experiment) == result for experiment,
    # result in self.experiments.items())``.
    experiments: dict = attr.ib()

    # A cache of transitions out of this state, mapping bytes to the states
    # that they lead to.
    transitions: dict = attr.ib(factory=dict)


class LStar:
    """This class holds the state for learning a DFA. The current DFA can be
    accessed as the ``dfa`` member of this class. Such a DFA becomes invalid
    as soon as ``learn`` has been called, and should only be used until the
    next call to ``learn``.

    Note that many of the DFA methods are on this class, but it is not itself
    a DFA. The reason for this is that it stores mutable state which can cause
    the structure of the learned DFA to change in potentially arbitrary ways,
    making all cached properties become nonsense.
    """

    def __init__(self, member):
        self.experiments = []
        self.__experiment_set = set()
        self.normalizer = IntegerNormalizer()

        self.__member_cache = {}
        self.__member = member
        self.__generation = 0

        # A list of all state objects that correspond to strings we have
        # seen and can demonstrate map to unique states.
        self.__states = [
            DistinguishedState(
                index=0,
                label=b"",
                accepting=self.member(b""),
                experiments={b"": self.member(b"")},
            )
        ]

        # When we're trying to figure out what state a string leads to we will
        # end up searching to find a suitable candidate. By putting states in
        # a self-organising list we ideally minimise the number of lookups.
        self.__self_organising_states = SelfOrganisingList(self.__states)

        self.start = 0

        self.__dfa_changed()

    def __dfa_changed(self):
        """Note that something has changed, updating the generation
        and resetting any cached state."""
        self.__generation += 1
        self.dfa = LearnedDFA(self)

    def is_accepting(self, i):
        """Equivalent to ``self.dfa.is_accepting(i)``"""
        return self.__states[i].accepting

    def label(self, i):
        """Returns the string label for state ``i``."""
        return self.__states[i].label

    def transition(self, i, c):
        """Equivalent to ``self.dfa.transition(i, c)```"""
        c = self.normalizer.normalize(c)
        state = self.__states[i]
        try:
            return state.transitions[c]
        except KeyError:
            pass

        # The state that we transition to when reading ``c`` is reached by
        # this string, because this state is reached by state.label. We thus
        # want our candidate for the transition to be some state with a label
        # equivalent to this string.
        #
        # We find such a state by looking for one such that all of its listed
        # experiments agree on the result for its state label and this string.
        string = state.label + bytes([c])

        # We keep track of some useful experiments for distinguishing this
        # string from other states, as this both allows us to more accurately
        # select the state to map to and, if necessary, create the new state
        # that this string corresponds to with a decent set of starting
        # experiments.
        accumulated = {}
        counts = Counter()

        def equivalent(t):
            """Checks if ``string`` could possibly lead to state ``t``."""
            for e, expected in accumulated.items():
                if self.member(t.label + e) != expected:
                    counts[e] += 1
                    return False

            for e, expected in t.experiments.items():
                result = self.member(string + e)
                if result != expected:
                    # We expect most experiments to return False so if we add
                    # only True ones to our collection of essential experiments
                    # we keep the size way down and select only ones that are
                    # likely to provide useful information in future.
                    if result:
                        accumulated[e] = result
                    return False
            return True

        try:
            destination = self.__self_organising_states.find(equivalent)
        except NotFound:
            i = len(self.__states)
            destination = DistinguishedState(
                index=i,
                label=string,
                experiments=accumulated,
                accepting=self.member(string),
            )
            self.__states.append(destination)
            self.__self_organising_states.add(destination)
        state.transitions[c] = destination.index
        return destination.index

    def member(self, s):
        """Check whether this string is a member of the language
        to be learned."""
        try:
            return self.__member_cache[s]
        except KeyError:
            result = self.__member(s)
            self.__member_cache[s] = result
            return result

    @property
    def generation(self):
        """Return an integer value that will be incremented
        every time the DFA we predict changes."""
        return self.__generation

    def learn(self, string):
        """Learn to give the correct answer on this string.
        That is, after this method completes we will have
        ``self.dfa.matches(s) == self.member(s)``.

        Note that we do not guarantee that this will remain
        true in the event that learn is called again with
        a different string. It is in principle possible that
        future learning will cause us to make a mistake on
        this string. However, repeatedly calling learn on
        each of a set of strings until the generation stops
        changing is guaranteed to terminate.
        """
        string = bytes(string)
        correct_outcome = self.member(string)

        # We don't want to check this inside the loop because it potentially
        # causes us to evaluate more of the states than we actually need to,
        # but if our model is mostly correct then this will be faster because
        # we only need to evaluate strings that are of the form
        # ``state + experiment``, which will generally be cached and/or needed
        # later.
        if self.dfa.matches(string) == correct_outcome:
            return

        # In the papers they assume that we only run this process
        # once, but this is silly - often when you've got a messy
        # string it will be wrong for many different reasons.
        #
        # Thus we iterate this to a fixed point where we repair
        # the DFA by repeatedly adding experiments until the DFA
        # agrees with the membership function on this string.

        # First we make sure that normalization is not the source of the
        # failure to match.
        while True:
            normalized = bytes(self.normalizer.normalize(c) for c in string)
            # We can correctly replace the string with its normalized version
            # so normalization is not the problem here.
            if self.member(normalized) == correct_outcome:
                string = normalized
                break
            alphabet = sorted(set(string), reverse=True)
            target = string
            for a in alphabet:

                def replace(b):
                    if a == b:
                        return target
                    return bytes(b if c == a else c for c in target)

                self.normalizer.distinguish(a, lambda x: self.member(replace(x)))
                target = replace(self.normalizer.normalize(a))
                assert self.member(target) == correct_outcome
            assert target != normalized
            self.__dfa_changed()

        if self.dfa.matches(string) == correct_outcome:
            return

        # Now we know normalization is correct we can attempt to determine if
        # any of our transitions are wrong.
        while True:
            dfa = self.dfa

            states = [dfa.start]

            def seems_right(n):
                """After reading n characters from s, do we seem to be
                in the right state?

                We determine this by replacing the first n characters
                of s with the label of the state we expect to be in.
                If we are in the right state, that will replace a substring
                with an equivalent one so must produce the same answer.
                """
                if n > len(string):
                    return False

                # Populate enough of the states list to know where we are.
                while n >= len(states):
                    states.append(dfa.transition(states[-1], string[len(states) - 1]))

                return self.member(dfa.label(states[n]) + string[n:]) == correct_outcome

            assert seems_right(0)

            n = find_integer(seems_right)

            # We got to the end without ever finding ourself in a bad
            # state, so we must correctly match this string.
            if n == len(string):
                assert dfa.matches(string) == correct_outcome
                break

            # Reading n characters does not put us in a bad state but
            # reading n + 1 does. This means that the remainder of
            # the string that we have not read yet is an experiment
            # that allows us to distinguish the state that we ended
            # up in from the state that we should have ended up in.

            source = states[n]
            character = string[n]
            wrong_destination = states[n + 1]

            # We've made an error in transitioning from ``source`` to
            # ``wrong_destination`` via ``character``. We now need to update
            # the DFA so that this transition no longer occurs. Note that we
            # do not guarantee that the transition is *correct* after this,
            # only that we don't make this particular error.
            assert self.transition(source, character) == wrong_destination

            labels_wrong_destination = self.dfa.label(wrong_destination)
            labels_correct_destination = self.dfa.label(source) + bytes([character])

            ex = string[n + 1 :]

            assert self.member(labels_wrong_destination + ex) != self.member(
                labels_correct_destination + ex
            )

            # Adding this experiment causes us to distinguish the wrong
            # destination from the correct one.
            self.__states[wrong_destination].experiments[ex] = self.member(
                labels_wrong_destination + ex
            )

            # We now clear the cached details that caused us to make this error
            # so that when we recalculate this transition we get to a
            # (hopefully now correct) different state.
            del self.__states[source].transitions[character]
            self.__dfa_changed()

            # We immediately recalculate the transition so that we can check
            # that it has changed as we expect it to have.
            new_destination = self.transition(source, string[n])
            assert new_destination != wrong_destination


class LearnedDFA(DFA):
    """This implements a lazily calculated DFA where states
    are labelled by some string that reaches them, and are
    distinguished by a membership test and a set of experiments."""

    def __init__(self, lstar):
        super().__init__()
        self.__lstar = lstar
        self.__generation = lstar.generation

    def __check_changed(self):
        if self.__generation != self.__lstar.generation:
            raise InvalidState(
                "The underlying L* model has changed, so this DFA is no longer valid. "
                "If you want to preserve a previously learned DFA for posterity, call "
                "canonicalise() on it first."
            )

    def label(self, i):
        self.__check_changed()
        return self.__lstar.label(i)

    @property
    def start(self):
        self.__check_changed()
        return self.__lstar.start

    def is_accepting(self, i):
        self.__check_changed()
        return self.__lstar.is_accepting(i)

    def transition(self, i, c):
        self.__check_changed()

        return self.__lstar.transition(i, c)

    @cached
    def successor_states(self, state):
        """Returns all of the distinct states that can be reached via one
        transition from ``state``, in the lexicographic order of the
        smallest character that reaches them."""
        seen = set()
        result = []
        for c in self.__lstar.normalizer.representatives():
            j = self.transition(state, c)
            if j not in seen:
                seen.add(j)
                result.append(j)
        return tuple(result)


class IntegerNormalizer:
    """A class for replacing non-negative integers with a
    "canonical" value that is equivalent for all relevant
    purposes."""

    def __init__(self):
        # We store canonical values as a sorted list of integers
        # with each value being treated as equivalent to the largest
        # integer in the list that is below it.
        self.__values = IntList([0])
        self.__cache = {}

    def __repr__(self):
        return f"IntegerNormalizer({list(self.__values)!r})"

    def __copy__(self):
        result = IntegerNormalizer()
        result.__values = IntList(self.__values)
        return result

    def representatives(self):
        yield from self.__values

    def normalize(self, value):
        """Return the canonical integer considered equivalent
        to ``value``."""
        try:
            return self.__cache[value]
        except KeyError:
            pass
        i = bisect_right(self.__values, value) - 1
        assert i >= 0
        return self.__cache.setdefault(value, self.__values[i])

    def distinguish(self, value, test):
        """Checks whether ``test`` gives the same answer for
        ``value`` and ``self.normalize(value)``. If it does
        not, updates the list of canonical values so that
        it does.

        Returns True if and only if this makes a change to
        the underlying canonical values."""
        canonical = self.normalize(value)
        if canonical == value:
            return False

        value_test = test(value)

        if test(canonical) == value_test:
            return False

        self.__cache.clear()

        def can_lower(k):
            new_canon = value - k
            if new_canon <= canonical:
                return False
            return test(new_canon) == value_test

        new_canon = value - find_integer(can_lower)

        assert new_canon not in self.__values

        insort(self.__values, new_canon)

        assert self.normalize(value) == new_canon
        return True
