# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import threading
from collections import Counter, defaultdict, deque
from math import inf

from hypothesis.internal.reflection import proxies


def cached(fn):
    @proxies(fn)
    def wrapped(self, *args):
        cache = self._DFA__cache(fn.__name__)
        try:
            return cache[args]
        except KeyError:
            return cache.setdefault(args, fn(self, *args))

    return wrapped


class DFA:
    """Base class for implementations of deterministic finite
    automata.

    This is abstract to allow for the possibility of states
    being calculated lazily as we traverse the DFA (which
    we make heavy use of in our L* implementation - see
    lstar.py for details).

    States can be of any hashable type.
    """

    def __init__(self):
        self.__caches = threading.local()

    def __cache(self, name):
        try:
            cache = getattr(self.__caches, name)
        except AttributeError:
            cache = {}
            setattr(self.__caches, name, cache)
        return cache

    @property
    def start(self):
        """Returns the starting state."""
        raise NotImplementedError

    def is_accepting(self, i):
        """Returns if state ``i`` is an accepting one."""
        raise NotImplementedError

    def transition(self, i, c):
        """Returns the state that i transitions to on reading
        character c from a string."""
        raise NotImplementedError

    @property
    def alphabet(self):
        return range(256)

    def transitions(self, i):
        """Iterates over all pairs (byte, state) of transitions
        which do not lead to dead states."""
        for c, j in self.raw_transitions(i):
            if not self.is_dead(j):
                yield c, j

    @cached
    def transition_counts(self, state):
        counts = Counter()
        for _, j in self.transitions(state):
            counts[j] += 1
        return list(counts.items())

    def matches(self, s):
        """Returns whether the string ``s`` is accepted
        by this automaton."""
        i = self.start
        for c in s:
            i = self.transition(i, c)
        return self.is_accepting(i)

    def all_matching_regions(self, string):
        """Return all pairs ``(u, v)`` such that ``self.matches(string[u:v])``."""

        # Stack format: (k, state, indices). After reading ``k`` characters
        # starting from any i in ``indices`` the DFA would be at ``state``.
        stack = [(0, self.start, range(len(string)))]

        results = []

        while stack:
            k, state, indices = stack.pop()

            # If the state is dead, abort early - no point continuing on
            # from here where there will be no more matches.
            if self.is_dead(state):
                continue

            # If the state is accepting, then every one of these indices
            # has a matching region of length ``k`` starting from it.
            if self.is_accepting(state):
                results.extend([(i, i + k) for i in indices])

            next_by_state = defaultdict(list)

            for i in indices:
                if i + k < len(string):
                    c = string[i + k]
                    next_by_state[self.transition(state, c)].append(i)
            for next_state, next_indices in next_by_state.items():
                stack.append((k + 1, next_state, next_indices))
        return results

    def max_length(self, i):
        """Returns the maximum length of a string that is
        accepted when starting from i."""
        if self.is_dead(i):
            return 0

        cache = self.__cache("max_length")

        try:
            return cache[i]
        except KeyError:
            pass

        # Naively we can calculate this as 1 longer than the
        # max length of the non-dead states this can immediately
        # transition to, but a) We don't want unbounded recursion
        # because that's how you get RecursionErrors and b) This
        # makes it hard to look for cycles. So we basically do
        # the recursion explicitly with a stack, but we maintain
        # a parallel set that tracks what's already on the stack
        # so that when we encounter a loop we can immediately
        # determine that the max length here is infinite.

        stack = [i]
        stack_set = {i}

        def pop():
            """Remove the top element from the stack, maintaining
            the stack set appropriately."""
            assert len(stack) == len(stack_set)
            j = stack.pop()
            stack_set.remove(j)
            assert len(stack) == len(stack_set)

        while stack:
            j = stack[-1]
            assert not self.is_dead(j)
            # If any of the children have infinite max_length we don't
            # need to check all of them to know that this state does
            # too.
            if any(cache.get(k) == inf for k in self.successor_states(j)):
                cache[j] = inf
                pop()
                continue

            # Recurse to the first child node that we have not yet
            # calculated max_length for.
            for k in self.successor_states(j):
                if k in stack_set:
                    # k is part of a loop and is known to be live
                    # (since we never push dead states on the stack),
                    # so it can reach strings of unbounded length.
                    assert not self.is_dead(k)
                    cache[k] = inf
                    break
                elif k not in cache and not self.is_dead(k):
                    stack.append(k)
                    stack_set.add(k)
                    break
            else:
                # All of j's successors have a known max_length or are dead,
                # so we can now compute a max_length for j itself.
                cache[j] = max(
                    (
                        1 + cache[k]
                        for k in self.successor_states(j)
                        if not self.is_dead(k)
                    ),
                    default=0,
                )

                # j is live so it must either be accepting or have a live child.
                assert self.is_accepting(j) or cache[j] > 0
                pop()
        return cache[i]

    @cached
    def has_strings(self, state, length):
        """Returns if any strings of length ``length`` are accepted when
        starting from state ``state``."""
        assert length >= 0

        cache = self.__cache("has_strings")

        try:
            return cache[state, length]
        except KeyError:
            pass

        pending = [(state, length)]
        seen = set()
        i = 0

        while i < len(pending):
            s, n = pending[i]
            i += 1
            if n > 0:
                for t in self.successor_states(s):
                    key = (t, n - 1)
                    if key not in cache and key not in seen:
                        pending.append(key)
                        seen.add(key)

        while pending:
            s, n = pending.pop()
            if n == 0:
                cache[s, n] = self.is_accepting(s)
            else:
                cache[s, n] = any(
                    cache.get((t, n - 1)) for t in self.successor_states(s)
                )

        return cache[state, length]

    def count_strings(self, state, length):
        """Returns the number of strings of length ``length``
        that are accepted when starting from state ``state``."""
        assert length >= 0
        cache = self.__cache("count_strings")

        try:
            return cache[state, length]
        except KeyError:
            pass

        pending = [(state, length)]
        seen = set()
        i = 0

        while i < len(pending):
            s, n = pending[i]
            i += 1
            if n > 0:
                for t in self.successor_states(s):
                    key = (t, n - 1)
                    if key not in cache and key not in seen:
                        pending.append(key)
                        seen.add(key)

        while pending:
            s, n = pending.pop()
            if n == 0:
                cache[s, n] = int(self.is_accepting(s))
            else:
                cache[s, n] = sum(
                    cache[t, n - 1] * k for t, k in self.transition_counts(s)
                )

        return cache[state, length]

    @cached
    def successor_states(self, state):
        """Returns all of the distinct states that can be reached via one
        transition from ``state``, in the lexicographic order of the
        smallest character that reaches them."""
        seen = set()
        result = []
        for _, j in self.raw_transitions(state):
            if j not in seen:
                seen.add(j)
                result.append(j)
        return tuple(result)

    def is_dead(self, state):
        """Returns True if no strings can be accepted
        when starting from ``state``."""
        return not self.is_live(state)

    def is_live(self, state):
        """Returns True if any strings can be accepted
        when starting from ``state``."""
        if self.is_accepting(state):
            return True

        # We work this out by calculating is_live for all nodes
        # reachable from state which have not already had it calculated.
        cache = self.__cache("is_live")
        try:
            return cache[state]
        except KeyError:
            pass

        # roots are states that we know already must be live,
        # either because we have previously calculated them to
        # be or because they are an accepting state.
        roots = set()

        # We maintain a backwards graph where ``j in backwards_graph[k]``
        # if there is a transition from j to k. Thus if a key in this
        # graph is live, so must all its values be.
        backwards_graph = defaultdict(set)

        # First we find all reachable nodes from i which have not
        # already been cached, noting any which are roots and
        # populating the backwards graph.

        explored = set()
        queue = deque([state])
        while queue:
            j = queue.popleft()
            if cache.get(j, self.is_accepting(j)):
                # If j can be immediately determined to be live
                # then there is no point in exploring beneath it,
                # because any effect of states below it is screened
                # off by the known answer for j.
                roots.add(j)
                continue

            if j in cache:
                # Likewise if j is known to be dead then there is
                # no point exploring beneath it because we know
                # that all nodes reachable from it must be dead.
                continue

            if j in explored:
                continue
            explored.add(j)

            for k in self.successor_states(j):
                backwards_graph[k].add(j)
                queue.append(k)

        marked_live = set()
        queue = deque(roots)
        while queue:
            j = queue.popleft()
            if j in marked_live:
                continue
            marked_live.add(j)
            for k in backwards_graph[j]:
                queue.append(k)
        for j in explored:
            cache[j] = j in marked_live

        return cache[state]

    def all_matching_strings_of_length(self, k):
        """Yields all matching strings whose length is ``k``, in ascending
        lexicographic order."""
        if k == 0:
            if self.is_accepting(self.start):
                yield b""
            return

        if not self.has_strings(self.start, k):
            return

        # This tracks a path through the DFA. We alternate between growing
        # it until it has length ``k`` and is in an accepting state, then
        # yielding that as a result, then modifying it so that the next
        # time we do that it will yield the lexicographically next matching
        # string.
        path = bytearray()

        # Tracks the states that are visited by following ``path`` from the
        # starting point.
        states = [self.start]

        while True:
            # First we build up our current best prefix to the lexicographically
            # first string starting with it.
            while len(path) < k:
                state = states[-1]
                for c, j in self.transitions(state):
                    if self.has_strings(j, k - len(path) - 1):
                        states.append(j)
                        path.append(c)
                        break
                else:
                    raise NotImplementedError("Should be unreachable")
            assert self.is_accepting(states[-1])
            assert len(states) == len(path) + 1
            yield bytes(path)

            # Now we want to replace this string with the prefix that will
            # cause us to extend to its lexicographic successor. This can
            # be thought of as just repeatedly moving to the next lexicographic
            # successor until we find a matching string, but we're able to
            # use our length counts to jump over long sequences where there
            # cannot be a match.
            while True:
                # As long as we are in this loop we are trying to move to
                # the successor of the current string.

                # If we've removed the entire prefix then we're done - no
                # successor is possible.
                if not path:
                    return

                if path[-1] == 255:
                    # If our last element is maximal then the we have to "carry
                    # the one" - our lexicographic successor must be incremented
                    # earlier than this.
                    path.pop()
                    states.pop()
                else:
                    # Otherwise increment by one.
                    path[-1] += 1
                    states[-1] = self.transition(states[-2], path[-1])

                    # If there are no strings of the right length starting from
                    # this prefix we need to keep going. Otherwise, this is
                    # the right place to be and we break out of our loop of
                    # trying to find the successor because it starts here.
                    if self.count_strings(states[-1], k - len(path)) > 0:
                        break

    def all_matching_strings(self, min_length=0):
        """Iterate over all strings matched by this automaton
        in shortlex-ascending order."""
        # max_length might be infinite, hence the while loop
        max_length = self.max_length(self.start)
        length = min_length
        while length <= max_length:
            yield from self.all_matching_strings_of_length(length)
            length += 1

    def raw_transitions(self, i):
        for c in self.alphabet:
            j = self.transition(i, c)
            yield c, j

    def canonicalise(self):
        """Return a canonical version of ``self`` as a ConcreteDFA.

        The DFA is not minimized, but nodes are sorted and relabelled
        and dead nodes are pruned, so two minimized DFAs for the same
        language will end up with identical canonical representatives.
        This is mildly important because it means that the output of
        L* should produce the same canonical DFA regardless of what
        order we happen to have run it in.
        """
        # We map all states to their index of appearance in depth
        # first search. This both is useful for canonicalising and
        # also allows for states that aren't integers.
        state_map = {}
        reverse_state_map = []
        accepting = set()

        seen = set()

        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            if state in state_map:
                continue
            i = len(reverse_state_map)
            if self.is_accepting(state):
                accepting.add(i)
            reverse_state_map.append(state)
            state_map[state] = i
            for _, j in self.transitions(state):
                if j in seen:
                    continue
                seen.add(j)
                queue.append(j)

        transitions = [
            {c: state_map[s] for c, s in self.transitions(t)} for t in reverse_state_map
        ]

        result = ConcreteDFA(transitions, accepting)
        assert self.equivalent(result)
        return result

    def equivalent(self, other):
        """Checks whether this DFA and other match precisely the same
        language.

        Uses the classic algorithm of Hopcroft and Karp (more or less):
        Hopcroft, John E. A linear algorithm for testing equivalence
        of finite automata. Vol. 114. Defense Technical Information Center, 1971.
        """

        # The basic idea of this algorithm is that we repeatedly
        # merge states that would be equivalent if the two start
        # states were. This starts by merging the two start states,
        # and whenever we merge two states merging all pairs of
        # states that are reachable by following the same character
        # from that point.
        #
        # Whenever we merge two states, we check if one of them
        # is accepting and the other non-accepting. If so, we have
        # obtained a contradiction and have made a bad merge, so
        # the two start states must not have been equivalent in the
        # first place and we return False.
        #
        # If the languages matched are different then some string
        # is contained in one but not the other. By looking at
        # the pairs of states visited by traversing the string in
        # each automaton in parallel, we eventually come to a pair
        # of states that would have to be merged by this algorithm
        # where one is accepting and the other is not. Thus this
        # algorithm always returns False as a result of a bad merge
        # if the two languages are not the same.
        #
        # If we successfully complete all merges without a contradiction
        # we can thus safely return True.

        # We maintain a union/find table for tracking merges of states.
        table = {}

        def find(s):
            trail = [s]
            while trail[-1] in table and table[trail[-1]] != trail[-1]:
                trail.append(table[trail[-1]])

            for t in trail:
                table[t] = trail[-1]

            return trail[-1]

        def union(s, t):
            s = find(s)
            t = find(t)
            table[s] = t

        alphabet = sorted(set(self.alphabet) | set(other.alphabet))

        queue = deque([((self.start, other.start))])
        while queue:
            self_state, other_state = queue.popleft()

            # We use a DFA/state pair for keys because the same value
            # may represent a different state in each DFA.
            self_key = (self, self_state)
            other_key = (other, other_state)

            # We have already merged these, no need to remerge.
            if find(self_key) == find(other_key):
                continue

            # We have found a contradiction, therefore the two DFAs must
            # not be equivalent.
            if self.is_accepting(self_state) != other.is_accepting(other_state):
                return False

            # Merge the two states
            union(self_key, other_key)

            # And also queue any logical consequences of merging those
            # two states for merging.
            for c in alphabet:
                queue.append(
                    (self.transition(self_state, c), other.transition(other_state, c))
                )
        return True


DEAD = "DEAD"


class ConcreteDFA(DFA):
    """A concrete representation of a DFA in terms of an explicit list
    of states."""

    def __init__(self, transitions, accepting, start=0):
        """
        * ``transitions`` is a list where transitions[i] represents the
          valid transitions out of state ``i``. Elements may be either dicts
          (in which case they map characters to other states) or lists. If they
          are a list they may contain tuples of length 2 or 3. A tuple ``(c, j)``
          indicates that this state transitions to state ``j`` given ``c``. A
          tuple ``(u, v, j)`` indicates this state transitions to state ``j``
          given any ``c`` with ``u <= c <= v``.
        * ``accepting`` is a set containing the integer labels of accepting
          states.
        * ``start`` is the integer label of the starting state.
        """
        super().__init__()
        self.__start = start
        self.__accepting = accepting
        self.__transitions = list(transitions)

    def __repr__(self):
        transitions = []
        # Particularly for including in source code it's nice to have the more
        # compact repr, so where possible we convert to the tuple based representation
        # which can represent ranges more compactly.
        for i in range(len(self.__transitions)):
            table = []
            for c, j in self.transitions(i):
                if not table or j != table[-1][-1] or c != table[-1][1] + 1:
                    table.append([c, c, j])
                else:
                    table[-1][1] = c
            transitions.append([(u, j) if u == v else (u, v, j) for u, v, j in table])

        start = "" if self.__start == 0 else f", start={self.__start!r}"
        return f"ConcreteDFA({transitions!r}, {self.__accepting!r}{start})"

    @property
    def start(self):
        return self.__start

    def is_accepting(self, i):
        return i in self.__accepting

    def transition(self, state, char):
        """Returns the state that i transitions to on reading
        character c from a string."""
        if state == DEAD:
            return DEAD

        table = self.__transitions[state]

        # Given long transition tables we convert them to
        # dictionaries for more efficient lookup.
        if not isinstance(table, dict) and len(table) >= 5:
            new_table = {}
            for t in table:
                if len(t) == 2:
                    new_table[t[0]] = t[1]
                else:
                    u, v, j = t
                    for c in range(u, v + 1):
                        new_table[c] = j
            self.__transitions[state] = new_table
            table = new_table

        if isinstance(table, dict):
            try:
                return self.__transitions[state][char]
            except KeyError:
                return DEAD
        else:
            for t in table:
                if len(t) == 2:
                    if t[0] == char:
                        return t[1]
                else:
                    u, v, j = t
                    if u <= char <= v:
                        return j
            return DEAD

    def raw_transitions(self, i):
        if i == DEAD:
            return
        transitions = self.__transitions[i]
        if isinstance(transitions, dict):
            yield from sorted(transitions.items())
        else:
            for t in transitions:
                if len(t) == 2:
                    yield t
                else:
                    u, v, j = t
                    for c in range(u, v + 1):
                        yield c, j
