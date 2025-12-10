from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine

class RewritingSystem:
    '''
    A class implementing rewriting systems for `FpGroup`s.

    References
    ==========
    .. [1] Epstein, D., Holt, D. and Rees, S. (1991).
           The use of Knuth-Bendix methods to solve the word problem in automatic groups.
           Journal of Symbolic Computation, 12(4-5), pp.397-414.

    .. [2] GAP's Manual on its KBMAG package
           https://www.gap-system.org/Manuals/pkg/kbmag-1.5.3/doc/manual.pdf

    '''
    def __init__(self, group):
        self.group = group
        self.alphabet = group.generators
        self._is_confluent = None

        # these values are taken from [2]
        self.maxeqns = 32767 # max rules
        self.tidyint = 100 # rules before tidying

        # _max_exceeded is True if maxeqns is exceeded
        # at any point
        self._max_exceeded = False

        # Reduction automaton
        self.reduction_automaton = None
        self._new_rules = {}

        # dictionary of reductions
        self.rules = {}
        self.rules_cache = deque([], 50)
        self._init_rules()


        # All the transition symbols in the automaton
        generators = list(self.alphabet)
        generators += [gen**-1 for gen in generators]
        # Create a finite state machine as an instance of the StateMachine object
        self.reduction_automaton = StateMachine('Reduction automaton for '+ repr(self.group), generators)
        self.construct_automaton()

    def set_max(self, n):
        '''
        Set the maximum number of rules that can be defined

        '''
        if n > self.maxeqns:
            self._max_exceeded = False
        self.maxeqns = n
        return

    @property
    def is_confluent(self):
        '''
        Return `True` if the system is confluent

        '''
        if self._is_confluent is None:
            self._is_confluent = self._check_confluence()
        return self._is_confluent

    def _init_rules(self):
        identity = self.group.free_group.identity
        for r in self.group.relators:
            self.add_rule(r, identity)
        self._remove_redundancies()
        return

    def _add_rule(self, r1, r2):
        '''
        Add the rule r1 -> r2 with no checking or further
        deductions

        '''
        if len(self.rules) + 1 > self.maxeqns:
            self._is_confluent = self._check_confluence()
            self._max_exceeded = True
            raise RuntimeError("Too many rules were defined.")
        self.rules[r1] = r2
        # Add the newly added rule to the `new_rules` dictionary.
        if self.reduction_automaton:
            self._new_rules[r1] = r2

    def add_rule(self, w1, w2, check=False):
        new_keys = set()

        if w1 == w2:
            return new_keys

        if w1 < w2:
            w1, w2 = w2, w1

        if (w1, w2) in self.rules_cache:
            return new_keys
        self.rules_cache.append((w1, w2))

        s1, s2 = w1, w2

        # The following is the equivalent of checking
        # s1 for overlaps with the implicit reductions
        # {g*g**-1 -> <identity>} and {g**-1*g -> <identity>}
        # for any generator g without installing the
        # redundant rules that would result from processing
        # the overlaps. See [1], Section 3 for details.

        if len(s1) - len(s2) < 3:
            if s1 not in self.rules:
                new_keys.add(s1)
                if not check:
                    self._add_rule(s1, s2)
            if s2**-1 > s1**-1 and s2**-1 not in self.rules:
                new_keys.add(s2**-1)
                if not check:
                    self._add_rule(s2**-1, s1**-1)

        # overlaps on the right
        while len(s1) - len(s2) > -1:
            g = s1[len(s1)-1]
            s1 = s1.subword(0, len(s1)-1)
            s2 = s2*g**-1
            if len(s1) - len(s2) < 0:
                if s2 not in self.rules:
                    if not check:
                        self._add_rule(s2, s1)
                    new_keys.add(s2)
            elif len(s1) - len(s2) < 3:
                new = self.add_rule(s1, s2, check)
                new_keys.update(new)

        # overlaps on the left
        while len(w1) - len(w2) > -1:
            g = w1[0]
            w1 = w1.subword(1, len(w1))
            w2 = g**-1*w2
            if len(w1) - len(w2) < 0:
                if w2 not in self.rules:
                    if not check:
                        self._add_rule(w2, w1)
                    new_keys.add(w2)
            elif len(w1) - len(w2) < 3:
                new = self.add_rule(w1, w2, check)
                new_keys.update(new)

        return new_keys

    def _remove_redundancies(self, changes=False):
        '''
        Reduce left- and right-hand sides of reduction rules
        and remove redundant equations (i.e. those for which
        lhs == rhs). If `changes` is `True`, return a set
        containing the removed keys and a set containing the
        added keys

        '''
        removed = set()
        added = set()
        rules = self.rules.copy()
        for r in rules:
            v = self.reduce(r, exclude=r)
            w = self.reduce(rules[r])
            if v != r:
                del self.rules[r]
                removed.add(r)
                if v > w:
                    added.add(v)
                    self.rules[v] = w
                elif v < w:
                    added.add(w)
                    self.rules[w] = v
            else:
                self.rules[v] = w
        if changes:
            return removed, added
        return

    def make_confluent(self, check=False):
        '''
        Try to make the system confluent using the Knuth-Bendix
        completion algorithm

        '''
        if self._max_exceeded:
            return self._is_confluent
        lhs = list(self.rules.keys())

        def _overlaps(r1, r2):
            len1 = len(r1)
            len2 = len(r2)
            result = []
            for j in range(1, len1 + len2):
                if (r1.subword(len1 - j, len1 + len2 - j, strict=False)
                       == r2.subword(j - len1, j, strict=False)):
                    a = r1.subword(0, len1-j, strict=False)
                    a = a*r2.subword(0, j-len1, strict=False)
                    b = r2.subword(j-len1, j, strict=False)
                    c = r2.subword(j, len2, strict=False)
                    c = c*r1.subword(len1 + len2 - j, len1, strict=False)
                    result.append(a*b*c)
            return result

        def _process_overlap(w, r1, r2, check):
                s = w.eliminate_word(r1, self.rules[r1])
                s = self.reduce(s)
                t = w.eliminate_word(r2, self.rules[r2])
                t = self.reduce(t)
                if s != t:
                    if check:
                        # system not confluent
                        return [0]
                    try:
                        new_keys = self.add_rule(t, s, check)
                        return new_keys
                    except RuntimeError:
                        return False
                return

        added = 0
        i = 0
        while i < len(lhs):
            r1 = lhs[i]
            i += 1
            # j could be i+1 to not
            # check each pair twice but lhs
            # is extended in the loop and the new
            # elements have to be checked with the
            # preceding ones. there is probably a better way
            # to handle this
            j = 0
            while j < len(lhs):
                r2 = lhs[j]
                j += 1
                if r1 == r2:
                    continue
                overlaps = _overlaps(r1, r2)
                overlaps.extend(_overlaps(r1**-1, r2))
                if not overlaps:
                    continue
                for w in overlaps:
                    new_keys = _process_overlap(w, r1, r2, check)
                    if new_keys:
                        if check:
                            return False
                        lhs.extend(new_keys)
                        added += len(new_keys)
                    elif new_keys == False:
                        # too many rules were added so the process
                        # couldn't complete
                        return self._is_confluent

                if added > self.tidyint and not check:
                    # tidy up
                    r, a = self._remove_redundancies(changes=True)
                    added = 0
                    if r:
                        # reset i since some elements were removed
                        i = min(lhs.index(s) for s in r)
                    lhs = [l for l in lhs if l not in r]
                    lhs.extend(a)
                    if r1 in r:
                        # r1 was removed as redundant
                        break

        self._is_confluent = True
        if not check:
            self._remove_redundancies()
        return True

    def _check_confluence(self):
        return self.make_confluent(check=True)

    def reduce(self, word, exclude=None):
        '''
        Apply reduction rules to `word` excluding the reduction rule
        for the lhs equal to `exclude`

        '''
        rules = {r: self.rules[r] for r in self.rules if r != exclude}
        # the following is essentially `eliminate_words()` code from the
        # `FreeGroupElement` class, the only difference being the first
        # "if" statement
        again = True
        new = word
        while again:
            again = False
            for r in rules:
                prev = new
                if rules[r]**-1 > r**-1:
                    new = new.eliminate_word(r, rules[r], _all=True, inverse=False)
                else:
                    new = new.eliminate_word(r, rules[r], _all=True)
                if new != prev:
                    again = True
        return new

    def _compute_inverse_rules(self, rules):
        '''
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        Arguments:
            rules (dictionary): Rules for which the inverse rules are to computed.

        Returns:
            Dictionary of inverse_rules.

        '''
        inverse_rules = {}
        for r in rules:
            rule_key_inverse = r**-1
            rule_value_inverse = (rules[r])**-1
            if (rule_value_inverse < rule_key_inverse):
                inverse_rules[rule_key_inverse] = rule_value_inverse
            else:
                inverse_rules[rule_value_inverse] = rule_key_inverse
        return inverse_rules

    def construct_automaton(self):
        '''
        Construct the automaton based on the set of reduction rules of the system.

        Automata Design:
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        The complete left hand side of the rules are the dead states of the automaton.

        '''
        self._add_to_automaton(self.rules)

    def _add_to_automaton(self, rules):
        '''
        Add new states and transitions to the automaton.

        Summary:
        States corresponding to the new rules added to the system are computed and added to the automaton.
        Transitions in the previously added states are also modified if necessary.

        Arguments:
            rules (dictionary) -- Dictionary of the newly added rules.

        '''
        # Automaton variables
        automaton_alphabet = []
        proper_prefixes = {}

        # compute the inverses of all the new rules added
        all_rules = rules
        inverse_rules = self._compute_inverse_rules(all_rules)
        all_rules.update(inverse_rules)

        # Keep track of the accept_states.
        accept_states = []

        for rule in all_rules:
            # The symbols present in the new rules are the symbols to be verified at each state.
            # computes the automaton_alphabet, as the transitions solely depend upon the new states.
            automaton_alphabet += rule.letter_form_elm
            # Compute the proper prefixes for every rule.
            proper_prefixes[rule] = []
            letter_word_array = list(rule.letter_form_elm)
            len_letter_word_array = len(letter_word_array)
            for i in range (1, len_letter_word_array):
                letter_word_array[i] = letter_word_array[i-1]*letter_word_array[i]
                # Add accept states.
                elem = letter_word_array[i-1]
                if elem not in self.reduction_automaton.states:
                    self.reduction_automaton.add_state(elem, state_type='a')
                    accept_states.append(elem)
            proper_prefixes[rule] = letter_word_array
            # Check for overlaps between dead and accept states.
            if rule in accept_states:
                self.reduction_automaton.states[rule].state_type = 'd'
                self.reduction_automaton.states[rule].rh_rule = all_rules[rule]
                accept_states.remove(rule)
            # Add dead states
            if rule not in self.reduction_automaton.states:
                self.reduction_automaton.add_state(rule, state_type='d', rh_rule=all_rules[rule])

        automaton_alphabet = set(automaton_alphabet)

        # Add new transitions for every state.
        for state in self.reduction_automaton.states:
            current_state_name = state
            current_state_type = self.reduction_automaton.states[state].state_type
            # Transitions will be modified only when suffixes of the current_state
            # belongs to the proper_prefixes of the new rules.
            # The rest are ignored if they cannot lead to a dead state after a finite number of transisitons.
            if current_state_type == 's':
                for letter in automaton_alphabet:
                    if letter in self.reduction_automaton.states:
                        self.reduction_automaton.states[state].add_transition(letter, letter)
                    else:
                        self.reduction_automaton.states[state].add_transition(letter, current_state_name)
            elif current_state_type == 'a':
                # Check if the transition to any new state in possible.
                for letter in automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

        # Add transitions for new states. All symbols used in the automaton are considered here.
        # Ignore this if `reduction_automaton.automaton_alphabet` = `automaton_alphabet`.
        if len(self.reduction_automaton.automaton_alphabet) != len(automaton_alphabet):
            for state in accept_states:
                current_state_name = state
                for letter in self.reduction_automaton.automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

    def reduce_using_automaton(self, word):
        '''
        Reduce a word using an automaton.

        Summary:
        All the symbols of the word are stored in an array and are given as the input to the automaton.
        If the automaton reaches a dead state that subword is replaced and the automaton is run from the beginning.
        The complete word has to be replaced when the word is read and the automaton reaches a dead state.
        So, this process is repeated until the word is read completely and the automaton reaches the accept state.

        Arguments:
            word (instance of FreeGroupElement) -- Word that needs to be reduced.

        '''
        # Modify the automaton if new rules are found.
        if self._new_rules:
            self._add_to_automaton(self._new_rules)
            self._new_rules = {}

        flag = 1
        while flag:
            flag = 0
            current_state = self.reduction_automaton.states['start']
            for i, s in enumerate(word.letter_form_elm):
                next_state_name = current_state.transitions[s]
                next_state = self.reduction_automaton.states[next_state_name]
                if next_state.state_type == 'd':
                    subst = next_state.rh_rule
                    word = word.substituted_word(i - len(next_state_name) + 1, i+1, subst)
                    flag = 1
                    break
                current_state = next_state
        return word
