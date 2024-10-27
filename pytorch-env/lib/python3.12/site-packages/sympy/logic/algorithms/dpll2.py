"""Implementation of DPLL algorithm

Features:
  - Clause learning
  - Watch literal scheme
  - VSIDS heuristic

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
"""

from collections import defaultdict
from heapq import heappush, heappop

from sympy.core.sorting import ordered
from sympy.assumptions.cnf import EncodedCNF

from sympy.logic.algorithms.lra_theory import LRASolver


def dpll_satisfiable(expr, all_models=False, use_lra_theory=False):
    """
    Check satisfiability of a propositional sentence.
    It returns a model rather than True when it succeeds.
    Returns a generator of all models if all_models is True.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.algorithms.dpll2 import dpll_satisfiable
    >>> dpll_satisfiable(A & ~B)
    {A: True, B: False}
    >>> dpll_satisfiable(A & ~A)
    False

    """
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    # Return UNSAT when False (encoded as 0) is present in the CNF
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False

    if use_lra_theory:
        lra, immediate_conflicts = LRASolver.from_encoded_cnf(expr)
    else:
        lra = None
        immediate_conflicts = []
    solver = SATSolver(expr.data + immediate_conflicts, expr.variables, set(), expr.symbols, lra_theory=lra)
    models = solver._find_model()

    if all_models:
        return _all_models(models)

    try:
        return next(models)
    except StopIteration:
        return False

    # Uncomment to confirm the solution is valid (hitting set for the clauses)
    #else:
        #for cls in clauses_int_repr:
            #assert solver.var_settings.intersection(cls)


def _all_models(models):
    satisfiable = False
    try:
        while True:
            yield next(models)
            satisfiable = True
    except StopIteration:
        if not satisfiable:
            yield False


class SATSolver:
    """
    Class for representing a SAT solver capable of
     finding a model to a boolean theory in conjunctive
     normal form.
    """

    def __init__(self, clauses, variables, var_settings, symbols=None,
                heuristic='vsids', clause_learning='none', INTERVAL=500,
                 lra_theory = None):

        self.var_settings = var_settings
        self.heuristic = heuristic
        self.is_unsatisfied = False
        self._unit_prop_queue = []
        self.update_functions = []
        self.INTERVAL = INTERVAL

        if symbols is None:
            self.symbols = list(ordered(variables))
        else:
            self.symbols = symbols

        self._initialize_variables(variables)
        self._initialize_clauses(clauses)

        if 'vsids' == heuristic:
            self._vsids_init()
            self.heur_calculate = self._vsids_calculate
            self.heur_lit_assigned = self._vsids_lit_assigned
            self.heur_lit_unset = self._vsids_lit_unset
            self.heur_clause_added = self._vsids_clause_added

            # Note: Uncomment this if/when clause learning is enabled
            #self.update_functions.append(self._vsids_decay)

        else:
            raise NotImplementedError

        if 'simple' == clause_learning:
            self.add_learned_clause = self._simple_add_learned_clause
            self.compute_conflict = self._simple_compute_conflict
            self.update_functions.append(self._simple_clean_clauses)
        elif 'none' == clause_learning:
            self.add_learned_clause = lambda x: None
            self.compute_conflict = lambda: None
        else:
            raise NotImplementedError

        # Create the base level
        self.levels = [Level(0)]
        self._current_level.varsettings = var_settings

        # Keep stats
        self.num_decisions = 0
        self.num_learned_clauses = 0
        self.original_num_clauses = len(self.clauses)

        self.lra = lra_theory

    def _initialize_variables(self, variables):
        """Set up the variable data structures needed."""
        self.sentinels = defaultdict(set)
        self.occurrence_count = defaultdict(int)
        self.variable_set = [False] * (len(variables) + 1)

    def _initialize_clauses(self, clauses):
        """Set up the clause data structures needed.

        For each clause, the following changes are made:
        - Unit clauses are queued for propagation right away.
        - Non-unit clauses have their first and last literals set as sentinels.
        - The number of clauses a literal appears in is computed.
        """
        self.clauses = [list(clause) for clause in clauses]

        for i, clause in enumerate(self.clauses):

            # Handle the unit clauses
            if 1 == len(clause):
                self._unit_prop_queue.append(clause[0])
                continue

            self.sentinels[clause[0]].add(i)
            self.sentinels[clause[-1]].add(i)

            for lit in clause:
                self.occurrence_count[lit] += 1

    def _find_model(self):
        """
        Main DPLL loop. Returns a generator of models.

        Variables are chosen successively, and assigned to be either
        True or False. If a solution is not found with this setting,
        the opposite is chosen and the search continues. The solver
        halts when every variable has a setting.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> list(l._find_model())
        [{1: True, 2: False, 3: False}, {1: True, 2: True, 3: True}]

        >>> from sympy.abc import A, B, C
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set(), [A, B, C])
        >>> list(l._find_model())
        [{A: True, B: False, C: False}, {A: True, B: True, C: True}]

        """

        # We use this variable to keep track of if we should flip a
        #  variable setting in successive rounds
        flip_var = False

        # Check if unit prop says the theory is unsat right off the bat
        self._simplify()
        if self.is_unsatisfied:
            return

        # While the theory still has clauses remaining
        while True:
            # Perform cleanup / fixup at regular intervals
            if self.num_decisions % self.INTERVAL == 0:
                for func in self.update_functions:
                    func()

            if flip_var:
                # We have just backtracked and we are trying to opposite literal
                flip_var = False
                lit = self._current_level.decision

            else:
                # Pick a literal to set
                lit = self.heur_calculate()
                self.num_decisions += 1

                # Stopping condition for a satisfying theory
                if 0 == lit:

                    # check if assignment satisfies lra theory
                    if self.lra:
                        for enc_var in self.var_settings:
                            res = self.lra.assert_lit(enc_var)
                            if res is not None:
                                break
                        res = self.lra.check()
                        self.lra.reset_bounds()
                    else:
                        res = None
                    if res is None or res[0]:
                        yield {self.symbols[abs(lit) - 1]:
                                    lit > 0 for lit in self.var_settings}
                    else:
                        self._simple_add_learned_clause(res[1])

                    while self._current_level.flipped:
                        self._undo()
                    if len(self.levels) == 1:
                        return
                    flip_lit = -self._current_level.decision
                    self._undo()
                    self.levels.append(Level(flip_lit, flipped=True))
                    flip_var = True
                    continue

                # Start the new decision level
                self.levels.append(Level(lit))

            # Assign the literal, updating the clauses it satisfies
            self._assign_literal(lit)

            # _simplify the theory
            self._simplify()

            # Check if we've made the theory unsat
            if self.is_unsatisfied:

                self.is_unsatisfied = False

                # We unroll all of the decisions until we can flip a literal
                while self._current_level.flipped:
                    self._undo()

                    # If we've unrolled all the way, the theory is unsat
                    if 1 == len(self.levels):
                        return

                # Detect and add a learned clause
                self.add_learned_clause(self.compute_conflict())

                # Try the opposite setting of the most recent decision
                flip_lit = -self._current_level.decision
                self._undo()
                self.levels.append(Level(flip_lit, flipped=True))
                flip_var = True

    ########################
    #    Helper Methods    #
    ########################
    @property
    def _current_level(self):
        """The current decision level data structure

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {2}], {1, 2}, set())
        >>> next(l._find_model())
        {1: True, 2: True}
        >>> l._current_level.decision
        0
        >>> l._current_level.flipped
        False
        >>> l._current_level.var_settings
        {1, 2}

        """
        return self.levels[-1]

    def _clause_sat(self, cls):
        """Check if a clause is satisfied by the current variable setting.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {-1}], {1}, set())
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l._clause_sat(0)
        False
        >>> l._clause_sat(1)
        True

        """
        for lit in self.clauses[cls]:
            if lit in self.var_settings:
                return True
        return False

    def _is_sentinel(self, lit, cls):
        """Check if a literal is a sentinel of a given clause.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._is_sentinel(2, 3)
        True
        >>> l._is_sentinel(-3, 1)
        False

        """
        return cls in self.sentinels[lit]

    def _assign_literal(self, lit):
        """Make a literal assignment.

        The literal assignment must be recorded as part of the current
        decision level. Additionally, if the literal is marked as a
        sentinel of any clause, then a new sentinel must be chosen. If
        this is not possible, then unit propagation is triggered and
        another literal is added to the queue to be set in the future.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l.var_settings
        {-3, -2, 1}

        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l._assign_literal(-1)
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l.var_settings
        {-1}

        """
        self.var_settings.add(lit)
        self._current_level.var_settings.add(lit)
        self.variable_set[abs(lit)] = True
        self.heur_lit_assigned(lit)

        sentinel_list = list(self.sentinels[-lit])

        for cls in sentinel_list:
            if not self._clause_sat(cls):
                other_sentinel = None
                for newlit in self.clauses[cls]:
                    if newlit != -lit:
                        if self._is_sentinel(newlit, cls):
                            other_sentinel = newlit
                        elif not self.variable_set[abs(newlit)]:
                            self.sentinels[-lit].remove(cls)
                            self.sentinels[newlit].add(cls)
                            other_sentinel = None
                            break

                # Check if no sentinel update exists
                if other_sentinel:
                    self._unit_prop_queue.append(other_sentinel)

    def _undo(self):
        """
        _undo the changes of the most recent decision level.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (-3, {-3, -2}, False)
        >>> l._undo()
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (0, {1}, False)

        """
        # Undo the variable settings
        for lit in self._current_level.var_settings:
            self.var_settings.remove(lit)
            self.heur_lit_unset(lit)
            self.variable_set[abs(lit)] = False

        # Pop the level off the stack
        self.levels.pop()

    #########################
    #      Propagation      #
    #########################
    """
    Propagation methods should attempt to soundly simplify the boolean
      theory, and return True if any simplification occurred and False
      otherwise.
    """
    def _simplify(self):
        """Iterate over the various forms of propagation to simplify the theory.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l.variable_set
        [False, False, False, False]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}

        >>> l._simplify()

        >>> l.variable_set
        [False, True, False, False]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, -1: set(), 2: {0, 3},
        ...3: {2, 4}}

        """
        changed = True
        while changed:
            changed = False
            changed |= self._unit_prop()
            changed |= self._pure_literal()

    def _unit_prop(self):
        """Perform unit propagation on the current theory."""
        result = len(self._unit_prop_queue) > 0
        while self._unit_prop_queue:
            next_lit = self._unit_prop_queue.pop()
            if -next_lit in self.var_settings:
                self.is_unsatisfied = True
                self._unit_prop_queue = []
                return False
            else:
                self._assign_literal(next_lit)

        return result

    def _pure_literal(self):
        """Look for pure literals and assign them when found."""
        return False

    #########################
    #      Heuristics       #
    #########################
    def _vsids_init(self):
        """Initialize the data structures needed for the VSIDS heuristic."""
        self.lit_heap = []
        self.lit_scores = {}

        for var in range(1, len(self.variable_set)):
            self.lit_scores[var] = float(-self.occurrence_count[var])
            self.lit_scores[-var] = float(-self.occurrence_count[-var])
            heappush(self.lit_heap, (self.lit_scores[var], var))
            heappush(self.lit_heap, (self.lit_scores[-var], -var))

    def _vsids_decay(self):
        """Decay the VSIDS scores for every literal.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.lit_scores
        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}

        >>> l._vsids_decay()

        >>> l.lit_scores
        {-3: -1.0, -2: -1.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -1.0}

        """
        # We divide every literal score by 2 for a decay factor
        #  Note: This doesn't change the heap property
        for lit in self.lit_scores.keys():
            self.lit_scores[lit] /= 2.0

    def _vsids_calculate(self):
        """
            VSIDS Heuristic Calculation

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.lit_heap
        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]

        >>> l._vsids_calculate()
        -3

        >>> l.lit_heap
        [(-2.0, -2), (-2.0, 2), (0.0, -1), (0.0, 1), (-2.0, 3)]

        """
        if len(self.lit_heap) == 0:
            return 0

        # Clean out the front of the heap as long the variables are set
        while self.variable_set[abs(self.lit_heap[0][1])]:
            heappop(self.lit_heap)
            if len(self.lit_heap) == 0:
                return 0

        return heappop(self.lit_heap)[1]

    def _vsids_lit_assigned(self, lit):
        """Handle the assignment of a literal for the VSIDS heuristic."""
        pass

    def _vsids_lit_unset(self, lit):
        """Handle the unsetting of a literal for the VSIDS heuristic.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l.lit_heap
        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]

        >>> l._vsids_lit_unset(2)

        >>> l.lit_heap
        [(-2.0, -3), (-2.0, -2), (-2.0, -2), (-2.0, 2), (-2.0, 3), (0.0, -1),
        ...(-2.0, 2), (0.0, 1)]

        """
        var = abs(lit)
        heappush(self.lit_heap, (self.lit_scores[var], var))
        heappush(self.lit_heap, (self.lit_scores[-var], -var))

    def _vsids_clause_added(self, cls):
        """Handle the addition of a new clause for the VSIDS heuristic.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.num_learned_clauses
        0
        >>> l.lit_scores
        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}

        >>> l._vsids_clause_added({2, -3})

        >>> l.num_learned_clauses
        1
        >>> l.lit_scores
        {-3: -1.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -2.0}

        """
        self.num_learned_clauses += 1
        for lit in cls:
            self.lit_scores[lit] += 1

    ########################
    #   Clause Learning    #
    ########################
    def _simple_add_learned_clause(self, cls):
        """Add a new clause to the theory.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.num_learned_clauses
        0
        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}

        >>> l._simple_add_learned_clause([3])

        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2], [3]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4, 5}}

        """
        cls_num = len(self.clauses)
        self.clauses.append(cls)

        for lit in cls:
            self.occurrence_count[lit] += 1

        self.sentinels[cls[0]].add(cls_num)
        self.sentinels[cls[-1]].add(cls_num)

        self.heur_clause_added(cls)

    def _simple_compute_conflict(self):
        """ Build a clause representing the fact that at least one decision made
        so far is wrong.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._simple_compute_conflict()
        [3]

        """
        return [-(level.decision) for level in self.levels[1:]]

    def _simple_clean_clauses(self):
        """Clean up learned clauses."""
        pass


class Level:
    """
    Represents a single level in the DPLL algorithm, and contains
    enough information for a sound backtracking procedure.
    """

    def __init__(self, decision, flipped=False):
        self.decision = decision
        self.var_settings = set()
        self.flipped = flipped
