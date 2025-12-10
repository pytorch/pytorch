"""Implements "A Fast Linear-Arithmetic Solver for DPLL(T)"

The LRASolver class defined in this file can be used
in conjunction with a SAT solver to check the
satisfiability of formulas involving inequalities.

Here's an example of how that would work:

    Suppose you want to check the satisfiability of
    the following formula:

    >>> from sympy.core.relational import Eq
    >>> from sympy.abc import x, y
    >>> f = ((x > 0) | (x < 0)) & (Eq(x, 0) | Eq(y, 1)) & (~Eq(y, 1) | Eq(1, 2))

    First a preprocessing step should be done on f. During preprocessing,
    f should be checked for any predicates such as `Q.prime` that can't be
    handled. Also unequality like `~Eq(y, 1)` should be split.

    I should mention that the paper says to split both equalities and
    unequality, but this implementation only requires that unequality
    be split.

    >>> f = ((x > 0) | (x < 0)) & (Eq(x, 0) | Eq(y, 1)) & ((y < 1) | (y > 1) | Eq(1, 2))

    Then an LRASolver instance needs to be initialized with this formula.

    >>> from sympy.assumptions.cnf import CNF, EncodedCNF
    >>> from sympy.assumptions.ask import Q
    >>> from sympy.logic.algorithms.lra_theory import LRASolver
    >>> cnf = CNF.from_prop(f)
    >>> enc = EncodedCNF()
    >>> enc.add_from_cnf(cnf)
    >>> lra, conflicts = LRASolver.from_encoded_cnf(enc)

    Any immediate one-lital conflicts clauses will be detected here.
    In this example, `~Eq(1, 2)` is one such conflict clause. We'll
    want to add it to `f` so that the SAT solver is forced to
    assign Eq(1, 2) to False.

    >>> f = f & ~Eq(1, 2)

    Now that the one-literal conflict clauses have been added
    and an lra object has been initialized, we can pass `f`
    to a SAT solver. The SAT solver will give us a satisfying
    assignment such as:

    (1 = 2): False
    (y = 1): True
    (y < 1): True
    (y > 1): True
    (x = 0): True
    (x < 0): True
    (x > 0): True

    Next you would pass this assignment to the LRASolver
    which will be able to determine that this particular
    assignment is satisfiable or not.

    Note that since EncodedCNF is inherently non-deterministic,
    the int each predicate is encoded as is not consistent. As a
    result, the code below likely does not reflect the assignment
    given above.

    >>> lra.assert_lit(-1) #doctest: +SKIP
    >>> lra.assert_lit(2) #doctest: +SKIP
    >>> lra.assert_lit(3) #doctest: +SKIP
    >>> lra.assert_lit(4) #doctest: +SKIP
    >>> lra.assert_lit(5) #doctest: +SKIP
    >>> lra.assert_lit(6) #doctest: +SKIP
    >>> lra.assert_lit(7) #doctest: +SKIP
    >>> is_sat, conflict_or_assignment = lra.check()

    As the particular assignment suggested is not satisfiable,
    the LRASolver will return unsat and a conflict clause when
    given that assignment. The conflict clause will always be
    minimal, but there can be multiple minimal conflict clauses.
    One possible conflict clause could be `~(x < 0) | ~(x > 0)`.

    We would then add whatever conflict clause is given to
    `f` to prevent the SAT solver from coming up with an
    assignment with the same conflicting literals. In this case,
    the conflict clause `~(x < 0) | ~(x > 0)` would prevent
    any assignment where both (x < 0) and (x > 0) were both
    true.

    The SAT solver would then find another assignment
    and we would check that assignment with the LRASolver
    and so on. Eventually either a satisfying assignment
    that the SAT solver and LRASolver agreed on would be found
    or enough conflict clauses would be added so that the
    boolean formula was unsatisfiable.


This implementation is based on [1]_, which includes a
detailed explanation of the algorithm and pseudocode
for the most important functions.

[1]_ also explains how backtracking and theory propagation
could be implemented to speed up the current implementation,
but these are not currently implemented.

TODO:
 - Handle non-rational real numbers
 - Handle positive and negative infinity
 - Implement backtracking and theory proposition
 - Simplify matrix by removing unused variables using Gaussian elimination

References
==========

.. [1] Dutertre, B., de Moura, L.:
       A Fast Linear-Arithmetic Solver for DPLL(T)
       https://link.springer.com/chapter/10.1007/11817963_11
"""
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.matrices.dense import eye
from sympy.assumptions import Predicate
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.ask import Q
from sympy.core import Dummy
from sympy.core.mul import Mul
from sympy.core.add import Add
from sympy.core.relational import Eq, Ne
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy.core.numbers import Rational, oo
from sympy.matrices.dense import Matrix

class UnhandledInput(Exception):
    """
    Raised while creating an LRASolver if non-linearity
    or non-rational numbers are present.
    """

# predicates that LRASolver understands and makes use of
ALLOWED_PRED = {Q.eq, Q.gt, Q.lt, Q.le, Q.ge}

# if true ~Q.gt(x, y) implies Q.le(x, y)
HANDLE_NEGATION = True

class LRASolver():
    """
    Linear Arithmetic Solver for DPLL(T) implemented with an algorithm based on
    the Dual Simplex method. Uses Bland's pivoting rule to avoid cycling.

    References
    ==========

    .. [1] Dutertre, B., de Moura, L.:
           A Fast Linear-Arithmetic Solver for DPLL(T)
           https://link.springer.com/chapter/10.1007/11817963_11
    """

    def __init__(self, A, slack_variables, nonslack_variables, enc_to_boundary, s_subs, testing_mode):
        """
        Use the "from_encoded_cnf" method to create a new LRASolver.
        """
        self.run_checks = testing_mode
        self.s_subs = s_subs  # used only for test_lra_theory.test_random_problems

        if any(not isinstance(a, Rational) for a in A):
            raise UnhandledInput("Non-rational numbers are not handled")
        if any(not isinstance(b.bound, Rational) for b in enc_to_boundary.values()):
            raise UnhandledInput("Non-rational numbers are not handled")
        m, n = len(slack_variables), len(slack_variables)+len(nonslack_variables)
        if m != 0:
            assert A.shape == (m, n)
        if self.run_checks:
            assert A[:, n-m:] == -eye(m)

        self.enc_to_boundary = enc_to_boundary  # mapping of int to Boundary objects
        self.boundary_to_enc = {value: key for key, value in enc_to_boundary.items()}
        self.A = A
        self.slack = slack_variables
        self.nonslack = nonslack_variables
        self.all_var = nonslack_variables + slack_variables

        self.slack_set = set(slack_variables)

        self.is_sat = True  # While True, all constraints asserted so far are satisfiable
        self.result = None  # always one of: (True, assignment), (False, conflict clause), None

    @staticmethod
    def from_encoded_cnf(encoded_cnf, testing_mode=False):
        """
        Creates an LRASolver from an EncodedCNF object
        and a list of conflict clauses for propositions
        that can be simplified to True or False.

        Parameters
        ==========

        encoded_cnf : EncodedCNF

        testing_mode : bool
            Setting testing_mode to True enables some slow assert statements
            and sorting to reduce nonterministic behavior.

        Returns
        =======

        (lra, conflicts)

        lra : LRASolver

        conflicts : list
            Contains a one-literal conflict clause for each proposition
            that can be simplified to True or False.

        Example
        =======

        >>> from sympy.core.relational import Eq
        >>> from sympy.assumptions.cnf import CNF, EncodedCNF
        >>> from sympy.assumptions.ask import Q
        >>> from sympy.logic.algorithms.lra_theory import LRASolver
        >>> from sympy.abc import x, y, z
        >>> phi = (x >= 0) & ((x + y <= 2) | (x + 2 * y - z >= 6))
        >>> phi = phi & (Eq(x + y, 2) | (x + 2 * y - z > 4))
        >>> phi = phi & Q.gt(2, 1)
        >>> cnf = CNF.from_prop(phi)
        >>> enc = EncodedCNF()
        >>> enc.from_cnf(cnf)
        >>> lra, conflicts = LRASolver.from_encoded_cnf(enc, testing_mode=True)
        >>> lra #doctest: +SKIP
        <sympy.logic.algorithms.lra_theory.LRASolver object at 0x7fdcb0e15b70>
        >>> conflicts #doctest: +SKIP
        [[4]]
        """
        # This function has three main jobs:
        # - raise errors if the input formula is not handled
        # - preprocesses the formula into a matrix and single variable constraints
        # - create one-literal conflict clauses from predicates that are always True
        #   or always False such as Q.gt(3, 2)
        #
        # See the preprocessing section of "A Fast Linear-Arithmetic Solver for DPLL(T)"
        # for an explanation of how the formula is converted into a matrix
        # and a set of single variable constraints.

        encoding = {}  # maps int to boundary
        A = []

        basic = []
        s_count = 0
        s_subs = {}
        nonbasic = []

        if testing_mode:
            # sort to reduce nondeterminism
            encoded_cnf_items = sorted(encoded_cnf.encoding.items(), key=lambda x: str(x))
        else:
            encoded_cnf_items = encoded_cnf.encoding.items()

        empty_var = Dummy()
        var_to_lra_var = {}
        conflicts = []

        for prop, enc in encoded_cnf_items:
            if isinstance(prop, Predicate):
                prop = prop(empty_var)
            if not isinstance(prop, AppliedPredicate):
                if prop == True:
                    conflicts.append([enc])
                    continue
                if prop == False:
                    conflicts.append([-enc])
                    continue

                raise ValueError(f"Unhandled Predicate: {prop}")

            assert prop.function in ALLOWED_PRED
            if prop.lhs == S.NaN or prop.rhs == S.NaN:
                raise ValueError(f"{prop} contains nan")
            if prop.lhs.is_imaginary or prop.rhs.is_imaginary:
                raise UnhandledInput(f"{prop} contains an imaginary component")
            if prop.lhs == oo or prop.rhs == oo:
                raise UnhandledInput(f"{prop} contains infinity")

            prop = _eval_binrel(prop)  # simplify variable-less quantities to True / False if possible
            if prop == True:
                conflicts.append([enc])
                continue
            elif prop == False:
                conflicts.append([-enc])
                continue
            elif prop is None:
                raise UnhandledInput(f"{prop} could not be simplified")

            expr = prop.lhs - prop.rhs
            if prop.function in [Q.ge, Q.gt]:
                expr = -expr

            # expr should be less than (or equal to) 0
            # otherwise prop is False
            if prop.function in [Q.le, Q.ge]:
                bool = (expr <= 0)
            elif prop.function in [Q.lt, Q.gt]:
                bool = (expr < 0)
            else:
                assert prop.function == Q.eq
                bool = Eq(expr, 0)

            if bool == True:
                conflicts.append([enc])
                continue
            elif bool == False:
                conflicts.append([-enc])
                continue


            vars, const = _sep_const_terms(expr)  # example: (2x + 3y + 2) --> (2x + 3y), (2)
            vars, var_coeff = _sep_const_coeff(vars)  # examples: (2x) --> (x, 2); (2x + 3y) --> (2x + 3y), (1)
            const = const / var_coeff

            terms = _list_terms(vars)  # example: (2x + 3y) --> [2x, 3y]
            for term in terms:
                term, _ = _sep_const_coeff(term)
                assert len(term.free_symbols) > 0
                if term not in var_to_lra_var:
                    var_to_lra_var[term] = LRAVariable(term)
                    nonbasic.append(term)

            if len(terms) > 1:
                if vars not in s_subs:
                    s_count += 1
                    d = Dummy(f"s{s_count}")
                    var_to_lra_var[d] = LRAVariable(d)
                    basic.append(d)
                    s_subs[vars] = d
                    A.append(vars - d)
                var = s_subs[vars]
            else:
                var = terms[0]

            assert var_coeff != 0

            equality = prop.function == Q.eq
            upper = var_coeff > 0 if not equality else None
            strict = prop.function in [Q.gt, Q.lt]
            b = Boundary(var_to_lra_var[var], -const, upper, equality, strict)
            encoding[enc] = b

        fs = [v.free_symbols for v in nonbasic + basic]
        assert all(len(syms) > 0 for syms in fs)
        fs_count = sum(len(syms) for syms in fs)
        if len(fs) > 0 and  len(set.union(*fs)) < fs_count:
            raise UnhandledInput("Nonlinearity is not handled")

        A, _ = linear_eq_to_matrix(A, nonbasic + basic)
        nonbasic = [var_to_lra_var[nb] for nb in nonbasic]
        basic = [var_to_lra_var[b] for b in basic]
        for idx, var in enumerate(nonbasic + basic):
            var.col_idx = idx

        return LRASolver(A, basic, nonbasic, encoding, s_subs, testing_mode), conflicts

    def reset_bounds(self):
        """
        Resets the state of the LRASolver to before
        anything was asserted.
        """
        self.result = None
        for var in self.all_var:
            var.lower = LRARational(-float("inf"), 0)
            var.lower_from_eq = False
            var.lower_from_neg = False
            var.upper = LRARational(float("inf"), 0)
            var.upper_from_eq= False
            var.lower_from_neg = False
            var.assign = LRARational(0, 0)

    def assert_lit(self, enc_constraint):
        """
        Assert a literal representing a constraint
        and update the internal state accordingly.

        Note that due to peculiarities of this implementation
        asserting ~(x > 0) will assert (x <= 0) but asserting
        ~Eq(x, 0) will not do anything.

        Parameters
        ==========

        enc_constraint : int
            A mapping of encodings to constraints
            can be found in `self.enc_to_boundary`.

        Returns
        =======

        None or (False, explanation)

        explanation : set of ints
            A conflict clause that "explains" why
            the literals asserted so far are unsatisfiable.
        """
        if abs(enc_constraint) not in self.enc_to_boundary:
            return None

        if not HANDLE_NEGATION and enc_constraint < 0:
            return None

        boundary = self.enc_to_boundary[abs(enc_constraint)]
        sym, c, negated = boundary.var, boundary.bound, enc_constraint < 0

        if boundary.equality and negated:
            return None # negated equality is not handled and should only appear in conflict clauses

        upper = boundary.upper != negated
        if boundary.strict != negated:
            delta = -1 if upper else 1
            c = LRARational(c, delta)
        else:
            c = LRARational(c, 0)

        if boundary.equality:
            res1 = self._assert_lower(sym, c, from_equality=True, from_neg=negated)
            if res1 and res1[0] == False:
                res = res1
            else:
                res2 = self._assert_upper(sym, c, from_equality=True, from_neg=negated)
                res =  res2
        elif upper:
            res = self._assert_upper(sym, c, from_neg=negated)
        else:
            res = self._assert_lower(sym, c, from_neg=negated)

        if self.is_sat and sym not in self.slack_set:
            self.is_sat = res is None
        else:
            self.is_sat = False

        return res

    def _assert_upper(self, xi, ci, from_equality=False, from_neg=False):
        """
        Adjusts the upper bound on variable xi if the new upper bound is
        more limiting. The assignment of variable xi is adjusted to be
        within the new bound if needed.

        Also calls `self._update` to update the assignment for slack variables
        to keep all equalities satisfied.
        """
        if self.result:
            assert self.result[0] != False
        self.result = None
        if ci >= xi.upper:
            return None
        if ci < xi.lower:
            assert (xi.lower[1] >= 0) is True
            assert (ci[1] <= 0) is True

            lit1, neg1 = Boundary.from_lower(xi)

            lit2 = Boundary(var=xi, const=ci[0], strict=ci[1] != 0, upper=True, equality=from_equality)
            if from_neg:
                lit2 = lit2.get_negated()
            neg2 = -1 if from_neg else 1

            conflict = [-neg1*self.boundary_to_enc[lit1], -neg2*self.boundary_to_enc[lit2]]
            self.result = False, conflict
            return self.result
        xi.upper = ci
        xi.upper_from_eq = from_equality
        xi.upper_from_neg = from_neg
        if xi in self.nonslack and xi.assign > ci:
            self._update(xi, ci)

        if self.run_checks and all(v.assign[0] != float("inf") and v.assign[0] != -float("inf")
                                   for v in self.all_var):
            M = self.A
            X = Matrix([v.assign[0] for v in self.all_var])
            assert all(abs(val) < 10 ** (-10) for val in M * X)

        return None

    def _assert_lower(self, xi, ci, from_equality=False, from_neg=False):
        """
        Adjusts the lower bound on variable xi if the new lower bound is
        more limiting. The assignment of variable xi is adjusted to be
        within the new bound if needed.

        Also calls `self._update` to update the assignment for slack variables
        to keep all equalities satisfied.
        """
        if self.result:
            assert self.result[0] != False
        self.result = None
        if ci <= xi.lower:
            return None
        if ci > xi.upper:
            assert (xi.upper[1] <= 0) is True
            assert (ci[1] >= 0) is True

            lit1, neg1 = Boundary.from_upper(xi)

            lit2 = Boundary(var=xi, const=ci[0], strict=ci[1] != 0, upper=False, equality=from_equality)
            if from_neg:
                lit2 = lit2.get_negated()
            neg2 = -1 if from_neg else 1

            conflict = [-neg1*self.boundary_to_enc[lit1],-neg2*self.boundary_to_enc[lit2]]
            self.result = False, conflict
            return self.result
        xi.lower = ci
        xi.lower_from_eq = from_equality
        xi.lower_from_neg = from_neg
        if xi in self.nonslack and xi.assign < ci:
            self._update(xi, ci)

        if self.run_checks and all(v.assign[0] != float("inf") and v.assign[0] != -float("inf")
                                   for v in self.all_var):
            M = self.A
            X = Matrix([v.assign[0] for v in self.all_var])
            assert all(abs(val) < 10 ** (-10) for val in M * X)

        return None

    def _update(self, xi, v):
        """
        Updates all slack variables that have equations that contain
        variable xi so that they stay satisfied given xi is equal to v.
        """
        i = xi.col_idx
        for j, b in enumerate(self.slack):
            aji = self.A[j, i]
            b.assign = b.assign + (v - xi.assign)*aji
        xi.assign = v

    def check(self):
        """
        Searches for an assignment that satisfies all constraints
        or determines that no such assignment exists and gives
        a minimal conflict clause that "explains" why the
        constraints are unsatisfiable.

        Returns
        =======

        (True, assignment) or (False, explanation)

        assignment : dict of LRAVariables to values
            Assigned values are tuples that represent a rational number
            plus some infinatesimal delta.

        explanation : set of ints
        """
        if self.is_sat:
            return True, {var: var.assign for var in self.all_var}
        if self.result:
            return self.result

        from sympy.matrices.dense import Matrix
        M = self.A.copy()
        basic = {s: i for i, s in enumerate(self.slack)}  # contains the row index associated with each basic variable
        nonbasic = set(self.nonslack)
        while True:
            if self.run_checks:
                # nonbasic variables must always be within bounds
                assert all(((nb.assign >= nb.lower) == True) and ((nb.assign <= nb.upper) == True) for nb in nonbasic)

                # assignments for x must always satisfy Ax = 0
                # probably have to turn this off when dealing with strict ineq
                if all(v.assign[0] != float("inf") and v.assign[0] != -float("inf")
                                   for v in self.all_var):
                    X = Matrix([v.assign[0] for v in self.all_var])
                    assert all(abs(val) < 10**(-10) for val in M*X)

                # check upper and lower match this format:
                # x <= rat + delta iff x < rat
                # x >= rat - delta iff x > rat
                # this wouldn't make sense:
                # x <= rat - delta
                # x >= rat + delta
                assert all(x.upper[1] <= 0 for x in self.all_var)
                assert all(x.lower[1] >= 0 for x in self.all_var)

            cand = [b for b in basic if b.assign < b.lower or b.assign > b.upper]

            if len(cand) == 0:
                return True, {var: var.assign for var in self.all_var}

            xi = min(cand, key=lambda v: v.col_idx) # Bland's rule
            i = basic[xi]

            if xi.assign < xi.lower:
                cand = [nb for nb in nonbasic
                        if (M[i, nb.col_idx] > 0 and nb.assign < nb.upper)
                        or (M[i, nb.col_idx] < 0 and nb.assign > nb.lower)]
                if len(cand) == 0:
                    N_plus = [nb for nb in nonbasic if M[i, nb.col_idx] > 0]
                    N_minus = [nb for nb in nonbasic if M[i, nb.col_idx] < 0]

                    conflict = []
                    conflict += [Boundary.from_upper(nb) for nb in N_plus]
                    conflict += [Boundary.from_lower(nb) for nb in N_minus]
                    conflict.append(Boundary.from_lower(xi))
                    conflict = [-neg*self.boundary_to_enc[c] for c, neg in conflict]
                    return False, conflict
                xj = min(cand, key=str)
                M = self._pivot_and_update(M, basic, nonbasic, xi, xj, xi.lower)

            if xi.assign > xi.upper:
                cand = [nb for nb in nonbasic
                        if (M[i, nb.col_idx] < 0 and nb.assign < nb.upper)
                        or (M[i, nb.col_idx] > 0 and nb.assign > nb.lower)]

                if len(cand) == 0:
                    N_plus = [nb for nb in nonbasic if M[i, nb.col_idx] > 0]
                    N_minus = [nb for nb in nonbasic if M[i, nb.col_idx] < 0]

                    conflict = []
                    conflict += [Boundary.from_upper(nb) for nb in N_minus]
                    conflict += [Boundary.from_lower(nb) for nb in N_plus]
                    conflict.append(Boundary.from_upper(xi))

                    conflict = [-neg*self.boundary_to_enc[c] for c, neg in conflict]
                    return False, conflict
                xj = min(cand, key=lambda v: v.col_idx)
                M = self._pivot_and_update(M, basic, nonbasic, xi, xj, xi.upper)

    def _pivot_and_update(self, M, basic, nonbasic, xi, xj, v):
        """
        Pivots basic variable xi with nonbasic variable xj,
        and sets value of xi to v and adjusts the values of all basic variables
        to keep equations satisfied.
        """
        i, j = basic[xi], xj.col_idx
        assert M[i, j] != 0
        theta = (v - xi.assign)*(1/M[i, j])
        xi.assign = v
        xj.assign = xj.assign + theta
        for xk in basic:
            if xk != xi:
                k = basic[xk]
                akj = M[k, j]
                xk.assign = xk.assign + theta*akj
        # pivot
        basic[xj] = basic[xi]
        del basic[xi]
        nonbasic.add(xi)
        nonbasic.remove(xj)
        return self._pivot(M, i, j)

    @staticmethod
    def _pivot(M, i, j):
        """
        Performs a pivot operation about entry i, j of M by performing
        a series of row operations on a copy of M and returning the result.
        The original M is left unmodified.

        Conceptually, M represents a system of equations and pivoting
        can be thought of as rearranging equation i to be in terms of
        variable j and then substituting in the rest of the equations
        to get rid of other occurances of variable j.

        Example
        =======

        >>> from sympy.matrices.dense import Matrix
        >>> from sympy.logic.algorithms.lra_theory import LRASolver
        >>> from sympy import var
        >>> Matrix(3, 3, var('a:i'))
        Matrix([
        [a, b, c],
        [d, e, f],
        [g, h, i]])

        This matrix is equivalent to:
        0 = a*x + b*y + c*z
        0 = d*x + e*y + f*z
        0 = g*x + h*y + i*z

        >>> LRASolver._pivot(_, 1, 0)
        Matrix([
        [ 0, -a*e/d + b, -a*f/d + c],
        [-1,       -e/d,       -f/d],
        [ 0,  h - e*g/d,  i - f*g/d]])

        We rearrange equation 1 in terms of variable 0 (x)
        and substitute to remove x from the other equations.

        0 = 0 + (-a*e/d + b)*y + (-a*f/d + c)*z
        0 = -x + (-e/d)*y + (-f/d)*z
        0 = 0 + (h - e*g/d)*y + (i - f*g/d)*z
        """
        _, _, Mij = M[i, :], M[:, j], M[i, j]
        if Mij == 0:
            raise ZeroDivisionError("Tried to pivot about zero-valued entry.")
        A = M.copy()
        A[i, :] = -A[i, :]/Mij
        for row in range(M.shape[0]):
            if row != i:
                A[row, :] = A[row, :] + A[row, j] * A[i, :]

        return A


def _sep_const_coeff(expr):
    """
    Example
    =======

    >>> from sympy.logic.algorithms.lra_theory import _sep_const_coeff
    >>> from sympy.abc import x, y
    >>> _sep_const_coeff(2*x)
    (x, 2)
    >>> _sep_const_coeff(2*x + 3*y)
    (2*x + 3*y, 1)
    """
    if isinstance(expr, Add):
        return expr, sympify(1)

    if isinstance(expr, Mul):
        coeffs = expr.args
    else:
        coeffs = [expr]

    var, const = [], []
    for c in coeffs:
        c = sympify(c)
        if len(c.free_symbols)==0:
            const.append(c)
        else:
            var.append(c)
    return Mul(*var), Mul(*const)


def _list_terms(expr):
    if not isinstance(expr, Add):
        return [expr]

    return expr.args


def _sep_const_terms(expr):
    """
    Example
    =======

    >>> from sympy.logic.algorithms.lra_theory import _sep_const_terms
    >>> from sympy.abc import x, y
    >>> _sep_const_terms(2*x + 3*y + 2)
    (2*x + 3*y, 2)
    """
    if isinstance(expr, Add):
        terms = expr.args
    else:
        terms = [expr]

    var, const = [], []
    for t in terms:
        if len(t.free_symbols) == 0:
            const.append(t)
        else:
            var.append(t)
    return sum(var), sum(const)


def _eval_binrel(binrel):
    """
    Simplify binary relation to True / False if possible.
    """
    if not (len(binrel.lhs.free_symbols) == 0 and len(binrel.rhs.free_symbols) == 0):
        return binrel
    if binrel.function == Q.lt:
        res = binrel.lhs < binrel.rhs
    elif binrel.function == Q.gt:
        res = binrel.lhs > binrel.rhs
    elif binrel.function == Q.le:
        res = binrel.lhs <= binrel.rhs
    elif binrel.function == Q.ge:
        res = binrel.lhs >= binrel.rhs
    elif binrel.function == Q.eq:
        res = Eq(binrel.lhs, binrel.rhs)
    elif binrel.function == Q.ne:
        res = Ne(binrel.lhs, binrel.rhs)

    if res == True or res == False:
        return res
    else:
        return None


class Boundary:
    """
    Represents an upper or lower bound or an equality between a symbol
    and some constant.
    """
    def __init__(self, var, const, upper, equality, strict=None):
        if not equality in [True, False]:
            assert equality in [True, False]


        self.var = var
        if isinstance(const, tuple):
            s = const[1] != 0
            if strict:
                assert s == strict
            self.bound = const[0]
            self.strict = s
        else:
            self.bound = const
            self.strict = strict
        self.upper = upper if not equality else None
        self.equality = equality
        self.strict = strict
        assert self.strict is not None

    @staticmethod
    def from_upper(var):
        neg = -1 if var.upper_from_neg else 1
        b = Boundary(var, var.upper[0], True, var.upper_from_eq, var.upper[1] != 0)
        if neg < 0:
            b = b.get_negated()
        return b, neg

    @staticmethod
    def from_lower(var):
        neg = -1 if var.lower_from_neg else 1
        b = Boundary(var, var.lower[0], False, var.lower_from_eq, var.lower[1] != 0)
        if neg < 0:
            b = b.get_negated()
        return b, neg

    def get_negated(self):
        return Boundary(self.var, self.bound, not self.upper, self.equality, not self.strict)

    def get_inequality(self):
        if self.equality:
            return Eq(self.var.var, self.bound)
        elif self.upper and self.strict:
            return self.var.var < self.bound
        elif not self.upper and self.strict:
            return self.var.var > self.bound
        elif self.upper:
            return self.var.var <= self.bound
        else:
            return self.var.var >= self.bound

    def __repr__(self):
        return repr("Boundary(" + repr(self.get_inequality()) + ")")

    def __eq__(self, other):
        other = (other.var, other.bound, other.strict, other.upper, other.equality)
        return (self.var, self.bound, self.strict, self.upper, self.equality) == other

    def __hash__(self):
        return hash((self.var, self.bound, self.strict, self.upper, self.equality))


class LRARational():
    """
    Represents a rational plus or minus some amount
    of arbitrary small deltas.
    """
    def __init__(self, rational, delta):
        self.value = (rational, delta)

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        return self.value == other.value

    def __add__(self, other):
        return LRARational(self.value[0] + other.value[0], self.value[1] + other.value[1])

    def __sub__(self, other):
        return LRARational(self.value[0] - other.value[0], self.value[1] - other.value[1])

    def __mul__(self, other):
        assert not isinstance(other, LRARational)
        return LRARational(self.value[0] * other, self.value[1] * other)

    def __getitem__(self, index):
        return self.value[index]

    def __repr__(self):
        return repr(self.value)


class LRAVariable():
    """
    Object to keep track of upper and lower bounds
    on `self.var`.
    """
    def __init__(self, var):
        self.upper = LRARational(float("inf"), 0)
        self.upper_from_eq = False
        self.upper_from_neg = False
        self.lower = LRARational(-float("inf"), 0)
        self.lower_from_eq = False
        self.lower_from_neg = False
        self.assign = LRARational(0,0)
        self.var = var
        self.col_idx = None

    def __repr__(self):
        return repr(self.var)

    def __eq__(self, other):
        if not isinstance(other, LRAVariable):
            return False
        return other.var == self.var

    def __hash__(self):
        return hash(self.var)
