"""Inference in propositional logic"""

from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module


def literal_symbol(literal):
    """
    The symbol in this literal (without the negation).

    Examples
    ========

    >>> from sympy.abc import A
    >>> from sympy.logic.inference import literal_symbol
    >>> literal_symbol(A)
    A
    >>> literal_symbol(~A)
    A

    """

    if literal is True or literal is False:
        return literal
    elif literal.is_Symbol:
        return literal
    elif literal.is_Not:
        return literal_symbol(literal.args[0])
    else:
        raise ValueError("Argument must be a boolean literal.")


def satisfiable(expr, algorithm=None, all_models=False, minimal=False, use_lra_theory=False):
    """
    Check satisfiability of a propositional sentence.
    Returns a model when it succeeds.
    Returns {true: true} for trivially true expressions.

    On setting all_models to True, if given expr is satisfiable then
    returns a generator of models. However, if expr is unsatisfiable
    then returns a generator containing the single element False.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import satisfiable
    >>> satisfiable(A & ~B)
    {A: True, B: False}
    >>> satisfiable(A & ~A)
    False
    >>> satisfiable(True)
    {True: True}
    >>> next(satisfiable(A & ~A, all_models=True))
    False
    >>> models = satisfiable((A >> B) & B, all_models=True)
    >>> next(models)
    {A: False, B: True}
    >>> next(models)
    {A: True, B: True}
    >>> def use_models(models):
    ...     for model in models:
    ...         if model:
    ...             # Do something with the model.
    ...             print(model)
    ...         else:
    ...             # Given expr is unsatisfiable.
    ...             print("UNSAT")
    >>> use_models(satisfiable(A >> ~A, all_models=True))
    {A: False}
    >>> use_models(satisfiable(A ^ A, all_models=True))
    UNSAT

    """
    if use_lra_theory:
        if algorithm is not None and algorithm != "dpll2":
            raise ValueError(f"Currently only dpll2 can handle using lra theory. {algorithm} is not handled.")
        algorithm = "dpll2"

    if algorithm is None or algorithm == "pycosat":
        pycosat = import_module('pycosat')
        if pycosat is not None:
            algorithm = "pycosat"
        else:
            if algorithm == "pycosat":
                raise ImportError("pycosat module is not present")
            # Silently fall back to dpll2 if pycosat
            # is not installed
            algorithm = "dpll2"

    if algorithm=="minisat22":
        pysat = import_module('pysat')
        if pysat is None:
            algorithm = "dpll2"

    if algorithm=="z3":
        z3 = import_module('z3')
        if z3 is None:
            algorithm = "dpll2"

    if algorithm == "dpll":
        from sympy.logic.algorithms.dpll import dpll_satisfiable
        return dpll_satisfiable(expr)
    elif algorithm == "dpll2":
        from sympy.logic.algorithms.dpll2 import dpll_satisfiable
        return dpll_satisfiable(expr, all_models, use_lra_theory=use_lra_theory)
    elif algorithm == "pycosat":
        from sympy.logic.algorithms.pycosat_wrapper import pycosat_satisfiable
        return pycosat_satisfiable(expr, all_models)
    elif algorithm == "minisat22":
        from sympy.logic.algorithms.minisat22_wrapper import minisat22_satisfiable
        return minisat22_satisfiable(expr, all_models, minimal)
    elif algorithm == "z3":
        from sympy.logic.algorithms.z3_wrapper import z3_satisfiable
        return z3_satisfiable(expr, all_models)

    raise NotImplementedError


def valid(expr):
    """
    Check validity of a propositional sentence.
    A valid propositional sentence is True under every assignment.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import valid
    >>> valid(A | ~A)
    True
    >>> valid(A | B)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Validity

    """
    return not satisfiable(Not(expr))


def pl_true(expr, model=None, deep=False):
    """
    Returns whether the given assignment is a model or not.

    If the assignment does not specify the value for every proposition,
    this may return None to indicate 'not obvious'.

    Parameters
    ==========

    model : dict, optional, default: {}
        Mapping of symbols to boolean values to indicate assignment.
    deep: boolean, optional, default: False
        Gives the value of the expression under partial assignments
        correctly. May still return None to indicate 'not obvious'.


    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import pl_true
    >>> pl_true( A & B, {A: True, B: True})
    True
    >>> pl_true(A & B, {A: False})
    False
    >>> pl_true(A & B, {A: True})
    >>> pl_true(A & B, {A: True}, deep=True)
    >>> pl_true(A >> (B >> A))
    >>> pl_true(A >> (B >> A), deep=True)
    True
    >>> pl_true(A & ~A)
    >>> pl_true(A & ~A, deep=True)
    False
    >>> pl_true(A & B & (~A | ~B), {A: True})
    >>> pl_true(A & B & (~A | ~B), {A: True}, deep=True)
    False

    """

    from sympy.core.symbol import Symbol

    boolean = (True, False)

    def _validate(expr):
        if isinstance(expr, Symbol) or expr in boolean:
            return True
        if not isinstance(expr, BooleanFunction):
            return False
        return all(_validate(arg) for arg in expr.args)

    if expr in boolean:
        return expr
    expr = sympify(expr)
    if not _validate(expr):
        raise ValueError("%s is not a valid boolean expression" % expr)
    if not model:
        model = {}
    model = {k: v for k, v in model.items() if v in boolean}
    result = expr.subs(model)
    if result in boolean:
        return bool(result)
    if deep:
        model = dict.fromkeys(result.atoms(), True)
        if pl_true(result, model):
            if valid(result):
                return True
        else:
            if not satisfiable(result):
                return False
    return None


def entails(expr, formula_set=None):
    """
    Check whether the given expr_set entail an expr.
    If formula_set is empty then it returns the validity of expr.

    Examples
    ========

    >>> from sympy.abc import A, B, C
    >>> from sympy.logic.inference import entails
    >>> entails(A, [A >> B, B >> C])
    False
    >>> entails(C, [A >> B, B >> C, A])
    True
    >>> entails(A >> B)
    False
    >>> entails(A >> (B >> A))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logical_consequence

    """
    if formula_set:
        formula_set = list(formula_set)
    else:
        formula_set = []
    formula_set.append(Not(expr))
    return not satisfiable(And(*formula_set))


class KB:
    """Base class for all knowledge bases"""
    def __init__(self, sentence=None):
        self.clauses_ = set()
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        raise NotImplementedError

    def ask(self, query):
        raise NotImplementedError

    def retract(self, sentence):
        raise NotImplementedError

    @property
    def clauses(self):
        return list(ordered(self.clauses_))


class PropKB(KB):
    """A KB for Propositional Logic.  Inefficient, with no indexing."""

    def tell(self, sentence):
        """Add the sentence's clauses to the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.tell(y)
        >>> l.clauses
        [y, x | y]

        """
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.add(c)

    def ask(self, query):
        """Checks if the query is true given the set of clauses.

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.tell(x & ~y)
        >>> l.ask(x)
        True
        >>> l.ask(y)
        False

        """
        return entails(query, self.clauses_)

    def retract(self, sentence):
        """Remove the sentence's clauses from the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.retract(x | y)
        >>> l.clauses
        []

        """
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.discard(c)
