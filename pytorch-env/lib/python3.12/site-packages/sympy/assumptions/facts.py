"""
Known facts in assumptions module.

This module defines the facts between unary predicates in ``get_known_facts()``,
and supports functions to generate the contents in
``sympy.assumptions.ask_generated`` file.
"""

from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
    Exclusive,)
from sympy.logic.inference import satisfiable


@cacheit
def get_composite_predicates():
    # To reduce the complexity of sat solver, these predicates are
    # transformed into the combination of primitive predicates.
    return {
        Q.real : Q.negative | Q.zero | Q.positive,
        Q.integer : Q.even | Q.odd,
        Q.nonpositive : Q.negative | Q.zero,
        Q.nonzero : Q.negative | Q.positive,
        Q.nonnegative : Q.zero | Q.positive,
        Q.extended_real : Q.negative_infinite | Q.negative | Q.zero | Q.positive | Q.positive_infinite,
        Q.extended_positive: Q.positive | Q.positive_infinite,
        Q.extended_negative: Q.negative | Q.negative_infinite,
        Q.extended_nonzero: Q.negative_infinite | Q.negative | Q.positive | Q.positive_infinite,
        Q.extended_nonpositive: Q.negative_infinite | Q.negative | Q.zero,
        Q.extended_nonnegative: Q.zero | Q.positive | Q.positive_infinite,
        Q.complex : Q.algebraic | Q.transcendental
    }


@cacheit
def get_known_facts(x=None):
    """
    Facts between unary predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        get_number_facts(x),
        get_matrix_facts(x)
    )
    return fact


@cacheit
def get_number_facts(x = None):
    """
    Facts between unary number predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # primitive predicates for extended real exclude each other.
        Exclusive(Q.negative_infinite(x), Q.negative(x), Q.zero(x),
            Q.positive(x), Q.positive_infinite(x)),

        # build complex plane
        Exclusive(Q.real(x), Q.imaginary(x)),
        Implies(Q.real(x) | Q.imaginary(x), Q.complex(x)),

        # other subsets of complex
        Exclusive(Q.transcendental(x), Q.algebraic(x)),
        Equivalent(Q.real(x), Q.rational(x) | Q.irrational(x)),
        Exclusive(Q.irrational(x), Q.rational(x)),
        Implies(Q.rational(x), Q.algebraic(x)),

        # integers
        Exclusive(Q.even(x), Q.odd(x)),
        Implies(Q.integer(x), Q.rational(x)),
        Implies(Q.zero(x), Q.even(x)),
        Exclusive(Q.composite(x), Q.prime(x)),
        Implies(Q.composite(x) | Q.prime(x), Q.integer(x) & Q.positive(x)),
        Implies(Q.even(x) & Q.positive(x) & ~Q.prime(x), Q.composite(x)),

        # hermitian and antihermitian
        Implies(Q.real(x), Q.hermitian(x)),
        Implies(Q.imaginary(x), Q.antihermitian(x)),
        Implies(Q.zero(x), Q.hermitian(x) | Q.antihermitian(x)),

        # define finity and infinity, and build extended real line
        Exclusive(Q.infinite(x), Q.finite(x)),
        Implies(Q.complex(x), Q.finite(x)),
        Implies(Q.negative_infinite(x) | Q.positive_infinite(x), Q.infinite(x)),

        # commutativity
        Implies(Q.finite(x) | Q.infinite(x), Q.commutative(x)),
    )
    return fact


@cacheit
def get_matrix_facts(x = None):
    """
    Facts between unary matrix predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # matrices
        Implies(Q.orthogonal(x), Q.positive_definite(x)),
        Implies(Q.orthogonal(x), Q.unitary(x)),
        Implies(Q.unitary(x) & Q.real_elements(x), Q.orthogonal(x)),
        Implies(Q.unitary(x), Q.normal(x)),
        Implies(Q.unitary(x), Q.invertible(x)),
        Implies(Q.normal(x), Q.square(x)),
        Implies(Q.diagonal(x), Q.normal(x)),
        Implies(Q.positive_definite(x), Q.invertible(x)),
        Implies(Q.diagonal(x), Q.upper_triangular(x)),
        Implies(Q.diagonal(x), Q.lower_triangular(x)),
        Implies(Q.lower_triangular(x), Q.triangular(x)),
        Implies(Q.upper_triangular(x), Q.triangular(x)),
        Implies(Q.triangular(x), Q.upper_triangular(x) | Q.lower_triangular(x)),
        Implies(Q.upper_triangular(x) & Q.lower_triangular(x), Q.diagonal(x)),
        Implies(Q.diagonal(x), Q.symmetric(x)),
        Implies(Q.unit_triangular(x), Q.triangular(x)),
        Implies(Q.invertible(x), Q.fullrank(x)),
        Implies(Q.invertible(x), Q.square(x)),
        Implies(Q.symmetric(x), Q.square(x)),
        Implies(Q.fullrank(x) & Q.square(x), Q.invertible(x)),
        Equivalent(Q.invertible(x), ~Q.singular(x)),
        Implies(Q.integer_elements(x), Q.real_elements(x)),
        Implies(Q.real_elements(x), Q.complex_elements(x)),
    )
    return fact



def generate_known_facts_dict(keys, fact):
    """
    Computes and returns a dictionary which contains the relations between
    unary predicates.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    All predicates in *keys* and *fact* must be unary and have same placeholder
    symbol.

    Parameters
    ==========

    keys : list of AppliedPredicate instances.

    fact : Fact between predicates in conjugated normal form.

    Examples
    ========

    >>> from sympy import Q, And, Implies
    >>> from sympy.assumptions.facts import generate_known_facts_dict
    >>> from sympy.abc import x
    >>> keys = [Q.even(x), Q.odd(x), Q.zero(x)]
    >>> fact = And(Implies(Q.even(x), ~Q.odd(x)),
    ...     Implies(Q.zero(x), Q.even(x)))
    >>> generate_known_facts_dict(keys, fact)
    {Q.even: ({Q.even}, {Q.odd}),
     Q.odd: ({Q.odd}, {Q.even, Q.zero}),
     Q.zero: ({Q.even, Q.zero}, {Q.odd})}
    """
    fact_cnf = to_cnf(fact)
    mapping = single_fact_lookup(keys, fact_cnf)

    ret = {}
    for key, value in mapping.items():
        implied = set()
        rejected = set()
        for expr in value:
            if isinstance(expr, AppliedPredicate):
                implied.add(expr.function)
            elif isinstance(expr, Not):
                pred = expr.args[0]
                rejected.add(pred.function)
        ret[key.function] = (implied, rejected)
    return ret


@cacheit
def get_known_facts_keys():
    """
    Return every unary predicates registered to ``Q``.

    This function is used to generate the keys for
    ``generate_known_facts_dict``.

    """
    # exclude polyadic predicates
    exclude = {Q.eq, Q.ne, Q.gt, Q.lt, Q.ge, Q.le}

    result = []
    for attr in Q.__class__.__dict__:
        if attr.startswith('__'):
            continue
        pred = getattr(Q, attr)
        if pred in exclude:
            continue
        result.append(pred)
    return result


def single_fact_lookup(known_facts_keys, known_facts_cnf):
    # Return the dictionary for quick lookup of single fact
    mapping = {}
    for key in known_facts_keys:
        mapping[key] = {key}
        for other_key in known_facts_keys:
            if other_key != key:
                if ask_full_inference(other_key, key, known_facts_cnf):
                    mapping[key].add(other_key)
                if ask_full_inference(~other_key, key, known_facts_cnf):
                    mapping[key].add(~other_key)
    return mapping


def ask_full_inference(proposition, assumptions, known_facts_cnf):
    """
    Method for inferring properties about objects.

    """
    if not satisfiable(And(known_facts_cnf, assumptions, proposition)):
        return False
    if not satisfiable(And(known_facts_cnf, assumptions, Not(proposition))):
        return True
    return None
