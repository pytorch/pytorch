"""
A module to implement logical predicates and assumption system.
"""

from .assume import (
    AppliedPredicate, Predicate, AssumptionsContext, assuming,
    global_assumptions
)
from .ask import Q, ask, register_handler, remove_handler
from .refine import refine
from .relation import BinaryRelation, AppliedBinaryRelation

__all__ = [
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming',
    'global_assumptions', 'Q', 'ask', 'register_handler', 'remove_handler',
    'refine',
    'BinaryRelation', 'AppliedBinaryRelation'
]
