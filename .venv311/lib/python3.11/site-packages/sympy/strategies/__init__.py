""" Rewrite Rules

DISCLAIMER: This module is experimental. The interface is subject to change.

A rule is a function that transforms one expression into another

    Rule :: Expr -> Expr

A strategy is a function that says how a rule should be applied to a syntax
tree. In general strategies take rules and produce a new rule

    Strategy :: [Rules], Other-stuff -> Rule

This allows developers to separate a mathematical transformation from the
algorithmic details of applying that transformation. The goal is to separate
the work of mathematical programming from algorithmic programming.

Submodules

strategies.rl         - some fundamental rules
strategies.core       - generic non-SymPy specific strategies
strategies.traverse   - strategies that traverse a SymPy tree
strategies.tools      - some conglomerate strategies that do depend on SymPy
"""

from . import rl
from . import traverse
from .rl import rm_id, unpack, flatten, sort, glom, distribute, rebuild
from .util import new
from .core import (
    condition, debug, chain, null_safe, do_one, exhaust, minimize, tryit)
from .tools import canon, typed
from . import branch

__all__ = [
    'rl',

    'traverse',

    'rm_id', 'unpack', 'flatten', 'sort', 'glom', 'distribute', 'rebuild',

    'new',

    'condition', 'debug', 'chain', 'null_safe', 'do_one', 'exhaust',
    'minimize', 'tryit',

    'canon', 'typed',

    'branch',
]
