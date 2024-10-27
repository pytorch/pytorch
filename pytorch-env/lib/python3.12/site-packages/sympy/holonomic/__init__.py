r"""
The :py:mod:`~sympy.holonomic` module is intended to deal with holonomic functions along
with various operations on them like addition, multiplication, composition,
integration and differentiation. The module also implements various kinds of
conversions such as converting holonomic functions to a different form and the
other way around.
"""

from .holonomic import (DifferentialOperator, HolonomicFunction, DifferentialOperators,
    from_hyper, from_meijerg, expr_to_holonomic)
from .recurrence import RecurrenceOperators, RecurrenceOperator, HolonomicSequence

__all__ = [
    'DifferentialOperator', 'HolonomicFunction', 'DifferentialOperators',
    'from_hyper', 'from_meijerg', 'expr_to_holonomic',

    'RecurrenceOperators', 'RecurrenceOperator', 'HolonomicSequence',
]
