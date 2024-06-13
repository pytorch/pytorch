# mypy: allow-untyped-defs
import functools
from inspect import signature
from .common_op_utils import _basic_validation

"""
Common utilities to register ops on ShardedTensor
and PartialTensor.
"""

def _register_op(op, func, op_table):
    """
    Performs basic validation and registers the provided op in the given
    op_table.
    """
    if len(signature(func).parameters) != 4:
        raise TypeError(
            f'Custom sharded op function expects signature: '
            f'(types, args, kwargs, process_group), but received '
            f'signature: {signature(func)}')

    op_table[op] = func

def _decorator_func(wrapped_func, op, op_table):
    """
    Decorator function to register the given ``op`` in the provided
    ``op_table``
    """

    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        _basic_validation(op, args, kwargs)
        return wrapped_func(types, args, kwargs, process_group)

    _register_op(op, wrapper, op_table)
    return wrapper
