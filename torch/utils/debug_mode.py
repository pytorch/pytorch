import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree

from collections.abc import Callable
from typing import Union, List, Optional, Iterable
from functools import partial

__all__ = ["DebugMode", ]

def embedding_check(func, args, kwargs):
    invalid_index_mask = args[0].shape[0] <= args[1]
    if torch.any(invalid_index_mask):
        msg = (f"{func}: Received invalid indices for embedding matrix of shape : {args[0].shape}."
               f" Invalid indices are {torch.masked_select(args[1], invalid_index_mask)}")
        raise IndexError(msg)

def extermal_check(func, args, kwargs, result, check):

    def check_fn(check, check_method):
        def error_if(t):
            if check_method(t):
                raise ValueError(f"{func} produced `{check}` in output")

        pytree.tree_map_only(torch.Tensor, lambda t: error_if(t), result)

    assert check in ('inf', 'nan')

    if check == 'inf':
        check_fn(check, lambda t: t.isinf().all())
    elif check == 'nan':
        check_fn(check, lambda t: t.isnan().all())


def indexing_checks(func, args, kwargs):
    if func in (torch.ops.aten.embedding, torch.ops.aten._embedding_bag):
        embedding_check(func, args, kwargs)

nan_check = partial(extermal_check, check='nan')
inf_check = partial(extermal_check, check='nan')

PRE_CHECKS = {'index': indexing_checks}
POST_CHECKS = {'nan': nan_check, 'inf': inf_check}
CHECKS = list(PRE_CHECKS.keys()) + list(POST_CHECKS.keys())

def get_pre_checks(check):
    # Handle case where check is str.
    if check == 'all':
        return tuple(PRE_CHECKS.values())
    if isinstance(check, str):
        return (PRE_CHECKS[check],)

    result = []
    # Handle case where check is List[str].
    for c in check:
        assert isinstance(c, str)
        if c in PRE_CHECKS:
            result.append(PRE_CHECKS[c])

    return tuple(result)


def get_post_checks(check):
    # Handle case where check is str.
    if check == 'all':
        return (nan_check, inf_check)
    if isinstance(check, str):
        return (POST_CHECKS[check],)

    result = []
    # Handle case where check is List[str].
    for c in check:
        if c in POST_CHECKS:
            result.append(POST_CHECKS[c])

    return tuple(result)


def to_iterable(checks):
    if checks is None:
        return ()
    if callable(checks):
        return (checks,)
    return tuple(checks)


def is_valid_check(check):
    if not (check == 'all' or check in CHECKS):
        valid_values = ['all', ] + CHECKS
        raise RuntimeError(f"Invalid value for check. Valid values are {valid_values} but received {check=}")


class DebugMode(TorchDispatchMode):
    r"""
    ``DebugMode`` is a context manager that returns more comprehensive error messages
    for certain set of operators and also allows for other checks like checking for extremal values
    in the computation.

    Besides the provided checks, user can pass their own checks to the debug mode, which will be run
    before or after calling any torch operator.

    Also, user can add a custom error handling for any uncaught exception from an operator.

    Arguments:
        checks: Checks to be run under DebugMode. Defaults to 'all' i.e. all availble checks.
        pre_checks: Callable which are called before running the operator.
            The callable should accept 3 arguments which are op, op_args and op_kwargs.
        post_checks: Callable which are called after running the operator.
            The callable should accept 4 arguments which are op, op_args, op_kwargs and result.
        custom_error_handler: Callable which is called when any uncaught exception is thrown from
            an operator. The callable should accept 4 arguments which are error, op, op_args and op_kwargs.

    Example usage

    .. code-block:: python
        with DebugMode():
            mod.sum().backward()

    """
    def __init__(self,
                 checks: Union[str, List[str]] = 'all',  # Selection of checks provided by PyTorch
                 # Additional pre-checks (callables) user can pass
                 # Arguments to the callable are op, op_args, op_kwargs
                 pre_checks: Optional[Union[Iterable[Callable], Callable]] = None,
                 # Additional post-checks (callables) user can pass
                 # Arguments to the callable are op, op_args, op_kwargs, result
                 post_checks: Optional[Union[Iterable[Callable], Callable]] = None,
                 # custom_error_handler for an uncaught error
                 custom_error_handler: Optional[Callable] = None) -> None:

        if isinstance(checks, str):
            is_valid_check(checks)
        else:
            for check in set(checks):
                assert isinstance(check, str)
                is_valid_check(check)

        self.checks = checks
        self.pre_checks = pre_checks
        self.post_checks = post_checks
        self.custom_error_handler = custom_error_handler

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        fn = func._overloadpacket

        # Pre-checks
        # Run user checks first as those would probably be more interesting to user
        for check in to_iterable(self.pre_checks) + get_pre_checks(self.checks):
            check(fn, args, kwargs)

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            if self.custom_error_handler is not None:
                exception = self.custom_error_handler(e, fn, args, kwargs)
                raise exception
            else:
                raise e

        # Post-checks (which require output of the function)
        # Run user checks first as those would probably be more interesting to user
        for check in to_iterable(self.post_checks) + get_post_checks(self.checks):
            check(fn, args, kwargs, result)

        return result
