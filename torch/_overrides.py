"""
Preliminary implementation of __torch_function__

TODO: rewrite this in C++ for performance.

NOTE: heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

"""

import collections
import functools
import os

from numpy.compat._inspect import getargspec

from torch import Tensor


_TORCH_FUNCTION = Tensor.__torch_function__
_TORCH_ONLY = [Tensor]


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __torch_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    Returns
    -------
    overloaded_types : collection of types
        Types of arguments from relevant_args with __torch_function__ methods.
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        if (arg_type not in overloaded_types and
                hasattr(arg_type, '__torch_function__')):

            # Create lists explicitly for the first type (usually the only one
            # done) to avoid setting up the iterator for overloaded_args.
            if overloaded_types:
                overloaded_types.append(arg_type)
                # By default, insert argument at the end, but if it is
                # subclass of another argument, insert it before that argument.
                # This ensures "subclasses before superclasses".
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, type(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = [arg_type]
                overloaded_args = [arg]

    return overloaded_types, overloaded_args


def implement_torch_function(
        implementation, public_api, relevant_args, args, kwargs):
    """Implement a function with checks for __torch_function__ overrides.

    Arguments
    ---------
    implementation : function
        Function that implements the operation on ``torch.Tensor`` without
        overrides when called like ``implementation(*args, **kwargs)``.
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    Result from calling `implementation()` or an `__torch_function__`
    method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    """
    # Check for __torch_function__ methods.
    types, overloaded_args = get_overloaded_types_and_args(relevant_args)
    # Short-cut for common cases: no overload or only Tensor overload
    # (directly or with subclasses that do not override __torch_function__).
    if (not overloaded_args or types == _TENSOR_ONLY or
            all(type(arg).__torch_function__ is _TORCH_FUNCTION
                for arg in overloaded_args)):
        return implementation(*args, **kwargs)

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Use `public_api` instead of `implemenation` so __torch_function__
        # implementations can do equality/identity comparisons.
        result = overloaded_arg.__torch_function__(
            public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    raise TypeError("no implementation found for '{}' on types that implement "
                    '__torch_function__: {}'
                    .format(func_name, list(map(type, overloaded_args))))

