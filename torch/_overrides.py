"""
Python implementation of __torch_function__

While most of the torch API and handling for __torch_function__ happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for __torch_function__ overrides as well. The main
developer-facing functionality in this file are handle_torch_function and 
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

NOTE: heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

"""


def _get_overloaded_args(relevant_args):
    """Returns a list of arguments on which to call __torch_function__.

    Checks arguments in relevant_args for __torch_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

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

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html

    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        if (arg_type not in overloaded_types and hasattr(arg_type, '__torch_function__')):
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

    return overloaded_args


def handle_torch_function(
        public_api, relevant_args, *args, **kwargs):
    """Implement a function with checks for __torch_function__ overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
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
    overloaded_args = _get_overloaded_args(relevant_args)

    # Call overrides
    for overloaded_arg in overloaded_args:
        # Use `public_api` instead of `implementation` so __torch_function__
        # implementations can do equality/identity comparisons.
        result = overloaded_arg.__torch_function__(public_api, args, kwargs)

        if result is not NotImplemented:
            return result

    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    raise TypeError("no implementation found for '{}' on types that implement "
                    '__torch_function__: {}'
                    .format(func_name, list(map(type, overloaded_args))))

def has_torch_function(relevant_args):
    """Check for __torch_function__ implementations in the elements of an iterable

    Arguments
    ---------
    relevant_args : iterable
        Iterable or aguments to check for __torch_function__ methods.

    Returns
    -------
    True if any of the elements of relevant_args have __torch_function__
    implementations, False otherwise.
    """
    return any(hasattr(a, '__torch_function__') for a in relevant_args)
