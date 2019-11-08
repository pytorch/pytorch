"""
Python implementation of __torch_function__

While most of the torch API and handling for __torch_function__ happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for __torch_function__ overrides as well. The main
developer-facing functionality in this file is the
torch_function_dispatch decorator. This function can be applied to
python functions in the torch.functional module to enable
__torch_function__ overrides for that function. See the examples in the
docstrings for torch_function_dispatch for details.

NOTE: heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

"""

import functools
import textwrap
from . import _six
if _six.PY3:
    from inspect import getfullargspec
    import collections
    ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')

    def getargspec(func):
        spec = getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
else:
    from inspect import getargspec

from .tensor import Tensor


_TENSOR_ONLY = [Tensor]

def _get_overloaded_types_and_args(relevant_args):
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

    return overloaded_types, overloaded_args


def _implement_torch_function(
        implementation, public_api, relevant_args, args, kwargs):
    """Implement a function with checks for __torch_function__ overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

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
    types, overloaded_args = _get_overloaded_types_and_args(relevant_args)
    # Short-cut for common cases: no overload or only Tensor overload
    # (directly or with subclasses that do not override __torch_function__).
    if not overloaded_args or types == _TENSOR_ONLY:
        return implementation(*args, **kwargs)

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


def _verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = getargspec(implementation)
    dispatcher_spec = getargspec(dispatcher)

    if (implementation_spec.args != dispatcher_spec.args or
        implementation_spec.varargs != dispatcher_spec.varargs or
        implementation_spec.keywords != dispatcher_spec.keywords or
        (bool(implementation_spec.defaults) !=
         bool(dispatcher_spec.defaults)) or
        (implementation_spec.defaults is not None and
         len(implementation_spec.defaults) !=
         len(dispatcher_spec.defaults))):
        raise RuntimeError('implementation and dispatcher for %s have '
                           'different function signatures' % implementation)


_wrapped_func_source = textwrap.dedent("""
    @functools.wraps(implementation)
    def {name}(*args, **kwargs):
        relevant_args = dispatcher(*args, **kwargs)
        return implement_torch_function(
            implementation, {name}, relevant_args, args, kwargs)
    """)

def torch_function_dispatch(dispatcher, module=None, verify=True):
    """Decorator for adding dispatch with the __torch_function__ protocol.

    If you define a function in Python and would like to permit user-defined
    tensor-like types to override it using __torch_function__, please apply this
    decorator on this function together with a custom dispatcher that indicates
    which arguments should be checked for the presence of __torch_function__.

    Suppose we'd like to apply this function to torch.frob, which has the
    following definition:

        def frob(input, bias, option=None):
            return input + bias

    We'd need to define a dispatcher for frob that has the same signature and
    returns the elements of the signature that should be checked for
    `__torch_function__`. If any of the arguments has a `__torch_function__`
    attribute, that function will be called to handle custom dispatch. Assuming
    that `bias` can be a tensor-like, our dispatcher would look like:

        def _frob_dispatcher(input, bias, option=None):
            return (input, bias)

    The dispatcher must return an iterable, so return a single-element tuple if
    only one argument should be checked. We would then modify the original
    definition for torch.frob to look like:

        @torch_function_dispatch(_frob_dispatcher)
        def frob(input, bias, option=None):
             return input + bias

    See ``torch/functional.py`` for more usage examples.

    Parameters
    ----------
    dispatcher : callable
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__torch_function__``.
    module : str, optional
        ``__module__`` attribute to set on new function, e.g.,
        ``module='torch'``.  By default, module is copied from the decorated
        function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.

    Returns
    -------
    dispatcher : callable
        Function suitable for decorating the implementation of a NumPy
        function.

    Notes
    -----
    The dispatcher should normally return a tuple containing all input
    arguments that may have a ``__torch_function__`` attribute.

    In some cases where that's not easily possible, e.g. ``torch.cat``, it is
    also valid (if a little slower) to make the dispatcher function a generator
    (i.e. use ``yield`` to return arguments one by one).

    """
    def decorator(implementation):
        if verify:
            _verify_matching_signatures(implementation, dispatcher)

        # Equivalently, we could define this function directly instead of using
        # exec. This version has the advantage of giving the helper function a
        # more interpretable name. Otherwise, the original function does not
        # show up at all in many cases, e.g., if it's written in C++ or if the
        # dispatcher gets an invalid keyword argument.
        source = _wrapped_func_source.format(name=implementation.__name__)

        source_object = compile(
            source, filename='<__torch_function__ internals>', mode='exec')
        scope = {
            'implementation': implementation,
            'dispatcher': dispatcher,
            'functools': functools,
            'implement_torch_function': _implement_torch_function,
        }
        _six.exec_(source_object, scope)

        public_api = scope[implementation.__name__]

        if module is not None:
            public_api.__module__ = module

        public_api._implementation = implementation

        return public_api

    return decorator
