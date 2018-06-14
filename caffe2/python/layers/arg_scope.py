## @package arg_scope
# Module caffe2.python.layers.arg_scope
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from contextlib import contextmanager
import six

_arg_stack = []

_decorated_func = {}


@contextmanager
def _push_arg_stack(arg_scope):
    try:
        _arg_stack.append(arg_scope)
        yield
    finally:
        _arg_stack.pop()


def _current_arg_scope():
    if _arg_stack:
        return _arg_stack[-1].copy()
    return {}


def _valid_override_args(func):
    args_with_defaults_length = len(func.__defaults__) if func.__defaults__ else 0
    return set(func.__code__.co_varnames[
        -args_with_defaults_length:func.__code__.co_argcount])


def _validate_func_args(func_key, args):
    invalid_args = set(args.keys()) - _decorated_func[func_key]
    if invalid_args:
        raise ValueError(
            "Trying to override an invalid argument {} for func {}".format(
                invalid_args, func_key))


def _validate_scope(scope):
    for func_key, args in six.iteritems(scope):
        _validate_func_args(func_key, args)


@contextmanager
def arg_scope(list_funcs_or_scope, **kwargs):
    """Overrides the default values for specific function.

    The two following uses are supported:
    @add_arg_scope
    def some_func(required_arg, optional_arg=None):
        ...

    with arg_scope([some_func], optional_arg=5):
        some_func(3)  # equivalient as some_func(3, 5)
        some_func(3, 6)  # equivalent as some_func(3, 6)

    or this can be used as:

    with arg_scope([some_func], optional_arg=5) as scope:
        pass

    with arg_scope(scope):
        some_func(3)  # equivalent as some_func(3, 5)

    However, only default arguments can be overridden. So the
    following code would raise an error:
    with_arg_scope([some_func], required_arg=6):
        pass

    Args:
        list_funcs_or_scope: A list or a dict. In case of a list, it should be
            a list of functions that has default arguments that needs to be
            overwritten. If a dict, then the content of the dict should directly
            be the arg scope to be used.
        **kwargs: Set of arguments to override the default.

    Returns:
        An overridden scope.

    Raises:
        ValueError: When incorrect arguments are passed to the function.
    """
    if isinstance(list_funcs_or_scope, dict):
        if kwargs:
            raise ValueError(
                "When providing a scope, one shouldn't also provide new " +
                "argumets to override")
        _validate_scope(list_funcs_or_scope)
        with _push_arg_stack(list_funcs_or_scope.copy()):
            yield list_funcs_or_scope
        return
    current_scope = _current_arg_scope()
    for func in list_funcs_or_scope:
        key_func = getattr(func, '_key_arg_scope', None)
        if key_func not in _decorated_func:
            raise ValueError(
                "The following function is not decorated with @add_arg_scope {}".format(
                    (func.__module__, func.__name__)))
        _validate_func_args(key_func, kwargs)
        if key_func in current_scope:
            current_func_scope = current_scope[key_func].copy()
            current_func_scope.update(kwargs)
            current_scope[key_func] = current_func_scope
        else:
            current_scope[key_func] = kwargs
    with _push_arg_stack(current_scope):
        yield current_scope


def add_arg_scope(func):
    """Allows the default arguments of the function to be overridden by arg_scope."""
    key_func = str(func)
    assert key_func not in _decorated_func, "Function already decorated"

    def func_with_new_defaults(*args, **kwargs):
        current_scope = _current_arg_scope()
        arg_scope = kwargs
        if key_func in current_scope:
            arg_scope = current_scope[key_func].copy()
            arg_scope.update(kwargs)
        return func(*args, **arg_scope)

    # Pass useful attributes
    if hasattr(func, "__name__"):
        func_with_new_defaults.__name__ = func.__name__
    if hasattr(func, "__module__"):
        func_with_new_defaults.__module__ = func.__module__
    if hasattr(func, "__doc__"):
        func_with_new_defaults.__doc__ = func.__doc__
    func_with_new_defaults._key_arg_scope = key_func
    _decorated_func[key_func] = _valid_override_args(func)
    return func_with_new_defaults
