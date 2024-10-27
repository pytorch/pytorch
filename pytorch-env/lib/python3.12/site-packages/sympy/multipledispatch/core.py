from __future__ import annotations
from typing import Any

import inspect

from .dispatcher import Dispatcher, MethodDispatcher, ambiguity_warn

# XXX: This parameter to dispatch isn't documented and isn't used anywhere in
# sympy. Maybe it should just be removed.
global_namespace: dict[str, Any] = {}


def dispatch(*types, namespace=global_namespace, on_ambiguity=ambiguity_warn):
    """ Dispatch function on the types of the inputs

    Supports dispatch on all non-keyword arguments.

    Collects implementations based on the function name.  Ignores namespaces.

    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0

    Specify an isolated namespace with the namespace keyword argument

    >>> my_namespace = dict()
    >>> @dispatch(int, namespace=my_namespace)
    ... def foo(x):
    ...     return x + 1

    Dispatch on instance methods within classes

    >>> class MyClass(object):
    ...     @dispatch(list)
    ...     def __init__(self, data):
    ...         self.data = data
    ...     @dispatch(int)
    ...     def __init__(self, datum): # noqa: F811
    ...         self.data = [datum]
    """
    types = tuple(types)

    def _(func):
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(
                name,
                MethodDispatcher(name))
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types, func, on_ambiguity=on_ambiguity)
        return dispatcher
    return _


def ismethod(func):
    """ Is func a method?

    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    signature = inspect.signature(func)
    return signature.parameters.get('self', None) is not None
