# mypy: allow-untyped-defs
import inspect
import sys

from .dispatcher import Dispatcher, MethodDispatcher


global_namespace = {}  # type: ignore[var-annotated]

__all__ = ["dispatch", "ismethod"]


def dispatch(*types, **kwargs):
    """Dispatch function on the types of the inputs
    Supports dispatch on all non-keyword arguments.
    Collects implementations based on the function name.  Ignores namespaces.
    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Example:
        >>> # xdoctest: +SKIP
        >>> @dispatch(int)
        ... def f(x):
        ...     return x + 1
        >>> @dispatch(float)
        ... def f(x):
        ...     return x - 1
        >>> # xdoctest: +SKIP
        >>> f(3)
        4
        >>> f(3.0)
        2.0
        >>> # Specify an isolated namespace with the namespace keyword argument
        >>> my_namespace = {}
        >>> @dispatch(int, namespace=my_namespace)
        ... def foo(x):
        ...     return x + 1
        >>> # Dispatch on instance methods within classes
        >>> class MyClass(object):
        ...     @dispatch(list)
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     @dispatch(int)
        ...     def __init__(self, datum):
        ...         self.data = [datum]
        >>> MyClass([1, 2, 3]).data
        [1, 2, 3]
        >>> MyClass(3).data
        [3]
    """
    namespace = kwargs.get("namespace", global_namespace)

    types = tuple(types)

    def _df(func):
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(  # type: ignore[union-attr]
                name,
                MethodDispatcher(name),
            )
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types, func)
        return dispatcher

    return _df


def ismethod(func):
    """Is func a method?
    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get("self", None) is not None
    else:
        if sys.version_info.major < 3:
            spec = inspect.getargspec(func)  # type: ignore[attr-defined]
        else:
            spec = inspect.getfullargspec(func)
        return spec and spec.args and spec.args[0] == "self"
