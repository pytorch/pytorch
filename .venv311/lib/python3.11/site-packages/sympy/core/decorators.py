"""
SymPy core decorators.

The purpose of this module is to expose decorators without any other
dependencies, so that they can be easily imported anywhere in sympy/core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import wraps
from .sympify import SympifyError, sympify


if TYPE_CHECKING:
    from typing import Callable, TypeVar, Union
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    T3 = TypeVar('T3')


def _sympifyit(arg, retval=None) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """
    decorator to smartly _sympify function arguments

    Explanation
    ===========

    @_sympifyit('other', NotImplemented)
    def add(self, other):
        ...

    In add, other can be thought of as already being a SymPy object.

    If it is not, the code is likely to catch an exception, then other will
    be explicitly _sympified, and the whole code restarted.

    if _sympify(arg) fails, NotImplemented will be returned

    See also
    ========

    __sympifyit
    """
    def deco(func):
        return __sympifyit(func, arg, retval)

    return deco


def __sympifyit(func, arg, retval=None):
    """Decorator to _sympify `arg` argument for function `func`.

       Do not use directly -- use _sympifyit instead.
    """

    # we support f(a,b) only
    if not func.__code__.co_argcount:
        raise LookupError("func not found")
    # only b is _sympified
    assert func.__code__.co_varnames[1] == arg
    if retval is None:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            return func(a, sympify(b, strict=True))

    else:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                # If an external class has _op_priority, it knows how to deal
                # with SymPy objects. Otherwise, it must be converted.
                if not hasattr(b, '_op_priority'):
                    b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval

    return __sympifyit_wrapper


def call_highest_priority(method_name: str
    ) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """A decorator for binary special methods to handle _op_priority.

    Explanation
    ===========

    Binary special methods in Expr and its subclasses use a special attribute
    '_op_priority' to determine whose special method will be called to
    handle the operation. In general, the object having the highest value of
    '_op_priority' will handle the operation. Expr and subclasses that define
    custom binary special methods (__mul__, etc.) should decorate those
    methods with this decorator to add the priority logic.

    The ``method_name`` argument is the name of the method of the other class
    that will be called.  Use this decorator in the following manner::

        # Call other.__rmul__ if other._op_priority > self._op_priority
        @call_highest_priority('__rmul__')
        def __mul__(self, other):
            ...

        # Call other.__mul__ if other._op_priority > self._op_priority
        @call_highest_priority('__mul__')
        def __rmul__(self, other):
        ...
    """
    def priority_decorator(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        @wraps(func)
        def binary_op_wrapper(self: T1, other: T2) -> T3:
            if hasattr(other, '_op_priority'):
                if other._op_priority > self._op_priority:  # type: ignore
                    f: Union[Callable[[T1], T3], None] = getattr(other, method_name, None)
                    if f is not None:
                        return f(self)
            return func(self, other)
        return binary_op_wrapper
    return priority_decorator


def sympify_method_args(cls: type[T1]) -> type[T1]:
    '''Decorator for a class with methods that sympify arguments.

    Explanation
    ===========

    The sympify_method_args decorator is to be used with the sympify_return
    decorator for automatic sympification of method arguments. This is
    intended for the common idiom of writing a class like :

    Examples
    ========

    >>> from sympy import Basic, SympifyError, S
    >>> from sympy.core.sympify import _sympify

    >>> class MyTuple(Basic):
    ...     def __add__(self, other):
    ...         try:
    ...             other = _sympify(other)
    ...         except SympifyError:
    ...             return NotImplemented
    ...         if not isinstance(other, MyTuple):
    ...             return NotImplemented
    ...         return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    In the above it is important that we return NotImplemented when other is
    not sympifiable and also when the sympified result is not of the expected
    type. This allows the MyTuple class to be used cooperatively with other
    classes that overload __add__ and want to do something else in combination
    with instance of Tuple.

    Using this decorator the above can be written as

    >>> from sympy.core.decorators import sympify_method_args, sympify_return

    >>> @sympify_method_args
    ... class MyTuple(Basic):
    ...     @sympify_return([('other', 'MyTuple')], NotImplemented)
    ...     def __add__(self, other):
    ...          return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    The idea here is that the decorators take care of the boiler-plate code
    for making this happen in each method that potentially needs to accept
    unsympified arguments. Then the body of e.g. the __add__ method can be
    written without needing to worry about calling _sympify or checking the
    type of the resulting object.

    The parameters for sympify_return are a list of tuples of the form
    (parameter_name, expected_type) and the value to return (e.g.
    NotImplemented). The expected_type parameter can be a type e.g. Tuple or a
    string 'Tuple'. Using a string is useful for specifying a Type within its
    class body (as in the above example).

    Notes: Currently sympify_return only works for methods that take a single
    argument (not including self). Specifying an expected_type as a string
    only works for the class in which the method is defined.
    '''
    # Extract the wrapped methods from each of the wrapper objects created by
    # the sympify_return decorator. Doing this here allows us to provide the
    # cls argument which is used for forward string referencing.
    for attrname, obj in cls.__dict__.items():
        if isinstance(obj, _SympifyWrapper):
            setattr(cls, attrname, obj.make_wrapped(cls))
    return cls


def sympify_return(*args):
    '''Function/method decorator to sympify arguments automatically

    See the docstring of sympify_method_args for explanation.
    '''
    # Store a wrapper object for the decorated method
    def wrapper(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        return _SympifyWrapper(func, args)  # type: ignore
    return wrapper


class _SympifyWrapper:
    '''Internal class used by sympify_return and sympify_method_args'''

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def make_wrapped(self, cls):
        func = self.func
        parameters, retval = self.args

        # XXX: Handle more than one parameter?
        [(parameter, expectedcls)] = parameters

        # Handle forward references to the current class using strings
        if expectedcls == cls.__name__:
            expectedcls = cls

        # Raise RuntimeError since this is a failure at import time and should
        # not be recoverable.
        nargs = func.__code__.co_argcount
        # we support f(a, b) only
        if nargs != 2:
            raise RuntimeError('sympify_return can only be used with 2 argument functions')
        # only b is _sympified
        if func.__code__.co_varnames[1] != parameter:
            raise RuntimeError('parameter name mismatch "%s" in %s' %
                    (parameter, func.__name__))

        @wraps(func)
        def _func(self, other):
            # XXX: The check for _op_priority here should be removed. It is
            # needed to stop mutable matrices from being sympified to
            # immutable matrices which breaks things in quantum...
            if not hasattr(other, '_op_priority'):
                try:
                    other = sympify(other, strict=True)
                except SympifyError:
                    return retval
            if not isinstance(other, expectedcls):
                return retval
            return func(self, other)

        return _func
