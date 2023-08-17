# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast

# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """
    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator, but:

    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """

    assert not (callable(ctx) and hasattr(ctx, '__enter__')), (
        f"Passed in {ctx} is both callable and also a valid context manager "
        "(has __enter__), making it ambiguous which interface to use.  If you "
        "intended to pass a context manager factory, rewrite your call as "
        "context_decorator(lambda: ctx()); if you intended to pass a context "
        "manager directly, rewrite your call as context_decorator(lambda: ctx)"
    )

    if not callable(ctx):
        def ctx_factory():
            return ctx
    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn("Decorating classes is deprecated and will be disabled in "
                          "future versions. You should only decorate functions or methods. "
                          "To preserve the current behavior of class decoration, you can "
                          "directly decorate the `__init__` method and nothing else.", stacklevel=2)
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses"""

    def __new__(cls, orig_func=None):
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)
