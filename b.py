import torch


"""Utilities for with-statement contexts.  See PEP 343."""
import _collections_abc
import contextlib
import abc
from functools import wraps
from types import GenericAlias


class AbstractContextManager(abc.ABC):
    """An abstract base class for context managers."""

    __class_getitem__ = classmethod(GenericAlias)

    __slots__ = ()

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractContextManager:
            return _collections_abc._check_methods(C, "__enter__", "__exit__")
        return NotImplemented


class AbstractAsyncContextManager(abc.ABC):
    """An abstract base class for asynchronous context managers."""

    __class_getitem__ = classmethod(GenericAlias)

    __slots__ = ()

    async def __aenter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractAsyncContextManager:
            return _collections_abc._check_methods(C, "__aenter__", "__aexit__")
        return NotImplemented


class ContextDecorator:
    "A base class or mixin that enables context managers to work as decorators."

    def _recreate_cm(self):
        """Return a recreated instance of self.

        Allows an otherwise one-shot context manager like
        _GeneratorContextManager to support use as
        a decorator via implicit recreation.

        This is a private interface just for _GeneratorContextManager.
        See issue #11647 for details.
        """
        return self

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)

        return inner


class AsyncContextDecorator:
    "A base class or mixin that enables async context managers to work as decorators."

    def _recreate_cm(self):
        """Return a recreated instance of self."""
        return self

    def __call__(self, func):
        @wraps(func)
        async def inner(*args, **kwds):
            async with self._recreate_cm():
                return await func(*args, **kwds)

        return inner


class _GeneratorContextManagerBase:
    """Shared functionality for @contextmanager and @asynccontextmanager."""

    def __init__(self, func, args, kwds):
        self.gen = func(*args, **kwds)
        self.func, self.args, self.kwds = func, args, kwds
        # Issue 19330: ensure context manager instances have good docstrings
        doc = getattr(func, "__doc__", None)
        if doc is None:
            doc = type(self).__doc__
        self.__doc__ = doc
        # Unfortunately, this still doesn't provide good help output when
        # inspecting the created context manager instances, since pydoc
        # currently bypasses the instance docstring and shows the docstring
        # for the class instead.
        # See http://bugs.python.org/issue19404 for more details.

    def _recreate_cm(self):
        # _GCMB instances are one-shot context managers, so the
        # CM must be recreated each time a decorated function is
        # called
        return self.__class__(self.func, self.args, self.kwds)


class _GeneratorContextManager(
    _GeneratorContextManagerBase,
    AbstractContextManager,
    ContextDecorator,
):
    """Helper for @contextmanager decorator."""

    def __enter__(self):
        # do not keep args and kwds alive unnecessarily
        # they are only needed for recreation, which is not possible anymore
        del self.args, self.kwds, self.func
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, typ, value, traceback):
        if typ is None:
            try:
                next(self.gen)
                # raise StopIteration
            except StopIteration:
                return False
            else:
                try:
                    raise RuntimeError("generator didn't stop")
                finally:
                    self.gen.close()
        else:
            if value is None:
                # Need to force instantiation so we can reliably
                # tell if we get the same exception back
                value = typ()
            try:
                self.gen.throw(value)
            except StopIteration as exc:
                # Suppress StopIteration *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed.
                return exc is not value
            except RuntimeError as exc:
                # Don't re-raise the passed in exception. (issue27122)
                if exc is value:
                    exc.__traceback__ = traceback
                    return False
                # Avoid suppressing if a StopIteration exception
                # was passed to throw() and later wrapped into a RuntimeError
                # (see PEP 479 for sync generators; async generators also
                # have this behavior). But do this only if the exception wrapped
                # by the RuntimeError is actually Stop(Async)Iteration (see
                # issue29692).
                if isinstance(value, StopIteration) and exc.__cause__ is value:
                    value.__traceback__ = traceback
                    return False
                raise
            except BaseException as exc:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if exc is not value:
                    raise
                exc.__traceback__ = traceback
                return False
            try:
                raise RuntimeError("generator didn't stop after throw()")
            finally:
                self.gen.close()


def contextmanager(func):
    @wraps(func)
    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)
    return helper


@contextlib.contextmanager
def set_default_dtype(dtype):
    # __init__
    saved_dtype = torch.get_default_dtype()
    try:
        # __enter__
        torch.set_default_dtype(dtype)
        yield
    finally:
        # __exit__
        torch.set_default_dtype(saved_dtype)


# @contextmanager
# def set_default_dtype(dtype):
#     saved_dtype = torch.get_default_dtype()
#     torch.set_default_dtype(dtype)
#     yield
#     torch.set_default_dtype(saved_dtype)


@torch.compile(backend="eager")
def f():
    with set_default_dtype(torch.float64):
        x = torch.tensor([3.0, 3.0 + 5.0j])
    return x


# y = f()
# print(y)
# assert y.dtype == torch.complex128
# def nonlocal_test():
#     z = 1
#     k = 2

#     def create_fn():
#         def fn(x):
#             nonlocal k, z
#             k = z
#         return fn

#     def run_fn(fn, x):
#         nonlocal z
#         z = 3
#         fn(x)
#         return x.cos()

#     @torch.compile(backend="eager", fullgraph=True)
#     def foo(x):
#         fn = create_fn()
#         return run_fn(fn, x)

#     x = torch.randn(2, 3)
#     foo(x)
#     print(f'{z=} - {k=}')
#     assert z == 3
#     assert k == 3

# nonlocal_test()


# z, k = 1, 2

# def create_fn():
#     def fn(x):
#         global z, k
#         k = z
#     return fn

# def run_fn(fn, x):
#     global z
#     z = 3
#     fn(x)
#     return x.cos()

# @torch.compile(backend="eager", fullgraph=True)
# def foo(x):
#     fn = create_fn()
#     return run_fn(fn, x)


# x = torch.randn(2, 3)
# foo(x)
# print(f'{z=} - {k=}')
# assert z == 3
# assert k == 3


@contextlib.contextmanager
def bar():
    try:
        yield 42
    finally:
        pass

@torch.compile(backend="eager", fullgraph=True)
def foo(t):
    y = t.sum()
    # with torch._functorch.vmap.vmap_increment_nesting(1, 'error') as lvl:
    with bar() as x:
        y += x
    return y

t = torch.randn(3)
foo(t)
