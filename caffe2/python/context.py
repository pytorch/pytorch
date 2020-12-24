## @package context
# Module caffe2.python.context

import inspect
import threading
import functools


class _ContextInfo(object):
    def __init__(self, cls, allow_default):
        self.cls = cls
        self.allow_default = allow_default
        self._local_stack = threading.local()

    @property
    def _stack(self):
        if not hasattr(self._local_stack, 'obj'):
            self._local_stack.obj = []
        return self._local_stack.obj

    def enter(self, value):
        self._stack.append(value)

    def exit(self, value):
        assert len(self._stack) > 0, 'Context %s is empty.' % self.cls
        assert self._stack.pop() == value

    def get_active(self, required=True):
        if len(self._stack) == 0:
            if not required:
                return None
            assert self.allow_default, (
                'Context %s is required but none is active.' % self.cls)
            self.enter(self.cls())
        return self._stack[-1]


class _ContextRegistry(object):
    def __init__(self):
        self._ctxs = {}

    def get(self, cls):
        if cls not in self._ctxs:
            assert issubclass(cls, Managed), "must be a context managed class, got {}".format(cls)
            self._ctxs[cls] = _ContextInfo(cls, allow_default=issubclass(cls, DefaultManaged))
        return self._ctxs[cls]


_CONTEXT_REGISTRY = _ContextRegistry()


def _context_registry():
    global _CONTEXT_REGISTRY
    return _CONTEXT_REGISTRY


def _get_managed_classes(obj):
    return [
        cls for cls in inspect.getmro(obj.__class__)
        if issubclass(cls, Managed) and cls != Managed and cls != DefaultManaged
    ]



class Managed(object):
    """
    Managed makes the inheritted class a context managed class.

        class Foo(Managed): ...

        with Foo() as f:
            assert f == Foo.current()
    """

    @classmethod
    def current(cls, value=None, required=True):
        ctx_info = _context_registry().get(cls)
        if value is not None:
            assert isinstance(value, cls), (
                'Wrong context type. Expected: %s, got %s.' % (cls, type(value)))
            return value
        return ctx_info.get_active(required=required)

    def __enter__(self):
        for cls in _get_managed_classes(self):
            _context_registry().get(cls).enter(self)
        return self

    def __exit__(self, *args):
        for cls in _get_managed_classes(self):
            _context_registry().get(cls).exit(self)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class DefaultManaged(Managed):
    """
    DefaultManaged is similar to Managed but if there is no parent when
    current() is called it makes a new one.
    """
    pass
