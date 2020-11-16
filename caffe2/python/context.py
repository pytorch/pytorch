## @package context
# Module caffe2.python.context





import threading
import six


class _ContextInfo(object):
    def __init__(self, cls, allow_default, arg_name):
        self.cls = cls
        self.allow_default = allow_default
        self.arg_name = arg_name
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

    def register(self, ctx_info):
        assert isinstance(ctx_info, _ContextInfo)
        assert (ctx_info.cls not in self._ctxs), (
            'Context %s already registered' % ctx_info.cls)
        self._ctxs[ctx_info.cls] = ctx_info

    def get(self, cls):
        assert cls in self._ctxs, 'Context %s not registered.' % cls
        return self._ctxs[cls]


_CONTEXT_REGISTRY = _ContextRegistry()


def _context_registry():
    global _CONTEXT_REGISTRY
    return _CONTEXT_REGISTRY


def __enter__(self):
    if self._prev_enter is not None:
        self._prev_enter()
    _context_registry().get(self._ctx_class).enter(self)
    return self


def __exit__(self, *args):
    _context_registry().get(self._ctx_class).exit(self)
    if self._prev_exit is not None:
        self._prev_exit(*args)


def __call__(self, func):
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        with self:
            return func(*args, **kwargs)
    return wrapper


@classmethod
def _current(cls, value=None, required=True):
    return _get_active_context(cls, value, required)


class define_context(object):
    def __init__(self, arg_name=None, allow_default=False):
        self.arg_name = arg_name
        self.allow_default = allow_default

    def __call__(self, cls):
        assert not hasattr(cls, '_ctx_class'), (
            '%s parent class (%s) already defines context.' % (
                cls, cls._ctx_class))
        cls._ctx_class = cls

        _context_registry().register(
            _ContextInfo(cls, self.allow_default, self.arg_name)
        )

        cls._prev_enter = cls.__enter__ if hasattr(cls, '__enter__') else None
        cls._prev_exit = cls.__exit__ if hasattr(cls, '__exit__') else None

        cls.__enter__ = __enter__
        cls.__exit__ = __exit__
        cls.__call__ = __call__
        cls.current = _current

        return cls


def _get_active_context(cls, val=None, required=True):
    ctx_info = _context_registry().get(cls)
    if val is not None:
        assert isinstance(val, cls), (
            'Wrong context type. Expected: %s, got %s.' % (cls, type(val)))
        return val
    return ctx_info.get_active(required=required)
