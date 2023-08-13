import functools
import sys
import types
import typing
from typing import Any, Dict

import sympy as real_sympy

from torch.fx.immutable_collections import immutable_dict, immutable_list

# mapping from real sympy class to our proxy of it
_wrapped_classes: Dict[type, type] = dict()

# we use this to de-duplicate all the objects we create
_wrapped_objects: Dict[Any, "SympyCacheProxy"] = dict()

# caches the result of every simpy call `(fn, *_key(args))` to return value
_cache: Dict[Any, Any] = dict()


def clear_cache():
    from .cached_sympy import Symbol

    Symbol(
        "cached_sympy_impl_clear_cache_dead_object"
    )  # ensure at least one dead object
    _cache.clear()
    global _wrapped_objects
    # min refcount will be for the object above
    min_refcount = min(sys.getrefcount(v) for v in _wrapped_objects.values())
    _wrapped_objects = {
        k: v for k, v in _wrapped_objects.items() if sys.getrefcount(v) > min_refcount
    }


class _missing:
    """A sentinel value for _cache lookup misses"""


def _identity(x):
    """Helper used by `_wrap`/`_unwrap`/`_key`"""
    return x


def _error(x):
    """Helper used by `_wrap`/`_unwrap`/`_key`"""
    if isinstance(x, types.FunctionType):
        raise NotImplementedError(x)
    raise NotImplementedError(
        f"{type(x)} not handled, you may need to replace `import sympy` "
        "with `import torch.utils._sympy.cached_sympy as sympy`."
    )


def _unwrap_proxy(x: "SympyCacheProxy"):
    """Helper used by `_unwrap`"""
    return x._wrapped_value


def _make_wrap_sympy(proxy_cls):
    """
    Build a handler used by `_wrap` for a proxy class

    Args:
        proxy_cls: SympyCacheProxy subclass

    Returns:
        A function to convert `sympy.*` objects into instances of proxy_cls
    """

    def wrap(result):
        wrap_obj = _wrapped_objects.get(result)
        if wrap_obj is None:
            assert not isinstance(result, int)
            wrap_obj = SympyCacheProxy(result)
            wrap_obj.__class__ = proxy_cls
            _wrapped_objects[result] = wrap_obj
        return wrap_obj

    return wrap


def _make_constant(val):
    """Helper used by `_wrap`/`_unwrap`/`_key`"""
    return lambda x: val


def _lazy_add_cls(x):
    """Cache miss handler for `_wrap` that will lazily create the needed proxy class"""
    cls = type(x)
    if cls.__module__.startswith("sympy"):
        _setup(cls)
        return _wrap_handlers[cls](x)
    return _error(x)


_common_handlers = {
    str: _identity,
    bool: _identity,
    int: _identity,
    float: _identity,
    type(NotImplemented): _identity,
    type(None): _identity,
}
_wrap_handlers = {
    **_common_handlers,
    real_sympy.core.function.FunctionClass: lambda t: _wrapped_classes[t],
    type: lambda t: _wrapped_classes[t],
    dict: lambda result: immutable_dict(
        (_wrap(k), _wrap(v)) for k, v in result.items()
    ),
    list: lambda result: immutable_list(_wrap(x) for x in result),
    tuple: lambda result: tuple(_wrap(x) for x in result),
    # TODO(jansel): we should make set() sorted + immutable, mutation footgun here
    set: lambda result: {_wrap(x) for x in result},
}
_unwrap_handlers = {
    **_common_handlers,
    type: _unwrap_proxy,
    dict: lambda x: {_unwrap(k): _unwrap(v) for k, v in x.items()},
    tuple: lambda x: tuple(_unwrap(y) for y in x),
    list: lambda x: [_unwrap(y) for y in x],
}
_key_handlers = {
    **_common_handlers,
    type: _identity,
    dict: lambda x: tuple((_key(k), _key(v)) for k, v in x.items()),
    tuple: lambda x: tuple(_key(y) for y in x),
    list: lambda x: tuple(_key(y) for y in x),
}


def _wrap(x):
    """Take a `real_sympy.*` object and return a `cached_sympy.*` object"""
    return _wrap_handlers.get(type(x), _lazy_add_cls)(x)


def _unwrap(x):
    """Take a `cached_sympy.*` object and return a `real_sympy.*` object"""
    return _unwrap_handlers.get(type(x), _error)(x)


def _key(x):
    """Generate the key used in cache lookups"""
    return _key_handlers.get(type(x), _error)(x)


def _wrap_fn(fn):
    """
    Creates a new version of a sympy function with caching.

    Args:
        fn: a function from `real_sympy.*` to `real_sympy.*` objects

    Returns:
        a function from `cached_sympy.*` to `cached_sympy.*` objects
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        key = (fn, *_key(args), *_key(kwargs))
        result = _cache.get(key, _missing)
        if result is _missing:
            # TODO(jansel): add some perf measurement here
            result = _wrap(
                fn(
                    *[_unwrap(x) for x in args],
                    **{k: _unwrap(v) for k, v in kwargs.items()},
                )
            )
            _cache[key] = result
        return result

    return wrapped


@typing.no_type_check
def _wrap_class(cls: type):
    """
    Args:
        cls: sympy class, e.g. `sympy.Integer`

    Returns:
        A newly create class that mimics `cls` with a caching layer
    """
    if cls in _wrapped_classes:
        return _wrapped_classes[cls]

    def __new__(cls, *args, **kwargs):
        return create(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        assert self._wrapped_value is not None

    create = _wrap_fn(cls)
    proxy_cls = type(cls.__name__, tuple([_setup(c) for c in cls.__bases__]), {})
    proxy_cls._wrapped_value = cls
    proxy_cls.__new__ = __new__
    proxy_cls.__init__ = __init__
    for k, v in cls.__dict__.items():
        if (
            k
            in (
                "__new__",
                "__init__",
                "__setstate__",
                "__getstate__",
                "__reduce__",
                "__reduce_ex__",
                "__init_subclass__",
            )
            or k in SympyCacheProxy.__dict__
        ):
            continue
        elif isinstance(v, property):
            # v.fset intentionally ignored
            setattr(proxy_cls, k, property(_wrap_fn(v.fget)))
        elif isinstance(v, staticmethod):
            setattr(proxy_cls, k, staticmethod(_wrap_fn(v.__func__)))
        elif isinstance(v, classmethod):
            setattr(proxy_cls, k, staticmethod(_wrap_fn(getattr(cls, k))))
        elif callable(v):
            setattr(proxy_cls, k, _wrap_fn(v))

    _wrapped_classes[cls] = proxy_cls
    _wrap_handlers[cls] = _make_wrap_sympy(proxy_cls)
    _unwrap_handlers[proxy_cls] = _unwrap_proxy
    _key_handlers[proxy_cls] = _identity
    return proxy_cls


def _setup(obj: Any):
    """
    Args:
        obj: an object from the `sympy.*` namespace

    Returns:
        A proxy object that mimics `obj` with caching
    """
    if obj is object:
        return SympyCacheProxy
    if isinstance(obj, type):
        assert obj in _wrapped_classes or obj.__module__.startswith("sympy"), obj
        return _wrap_class(obj)
    if isinstance(obj, real_sympy.Basic):
        _wrap_class(type(obj))
        return _wrap(obj)
    if isinstance(obj, types.FunctionType):
        return _wrap_fn(obj)
    raise NotImplementedError(obj)


class SympyCacheProxy:
    """
    Base class for all the proxy objects created by `_wrap_class`
    """

    def __init__(self, obj):
        self._wrapped_value = obj

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if other.__class__ in _wrapped_classes:
            return self is other
        return _unwrap(self) == _unwrap(other)

    def __reduce__(self):
        # TODO(jansel): figure out why we are calling copy.deepcopy and hitting this, that sounds slow
        return (_wrap, (self._wrapped_value,))

    def replace(self, query, value, map=False, simultaneous=True, exact=None):
        if isinstance(value, types.FunctionType):

            def callback(*args, **kwargs):
                return _unwrap(
                    value(
                        *[_wrap(x) for x in args],
                        **{k: _wrap(v) for k, v in kwargs.items()},
                    )
                )

            # for callback-style replace() we can't use the cache
            return _wrap(
                _unwrap(self).replace(
                    _unwrap(query),
                    callback,
                    map=_unwrap(map),
                    simultaneous=_unwrap(simultaneous),
                    exact=_unwrap(exact),
                )
            )

        real_self = _unwrap(self)
        real_replace = real_self.__class__.replace
        return _wrap_fn(real_replace)(
            real_self,
            _unwrap(value),
            _unwrap(map),
            _unwrap(simultaneous),
            _unwrap(exact),
        )

    @classmethod
    def _install_passthrough(cls, name):
        def fn(self):
            return getattr(self._wrapped_value, name)

        setattr(cls, name, property(fn))


SympyCacheProxy._install_passthrough("name")
for _name in dir(real_sympy.true):
    if _name.startswith("is_"):
        SympyCacheProxy._install_passthrough(_name)
