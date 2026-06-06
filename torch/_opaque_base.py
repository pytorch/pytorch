# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None
_MISSING = object()


def _get_pybind_opaque_base():
    for base in OpaqueBase.__mro__[1:]:
        if base.__module__ == "torch._C" and base.__name__ == "_OpaqueBase":
            return base
    raise TypeError("OpaqueBase has not been initialized with a pybind base")


def _rebuild_opaque_base(cls, newargs=(), newkwargs=None):
    if newkwargs is None:
        newkwargs = {}

    instance = cls.__new__(cls, *newargs, **newkwargs)
    if not _is_instance_of_type(instance, cls):
        raise TypeError(f"{cls.__name__}.__new__ did not return an instance")

    pybind_opaque_base = _get_pybind_opaque_base()
    pybind_opaque_base.__init__(instance)
    object.__setattr__(instance, "_opaque_base_initialized", True)
    return instance


def _set_opaque_base_state(instance, state):
    setstate = _find_pickle_method(type(instance), "__setstate__")
    if setstate is not None:
        setstate(instance, state)
        return

    if state is None:
        return

    if isinstance(state, tuple) and len(state) == 2:
        dict_state, slot_state = state
    else:
        dict_state, slot_state = state, None

    if dict_state:
        instance.__dict__.update(dict_state)
    if slot_state:
        for name, value in slot_state.items():
            setattr(instance, name, value)


def _find_pickle_method(cls, name):
    for base in cls.__mro__:
        method = base.__dict__.get(name)
        if method is None:
            continue
        if base in {OpaqueBase, object} or base.__module__ == "pybind11_builtins":
            continue
        return method
    return None


def _strip_opaque_base_state(state):
    if isinstance(state, dict):
        state = dict(state)
        state.pop("_opaque_base_initialized", None)
        return state

    if isinstance(state, tuple) and len(state) == 2:
        dict_state, slot_state = state
        return _strip_opaque_base_state(dict_state), slot_state

    return state


def _is_instance_of_type(instance, cls):
    instance_type = type(instance)
    return instance_type is cls or issubclass(instance_type, cls)


def _is_opaque_base_instance(cls, instance, base_instancecheck):
    # When checking against OpaqueBase itself (not a concrete subclass),
    # use normal inheritance first, then delegate to the registration system
    # which also covers value types and FakeScriptObject wrappers.
    if cls is OpaqueBase:
        if base_instancecheck(instance):
            return True

        from torch._library.opaque_object import is_opaque_value

        return is_opaque_value(instance)

    if base_instancecheck(instance):
        return True

    # Check FakeScriptObject before hasattr to avoid triggering custom
    # __getattr__ on arbitrary user objects (e.g. dict-like objects that
    # raise KeyError on unknown attributes).
    # e.g. test/dynamo/test_dynamic_shapes.py -k test_user_getattr1_dynamic_shapes
    global _FakeScriptObject_cls
    if _FakeScriptObject_cls is None:
        from torch._library.fake_class_registry import FakeScriptObject

        _FakeScriptObject_cls = FakeScriptObject
    if isinstance(instance, _FakeScriptObject_cls) and hasattr(instance, "real_obj"):
        return base_instancecheck(instance.real_obj)

    return False


class OpaqueBaseMeta(type):
    def __instancecheck__(cls, instance):
        return _is_opaque_base_instance(cls, instance, super().__instancecheck__)


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass


def _install_opaque_base(_PybindOpaqueBase: type) -> tuple[type, type]:
    """Install OpaqueBase on top of a pybind-registered marker base.

    Pybind assumes explicit Python bases passed to py::class_ also have pybind
    type information.  The C extension calls this with a tiny pybind class so
    pybinded opaque types can inherit from OpaqueBase directly.
    """
    global OpaqueBaseMeta, OpaqueBase

    if getattr(OpaqueBase, "_pybind_backed", False):
        return OpaqueBaseMeta, OpaqueBase

    class OpaqueBaseMeta(
        type(_PybindOpaqueBase),  # pyrefly: ignore [invalid-inheritance]
    ):
        def __call__(cls: type, *args, **kwargs):
            if _needs_pybind_meta_call(cls):
                return super().__call__(*args, **kwargs)
            return _construct_python_opaque(cls, args, kwargs)

        def __instancecheck__(cls, instance):
            return _is_opaque_base_instance(cls, instance, super().__instancecheck__)

    def _construct_python_opaque(cls, args, kwargs):
        class_constructing = cls.__dict__.get("_opaque_base_constructing", _MISSING)
        cls._opaque_base_constructing = True
        try:
            instance = cls.__new__(cls, *args, **kwargs)
            if not _is_instance_of_type(instance, cls):
                return instance

            init = _find_python_init(cls)
            if init is not None:
                init(instance, *args, **kwargs)
            _ensure_opaque_base_initialized(instance)
            return instance
        finally:
            if class_constructing is _MISSING:
                delattr(cls, "_opaque_base_constructing")
            else:
                cls._opaque_base_constructing = class_constructing

    class OpaqueBase(_PybindOpaqueBase, metaclass=OpaqueBaseMeta):
        def __init__(self, *args, **kwargs):
            _ensure_opaque_base_initialized(self)

            init = _find_python_init_after_opaque_base(type(self))
            if init is not None:
                init(self, *args, **kwargs)

        def __reduce_ex__(self, protocol):
            if _needs_pybind_meta_call(type(self)):
                return object.__reduce_ex__(self, protocol)

            reduce = _find_pickle_method(type(self), "__reduce__")
            if reduce is not None:
                return reduce(self)

            getnewargs_ex = _find_pickle_method(type(self), "__getnewargs_ex__")
            if getnewargs_ex is not None:
                newargs, newkwargs = getnewargs_ex(self)
            else:
                getnewargs = _find_pickle_method(type(self), "__getnewargs__")
                newargs = getnewargs(self) if getnewargs is not None else ()
                newkwargs = {}

            getstate = _find_pickle_method(type(self), "__getstate__")
            state = (
                getstate(self) if getstate is not None else object.__getstate__(self)
            )
            state = _strip_opaque_base_state(state)
            return _rebuild_opaque_base, (type(self), newargs, newkwargs), state

        def __setstate__(self, state):
            _set_opaque_base_state(self, state)

    def _needs_pybind_meta_call(cls):
        if OpaqueBase not in cls.__mro__:
            return True
        for base in cls.__mro__:
            if base is OpaqueBase:
                return False
            if _is_pybind_init(base.__dict__.get("__init__")):
                return True
        return False

    def _find_python_init(cls):
        for base in cls.__mro__:
            if (
                base in {OpaqueBase, _PybindOpaqueBase, object}
                or base.__module__ == "pybind11_builtins"
            ):
                continue
            init = base.__dict__.get("__init__")
            if init is None or init is object.__init__ or _is_pybind_init(init):
                continue
            return init
        return None

    def _find_python_init_after_opaque_base(cls):
        after_opaque_base = False
        for base in cls.__mro__:
            if base is OpaqueBase:
                after_opaque_base = True
                continue
            if not after_opaque_base:
                continue
            if (
                base in {_PybindOpaqueBase, object}
                or base.__module__ == "pybind11_builtins"
            ):
                continue
            init = base.__dict__.get("__init__")
            if init is None or init is object.__init__ or _is_pybind_init(init):
                continue
            return init
        return None

    def _is_pybind_init(init):
        return type(init).__name__ == "instancemethod"

    def _ensure_opaque_base_initialized(instance):
        try:
            initialized = object.__getattribute__(instance, "_opaque_base_initialized")
        except AttributeError:
            initialized = False
        if initialized:
            return
        _PybindOpaqueBase.__init__(instance)
        object.__setattr__(instance, "_opaque_base_initialized", True)

    def _pybind_instancecheck(cls, instance):
        if OpaqueBase in getattr(cls, "__mro__", ()):
            return _is_opaque_base_instance(
                cls, instance, lambda obj: type.__instancecheck__(cls, obj)
            )
        return type.__instancecheck__(cls, instance)

    type(
        _PybindOpaqueBase
    ).__instancecheck__ = _pybind_instancecheck  # pyrefly: ignore [bad-assignment]
    OpaqueBase._pybind_backed = True
    OpaqueBaseMeta.__qualname__ = "OpaqueBaseMeta"
    OpaqueBase.__qualname__ = "OpaqueBase"
    globals()["OpaqueBaseMeta"] = OpaqueBaseMeta
    globals()["OpaqueBase"] = OpaqueBase
    return OpaqueBaseMeta, OpaqueBase
