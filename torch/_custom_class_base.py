# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None
_MISSING_PICKLE_METHOD = object()


def _get_pybind_custom_class_base():
    for base in CustomClassBase.__mro__[1:]:
        if base.__module__ == "torch._C" and base.__name__ == "_OpaqueBase":
            return base
    raise TypeError("CustomClassBase has not been initialized with a pybind base")


def _rebuild_custom_class_base(cls, newargs=(), newkwargs=None):
    if newkwargs is None:
        newkwargs = {}

    instance = cls.__new__(cls, *newargs, **newkwargs)
    if not _is_instance_of_type(instance, cls):
        raise TypeError(f"{cls.__name__}.__new__ did not return an instance")

    _ensure_custom_class_base_initialized(instance)
    return instance


def _set_custom_class_base_state(instance, state):
    setstate = _find_pickle_method(instance, "__setstate__")
    if setstate is not _MISSING_PICKLE_METHOD:
        setstate(state)
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


def _find_pickle_method(instance, name):
    cls = type(instance)
    for base in cls.__mro__:
        method = base.__dict__.get(name, _MISSING_PICKLE_METHOD)
        if method is _MISSING_PICKLE_METHOD:
            continue
        if base in {CustomClassBase, object} or base.__module__ == "pybind11_builtins":
            continue
        descriptor_get = getattr(method, "__get__", None)
        return descriptor_get(instance, cls) if descriptor_get is not None else method
    return _MISSING_PICKLE_METHOD


def _get_object_state(instance):
    try:
        dict_state = dict(object.__getattribute__(instance, "__dict__"))
    except AttributeError:
        dict_state = None

    slot_state = {}
    for name in _slot_names(type(instance)):
        if name in {"__dict__", "__weakref__"}:
            continue
        try:
            slot_state[name] = object.__getattribute__(instance, name)
        except AttributeError:
            pass

    if slot_state:
        return dict_state, slot_state
    return dict_state


def _slot_names(cls):
    names = []
    for base in cls.__mro__:
        slots = base.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{base.__name__.lstrip('_')}{name}"
            names.append(name)
    return names


def _is_instance_of_type(instance, cls):
    instance_type = type(instance)
    return instance_type is cls or issubclass(instance_type, cls)


def _is_custom_class_base_instance(cls, instance, base_instancecheck):
    # When checking against CustomClassBase itself (not a concrete subclass),
    # use normal inheritance first, then delegate to the registration system
    # which also covers constant types and FakeScriptObject wrappers.
    if cls is CustomClassBase:
        if base_instancecheck(instance):
            return True

        from torch._library.opaque_object import is_custom_class_obj

        return is_custom_class_obj(instance)

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


def _ensure_custom_class_base_initialized(instance):
    _get_pybind_custom_class_base().__init__(instance)


def _ensure_custom_class_base_metaclass(cls):
    pybind_meta = type(_get_pybind_custom_class_base())
    pending = []

    def validate(current_cls):
        if not isinstance(current_cls, CustomClassBaseMeta):
            if type(current_cls) is not pybind_meta:
                raise TypeError(
                    f"{current_cls.__name__} must use a metaclass derived from "
                    "CustomClassBaseMeta"
                )
            pending.append(current_cls)
        for subclass in current_cls.__subclasses__():
            validate(subclass)

    validate(cls)
    for current_cls in pending:
        current_cls.__class__ = CustomClassBaseMeta


class CustomClassBaseMeta(type):
    def __instancecheck__(cls, instance):
        return _is_custom_class_base_instance(cls, instance, super().__instancecheck__)


class CustomClassBase(metaclass=CustomClassBaseMeta):
    pass


def _install_custom_class_base(_PybindCustomClassBase: type) -> tuple[type, type]:
    """Install CustomClassBase on top of a pybind-compatible marker base.

    Pybind assumes explicit Python bases passed to py::class_ also have pybind
    type information. The C extension provides a synthetic pybind-compatible
    base type with no C++ value holder so unrelated pybind classes can inherit
    from CustomClassBase without adding a fake C++ base subobject.
    """
    global CustomClassBaseMeta, CustomClassBase, OpaqueBaseMeta, OpaqueBase

    if getattr(CustomClassBase, "_pybind_backed", False):
        return CustomClassBaseMeta, CustomClassBase

    class CustomClassBaseMeta(
        type(_PybindCustomClassBase),  # pyrefly: ignore [invalid-inheritance]
    ):
        def __instancecheck__(cls, instance):
            return _is_custom_class_base_instance(
                cls, instance, super().__instancecheck__
            )

    class CustomClassBase(_PybindCustomClassBase, metaclass=CustomClassBaseMeta):
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            # pyrefly: ignore [no-matching-overload]
            instance = _PybindCustomClassBase.__new__(cls)
            if not _needs_pybind_meta_call(cls):
                _ensure_custom_class_base_initialized(instance)
            return instance

        def __init__(self, *args, **kwargs):
            if _needs_pybind_meta_call(type(self)):
                return
            init = _find_python_init_after_custom_class_base(type(self))
            if init is not None:
                init(self, *args, **kwargs)
            elif (args or kwargs) and not _has_python_new_before_custom_class_base(
                type(self)
            ):
                raise TypeError(f"{type(self).__name__}() takes no arguments")

        def __reduce_ex__(self, protocol):
            if _needs_pybind_meta_call(type(self)):
                return object.__reduce_ex__(self, protocol)

            reduce = _find_pickle_method(self, "__reduce__")
            if reduce is not _MISSING_PICKLE_METHOD:
                return reduce()

            getnewargs_ex = _find_pickle_method(self, "__getnewargs_ex__")
            if getnewargs_ex is not _MISSING_PICKLE_METHOD:
                newargs, newkwargs = getnewargs_ex()
            else:
                getnewargs = _find_pickle_method(self, "__getnewargs__")
                newargs = (
                    getnewargs() if getnewargs is not _MISSING_PICKLE_METHOD else ()
                )
                newkwargs = {}

            getstate = _find_pickle_method(self, "__getstate__")
            state = (
                getstate()
                if getstate is not _MISSING_PICKLE_METHOD
                else _get_object_state(self)
            )
            return _rebuild_custom_class_base, (type(self), newargs, newkwargs), state

        def __setstate__(self, state):
            _set_custom_class_base_state(self, state)

    def _needs_pybind_meta_call(cls):
        if CustomClassBase not in cls.__mro__:
            return True
        for base in cls.__mro__:
            if base is CustomClassBase:
                return False
            if _is_pybind_init(base.__dict__.get("__init__")):
                return True
        return False

    def _find_python_init_after_custom_class_base(cls):
        after_custom_class_base = False
        for base in cls.__mro__:
            if base is CustomClassBase:
                after_custom_class_base = True
                continue
            if not after_custom_class_base:
                continue
            if (
                base in {_PybindCustomClassBase, object}
                or base.__module__ == "pybind11_builtins"
            ):
                continue
            init = base.__dict__.get("__init__")
            if init is None or init is object.__init__ or _is_pybind_init(init):
                continue
            return init
        return None

    def _has_python_new_before_custom_class_base(cls):
        for base in cls.__mro__:
            if base is CustomClassBase:
                return False
            if "__new__" in base.__dict__:
                return True
        return False

    def _is_pybind_init(init):
        # This is pybind11's internal method wrapper type name. PyTorch vendors
        # pybind11, so the dependency is stable and covered by opaque tests.
        return type(init).__name__ == "instancemethod"

    CustomClassBase._pybind_backed = True
    CustomClassBaseMeta.__qualname__ = "CustomClassBaseMeta"
    CustomClassBase.__qualname__ = "CustomClassBase"
    globals()["CustomClassBaseMeta"] = CustomClassBaseMeta
    globals()["CustomClassBase"] = CustomClassBase
    globals()["OpaqueBaseMeta"] = CustomClassBaseMeta
    globals()["OpaqueBase"] = CustomClassBase
    return CustomClassBaseMeta, CustomClassBase


# Backward compatibility aliases.
OpaqueBaseMeta = CustomClassBaseMeta
OpaqueBase = CustomClassBase
_install_opaque_base = _install_custom_class_base
_rebuild_opaque_base = _rebuild_custom_class_base
