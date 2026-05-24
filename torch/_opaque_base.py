# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None


def _is_opaque_base_instance(cls, instance, base_instancecheck):
    # When checking against OpaqueBase itself (not a concrete subclass),
    # delegate to the registration system which correctly covers all
    # opaque types (value types, metaclass-only reference types, and
    # FakeScriptObject wrappers).
    if cls is OpaqueBase:
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
        def __call__(cls, *args, **kwargs):
            # pybind11_type.__call__ requires pybind base __init__ to run for
            # Python subclasses.  OpaqueBase is only a marker for those classes.
            return type.__call__(cls, *args, **kwargs)

        def __instancecheck__(cls, instance):
            return _is_opaque_base_instance(cls, instance, super().__instancecheck__)

    class OpaqueBase(_PybindOpaqueBase, metaclass=OpaqueBaseMeta):
        pass

    OpaqueBase._pybind_backed = True
    OpaqueBaseMeta.__qualname__ = "OpaqueBaseMeta"
    OpaqueBase.__qualname__ = "OpaqueBase"
    globals()["OpaqueBaseMeta"] = OpaqueBaseMeta
    globals()["OpaqueBase"] = OpaqueBase
    return OpaqueBaseMeta, OpaqueBase
