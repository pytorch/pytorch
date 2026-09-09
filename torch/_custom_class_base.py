# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None


class CustomClassBaseMeta(type):
    def __instancecheck__(cls, instance):
        # When checking against CustomClassBase itself (not a concrete subclass),
        # delegate to the registration system which correctly covers all
        # custom classes (constant types, metaclass-only symbolic types, and
        # FakeScriptObject wrappers).
        if cls is CustomClassBase:
            from torch._library.opaque_object import is_custom_class_obj

            return is_custom_class_obj(instance)

        if super().__instancecheck__(instance):
            return True

        # Check FakeScriptObject before hasattr to avoid triggering custom
        # __getattr__ on arbitrary user objects (e.g. dict-like objects that
        # raise KeyError on unknown attributes).
        # e.g. test/dynamo/test_dynamic_shapes.py -k test_user_getattr1_dynamic_shapes
        global _FakeScriptObject_cls
        if _FakeScriptObject_cls is None:
            from torch._library.fake_class_registry import FakeScriptObject

            _FakeScriptObject_cls = FakeScriptObject
        if isinstance(instance, _FakeScriptObject_cls) and hasattr(
            instance, "real_obj"
        ):
            return super().__instancecheck__(instance.real_obj)

        return False


class CustomClassBase(metaclass=CustomClassBaseMeta):
    pass


# Backward compatibility aliases
OpaqueBaseMeta = CustomClassBaseMeta
OpaqueBase = CustomClassBase
