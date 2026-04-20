class OpaqueBaseMeta(type):
    def __instancecheck__(cls, instance):
        # When checking against OpaqueBase itself (not a concrete subclass),
        # delegate to the registration system which correctly covers all
        # opaque types (value types, metaclass-only reference types, and
        # FakeScriptObject wrappers).
        if cls is OpaqueBase:
            from torch._library.opaque_object import is_opaque_value

            return is_opaque_value(instance)

        if super().__instancecheck__(instance):
            return True

        from torch._library.fake_class_registry import FakeScriptObject

        # Check FakeScriptObject before hasattr to avoid triggering custom
        # __getattr__ on arbitrary user objects (e.g. dict-like objects that
        # raise KeyError on unknown attributes). FakeScriptObject has
        # well-defined attribute access so hasattr is safe on it.
        if isinstance(instance, FakeScriptObject) and hasattr(instance, "real_obj"):
            return super().__instancecheck__(instance.real_obj)

        return False


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass
