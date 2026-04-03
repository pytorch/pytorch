class OpaqueBaseMeta(type):
    def __instancecheck__(cls, instance):
        if super().__instancecheck__(instance):
            return True

        if hasattr(instance, "real_obj"):
            from torch._library.fake_class_registry import FakeScriptObject

            if isinstance(instance, FakeScriptObject):
                return super().__instancecheck__(instance.real_obj)

        return False


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass
