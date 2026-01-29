class OpaqueBaseMeta(type):
    def __instancecheck__(cls, instance):
        from torch._library.fake_class_registry import FakeScriptObject

        return super().__instancecheck__(instance) or (
            isinstance(instance, FakeScriptObject)
            and super().__instancecheck__(instance.real_obj)
        )


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass
