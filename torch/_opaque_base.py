_FakeScriptObject = None


def _get_fake_script_object():
    global _FakeScriptObject
    if _FakeScriptObject is None:
        from torch._library.fake_class_registry import FakeScriptObject

        _FakeScriptObject = FakeScriptObject
    return _FakeScriptObject


class OpaqueBaseMeta(type):
    def __instancecheck__(cls, instance):
        if super().__instancecheck__(instance):
            return True
        FakeScriptObject = _get_fake_script_object()
        return isinstance(instance, FakeScriptObject) and super().__instancecheck__(
            instance.real_obj
        )


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass
