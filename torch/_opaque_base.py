import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None


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


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass


_OPAQUE_BASE_META_CALL: Any = OpaqueBaseMeta.__call__
_reference_opaque_creation_guard_lock = threading.RLock()
_reference_opaque_creation_guard_depth = 0


def _guarded_opaque_base_meta_call(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
    from torch._library.opaque_object import is_opaque_reference_type

    if is_opaque_reference_type(cls):
        from torch._guards import active_fake_mode

        if active_fake_mode() is not None:
            from torch.fx.experimental.proxy_tensor import get_proxy_mode

            if get_proxy_mode() is not None:
                raise RuntimeError(
                    f"A reference-type opaque object ({cls.__name__}) was "
                    "created during tracing. This would bake the object "
                    "into the compiled graph as a constant with no guards, "
                    "and can cause silent correctness errors on subsequent "
                    "calls with different inputs. Create the opaque object "
                    "before torch.compile and pass it as an argument."
                )
    return _OPAQUE_BASE_META_CALL(cls, *args, **kwargs)


@contextmanager
def enable_reference_opaque_creation_guard() -> Generator[None, None, None]:
    global _reference_opaque_creation_guard_depth

    with _reference_opaque_creation_guard_lock:
        if _reference_opaque_creation_guard_depth == 0:
            OpaqueBaseMeta.__call__ = _guarded_opaque_base_meta_call  # type: ignore[method-assign]
        _reference_opaque_creation_guard_depth += 1

    try:
        yield
    finally:
        with _reference_opaque_creation_guard_lock:
            _reference_opaque_creation_guard_depth -= 1
            if _reference_opaque_creation_guard_depth == 0:
                OpaqueBaseMeta.__call__ = _OPAQUE_BASE_META_CALL  # type: ignore[method-assign]
