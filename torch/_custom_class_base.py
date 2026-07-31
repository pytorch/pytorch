import threading
import weakref
from contextvars import ContextVar, Token


# Cached lazily on first __instancecheck__ miss to avoid an import cycle at
# module load (FakeScriptObject's module imports torch, which imports us).
_FakeScriptObject_cls: type | None = None
# Weakly track allocation order so FX can distinguish objects that predate a
# trace from objects created while that trace is running.
_creation_serial = 0
_creation_registry: dict[int, tuple[weakref.ReferenceType[object], int]] = {}
_canonical_creation_registry: dict[
    tuple[type[object], int],
    tuple[dict[int, weakref.ReferenceType[object]], int],
] = {}
_active_creation_epoch: ContextVar[int | None] = ContextVar(
    "custom_class_creation_epoch", default=None
)
_creation_lock = threading.RLock()


def _canonical_creation_key(obj: object) -> tuple[type[object], int] | None:
    obj_type = type(obj)
    # Generator arguments can be rewrapped while retaining the same C++ object.
    if obj_type.__module__ == "torch._C" and obj_type.__name__ == "Generator":
        cdata = object.__getattribute__(obj, "_cdata")
        if isinstance(cdata, int):
            return obj_type, cdata
    return None


def _get_live_canonical_entry(
    key: tuple[type[object], int],
) -> tuple[dict[int, weakref.ReferenceType[object]], int] | None:
    entry = _canonical_creation_registry.get(key)
    if entry is None:
        return None
    refs = entry[0]
    for obj_id, ref in tuple(refs.items()):
        if ref() is None:
            del refs[obj_id]
    if not refs:
        del _canonical_creation_registry[key]
        return None
    return entry


def _record_creation(obj: object) -> int | None:
    global _creation_serial

    with _creation_lock:
        obj_id = id(obj)
        existing = _creation_registry.get(obj_id)
        if existing is not None and existing[0]() is obj:
            return existing[1]

        canonical_key = _canonical_creation_key(obj)
        canonical_entry = (
            _get_live_canonical_entry(canonical_key)
            if canonical_key is not None
            else None
        )

        def remove(ref: weakref.ReferenceType[object]) -> None:
            with _creation_lock:
                current = _creation_registry.get(obj_id)
                if current is not None and current[0] is ref:
                    del _creation_registry[obj_id]
                if canonical_key is not None:
                    current = _canonical_creation_registry.get(canonical_key)
                    if current is not None:
                        refs = current[0]
                        if refs.get(obj_id) is ref:
                            del refs[obj_id]
                        if not refs:
                            del _canonical_creation_registry[canonical_key]

        try:
            ref = weakref.ref(obj, remove)
        except TypeError:
            return None

        if canonical_entry is None:
            _creation_serial += 1
            serial = _creation_serial
        else:
            serial = canonical_entry[1]
        _creation_registry[obj_id] = (ref, serial)
        if canonical_key is not None:
            if canonical_entry is None:
                _canonical_creation_registry[canonical_key] = ({obj_id: ref}, serial)
            else:
                canonical_entry[0][obj_id] = ref
        return serial


def _enter_custom_class_creation_epoch() -> tuple[int, Token[int | None]]:
    with _creation_lock:
        epoch = _creation_serial
    active_epoch = _active_creation_epoch.get()
    if active_epoch is not None:
        epoch = min(epoch, active_epoch)
    return epoch, _active_creation_epoch.set(epoch)


def _exit_custom_class_creation_epoch(token: Token[int | None]) -> None:
    _active_creation_epoch.reset(token)


def _get_custom_class_creation_serial(obj: object) -> int | None:
    with _creation_lock:
        existing = _creation_registry.get(id(obj))
        if existing is not None and existing[0]() is obj:
            return existing[1]
        canonical_key = _canonical_creation_key(obj)
        if canonical_key is not None:
            existing = _get_live_canonical_entry(canonical_key)
            if existing is not None:
                return existing[1]
        return None


class CustomClassBaseMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        _record_creation(obj)
        return obj

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
    def __new__(cls, *args, **kwargs):
        # copy.copy calls __new__ directly and bypasses the metaclass __call__.
        new = super().__new__
        if new is object.__new__:
            obj = new(cls)
        else:
            obj = new(cls, *args, **kwargs)
        _record_creation(obj)
        return obj


# Backward compatibility aliases
OpaqueBaseMeta = CustomClassBaseMeta
OpaqueBase = CustomClassBase
