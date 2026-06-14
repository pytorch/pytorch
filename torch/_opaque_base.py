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

    def __call__(cls, *args, **kwargs):
        from torch._library.opaque_object import is_opaque_reference_type

        # Reference types registered via the metaclass alone may not inherit
        # OpaqueBase, so read the opt-out defensively.
        allow_mid_trace = getattr(
            cls, "_allow_opaque_reference_mid_trace_creation", False
        )
        if is_opaque_reference_type(cls) and not allow_mid_trace:
            from torch._guards import active_fake_mode

            if active_fake_mode() is not None:
                from torch.fx.experimental.proxy_tensor import get_proxy_mode

                if get_proxy_mode() is not None:
                    raise RuntimeError(
                        f"A reference-type opaque object ({cls.__name__}) was "
                        f"created during tracing (e.g. inside "
                        f"register_torch_dispatch). This would bake the object "
                        f"into the compiled graph as a constant with no guards, "
                        f"and might cause silent correctness errors on subsequent "
                        f"calls with different inputs. Create the opaque object "
                        f"before torch.compile and pass it as an argument. If "
                        f"instances of this type are always safe to create "
                        f"mid-trace, set "
                        f"`_allow_opaque_reference_mid_trace_creation = True` on "
                        f"the class to opt out of this check."
                    )
        return super().__call__(*args, **kwargs)


class OpaqueBase(metaclass=OpaqueBaseMeta):
    pass
