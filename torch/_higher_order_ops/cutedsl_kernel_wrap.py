"""User-defined CuTeDSL kernels in PT2: the counterpart of triton_kernel_wrap.py."""

from torch._higher_order_ops.user_kernel_hop import ConstantRules


def _is_cutlass_type(cls: type) -> bool:
    """Whether ``cls`` comes from the cutlass package (not cutlass_library, say)."""
    module = getattr(cls, "__module__", "") or ""
    return module == "cutlass" or module.startswith("cutlass.")


class _CuteDSLConstantRules(ConstantRules):
    def is_dsl_value(self, value: object) -> bool:
        # Keyed by its own equality, never by its attributes: host code makes only
        # cutlass values that are singletons or compare by value (dtype classes, enum
        # members, op dataclasses); layouts and atoms exist only in a @cute.jit trace.
        return _is_cutlass_type(type(value))

    def scalar_value(self, value: object) -> bool | int | float | None:
        # A cutlass Numeric, such as cutlass.Float32(2.0), is keyed by its value; other
        # cutlass values with a .value (a cute.ScaledBasis also has a mode, an enum
        # member is a singleton) are not. Matched by name: cutlass need not be imported.
        if not any(
            c.__name__ == "Numeric" and c.__module__ == "cutlass.base_dsl.typing"
            for c in type(value).__mro__
        ):
            return None
        inner = getattr(value, "value", None)
        return inner if isinstance(inner, (bool, int, float)) else None


CUTEDSL_CONSTANT_RULES = _CuteDSLConstantRules()
