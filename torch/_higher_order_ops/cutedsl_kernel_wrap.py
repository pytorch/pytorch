"""User-defined CuTeDSL kernels in PT2: the counterpart of triton_kernel_wrap.py."""

from torch._higher_order_ops.user_kernel_hop import ConstantRules


def _is_cutlass_type(cls: type) -> bool:
    """Whether ``cls`` comes from the cutlass package (not cutlass_library, say)."""
    module = getattr(cls, "__module__", "") or ""
    return module == "cutlass" or module.startswith("cutlass.")


class _CuteDSLConstantRules(ConstantRules):
    def is_dsl_value(self, value: object) -> bool:
        return _is_cutlass_type(type(value))

    def scalar_value(self, value: object) -> bool | int | float | None:
        # A cutlass Numeric, such as cutlass.Float32(2.0), is keyed by its value.
        inner = getattr(value, "value", None) if self.is_dsl_value(value) else None
        return inner if isinstance(inner, (bool, int, float)) else None


CUTEDSL_CONSTANT_RULES = _CuteDSLConstantRules()
