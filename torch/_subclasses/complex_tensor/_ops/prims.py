import torch

from .._core import ComplexTensor
from .common import (
    _get_func_name,
    complex_to_real_dtype,
    register_complex,
    register_force_test,
    register_inductor,
    split_complex_tensor,
)


prims = torch.ops.prims
aten = torch.ops.aten


# TODO (hameerabbasi): Not being tested
@register_force_test(prims.convert_element_type)
def convert_element_type_impl(x: ComplexTensor, dtype: torch.dtype) -> ComplexTensor:
    dtype = complex_to_real_dtype(dtype)
    u, v = split_complex_tensor(x)
    u_out = prims.convert_element_type(u, dtype)
    v_out = prims.convert_element_type(v, dtype)

    return ComplexTensor(u_out, v_out)


@register_complex(prims.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    return aten._conj_physical(self)


@register_complex(prims.conj)
def conj_impl(self: ComplexTensor) -> ComplexTensor:
    return aten._conj(self)


INDUCTOR_OPS = [getattr(prims, op) for op in prims if op.startswith(("fft_", "_fft_"))]

for inductor_op in INDUCTOR_OPS:
    globals()[_get_func_name(inductor_op)] = register_force_test(
        inductor_op, register_inductor(inductor_op)
    )

del inductor_op
