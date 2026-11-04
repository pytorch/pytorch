# Shared torch <-> cute dtype mapping for traits, scratch allocation, and overrides. This is
# distinct from quack's map, which maps bool to Uint8 and omits float64. Import lazily because
# this module imports cutlass.

import cutlass
from cutlass import Float32, Float64, Int32

import torch


# torch dtype -> cute numeric type. Extend as new dtypes are supported.
torch2cute = {
    torch.float32: Float32,
    torch.float64: Float64,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: Int32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
}

# Inverse mapping for allocating torch scratch from a trait's cute accumulator dtype.
cute2torch = {v: k for k, v in torch2cute.items()}
