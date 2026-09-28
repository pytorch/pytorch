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
}

# Inverse mapping for allocating torch scratch from a trait's cute accumulator dtype.
cute2torch = {v: k for k, v in torch2cute.items()}
