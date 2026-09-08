# The torch <-> cute dtype correspondence, for every CuteDSL native op. Its own module because
# a trait names its accumulator in cute types, a driver allocates matching torch scratch, and an
# override reads what it was handed -- three different packages.
#
# NOT the vendored quack `torch2cute_dtype_map`, which maps torch.bool to Uint8 rather than
# Boolean and carries no float64. Imports cutlass at module scope, so bind it lazily (see
# test_no_dsl_imports_after_import_torch).

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

# The inverse, for sizing a SCRATCH buffer from an accumulator type (a trait's field dtypes are
# cute types, and a partials buffer has to be allocated in torch).
cute2torch = {v: k for k, v in torch2cute.items()}
