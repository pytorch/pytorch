# The torch <-> cute dtype correspondence, for every CuteDSL native op.
#
# Its own module because it is the one piece of the DSL layer that is neither launch glue nor kernel:
# a trait names its accumulator in cute types, a driver allocates the matching scratch in torch, and
# an override reads the element type of whatever it was handed. Those live in different packages, so
# a map that sat with any one of them would be imported for its dtype table alone.
#
# NOT the vendored `torch/_vendor/quack/cute_dsl_utils.py::torch2cute_dtype_map`, which is third-party
# and differs where it matters: it maps torch.bool to Uint8 rather than Boolean, and carries no
# float64. Two maps with the same name and different semantics would be worse than one of each.
#
# Imports cutlass at module scope, so it stays off the `import torch` path -- bind it lazily or from
# inside a function body (see test_no_dsl_imports_after_import_torch).

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
