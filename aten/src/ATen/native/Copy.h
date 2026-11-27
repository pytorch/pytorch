#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/ScalarType.h>

namespace at {

class Tensor;
struct TensorIterator;
class TensorBase;

namespace native {

using copy_fn = void (*)(TensorIterator&, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub)

TORCH_API void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src);

// Raises for casts to or from a shell dtype (e.g. float4_e2m1fn_x2), which have
// no defined value conversion. Kernels that convert without going through
// copy_ call it to reject the same casts.
TORCH_API void check_shell_dtype_cast(ScalarType src, ScalarType dst);

} // namespace native
} // namespace at
