#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
}

namespace at::native {

using addr_fn = void (*)(TensorIterator &, const Scalar& beta, const Scalar& alpha);
DECLARE_DISPATCH(addr_fn, addr_stub)

namespace mps {
    TORCH_API void linalg_solve_out_mps_impl(const TensorBase& A, const TensorBase& B, bool left, bool check_errors, const TensorBase& result, const TensorBase& LU, const TensorBase& pivots, const TensorBase& info);
}
} // namespace at::native
