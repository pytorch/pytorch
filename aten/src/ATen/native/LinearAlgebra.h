#pragma once

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
    void linalg_solve_out_mps_impl(const Tensor& A, const Tensor& B, bool left, bool check_errors, const Tensor& result, const Tensor& LU, const Tensor& pivots, const Tensor& info);
}
} // namespace at::native
