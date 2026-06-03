//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/vdot_native.h>
#endif

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at::native {

namespace mps {

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1,
              "1D tensors expected, but got ",
              self.dim(),
              "D and ",
              other.dim(),
              "D tensors");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(),
              "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(),
              " and ",
              other.scalar_type());
  TORCH_CHECK(self.numel() == other.numel(),
              "inconsistent tensor size, expected tensor [",
              self.numel(),
              "] and src [",
              other.numel(),
              "] to have the same number of elements, but got ",
              self.numel(),
              " and ",
              other.numel(),
              " elements respectively");
  TORCH_CHECK(self.device() == other.device(),
              "Expected all tensors to be on the same device. Found: ",
              self.device(),
              ", ",
              other.device());
}
} // namespace mps

Tensor dot_mps(const Tensor& self, const Tensor& other) {
  using namespace mps;

  if (self.numel() == 0 && other.numel() == 0) {
    return zeros({}, self.options());
  }

  dot_check(self, other);

  // Route the inner product through the hand-written Metal GEMM kernels (no
  // MPSGraph): (1, K) @ (K, 1) -> (1, 1) -> 0-D scalar. at::mm dispatches by dtype
  // (real / integer -> mps_gemm, complex -> mps_gemm_complex) and resolves conj
  // views, so dot matches metalBLAS (which also lowers dot to a matmul). bool has
  // no GEMM kernel, so compute it in int32 and cast back (sum of ANDs != 0).
  const int64_t K = self.numel();
  if (self.scalar_type() == kBool) {
    return at::mm(self.to(kInt).reshape({1, K}), other.to(kInt).reshape({K, 1}))
        .reshape({})
        .to(kBool);
  }
  return at::mm(self.reshape({1, K}), other.reshape({K, 1})).reshape({});
}

Tensor vdot_mps(const Tensor& self, const Tensor& other) {
  // For real dtypes, vdot is identical to dot
  if (!self.is_complex()) {
    return dot_mps(self, other);
  }

  return dot_mps(self.conj(), other);
}

static Tensor& addmv_out_mps_impl(const Tensor& self,
                                  const Tensor& mat,
                                  const Tensor& vec,
                                  const Scalar& beta_,
                                  const Scalar& alpha_,
                                  Tensor& result) {
  using namespace mps;

  TORCH_CHECK(mat.is_mps());
  TORCH_CHECK(vec.is_mps());
  TORCH_CHECK(result.is_mps());
  TORCH_CHECK(self.is_mps());

  if (result.numel() == 0) {
    return result;
  }

  // result = beta*self + alpha*(mat @ vec). at::mm routes the matrix-vector
  // product through the MPSGraph-free GEMM kernels (float/integer) or the complex
  // decomposition; the alpha/beta epilogue is a couple of elementwise ops. For
  // integer outputs alpha/beta truncate to the integer type (matching torch's
  // integer addmv, e.g. alpha=0.6 -> 0), since a float scalar can't scale an int
  // tensor in place.
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  const bool is_int = c10::isIntegralType(result.scalar_type(), /*includeBool=*/false);
  const Scalar a = is_int ? Scalar(alpha_.toLong()) : alpha_;
  const Scalar b = is_int ? Scalar(beta_.toLong()) : beta_;
  Tensor out = at::mm(mat, vec.unsqueeze(1)).reshape(result.sizes()).mul_(a);
  if (beta_.toComplexDouble() != 0.0) {
    out.add_(*self_, b);
  }
  result.copy_(out);
  return result;
}

TORCH_IMPL_FUNC(addmv_out_mps)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta_,
 const Scalar& alpha_,
 const Tensor& result) {
  addmv_out_mps_impl(self, mat, vec, beta_, alpha_, const_cast<Tensor&>(result));
}

} // namespace at::native
