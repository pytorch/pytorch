//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/vdot_native.h>
#endif

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at::native {

namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReduceOps_metallib.h>
#endif

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

  const auto dtype = self.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(dtype) || dtype == kComplexFloat ||
                                  (isIntegralType(dtype, false) && !isBarebonesUnsignedType(dtype)),
                              "\"dot\" not implemented for '",
                              dtype,
                              "'");

  if (self.numel() == 0 & other.numel() == 0) {
    return zeros({}, self.options());
  }

  dot_check(self, other);

  using namespace std::string_view_literals;
  const auto acc_type = isIntegralType(dtype, false) ? (dtype == kLong ? kLong : kInt) : toOpMathType(dtype);
  const auto numel = self.numel();
  const auto num_partials =
      std::min<int64_t>(DOT_MAX_TGS, at::ceil_div<int64_t>(numel, MAX_THREADGROUP_SIZE * SUM_NCHAINS));
  const DotParams<> params{.numel = numel,
                           .num_tgs = num_partials,
                           .a_stride = self.stride(0),
                           .b_stride = other.stride(0),
                           .a_conj = self.is_conj(),
                           .b_conj = other.is_conj()};
  const auto partials = at::empty({num_partials}, self.options().dtype(acc_type));
  const auto layout = self.stride(0) == 1 && other.stride(0) == 1 ? "dense"sv : "strided"sv;
  const auto kernel = fmt::format("dot_reduce_{}_{}", layout, scalarToMetalTypeString(dtype));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc(kernel);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, other, partials, params);
      mtl_dispatch1DJob(encoder, pso, num_partials * [pso maxTotalThreadsPerThreadgroup]);
    }
  });
  return (num_partials == 1 ? partials.squeeze(0) : partials.sum(acc_type)).to(dtype);
}

Tensor vdot_mps(const Tensor& self, const Tensor& other) {
  // For real dtypes, vdot is identical to dot
  if (!self.is_complex()) {
    return dot_mps(self, other);
  }

  return dot_mps(self.conj(), other);
}

TORCH_IMPL_FUNC(addmv_out_mps)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta_,
 const Scalar& alpha_,
 const Tensor& result) {
  auto out = result.unsqueeze(1);
  at::addmm_out(out, self.expand({mat.size(0)}).unsqueeze(1), mat, vec.unsqueeze(1), beta_, alpha_);
}

} // namespace at::native
