#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_pdist_backward_native.h>
#include <ATen/ops/_pdist_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sum.h>
#endif

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Distance_metallib.h>
#endif

inline int64_t pdist_mode(double p, bool backward) {
  if (p == 1.0)
    return 1;
  if (p == 2.0)
    return 2;
  if (std::isinf(p))
    return 3;

  if (!backward && p == 0.0)
    return 0;
  if (backward && p < 2.0)
    return 5;

  return 4;
}

} // namespace

Tensor _pdist_forward_mps(const Tensor& self, const double p) {
  TORCH_CHECK(self.numel() > 0, "Input tensor is empty");
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  TORCH_CHECK(self.is_mps(), "_pdist_forward only supports MPS tensors");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "_pdist_forward only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "_pdist_forward only supports non-negative p values");

  const int64_t n = self.size(0);
  Tensor result = at::empty({0}, self.options(), MemoryFormat::Contiguous);
  if (n <= 1) {
    result.resize_({0});
    return result;
  }

  const int64_t m = self.size(1);
  const int64_t combs = n * (n - 1) / 2;
  result.resize_({combs});

  if (m == 0) {
    result.fill_(0);
    return result;
  }

  // Match CPU legacy private-op behavior for dim > 2:
  // use flattened-row indexing width m, but only compute the first combs outputs.
  const int64_t n_for_pairs = (self.dim() == 2) ? n : self.numel() / m;

  const float p_val = static_cast<float>(p);
  const int64_t mode = pdist_mode(p, /*backward=*/false);
  const std::string kernel = fmt::format("pdist_forward_{}", scalarToMetalTypeString(self));

  MPSStream* mps_stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = mps_stream->commandEncoder();
      id<MTLComputePipelineState> pdist_pso = lib.getPipelineStateForFunc(kernel);

      getMPSProfiler().beginProfileKernel(pdist_pso, "pdist_forward", {self});
      [compute_encoder setComputePipelineState:pdist_pso];
      mtl_setArgs(compute_encoder, result, self, n_for_pairs, m, p_val, mode);
      mtl_dispatch1DJob(compute_encoder, pdist_pso, result.numel());
      getMPSProfiler().endProfileKernel(pdist_pso);
    }
  });

  return result;
}

Tensor _pdist_backward_mps(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  TORCH_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  TORCH_CHECK(self.is_mps() && grad.is_mps() && pdist.is_mps(), "_pdist_backward only supports MPS tensors");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "_pdist_backward only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "_pdist_backward only supports non-negative p values");

  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return result;
  }

  const int64_t n = self.size(0);
  const int64_t m = self.size(1);
  const int64_t combs = pdist.numel();
  const int64_t grad_stride = grad.stride(0);
  const float p_val = static_cast<float>(p);
  const int64_t mode = pdist_mode(p, /*backward=*/true);
  const std::string kernel = fmt::format("pdist_backward_{}", scalarToMetalTypeString(self));

  Tensor buffer = at::empty({n - 1, n, m}, result.options(), MemoryFormat::Contiguous);
  Tensor result_for_kernel =
      (self.dim() == 2) ? result : at::empty({n, m}, self.options(), MemoryFormat::Contiguous);

  MPSStream* mps_stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = mps_stream->commandEncoder();
      id<MTLComputePipelineState> pdist_backward_pso = lib.getPipelineStateForFunc(kernel);

      getMPSProfiler().beginProfileKernel(pdist_backward_pso, "pdist_backward", {grad, self, pdist});
      [compute_encoder setComputePipelineState:pdist_backward_pso];
      mtl_setArgs(compute_encoder, buffer, grad, self, pdist, grad_stride, n, m, combs, p_val, mode);
      mtl_dispatch1DJob(compute_encoder, pdist_backward_pso, combs * m);
      getMPSProfiler().endProfileKernel(pdist_backward_pso);
    }
  });

  at::sum_out(result_for_kernel, buffer, 0);
  if (self.dim() != 2) {
    result.fill_(0);
    result.view({-1}).narrow(0, 0, n * m).copy_(result_for_kernel.view({-1}));
  }
  return result;
}

} // namespace at::native
