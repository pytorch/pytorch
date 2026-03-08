#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Distance.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Distance.h>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
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

inline uint32_t checked_uint32(int64_t v, const char* name) {
  TORCH_CHECK(v >= 0 && static_cast<uint64_t>(v) <= std::numeric_limits<uint32_t>::max(),
              name,
              " must fit into uint32 for MPS pdist kernels, got: ",
              v);
  return static_cast<uint32_t>(v);
}

} // namespace

static void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, const double p) {
  const int64_t m = self.size(1);
  const int64_t n = (self.dim() == 2) ? self.size(0) : self.numel() / m;
  PdistForwardParams params;
  params.n = checked_uint32(n, "n");
  params.m = checked_uint32(m, "m");
  params.p = static_cast<float>(p);
  params.mode = static_cast<uint32_t>(pdist_mode(p, /*backward=*/false));
  const std::string kernel = fmt::format("pdist_forward_{}", scalarToMetalTypeString(self));

  MPSStream* mps_stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = mps_stream->commandEncoder();
      id<MTLComputePipelineState> pdist_pso = lib.getPipelineStateForFunc(kernel);

      getMPSProfiler().beginProfileKernel(pdist_pso, "pdist_forward", {self});
      [compute_encoder setComputePipelineState:pdist_pso];
      mtl_setArgs(compute_encoder, result, self, params);
      mtl_dispatch1DJob(compute_encoder, pdist_pso, result.numel());
      getMPSProfiler().endProfileKernel(pdist_pso);
    }
  });
}

static void pdist_backward_kernel_impl(Tensor& result,
                                       const Tensor& grad,
                                       const Tensor& self,
                                       const double p,
                                       const Tensor& pdist) {
  result.fill_(0);
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    return;
  }

  const int64_t n = self.size(0);
  const int64_t m = self.size(1);
  const int64_t combs = pdist.numel();
  PdistBackwardParams params;
  params.grad_stride = checked_uint32(grad.stride(0), "grad stride");
  params.n = checked_uint32(n, "n");
  params.m = checked_uint32(m, "m");
  params.p = static_cast<float>(p);
  params.mode = static_cast<uint32_t>(pdist_mode(p, /*backward=*/true));
  const std::string kernel = fmt::format("pdist_backward_{}", scalarToMetalTypeString(self));

  Tensor buffer = at::empty({n - 1, n, m}, result.options(), MemoryFormat::Contiguous);
  Tensor result_for_kernel = (self.dim() == 2) ? result : at::empty({n, m}, self.options(), MemoryFormat::Contiguous);

  MPSStream* mps_stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mps_stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = mps_stream->commandEncoder();
      id<MTLComputePipelineState> pdist_backward_pso = lib.getPipelineStateForFunc(kernel);

      getMPSProfiler().beginProfileKernel(pdist_backward_pso, "pdist_backward", {grad, self, pdist});
      [compute_encoder setComputePipelineState:pdist_backward_pso];
      mtl_setArgs(compute_encoder, buffer, grad, self, pdist, params);
      mtl_dispatch1DJob(compute_encoder, pdist_backward_pso, combs * m);
      getMPSProfiler().endProfileKernel(pdist_backward_pso);
    }
  });

  at::sum_out(result_for_kernel, buffer, 0);
  if (self.dim() != 2) {
    result.fill_(0);
    result.view({-1}).narrow(0, 0, n * m).copy_(result_for_kernel.view({-1}));
  }
}

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl)
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl)

} // namespace at::native
