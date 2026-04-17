#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eye_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Eye_metallib.h>
#endif

Tensor& eye_out_mps(int64_t n, Tensor& result) {
  // the default value of `m` equals to `n`
  return eye_out_mps(n, n, result);
}

Tensor& eye_out_mps(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});

  if (result.numel() == 0)
    return result;

  MPSStream* mpsStream = getCurrentMPSStream();
  auto stride0 = result.stride(0);
  auto stride1 = result.stride(1);

  // Small tensors: single-pass 2D kernel (one dispatch, no zero_() overhead).
  // Large tensors: zero_() + diagonal fill (memset is faster than n*m branching writes).
  constexpr int64_t kSinglePassThreshold = 1 << 22;

  if (n * m <= kSinglePassThreshold) {
    auto key = "eye_" + scalarToMetalTypeString(result);
    id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
    id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);

    // Map x to the smaller stride for coalesced writes
    bool swap = stride0 < stride1;
    auto x_stride = swap ? stride0 : stride1;
    auto y_stride = swap ? stride1 : stride0;
    auto grid_x = swap ? static_cast<NSUInteger>(n) : static_cast<NSUInteger>(m);
    auto grid_y = swap ? static_cast<NSUInteger>(m) : static_cast<NSUInteger>(n);

    dispatch_sync(mpsStream->queue(), ^() {
      @autoreleasepool {
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, result, y_stride, x_stride);

        auto maxTg = [pso maxTotalThreadsPerThreadgroup];
        auto tg_x = std::min(grid_x, maxTg);
        auto tg_y = std::min(grid_y, std::max(maxTg / tg_x, static_cast<NSUInteger>(1)));
        [computeEncoder dispatchThreads:MTLSizeMake(grid_x, grid_y, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
      }
    });
  } else {
    result.zero_();
    int64_t sz = std::min(n, m);
    int64_t diag_stride = stride0 + stride1;
    auto key = "eye_diag_" + scalarToMetalTypeString(result);
    id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
    id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);

    dispatch_sync(mpsStream->queue(), ^() {
      @autoreleasepool {
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, result, diag_stride);
        mtl_dispatch1DJob(computeEncoder, pso, sz);
      }
    });
  }

  return result;
}

} // namespace at::native
