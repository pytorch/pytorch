//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/group_norm.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GroupNorm.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GroupNorm_metallib.h>
#endif

} // namespace mps

static void GroupNormKernelImpl(const Tensor& X,
                                const Tensor& gamma,
                                const Tensor& beta,
                                int64_t N,
                                int64_t C,
                                int64_t HxW,
                                int64_t group,
                                double eps,
                                Tensor& Y,
                                Tensor& mean,
                                Tensor& rstd) {
  using namespace mps;
  // The following asserts are satisfied by the caller of `native_group_norm`
  TORCH_INTERNAL_ASSERT(X.numel() == N * C * HxW);
  TORCH_INTERNAL_ASSERT(!gamma.defined() || gamma.numel() == C);
  TORCH_INTERNAL_ASSERT(!beta.defined() || beta.numel() == C);
  TORCH_INTERNAL_ASSERT(X.is_contiguous());
  TORCH_INTERNAL_ASSERT(!(gamma.defined() && beta.defined()) || (gamma.scalar_type() == beta.scalar_type()));

  if (N == 0) {
    return;
  }

  uint32_t channels_per_group = C / group;
  uint32_t elements_per_group = channels_per_group * HxW;

  GroupNormParams params;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = channels_per_group;
  params.elements_per_group = elements_per_group;
  params.eps = eps;

  // Create one threadgroup for each separate group of the input.
  uint32_t num_threadgroups = N * group;

  // Within a threadgroup, create one thread for each block, rounded up to the
  // nearest multiple of 32 so we use the whole simdgroup, but with 1024 threads
  // maximum.
  uint32_t blocks_per_threadgroup = (elements_per_group + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint32_t simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  uint32_t threads_per_threadgroup = std::min(uint32_t(1024), 32 * simdgroups_per_threadgroup);

  MPSStream* stream = getCurrentMPSStream();
  auto gamma_opt = gamma.defined() ? std::make_optional(gamma) : std::nullopt;
  auto beta_opt = beta.defined() ? std::make_optional(beta) : std::nullopt;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state =
          lib.getPipelineStateForFunc(fmt::format("group_norm_{}_{}_{}",
                                                  scalarToMetalTypeString(X),
                                                  gamma.defined() ? scalarToMetalTypeString(gamma) : "void",
                                                  beta.defined() ? scalarToMetalTypeString(beta) : "void"));
      getMPSProfiler().beginProfileKernel(pipeline_state, "group_norm", {X});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, Y, mean, rstd, X, gamma_opt, beta_opt, params);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);

} // namespace at::native
