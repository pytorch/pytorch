#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/mps/EmptyTensor.h>
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
  TORCH_INTERNAL_ASSERT(mean.scalar_type() == rstd.scalar_type());

  if (X.numel() == 0) {
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
          lib.getPipelineStateForFunc(fmt::format("group_norm_{}_{}_{}_{}",
                                                  scalarToMetalTypeString(X),
                                                  scalarToMetalTypeString(mean),
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

static void group_norm_backward_x(const Tensor& dY,
                                  const Tensor& X,
                                  const Tensor& mean,
                                  const Tensor& rstd,
                                  const Tensor& gamma,
                                  int64_t N,
                                  int64_t C,
                                  int64_t HxW,
                                  int64_t group,
                                  Tensor& dX) {
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  auto gamma_opt = gamma.defined() ? std::make_optional(gamma) : std::nullopt;
  uint32_t channels_per_group = C / group;
  uint32_t elements_per_group = channels_per_group * HxW;

  GroupNormBackwardXParams params;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = channels_per_group;
  params.elements_per_group = elements_per_group;

  uint32_t num_threadgroups = N * group;
  uint32_t blocks_per_threadgroup = (elements_per_group + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint32_t simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  uint32_t threads_per_threadgroup = std::min(uint32_t(1024), 32 * simdgroups_per_threadgroup);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state =
          lib.getPipelineStateForFunc(fmt::format("group_norm_backward_x_{}_{}_{}",
                                                  scalarToMetalTypeString(dY),
                                                  scalarToMetalTypeString(mean),
                                                  gamma.defined() ? scalarToMetalTypeString(gamma) : "void"));
      getMPSProfiler().beginProfileKernel(pipeline_state, "group_norm_backward_x", {dY, X});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, dX, dY, X, mean, rstd, gamma_opt, params);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

static void group_norm_backward_affine(const Tensor& dY,
                                       const Tensor& X,
                                       const Tensor& mean,
                                       const Tensor& rstd,
                                       Tensor& dgamma,
                                       Tensor& dbeta,
                                       int64_t N,
                                       int64_t C,
                                       int64_t HxW,
                                       int64_t group) {
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  uint32_t N_times_HxW = N * HxW;

  GroupNormBackwardAffineParams params;
  params.N_times_HxW = N_times_HxW;
  params.C = C;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = C / group;

  uint32_t blocks_per_threadgroup = (N_times_HxW + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint32_t simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  uint32_t threads_per_threadgroup = std::min(uint32_t(1024), 32 * simdgroups_per_threadgroup);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(fmt::format("group_norm_backward_affine_{}_{}_{}",
                                                                    scalarToMetalTypeString(dY),
                                                                    scalarToMetalTypeString(mean),
                                                                    scalarToMetalTypeString(dgamma)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "group_norm_backward_affine", {dY, X});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, dgamma, dbeta, dY, X, mean, rstd, params);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(C, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

static void GroupNormBackwardKernelImpl(const Tensor& dY,
                                        const Tensor& X,
                                        const Tensor& mean,
                                        const Tensor& rstd,
                                        const Tensor& gamma,
                                        int64_t N,
                                        int64_t C,
                                        int64_t HxW,
                                        int64_t group,
                                        Tensor& dX,
                                        Tensor& dgamma,
                                        Tensor& dbeta) {
  if (X.numel() == 0) {
    if (dgamma.defined()) {
      dgamma.zero_();
    }
    if (dbeta.defined()) {
      dbeta.zero_();
    }
    return;
  }
  if (dX.defined()) {
    group_norm_backward_x(dY, X, mean, rstd, gamma, N, C, HxW, group, dX);
  }
  if (dgamma.defined() || dbeta.defined()) {
    // If only one of either dgamma or dbeta is defined, allocate a temporary
    // for the other, so that the kernel doesn't need to switch on it.
    const Tensor& affine_ref = dgamma.defined() ? dgamma : dbeta;
    Tensor dgamma_out = dgamma.defined() ? dgamma : at::detail::empty_mps({C}, affine_ref.options());
    Tensor dbeta_out = dbeta.defined() ? dbeta : at::detail::empty_mps({C}, affine_ref.options());
    group_norm_backward_affine(dY, X, mean, rstd, dgamma_out, dbeta_out, N, C, HxW, group);
  }
}

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace at::native
