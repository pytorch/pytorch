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

template <typename idx_T = uint32_t>
static void group_norm_forward(const Tensor& X,
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
  static_assert(std::is_same_v<idx_T, uint32_t> || std::is_same_v<idx_T, uint64_t>);

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

  idx_T channels_per_group = C / group;
  idx_T elements_per_group = channels_per_group * HxW;

  GroupNormParams<idx_T> params;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = channels_per_group;
  params.elements_per_group = elements_per_group;
  params.eps = eps;

  auto N_times_group = N * group;
  TORCH_CHECK((N_times_group) < (uint64_t(1) << 32), "`N` times `num_groups` must be less than 2^32");

  // Create one threadgroup for each separate group of the input.
  idx_T num_threadgroups = N_times_group;

  // Within a threadgroup, create one thread for each block, rounded up to the
  // nearest multiple of 32 so we use the whole simdgroup, but with 1024 threads
  // maximum.
  idx_T blocks_per_threadgroup = (elements_per_group + idx_T(BLOCK_SIZE) - 1) / idx_T(BLOCK_SIZE);
  idx_T simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  idx_T threads_per_threadgroup = std::min(idx_T(1024), 32 * simdgroups_per_threadgroup);

  MPSStream* stream = getCurrentMPSStream();
  auto gamma_opt = gamma.defined() ? std::make_optional(gamma) : std::nullopt;
  auto beta_opt = beta.defined() ? std::make_optional(beta) : std::nullopt;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state =
          lib.getPipelineStateForFunc(fmt::format("group_norm_{}_{}_{}_{}_{}",
                                                  scalarToMetalTypeString(X),
                                                  scalarToMetalTypeString(mean),
                                                  scalarToMetalTypeString(gamma_opt),
                                                  scalarToMetalTypeString(beta_opt),
                                                  std::is_same_v<idx_T, uint32_t> ? "uint32_t" : "uint64_t"));
      getMPSProfiler().beginProfileKernel(pipeline_state, "group_norm", {X});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, Y, mean, rstd, X, gamma_opt, beta_opt, params);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

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
  if (X.numel() >= (uint64_t(1) << 32)) {
    group_norm_forward<uint64_t>(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  } else {
    group_norm_forward(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  }
}

template <typename idx_T = uint32_t>
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
  static_assert(std::is_same_v<idx_T, uint32_t> || std::is_same_v<idx_T, uint64_t>);

  MPSStream* stream = getCurrentMPSStream();
  auto gamma_opt = gamma.defined() ? std::make_optional(gamma) : std::nullopt;
  idx_T channels_per_group = C / group;
  idx_T elements_per_group = channels_per_group * HxW;

  GroupNormParams<idx_T> params;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = channels_per_group;
  params.elements_per_group = elements_per_group;

  auto N_times_group = N * group;
  TORCH_CHECK((N_times_group) < (uint64_t(1) << 32), "`N` times `num_groups` must be less than 2^32");

  idx_T num_threadgroups = N_times_group;
  idx_T blocks_per_threadgroup = (elements_per_group + idx_T(BLOCK_SIZE) - 1) / idx_T(BLOCK_SIZE);
  idx_T simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  idx_T threads_per_threadgroup = std::min(idx_T(1024), 32 * simdgroups_per_threadgroup);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state =
          lib.getPipelineStateForFunc(fmt::format("group_norm_backward_x_{}_{}_{}_{}",
                                                  scalarToMetalTypeString(dY),
                                                  scalarToMetalTypeString(mean),
                                                  scalarToMetalTypeString(gamma_opt),
                                                  std::is_same_v<idx_T, uint32_t> ? "uint32_t" : "uint64_t"));
      getMPSProfiler().beginProfileKernel(pipeline_state, "group_norm_backward_x", {dY, X});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, dX, dY, X, mean, rstd, gamma_opt, params);
      [compute_encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

template <typename idx_T = uint32_t>
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
  static_assert(std::is_same_v<idx_T, uint32_t> || std::is_same_v<idx_T, uint64_t>);

  MPSStream* stream = getCurrentMPSStream();
  idx_T N_times_HxW = N * HxW;

  GroupNormParams<idx_T> params;
  params.N_times_HxW = N_times_HxW;
  params.C = C;
  params.HxW = HxW;
  params.num_groups = group;
  params.channels_per_group = C / group;

  idx_T blocks_per_threadgroup = (N_times_HxW + idx_T(BLOCK_SIZE) - 1) / idx_T(BLOCK_SIZE);
  idx_T simdgroups_per_threadgroup = (blocks_per_threadgroup + 31) / 32;
  idx_T threads_per_threadgroup = std::min(idx_T(1024), 32 * simdgroups_per_threadgroup);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state =
          lib.getPipelineStateForFunc(fmt::format("group_norm_backward_affine_{}_{}_{}_{}",
                                                  scalarToMetalTypeString(dY),
                                                  scalarToMetalTypeString(mean),
                                                  scalarToMetalTypeString(dgamma),
                                                  std::is_same_v<idx_T, uint32_t> ? "uint32_t" : "uint64_t"));
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
    if (X.numel() >= (uint64_t(1) << 32)) {
      group_norm_backward_x<uint64_t>(dY, X, mean, rstd, gamma, N, C, HxW, group, dX);
    } else {
      group_norm_backward_x(dY, X, mean, rstd, gamma, N, C, HxW, group, dX);
    }
  }
  if (dgamma.defined() || dbeta.defined()) {
    // If only one of either dgamma or dbeta is defined, create a temporary for
    // the other, so that the kernel doesn't need to switch on it.
    const Tensor& affine_ref = dgamma.defined() ? dgamma : dbeta;
    Tensor dgamma_out = dgamma.defined() ? dgamma : at::detail::empty_mps({C}, affine_ref.options());
    Tensor dbeta_out = dbeta.defined() ? dbeta : at::detail::empty_mps({C}, affine_ref.options());
    if (X.numel() >= (uint64_t(1) << 32)) {
      group_norm_backward_affine<uint64_t>(dY, X, mean, rstd, dgamma_out, dbeta_out, N, C, HxW, group);
    } else {
      group_norm_backward_affine(dY, X, mean, rstd, dgamma_out, dbeta_out, N, C, HxW, group);
    }
  }
}

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace at::native
