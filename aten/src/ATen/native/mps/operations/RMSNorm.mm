#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#endif
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/RMSNorm.h>
#include <fmt/format.h>

namespace at::native::mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RMSNorm_metallib.h>
#endif

Tensor rms_norm_mps_kernel(const Tensor& input,
                           c10::SymIntArrayRef normalized_shape,
                           const Tensor& weight,
                           const double eps) {
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "Expected contiguous input and weight tensors");
  auto output = at::empty_like(input);
  const int normalized_ndim = normalized_shape.size();
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int axis = input_ndim - normalized_ndim;
  const size_t M = static_cast<size_t>(c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis));
  const size_t N = static_cast<size_t>(c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend()));

  static constexpr int SIMD_SIZE = 32;
  static constexpr int N_READS = 4;
  static constexpr int LOOPED_LIMIT = 4096;
  const std::string name = N > LOOPED_LIMIT ? "rms_norm_looped" : "rms_norm";

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = fmt::format("{}_{}", name, scalarToMetalTypeString(output));
      id<MTLComputePipelineState> rms_norm_pso = lib.getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:rms_norm_pso];
      mtl_setArgs(computeEncoder, input, weight, output, eps, N, 1);

      const auto maxThreadsPerGroup = static_cast<size_t>([rms_norm_pso maxTotalThreadsPerThreadgroup]);
      size_t threadgroup_size = maxThreadsPerGroup;
      if (N <= LOOPED_LIMIT) {
        size_t threadgroup_needed = (N + N_READS - 1) / N_READS;
        size_t simds_needed = (threadgroup_needed + SIMD_SIZE - 1) / SIMD_SIZE;
        size_t threadgroup_size = SIMD_SIZE * simds_needed;
        assert(threadgroup_size <= maxThreadsPerGroup);
      }
      size_t n_threads = M * threadgroup_size;

      [computeEncoder dispatchThreads:MTLSizeMake(n_threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    }
  });

  return output;
}

} // namespace at::native::mps
