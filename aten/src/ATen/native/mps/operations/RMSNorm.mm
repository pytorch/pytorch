#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/rms_norm_native.h>
#endif
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

namespace at::native {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RMSNorm_metallib.h>
#endif

static Tensor rms_norm_mps_kernel(const Tensor& input,
                                  IntArrayRef normalized_shape,
                                  const Tensor& weight,
                                  const double eps) {
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

Tensor rms_norm_mps(const Tensor& input,
                    IntArrayRef normalized_shape,
                    const std::optional<Tensor>& weight_opt,
                    const std::optional<double> eps_opt) {
  std::vector<at::SymInt> normalized_shape_symint;
  for (const auto& val : normalized_shape) {
    normalized_shape_symint.emplace_back(at::SymInt(val));
  }

  if (weight_opt.has_value()) {
    const Tensor weight = weight_opt.value();
    const bool any_nested = input.is_nested() || weight.is_nested();
    const bool any_inputs_require_grad = input.requires_grad() || weight.requires_grad();
    const bool all_contiguous = input.is_contiguous() && weight.is_contiguous();
    const bool is_input_fp = input.dtype() == kBFloat16 || input.dtype() == kHalf || input.dtype() == kFloat;
    const bool is_weight_fp = weight.dtype() == kBFloat16 || weight.dtype() == kHalf || weight.dtype() == kFloat;

    if (!(at::GradMode::is_enabled() && any_inputs_require_grad) && all_contiguous && !any_nested && is_input_fp &&
        is_weight_fp) {
      _check_rms_norm_inputs_symint(input, normalized_shape_symint, weight);
      double eps_val;
      if (!eps_opt.has_value()) {
        eps_val = std::numeric_limits<at::scalar_value_type<double>::type>::epsilon();
      } else {
        eps_val = eps_opt.value();
      }
      return rms_norm_mps_kernel(input, normalized_shape, weight, eps_val);
    }
  }

  return rms_norm_symint(input, normalized_shape_symint, weight_opt, eps_opt);
}

} // namespace at::native
