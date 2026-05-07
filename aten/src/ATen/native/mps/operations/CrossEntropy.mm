#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/CrossEntropyKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_mps_fused_cross_entropy_backward_native.h>
#include <ATen/ops/_mps_fused_cross_entropy_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& ce_lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/CrossEntropyKernel_metallib.h>
static auto& ce_lib = lib;
#endif

} // namespace mps

std::tuple<Tensor, Tensor> _mps_fused_cross_entropy_forward(
    const Tensor& logits,
    const Tensor& target,
    int64_t ignore_index,
    double label_smoothing) {
  using namespace mps;
  TORCH_CHECK(
      logits.dim() == 2,
      "fused cross entropy expects 2D logits, got ",
      logits.dim(),
      "D");
  TORCH_CHECK(
      logits.is_contiguous(),
      "fused cross entropy expects contiguous logits");
  TORCH_CHECK(
      logits.scalar_type() == kFloat || logits.scalar_type() == kHalf ||
          logits.scalar_type() == kBFloat16,
      "fused cross entropy supports float32/float16/bfloat16, got ",
      logits.scalar_type());

  int64_t B = logits.size(0);
  int64_t V = logits.size(1);

  auto loss = at::empty({B}, logits.options().dtype(kFloat));
  auto lse = at::empty({B}, logits.options().dtype(kFloat));

  Tensor target_i64 = target.to(kLong);

  CrossEntropyParams params = {};
  params.vocab_size = static_cast<uint32_t>(V);
  params.batch_size = static_cast<uint32_t>(B);
  params.ignore_index = static_cast<int32_t>(ignore_index);
  params.label_smoothing = static_cast<float>(label_smoothing);

  std::string kname = fmt::format(
      "cross_entropy_forward_{}", scalarToMetalTypeString(logits));

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pso = ce_lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, logits, target_i64, loss, lse, params);

      uint32_t tg_size = 1024;
      tg_size = std::min(
          tg_size,
          static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
      tg_size = (tg_size / [pso threadExecutionWidth]) *
          [pso threadExecutionWidth];

      [computeEncoder
          dispatchThreadgroups:MTLSizeMake(B, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
  });

  return {loss, lse};
}

Tensor _mps_fused_cross_entropy_backward(
    const Tensor& grad_output,
    const Tensor& logits,
    const Tensor& target,
    const Tensor& lse,
    int64_t ignore_index,
    double label_smoothing) {
  using namespace mps;

  int64_t B = logits.size(0);
  int64_t V = logits.size(1);

  Tensor grad_out_c = grad_output.contiguous();
  auto grad_input = at::empty_like(logits);
  Tensor target_i64 = target.to(kLong);

  CrossEntropyParams params = {};
  params.vocab_size = static_cast<uint32_t>(V);
  params.batch_size = static_cast<uint32_t>(B);
  params.ignore_index = static_cast<int32_t>(ignore_index);
  params.label_smoothing = static_cast<float>(label_smoothing);

  std::string kname = fmt::format(
      "cross_entropy_backward_{}", scalarToMetalTypeString(logits));

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pso = ce_lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(
          computeEncoder, grad_out_c, logits, target_i64, lse, grad_input,
          params);

      uint32_t tg_size = 1024;
      tg_size = std::min(
          tg_size,
          static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
      tg_size = (tg_size / [pso threadExecutionWidth]) *
          [pso threadExecutionWidth];

      [computeEncoder
          dispatchThreadgroups:MTLSizeMake(B, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
  });

  return grad_input;
}

} // namespace at::native
