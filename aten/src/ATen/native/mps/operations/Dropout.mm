#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorOperators.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>
#include <ATen/ops/ones_like.h>
#endif

namespace at::native {
namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Dropout_metallib.h>
#endif
} // namespace mps

std::tuple<Tensor, Tensor> native_dropout_mps(const Tensor& input, double p, std::optional<bool> train) {
  if (input.numel() == 0 || !train.value_or(false) || p == 0) {
    return {input.clone(), at::ones_like(input, input.options().dtype(c10::kBool))};
  }

  using namespace mps;

  TORCH_CHECK(isFloatingType(input.scalar_type()),
              "native_dropout_mps: input must be a floating-point tensor, got ",
              input.scalar_type());

  const float p_comp = static_cast<float>(1.0 - p);
  const float scale = p_comp == 0.0f ? 0.0f : 1.0f / p_comp;

  // The fused kernel walks contiguous buffers; force a contiguous copy for
  // strided inputs so we keep the kernel as a simple linear pass.
  Tensor input_c = input.contiguous();
  Tensor output = at::empty_like(input_c);
  Tensor mask = at::empty_like(input_c, input_c.options().dtype(c10::kBool));

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(std::nullopt, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("dropout_fwd_" + scalarToMetalTypeString(input_c));

    int64_t seed;
    int64_t base_offset;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      seed = static_cast<int64_t>(mps_gen->current_seed());
      base_offset = static_cast<int64_t>(mps_gen->get_offset());
      mps_gen->set_offset(base_offset + input_c.numel());
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        auto numel = static_cast<uint32_t>(input_c.numel());
        mtl_setArgs(computeEncoder,
                    output,
                    mask,
                    input_c,
                    std::array<float, 2>{p_comp, scale},
                    std::array<long, 2>{seed, base_offset});
        mtl_setBytes(computeEncoder, numel, 5);
        mtl_dispatch1DJob(computeEncoder, pso, (numel + 3) / 4);
      }
    });
  }

  return {std::move(output), std::move(mask)};
}

Tensor native_dropout_backward_mps(const Tensor& grad, const Tensor& mask, double scale) {
  TORCH_CHECK(isFloatingType(grad.scalar_type()),
              "native_dropout_backward_mps: grad must be floating-point, got ",
              grad.scalar_type());
  TORCH_CHECK(
      mask.scalar_type() == c10::kBool, "native_dropout_backward_mps: mask must be bool, got ", mask.scalar_type());

  using namespace mps;

  Tensor grad_c = grad.contiguous();
  Tensor mask_c = mask.contiguous();
  Tensor grad_input = at::empty_like(grad_c);

  if (grad_input.numel() == 0) {
    return grad_input;
  }

  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc("dropout_bwd_" + scalarToMetalTypeString(grad_c));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, grad_input, grad_c, mask_c, static_cast<float>(scale));
        mtl_dispatch1DJob(computeEncoder, pso, grad_input.numel());
      }
    });
  }

  return grad_input;
}

} // namespace at::native
