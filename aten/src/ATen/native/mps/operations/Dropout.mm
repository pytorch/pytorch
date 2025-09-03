#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorOperators.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>
#include <ATen/ops/ones_like.h>
#endif

namespace at::native {

static Tensor native_dropout_mask_and_scale(const Tensor& input, const Tensor& mask, float scale) {
  auto output = at::empty_like(input);
  mps::binary_op_kernel("native_dropout_mask_and_scale", input, mask, output, scale);
  return output;
}

std::tuple<Tensor, Tensor> native_dropout_mps(const Tensor& input, double p, std::optional<bool> train) {
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }

  Tensor mask;
  Tensor output;

  if (!train.has_value() || *train) {
    float p_comp = 1.0f - p;
    mask = at::empty_like(input, input.options().dtype(c10::kBool));
    mask.bernoulli_(p_comp);
    auto scale = p_comp == 0 ? 0.0f : 1.0f / p_comp;
    output = native_dropout_mask_and_scale(input, mask, scale);
  } else {
    mask = at::ones_like(input, input.options().dtype(c10::kBool));
    output = input.clone();
  }
  return std::make_tuple(output, mask);
}

Tensor native_dropout_backward_mps(const Tensor& grad, const Tensor& mask, double scale) {
  return native_dropout_mask_and_scale(grad, mask, scale);
}

} // namespace at::native