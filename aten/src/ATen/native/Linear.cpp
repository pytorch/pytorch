#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  if (input1.dim() != input2.dim()) {
    throw std::runtime_error("Inputs should have the same number of dimensions");
  }
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    if (input1.size(i) != input2.size(i)) {
      throw std::runtime_error("Batch dimensions of inputs do not match");
    }
  }
  if (input1.size(input1.dim() - 1) != weight.size(1) || input2.size(input2.dim() - 1) != weight.size(2)) {
    throw std::runtime_error("Input sizes do not match weight sizes");
  }
  if (bias.defined() && bias.size(0) != weight.size(0)) {
    throw std::runtime_error("Bias sizes does not match weight size");
  }

  auto b_input1 = input1.unsqueeze(-2).unsqueeze(-2);
  auto b_input2 = input2.unsqueeze(-2).unsqueeze(-1);

  auto output = at::matmul(at::matmul(b_input1, weight), b_input2);
  output = output.squeeze(-1).squeeze(-2).sum(-1);
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

}}  // namespace at::native
