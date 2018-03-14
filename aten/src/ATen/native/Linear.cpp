#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
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
