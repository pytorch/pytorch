#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"


namespace at { namespace native {

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  AT_ASSERT(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got %d and %d",
            input1.dim(), input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    AT_ASSERT(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim %d: got %d and %d",
              i, input1.size(i), input2.size(i));
  }
  AT_ASSERT(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got %d but expected %d",
            input1.size(input1.dim() - 1), weight.size(1));
  AT_ASSERT(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got %d but expected %d",
            input2.size(input2.dim() - 1), weight.size(2));
  AT_ASSERT(bias.defined() && bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got %d but expected %d",
            bias.size(0), weight.size(0));

  auto b_input1 = input1.unsqueeze(-2).unsqueeze(-2);
  auto b_input2 = input2.unsqueeze(-2).unsqueeze(-1);

  auto output = at::matmul(at::matmul(b_input1, weight), b_input2);
  output = output.squeeze(-1).squeeze(-1);
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

}}  // namespace at::native
