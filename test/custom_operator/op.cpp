#include <torch/op.h>

#include <cstddef>
#include <vector>

std::vector<at::Tensor> custom_op(
    at::Tensor tensor,
    double scalar,
    int64_t repeat) {
  std::vector<at::Tensor> output;
  output.reserve(repeat);
  for (int64_t i = 0; i < repeat; ++i) {
    output.push_back(tensor * scalar);
  }
  return output;
}

static torch::RegisterOperators registry("custom::op", &custom_op);
