#include <torch/nn/modules/embedding.h>

#include <cstdint>

namespace torch {
namespace nn {

Embedding::Embedding(int64_t count, int64_t dimension)
    : count_(count), dimension_(dimension) {}

void Embedding::reset() {
  table_ = register_parameter(
      "table", at::CPU(at::kFloat).empty({count_, dimension_}));
  table_.data().normal_(0, 1);
}

std::vector<Variable> Embedding::forward(std::vector<Variable> input) {
  return {at::embedding(table_, /*indices=*/input[0])};
}

} // namespace nn
} // namespace torch
