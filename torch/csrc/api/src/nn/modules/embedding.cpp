#include <torch/nn/modules/embedding.h>

#include <torch/tensor.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

EmbeddingOptions::EmbeddingOptions(int64_t count, int64_t dimension)
    : count_(count), dimension_(dimension) {}

EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options)
    : options_(std::move(options)) {
  reset();
}

void EmbeddingImpl::reset() {
  table_ = register_parameter(
      "table", torch::empty({options_.count_, options_.dimension_}));
  table_.data().normal_(0, 1);
}

Tensor EmbeddingImpl::forward(Tensor input) {
  return torch::embedding(table_, /*indices=*/input);
}

const EmbeddingOptions& EmbeddingImpl::options() const noexcept {
  return options_;
}

} // namespace nn
} // namespace torch
