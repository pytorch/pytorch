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
    : options(std::move(options)) {
  reset();
}

void EmbeddingImpl::reset() {
  table = register_parameter(
      "table", torch::empty({options.count_, options.dimension_}));
  table.data().normal_(0, 1);
}

Tensor EmbeddingImpl::forward(Tensor input) {
  return torch::embedding(table, /*indices=*/input);
}
} // namespace nn
} // namespace torch
