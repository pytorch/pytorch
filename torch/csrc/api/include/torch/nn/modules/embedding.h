#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

struct EmbeddingOptions {
  EmbeddingOptions(int64_t count, int64_t dimension);
  TORCH_ARG(int64_t, count);
  TORCH_ARG(int64_t, dimension);
};

class EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  explicit EmbeddingImpl(EmbeddingOptions options);

  void reset() override;
  Tensor forward(Tensor);

  EmbeddingOptions options;
  Tensor table;
};

TORCH_MODULE(Embedding);

} // namespace nn
} // namespace torch
