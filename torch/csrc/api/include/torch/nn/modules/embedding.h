#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class Embedding : public torch::nn::CloneableModule<Embedding> {
 public:
  Embedding(uint32_t num_embeddings, uint32_t embedding_dim);

  variable_list forward(variable_list) override;

  uint32_t num_embeddings, embedding_dim;
  Variable weight;
};
}} // namespace torch::nn
