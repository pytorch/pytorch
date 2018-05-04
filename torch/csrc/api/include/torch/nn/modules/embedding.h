#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class Embedding : public torch::nn::CloneableModule<Embedding> {
 public:
  Embedding(uint32_t num_embeddings, uint32_t embedding_dim)
      : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {}

  variable_list forward(variable_list) override;
  void reset_parameters() override;
  void initialize_parameters() override;

  Variable weight;
  uint32_t num_embeddings, embedding_dim;
};
}} // namespace torch::nn
