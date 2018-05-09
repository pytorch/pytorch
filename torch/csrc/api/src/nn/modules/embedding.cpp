#include <torch/nn/modules/embedding.h>

namespace torch { namespace nn {

Embedding::Embedding(uint32_t num_embeddings, uint32_t embedding_dim)
    : CloneableModule<Embedding>("Embedding"),
      num_embeddings(num_embeddings),
      embedding_dim(embedding_dim) {}

variable_list Embedding::forward(variable_list input) {
  auto x = input[0];
  return variable_list({at::embedding(weight, x, -1, false, false)});
}

void Embedding::reset_parameters() {
  for (auto& p : parameters()) {
    p.second.data().normal_(0, 1);
  }
}

void Embedding::initialize_parameters() {
  weight = this->add(
      Var(at::CPU(at::kFloat).empty({num_embeddings, embedding_dim})),
      "weight");
}

}} // namespace torch::nn
