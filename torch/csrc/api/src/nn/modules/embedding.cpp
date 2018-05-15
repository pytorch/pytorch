#include <torch/nn/modules/embedding.h>

namespace torch { namespace nn {

Embedding::Embedding(uint32_t num_embeddings, uint32_t embedding_dim)
    : num_embeddings(num_embeddings),
      embedding_dim(embedding_dim),
      weight(Var(at::CPU(at::kFloat).empty({num_embeddings, embedding_dim}))) {
  add(weight, "weight");
  weight.data().normal_(0, 1);
}

variable_list Embedding::forward(variable_list input) {
  auto x = input[0];
  return variable_list({at::embedding(weight, x, -1, false, false)});
}

}} // namespace torch::nn
