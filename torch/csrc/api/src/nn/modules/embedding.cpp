#include <torch/nn/modules/embedding.h>

namespace torch { namespace nn {
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
      Var(DefaultTensor(at::kFloat).tensor({num_embeddings, embedding_dim}),
          true),
      "weight");
}

}} // namespace torch::nn
