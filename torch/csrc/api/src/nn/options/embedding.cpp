#include <torch/nn/options/embedding.h>

namespace torch {
namespace nn {
EmbeddingOptions::EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim) :
 num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {}

EmbeddingBagOptions::EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim) :
 num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {}
} // namespace nn
} // namespace torch
