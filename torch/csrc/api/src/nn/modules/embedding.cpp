#include <torch/nn/modules/embedding.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

EmbeddingOptions::EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim) : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {}

EmbeddingImpl::EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim, c10::optional<int64_t> padding_idx, c10::optional<float> max_norm,
float norm_type, bool scale_grad_by_freq, bool sparse, c10::optional<torch::Tensor> weight): EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)){
  if (padding_idx != c10::nullopt){
    if(*padding_idx > 0){
      assert((padding_idx < num_embeddings) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((padding_idx >= -num_embeddings) && "Padding_idx must be within num_embedding");
      *padding_idx = *padding_idx+num_embeddings;
      options.padding_idx_ = padding_idx;
    }
  }
  if(max_norm != c10::nullopt){
    options.max_norm(max_norm);
  }
  options.norm_type(norm_type);
  options.scale_grad_by_freq(scale_grad_by_freq);

  if (!weight.has_value()){
  // if (weight ==  c10::nullopt){
    options.weight(torch::empty({num_embeddings, embedding_dim}));
    EmbeddingImpl::reset();
  }
  else{
    assert((((*weight).size(0) == num_embeddings) && ((*weight).size(1) == embedding_dim)) && "Shape of weight does not match num_embeddings and embedding_dim");
  }
}

void EmbeddingImpl::reset() {
  (*(options.weight_)).Tensor::normal_(0, 1);
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  if(options.padding_idx() != c10::nullopt){
    stream << ",padding_idx=" << *options.padding_idx();
  }
  if(options.max_norm() != c10::nullopt){
    stream << ",max_norm=" << *options.max_norm();
  }
  if(options.norm_type() != 2){
    stream << ",norm_type=" << options.norm_type();
  }
  if(options.scale_grad_by_freq()){
    stream << ",scale_grad_by_freq=" << options.scale_grad_by_freq();
  }
  if(options.sparse()){
    stream << ",sparse=" << options.sparse();
  }
  stream << ")";
}

Tensor EmbeddingImpl::forward(const Tensor& input) {
  if(options.padding_idx() != c10::nullopt){
    if(*options.padding_idx() > 0){
      assert((*options.padding_idx() < (*options.weight()).size(0)) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((*options.padding_idx() >= -(*options.weight()).size(0)) && "Padding_idx must be within num_embedding");
      options.padding_idx(*options.padding_idx() + (*options.weight_).size(0));
    }
  }
  else{
    options.padding_idx(-1);
  }

  if(options.max_norm() != c10::nullopt){
    input.contiguous();
  }
  return torch::embedding(*options.weight(), /*indices=*/input, *options.padding_idx(), options.scale_grad_by_freq(), options.sparse());
}
} // namespace nn
} // namespace torch
