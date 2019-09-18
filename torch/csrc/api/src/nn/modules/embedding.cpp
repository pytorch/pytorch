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

EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options): options(options) {
  EmbeddingImpl::reset();
}

void EmbeddingImpl::reset() {
  if (options.padding_idx() != c10::nullopt){
    if(*options.padding_idx() > 0){
      assert((*options.padding_idx() < *options.num_embeddings()) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((*options.padding_idx() >= -(*options.num_embeddings())) && "Padding_idx must be within num_embedding");
      options.padding_idx(*options.num_embeddings() + *options.padding_idx());
    }
  }
  if (!options._weight().has_value()){
    torch::nn::init.normal_(weight);
  }
  else{
    weight = options._weight();
    assert((weight.size(0) == options.num_embeddings()) && (weight.size(1) == embedding_dim) && "Shape of _weight does not match num_embeddings and embedding_dim");
  }
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
      assert((*options.padding_idx() < (weight.size(0)) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((*options.padding_idx() >= -(weight.size(0)) && "Padding_idx must be within num_embedding");
      options.padding_idx((weight.size(0) + *options.padding_idx());
    }
  }
  else{
    options.padding_idx(-1);
  }

  if(options.max_norm() != c10::nullopt){
    input = input.contiguous();
    torch::NoGradGuard no_grad;
    torch::embedding_renorm(weight, input, *options.max_norm(), options.norm_type());
  }
  return torch::embedding(weight, input, *options.padding_idx(), options.scale_grad_by_freq(), options.sparse());
}
} // namespace nn
} // namespace torch
