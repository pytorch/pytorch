#include <torch/nn/modules/embedding.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

EmbeddingOptions::EmbeddingOptions(int64_t count, int64_t dimension) : count_(count), dimension_(dimension) {}

EmbeddingImpl::EmbeddingImpl(int64_t count, int64_t dimension, c10::optional<int64_t> padding_idx, c10::optional<float> max_norm,
float norm_type, bool scale_grad_by_freq, bool sparse, c10::optional<torch::Tensor> weight): EmbeddingImpl(EmbeddingOptions(count, dimension)){
  if (padding_idx != c10::nullopt){
    if(*padding_idx > 0){
      assert((padding_idx < count) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((padding_idx >= -count) && "Padding_idx must be within num_embedding");
      *padding_idx = *padding_idx+count;
      options.padding_idx_ = padding_idx;
    }
  }
  if(max_norm != c10::nullopt){
    options.max_norm_ = max_norm;
  }
  options.norm_type_ = norm_type;
  options.scale_grad_by_freq_ = scale_grad_by_freq;

  if (!weight.has_value()){
  // if (weight ==  c10::nullopt){
    options.weight_ = torch::empty({count, dimension});
    EmbeddingImpl::reset();
  }
  else{
    assert((((*weight).size(0) == count) && ((*weight).size(1) == dimension)) && "Shape of weight does not match num_embeddings and embedding_dim");
  }
}

void EmbeddingImpl::reset() {
  (*(options.weight_)).Tensor::normal_(0, 1);
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(count=" << options.count_
         << ", dimension=" << options.dimension_;
  if(options.padding_idx_ != c10::nullopt){
    stream << ",padding_idx=" << *options.padding_idx_;
  }
  if(options.max_norm_ != c10::nullopt){
    stream << ",max_norm=" << *options.max_norm_;
  }
  if(options.norm_type_ != 2){
    stream << ",norm_type=" << options.norm_type_;
  }
  if(options.scale_grad_by_freq_){
    stream << ",scale_grad_by_freq=" << options.scale_grad_by_freq_;
  }
  if(options.sparse_){
    stream << ",sparse=" << options.sparse_;
  }
  stream << ")";
}

Tensor EmbeddingImpl::forward(const Tensor& input) {
  if(options.padding_idx_ != c10::nullopt){
    if(*options.padding_idx_ > 0){
      assert((*options.padding_idx_ < (*options.weight_).size(0)) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((*options.padding_idx_ >= -(*options.weight_).size(0)) && "Padding_idx must be within num_embedding");
      *options.padding_idx_ += (*options.weight_).size(0);
    }
  }
  else{
    options.padding_idx_ = -1;
  }

  if(options.max_norm_ != c10::nullopt){
    input.contiguous();
  }
  return torch::embedding(*options.weight_, /*indices=*/input, *options.padding_idx_, options.scale_grad_by_freq_, options.sparse_);
}
} // namespace nn
} // namespace torch
