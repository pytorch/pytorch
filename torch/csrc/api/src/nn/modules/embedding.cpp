#include <torch/nn/modules/embedding.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

EmbeddingOptions::EmbeddingOptions(int64_t count, int64_t dimension, int64_t padding_idx=0, float max_norm=0,
float norm_type=2., bool scale_grad_by_freq=false, bool sparse=false)
    : count_(count), dimension_(dimension) {}

EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options) : options(options) {
  if (options.padding_idx_ == 0){
    if(options.padding_idx_ > 0){
      assert((options.padding_idx_ < options.count_) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((options.padding_idx_ >= -options.count_) && "Padding_idx must be within num_embedding");
      options.padding_idx_ += options.count_;
    }
  }
  //check weight and call reset accordingly

  // if (options.weight == )
  //   // self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
  //   // self.reset_parameters()
  // else
    // assert list(_weight.shape) == [num_embeddings, embedding_dim],
    //   'Shape of weight does not match num_embeddings and embedding_dim'
    //   self.weight = Parameter(_weight)
  //reset();
}

void EmbeddingImpl::reset() {
  weight = register_parameter(
      "weight", torch::empty({options.count_, options.dimension_}));
  NoGradGuard guard;
  weight.normal_(0, 1);
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(count=" << options.count_
         << ", dimension=" << options.dimension_ << ")";
}

Tensor EmbeddingImpl::forward(const Tensor& input) {
  if(options.padding_idx_ !=0){
    if(options.padding_idx_ > 0){
      assert((options.padding_idx_ < options.weight_.size(0)) && "Padding_idx must be within num_embeddings");
    }
    else{
      assert((options.padding_idx_ >= -options.weight_.size(0)) && "Padding_idx must be within num_embedding");
      options.padding_idx_ += options.weight_.size(0);
    }
  }
  else{
    options.padding_idx_ = -1;
  }
  if(options.max_norm_ != 0){

  }
  // if max_norm is not None:
  //       # `embedding_renorm_` will call .contiguous() on input anyways, so we
  //       # call it here and take advantage of the improved locality in the
  //       # `embedding` call below too.
  //       input = input.contiguous()
  //       # XXX: equivalent to
  //       # with torch.no_grad():
  //       #   torch.nembedding_renorm_
  //       # remove once script supports set_grad_enabled
  //       _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
  //   return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
  return torch::embedding(weight, /*indices=*/input, options.padding_idx_, options.scale_grad_by_freq_, options.sparse_);
}
} // namespace nn
} // namespace torch
