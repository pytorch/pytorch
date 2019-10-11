#include <torch/nn/modules/embedding.h>

#include <torch/types.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

  LayerNorm::LayerNorm() {

  }

  void LayerNorm::reset() {

  }

  void LayerNorm::pretty_print(std::ostream& stream) const {

  }

  torch::Tensor EmbeddingImpl::forward(const Tensor& input) {

  }
  // EmbeddingImpl::EmbeddingImpl(const EmbeddingOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  //   reset();
  // }

  // void EmbeddingImpl::reset() {
  //   if (options.padding_idx() != c10::nullopt) {
  //     if (*options.padding_idx() > 0) {
  //       TORCH_CHECK(*options.padding_idx() < options.num_embeddings(), "Padding_idx must be within num_embeddings");
  //     }
  //     else if (*options.padding_idx() < 0) {
  //       TORCH_CHECK(*options.padding_idx() >= -(options.num_embeddings()), "Padding_idx must be within num_embedding");
  //       options.padding_idx(options.num_embeddings() + *options.padding_idx());
  //     }
  //   }
  //
  //   if (options._weight() == c10::nullopt) {
  //     weight = register_parameter(
  //         "weight", torch::empty({options.num_embeddings(), options.embedding_dim()}));
  //     torch::nn::init::normal_(weight);
  //     if (options.padding_idx() != c10::nullopt) {
  //       torch::NoGradGuard no_grad;
  //       weight[*options.padding_idx()].fill_(0);
  //     }
  //   } else {
  //     TORCH_CHECK((*options._weight()).sizes() == torch::IntArrayRef({options.num_embeddings(), options.embedding_dim()}), "Shape of _weight does not match num_embeddings and embedding_dim");
  //     weight = register_parameter("weight", *options._weight());
  //   }
  // }
  //
  // void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  //   stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
  //          << ", embedding_dim=" << options.embedding_dim();
  //   if (options.padding_idx() != c10::nullopt) {
  //     stream << ", padding_idx=" << *options.padding_idx();
  //   }
  //   if (options.max_norm() != c10::nullopt) {
  //     stream << ", max_norm=" << *options.max_norm();
  //   }
  //   if (options.norm_type() != 2) {
  //     stream << ", norm_type=" << options.norm_type();
  //   }
  //   if (options.scale_grad_by_freq()) {
  //     stream << ", scale_grad_by_freq=" << std::boolalpha << options.scale_grad_by_freq();
  //   }
  //   if (options.sparse()) {
  //     stream << ", sparse=" << std::boolalpha << options.sparse();
  //   }
  //   stream << ")";
  // }
  //
  // torch::Tensor EmbeddingImpl::forward(const Tensor& input) {
  //   if (options.padding_idx() != c10::nullopt) {
  //     if (*options.padding_idx() > 0) {
  //       TORCH_CHECK(*options.padding_idx() < weight.size(0), "Padding_idx must be within num_embeddings");
  //     }
  //     else if (*options.padding_idx() < 0) {
  //       TORCH_CHECK(*options.padding_idx() >= -weight.size(0), "Padding_idx must be within num_embedding");
  //       options.padding_idx(weight.size(0) + *options.padding_idx());
  //     }
  //   } else {
  //     options.padding_idx(-1);
  //   }
  //
  //   if (options.max_norm() != c10::nullopt) {
  //     torch::NoGradGuard no_grad;
  //     torch::embedding_renorm_(weight, input.contiguous(), *options.max_norm(), options.norm_type());
  //   }
  //   return torch::embedding(weight, input.contiguous(), *options.padding_idx(), options.scale_grad_by_freq(), options.sparse());
  // }
} // namespace nn
} // namespace torch
