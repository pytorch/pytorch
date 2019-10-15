#include <torch/nn/modules/normalization.h>

#include <torch/cuda.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <ostream>
#include <utility>

namespace torch {
namespace nn {

LayerNormImpl::LayerNormImpl(const LayerNormOptions& options_) : options(options_) {
  reset();
}

void LayerNormImpl::reset() {
  if (options.elementwise_affine()) {
    weight = register_parameter("weight", torch::ones(torch::IntArrayRef(options.normalized_shape())));
    bias = register_parameter("bias", torch::zeros(torch::IntArrayRef(options.normalized_shape())));
  } else {
    weight = register_parameter("weight", torch::empty(torch::IntArrayRef(options.normalized_shape())));
    bias = register_parameter("bias", torch::empty(torch::IntArrayRef(options.normalized_shape())));
  }
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LayerNorm(normalized_shape=" << torch::IntArrayRef(options.normalized_shape())
         << ", elementwise_affine=" << options.elementwise_affine() << ", eps=" << options.eps()
         << ")";
}

torch::Tensor LayerNormImpl::forward(const Tensor& input) {
  return torch::layer_norm(input, torch::IntArrayRef(options.normalized_shape()), weight, bias, options.eps(),
                          at::globalContext().userEnabledCuDNN());
}
} // namespace nn
} // namespace torch
