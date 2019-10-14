#include <torch/nn/modules/normalization.h>

#include <torch/cuda.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

LayerNormImpl::LayerNormImpl(const LayerNormOptions& options_) : options(options_) {
  reset();
}

// todo - how to store the shape? checks based on elementwise_affine?
void LayerNormImpl::reset() {
  options.elementwise_affine(elementwise_affine);
  options.eps(eps);
  if(options.elementwise_affine()) {
    weight = register_parameter("weight", torch::ones(options.normalized_shape()));
    bias = register_parameter("bias", torch::zeros(options.normalized_shape()));
  } else {
    weight = register_parameter("weight", torch::empty(options.normalized_shape()));
    bias = register_parameter("bias", torch::empty(options.normalized_shape()));
  }
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LayerNorm(normalized_shape=" << options.normalized_shape()
         << ", elementwise_affine=" << options.elementwise_affine() << ", eps=" << options.eps()
         << ")";
}

//todo- where to get this: torch.backends.cudnn.enabled value from
torch::Tensor LayerNormImpl::forward(const Tensor& input) {
  return torch::layer_norm(input, options.normalized_shape(), weight, bias, options.eps(),
                          false/*torch.backends.cudnn.enabled*/)
}
} // namespace nn
} // namespace torch
