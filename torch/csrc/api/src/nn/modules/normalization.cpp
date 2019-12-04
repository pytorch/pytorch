#include <torch/nn/modules/normalization.h>

#include <torch/cuda.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <ostream>
#include <utility>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

LayerNormImpl::LayerNormImpl(const LayerNormOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  reset();
}

void LayerNormImpl::reset() {
  if (options.elementwise_affine()) {
    weight = register_parameter("weight", torch::empty(options.normalized_shape()));
    bias = register_parameter("bias", torch::empty(options.normalized_shape()));
  } else {
    weight = register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  reset_parameters();
}

void LayerNormImpl::reset_parameters() {
  if (options.elementwise_affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LayerNorm(" << torch::IntArrayRef(options.normalized_shape())
         << ", eps=" << options.eps()
         << ", elementwise_affine=" << options.elementwise_affine()
         << ")";
}

torch::Tensor LayerNormImpl::forward(const Tensor& input) {
  return F::detail::layer_norm(input, options.normalized_shape(), weight, bias, options.eps());
}

// ============================================================================

LocalResponseNormImpl::LocalResponseNormImpl(const LocalResponseNormOptions& options_)
    : options(options_) {}

Tensor LocalResponseNormImpl::forward(const Tensor& input) {
  return F::detail::local_response_norm(input, options.size(), options.alpha(), options.beta(), options.k());
}

void LocalResponseNormImpl::reset() {}

void LocalResponseNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LocalResponseNorm(" <<  options.size()
         << ", alpha=" << options.alpha()
         << ", beta=" << options.beta()
         << ", k=" << options.k()
         << ")";
}

// ============================================================================

void CrossMapLRN2dImpl::reset() {}

void CrossMapLRN2dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::CrossMapLRN2d(" << options.size()
         << ", alpha=" << options.alpha()
         << ", beta=" << options.beta()
         << ", k=" << options.k()
         << ")";
}

torch::Tensor CrossMapLRN2dImpl::forward(const torch::Tensor& input) {
  return functions::CrossMapLRN2d::apply(input, options);
}

// ============================================================================

GroupNormImpl::GroupNormImpl(const GroupNormOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  reset();
}

void GroupNormImpl::reset() {
  if (options.affine()) {
    weight = register_parameter("weight", torch::empty(options.num_channels()));
    bias = register_parameter("bias", torch::empty(options.num_channels()));
  } else {
    weight = register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  reset_parameters();
}

void GroupNormImpl::reset_parameters() {
  if (options.affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

torch::Tensor GroupNormImpl::forward(const Tensor& input) {
  return F::detail::group_norm(input, options.num_groups(), weight, bias, options.eps());
}

void GroupNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::GroupNorm(" << options.num_groups()
         << ", " << options.num_channels()
         << ", eps=" << options.eps()
         << ", affine=" << options.affine()
         << ")";
}

} // namespace nn
} // namespace torch
