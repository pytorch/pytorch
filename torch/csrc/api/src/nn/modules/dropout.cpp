#include <torch/nn/modules/dropout.h>
#include <torch/nn/functional/dropout.h>

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

Tensor DropoutImpl::forward(Tensor input) {
  return F::detail::dropout(input, options.p(),
      is_training(), options.inplace());
}

void DropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Dropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor Dropout2dImpl::forward(Tensor input) {
  return F::detail::dropout2d(input, options.p(),
      is_training(), options.inplace());
}

void Dropout2dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Dropout2d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor Dropout3dImpl::forward(Tensor input) {
  return F::detail::dropout3d(input, options.p(),
      is_training(), options.inplace());
}

void Dropout3dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Dropout3d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

FeatureDropoutImpl::FeatureDropoutImpl(double p)
  : detail::_DropoutNd<FeatureDropoutImpl>(p) {
  TORCH_WARN("torch::nn::FeatureDropout module is deprecated."
             "Use Dropout{2,3}d instead.");
}

FeatureDropoutImpl::FeatureDropoutImpl(const FeatureDropoutOptions& options_)
  : detail::_DropoutNd<FeatureDropoutImpl>(options_) {
  TORCH_WARN("torch::nn::FeatureDropout module is deprecated."
             "Use Dropout{2,3}d instead.");
}

Tensor FeatureDropoutImpl::forward(const Tensor& input) {
  return torch::feature_dropout(input, options.p(), is_training());
}

void FeatureDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::FeatureDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor AlphaDropoutImpl::forward(const Tensor& input) {
  return F::detail::alpha_dropout(input, options.p(), is_training(), /*inplace=*/false);
}

void AlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::AlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor FeatureAlphaDropoutImpl::forward(const Tensor& input) {
  return F::detail::feature_alpha_dropout(input, options.p(), is_training(), /*inplace=*/false);
}

void FeatureAlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::FeatureAlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

} // namespace nn
} // namespace torch
