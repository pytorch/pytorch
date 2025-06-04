#include <torch/nn/functional/dropout.h>
#include <torch/nn/modules/dropout.h>

#include <torch/types.h>

#include <c10/util/Exception.h>

#include <ostream>
#include <utility>

namespace F = torch::nn::functional;

namespace torch::nn {

Tensor DropoutImpl::forward(Tensor input) {
  return F::detail::dropout(
      std::move(input), options.p(), is_training(), options.inplace());
}

void DropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Dropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor Dropout2dImpl::forward(Tensor input) {
  return F::detail::dropout2d(
      std::move(input), options.p(), is_training(), options.inplace());
}

void Dropout2dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Dropout2d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor Dropout3dImpl::forward(Tensor input) {
  return F::detail::dropout3d(
      std::move(input), options.p(), is_training(), options.inplace());
}

void Dropout3dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Dropout3d(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor AlphaDropoutImpl::forward(const Tensor& input) {
  return F::detail::alpha_dropout(
      input, options.p(), is_training(), /*inplace=*/false);
}

void AlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::AlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

// ============================================================================

Tensor FeatureAlphaDropoutImpl::forward(const Tensor& input) {
  return F::detail::feature_alpha_dropout(
      input, options.p(), is_training(), /*inplace=*/false);
}

void FeatureAlphaDropoutImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::FeatureAlphaDropout(p=" << options.p()
         << ", inplace=" << options.inplace() << ")";
}

} // namespace torch::nn
