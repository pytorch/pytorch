#include <torch/nn/modules/loss.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

L1LossImpl::L1LossImpl(const L1LossOptions& options_) : options(options_) {}

void L1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::L1Loss";
}

Tensor L1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::l1_loss(input, target, options);
}

// ============================================================================

KLDivLossImpl::KLDivLossImpl(const KLDivLossOptions& options_)
    : options(options_) {}

void KLDivLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::KLDivLoss";
}

Tensor KLDivLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::kl_div(input, target, options);
}

// ============================================================================

MSELossImpl::MSELossImpl(const MSELossOptions& options_) : options(options_) {}

void MSELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MSELoss";
}

Tensor MSELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::mse_loss(input, target, options);
}

// ============================================================================

BCELossImpl::BCELossImpl(const BCELossOptions& options_) : options(options_) {
  register_parameter("weight", options.weight());
}

void BCELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::BCELoss";
}

Tensor BCELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::binary_cross_entropy(input, target, options);
}

// ============================================================================

HingeEmbeddingLossImpl::HingeEmbeddingLossImpl(
    const HingeEmbeddingLossOptions& options_)
    : options(options_) {}

void HingeEmbeddingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::HingeEmbeddingLoss(margin=" << options.margin() << ")";
}

Tensor HingeEmbeddingLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::hinge_embedding_loss(input, target, options);
}

} // namespace nn
} // namespace torch
