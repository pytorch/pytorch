#include <torch/nn/modules/loss.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

L1LossImpl::L1LossImpl(const torch::nn::L1LossOptions& options_)
    : options(options_) {}

void L1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::L1Loss";
}

Tensor L1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return torch::l1_loss(input, target, options.reduction());
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

// ============================================================================

MultiMarginLossImpl::MultiMarginLossImpl(
    const MultiMarginLossOptions& options_)
    : options(options_) {
      reset();
    }

void MultiMarginLossImpl::reset() {
  register_buffer("weight", options.weight().value());
}

void MultiMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiMarginLoss(p=" << options.p() << 
            ", margin=" << options.margin() <<
            ", weight=" << options.weight().value() <<
            ", reduction=" << options.reduction() << ")";
}

Tensor MultiMarginLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::multi_margin_loss(input, target, options);
}

} // namespace nn
} // namespace torch
