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

CosineEmbeddingLossImpl::CosineEmbeddingLossImpl(
    const CosineEmbeddingLossOptions& options_)
    : options(options_) {}

void CosineEmbeddingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CosineEmbeddingLoss(margin=" << options.margin() << ")";
}

Tensor CosineEmbeddingLossImpl::forward(
    const Tensor& input1,
    const Tensor& input2,
    const Tensor& target) {
  return F::cosine_embedding_loss(input1, input2, target, options);
}

// ============================================================================

SoftMarginLossImpl::SoftMarginLossImpl(
    const torch::nn::SoftMarginLossOptions& options_) : options(options_) {}

void SoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SoftMarginLoss";
}

Tensor SoftMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::soft_margin_loss(input, target, options);
}

} // namespace nn
} // namespace torch
