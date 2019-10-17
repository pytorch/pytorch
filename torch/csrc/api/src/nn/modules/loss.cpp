#include <torch/nn/modules/loss.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

L1LossImpl::L1LossImpl(const torch::nn::L1LossOptions& options_)
    : options(options_) {}

void L1LossImpl::reset() {}

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

void HingeEmbeddingLossImpl::reset() {}

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
    const MultiMarginLossOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {
      reset();
}

void MultiMarginLossImpl::reset() {
  TORCH_CHECK((options.p() == 1) || (options.p() == 2), "only p == 1 and p == 2 supported");
  TORCH_CHECK(!options.weight().defined() || options.weight().dim() == 1);

  register_buffer("weight", options.weight());
}

void MultiMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiMarginLoss(p=" << options.p() << 
            ", margin=" << options.margin() <<
            ", weight=" << options.weight() <<
            ", reduction=" << options.reduction() << ")";
}

Tensor MultiMarginLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::multi_margin_loss(input, target, options);
}

// ============================================================================
  
CosineEmbeddingLossImpl::CosineEmbeddingLossImpl(
    const CosineEmbeddingLossOptions& options_)
    : options(options_) {}

void CosineEmbeddingLossImpl::reset() {}

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

MultiLabelSoftMarginLossImpl::MultiLabelSoftMarginLossImpl(
    const torch::nn::MultiLabelSoftMarginLossOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {
  reset();
}

void MultiLabelSoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiLabelSoftMarginLoss()";
}

void MultiLabelSoftMarginLossImpl::reset() {
  register_buffer("weight", options.weight());
}

Tensor MultiLabelSoftMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::multilabel_soft_margin_loss(input,
    target,
    options);
}

// ============================================================================

TripletMarginLossImpl::TripletMarginLossImpl(
    const TripletMarginLossOptions& options_)
    : options(options_) {}

void TripletMarginLossImpl::reset() {}

void TripletMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::TripletMarginLoss(margin=" << options.margin() << 
            ", p=" << options.p() <<
            ", eps=" << options.eps() << std::boolalpha <<
            ", swap=" << options.swap() << ")";
}

Tensor TripletMarginLossImpl::forward(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative) {
  return F::triplet_margin_loss(anchor, positive, negative, options);
}

// ============================================================================

MultiLabelMarginLossImpl::MultiLabelMarginLossImpl(
    const torch::nn::MultiLabelMarginLossOptions& options_)
    : options(options_) {}

void MultiLabelMarginLossImpl::reset() {}

void MultiLabelMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiLabelMarginLoss()";
}

Tensor MultiLabelMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::multilabel_margin_loss(input, target, options);
}

// ============================================================================

SoftMarginLossImpl::SoftMarginLossImpl(
    const torch::nn::SoftMarginLossOptions& options_) : options(options_) {}

void SoftMarginLossImpl::reset() {}

void SoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SoftMarginLoss()";
}

Tensor SoftMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::soft_margin_loss(input, target, options);
}

} // namespace nn
} // namespace torch
