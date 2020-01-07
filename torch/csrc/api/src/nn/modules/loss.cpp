#include <torch/nn/modules/loss.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

L1LossImpl::L1LossImpl(const L1LossOptions& options_) : options(options_) {}

void L1LossImpl::reset() {}

void L1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::L1Loss()";
}

Tensor L1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::l1_loss(input, target, options.reduction());
}

// ============================================================================

KLDivLossImpl::KLDivLossImpl(const KLDivLossOptions& options_)
    : options(options_) {}

void KLDivLossImpl::reset() {}

void KLDivLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::KLDivLoss()";
}

Tensor KLDivLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::kl_div(input, target, options.reduction());
}

// ============================================================================

MSELossImpl::MSELossImpl(const MSELossOptions& options_) : options(options_) {}

void MSELossImpl::reset() {}

void MSELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MSELoss()";
}

Tensor MSELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::mse_loss(input, target, options.reduction());
}

// ============================================================================

BCELossImpl::BCELossImpl(const BCELossOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  reset();
}

void BCELossImpl::reset() {
  register_buffer("weight", options.weight());
}

void BCELossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::BCELoss()";
}

Tensor BCELossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::binary_cross_entropy(input, target, options.weight(), options.reduction());
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
  return F::detail::hinge_embedding_loss(input, target, options.margin(), options.reduction());
}

// ============================================================================

MultiMarginLossImpl::MultiMarginLossImpl(
    const MultiMarginLossOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {
  reset();
}

void MultiMarginLossImpl::reset() {
  TORCH_CHECK(
      (options.p() == 1) || (options.p() == 2),
      "only p == 1 and p == 2 supported");
  TORCH_CHECK(!options.weight().defined() || options.weight().dim() == 1);

  register_buffer("weight", options.weight());
}

void MultiMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiMarginLoss(p=" << options.p()
         << ", margin=" << options.margin() << ", weight=" << options.weight()
         << ", reduction=" << enumtype::get_enum_name(options.reduction()) << ")";
}

Tensor MultiMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::multi_margin_loss(input, target, options.p(), options.margin(), options.weight(), options.reduction());
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
  return F::detail::cosine_embedding_loss(input1, input2, target, options.margin(), options.reduction());
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

Tensor MultiLabelSoftMarginLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::multilabel_soft_margin_loss(input, target, options.weight(), options.reduction());
}

// ============================================================================

TripletMarginLossImpl::TripletMarginLossImpl(
    const TripletMarginLossOptions& options_)
    : options(options_) {}

void TripletMarginLossImpl::reset() {}

void TripletMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::TripletMarginLoss(margin=" << options.margin()
         << ", p=" << options.p() << ", eps=" << options.eps() << std::boolalpha
         << ", swap=" << options.swap() << ")";
}

Tensor TripletMarginLossImpl::forward(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative) {
  return F::detail::triplet_margin_loss(
    anchor,
    positive,
    negative,
    options.margin(),
    options.p(),
    options.eps(),
    options.swap(),
    options.reduction());
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
  return F::detail::multilabel_margin_loss(input, target, options.reduction());
}

// ============================================================================

SoftMarginLossImpl::SoftMarginLossImpl(
    const torch::nn::SoftMarginLossOptions& options_) : options(options_) {}

void SoftMarginLossImpl::reset() {}

void SoftMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SoftMarginLoss()";
}

Tensor SoftMarginLossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::soft_margin_loss(input, target, options.reduction());
}

// ============================================================================

SmoothL1LossImpl::SmoothL1LossImpl(
    const torch::nn::SmoothL1LossOptions& options_) : options(options_) {}

void SmoothL1LossImpl::reset() {}

void SmoothL1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SmoothL1Loss";
}

Tensor SmoothL1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return F::detail::smooth_l1_loss(input, target, options.reduction());
}
  
// ============================================================================
  
CTCLossImpl::CTCLossImpl(const CTCLossOptions& options_) : options(options_) {}

void CTCLossImpl::reset() {}

void CTCLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CTCLoss()";
}

Tensor CTCLossImpl::forward(const Tensor& log_probs, const Tensor& targets,
                 const Tensor& input_lengths, const Tensor& target_lengths) {
  return F::detail::ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    options.blank(),
    options.reduction(),
    options.zero_infinity());
}

// ============================================================================

PoissonNLLLossImpl::PoissonNLLLossImpl(const PoissonNLLLossOptions& options_)
  : options(options_) {}

void PoissonNLLLossImpl::reset() {}

void PoissonNLLLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PoissonNLLLoss()";
}

Tensor PoissonNLLLossImpl::forward(
  const Tensor& log_input, const Tensor& target) {
  return F::detail::poisson_nll_loss(
    log_input, target,
    options.log_input(), options.full(), options.eps(), options.reduction());
}

// ============================================================================

MarginRankingLossImpl::MarginRankingLossImpl(
  const MarginRankingLossOptions& options_) : options(options_) {}

void MarginRankingLossImpl::reset() {}

void MarginRankingLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MarginRankingLoss()";
}

Tensor MarginRankingLossImpl::forward(const Tensor& input1,
    const Tensor& input2, const Tensor& target) {
  return F::detail::margin_ranking_loss(input1, input2, target, options.margin(), options.reduction());
}

// ============================================================================

NLLLossImpl::NLLLossImpl(
    const NLLLossOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {
  reset();
}

void NLLLossImpl::reset() {
  weight = register_buffer("weight", options.weight());
}

void NLLLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::NLLLoss()";
}

Tensor NLLLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::nll_loss(
    input,
    target,
    weight,
    options.ignore_index(),
    options.reduction());
}

// ============================================================================

CrossEntropyLossImpl::CrossEntropyLossImpl(
    const CrossEntropyLossOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {
  reset();
}

void CrossEntropyLossImpl::reset() {
  weight = register_buffer("weight", options.weight());
}

void CrossEntropyLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CrossEntropyLoss()";
}

Tensor CrossEntropyLossImpl::forward(
    const Tensor& input,
    const Tensor& target) {
  return F::detail::cross_entropy(
    input,
    target,
    weight,
    options.ignore_index(),
    options.reduction());
}

// ============================================================================

BCEWithLogitsLossImpl::BCEWithLogitsLossImpl(
  const BCEWithLogitsLossOptions& options_) : options(options_) {
  reset();
}

void BCEWithLogitsLossImpl::reset() {
  weight = register_buffer("weight", options.weight());
  pos_weight = register_buffer("pos_weight", options.pos_weight());
}

void BCEWithLogitsLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::BCEWithLogitsLoss()";
}

Tensor BCEWithLogitsLossImpl::forward(
  const Tensor& input, const Tensor& target) {
  return F::detail::binary_cross_entropy_with_logits(input, target,
    options.weight(), options.reduction(), options.pos_weight());
}

} // namespace nn
} // namespace torch
