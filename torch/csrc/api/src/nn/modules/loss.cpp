#include <torch/nn/modules/loss.h>
<<<<<<< HEAD
=======
using namespace std
>>>>>>> Add C++ Frontend - triplet-margin-loss
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

TripletMarginLossImpl::TripletMarginLossImpl(
    const TripletMarginLossOptions& options_)
    : options(options_) {}

void TripletMarginLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::TripletMarginLoss(margin=" << options.margin() << 
            ", p=" << options.p() <<
            ", eps=" << options.eps() << std::boolalpha <<
<<<<<<< HEAD
            ", swap=" << options.swap() <<
=======
            ", swap=" <<options.swap() <<
>>>>>>> Add C++ Frontend - triplet-margin-loss
            ", reduction=" << options.reduction() << ")";
}

Tensor TripletMarginLossImpl::forward(
    const Tensor& anchor,
    const Tensor& positive,
    const Tensor& negative) {
  return F::triplet_margin_loss(anchor, positive, negative, options);
}

} // namespace nn
} // namespace torch
