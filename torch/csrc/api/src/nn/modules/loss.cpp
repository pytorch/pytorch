#include <torch/nn/modules/loss.h>

namespace torch {
namespace nn {

L1LossImpl::L1LossImpl(torch::nn::L1LossOptions options)
    : options(options) {}

void L1LossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::L1Loss";
}

Tensor L1LossImpl::forward(const Tensor& input, const Tensor& target) {
  return torch::l1_loss(input, target, options.reduction());
}

} // namespace nn
} // namespace torch
