#include <torch/nn/modules/activation.h>
#include <torch/nn/functional/activation.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

ELUImpl::ELUImpl(const ELUOptions& options_) : options(options_) {}

Tensor ELUImpl::forward(const Tensor& input) {
  return F::elu(input, options);
}

void ELUImpl::reset() {}

void ELUImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::ELU(alpha=" << options.alpha()
         << ", inplace=" << options.inplace() << ")";
}

} // namespace nn
} // namespace torch
