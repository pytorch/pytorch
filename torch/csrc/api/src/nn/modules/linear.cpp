#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

void IdentityImpl::reset() {}

void IdentityImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Identity()";
}

Tensor IdentityImpl::forward(const Tensor& input) {
  return input;
}

// ============================================================================

LinearImpl::LinearImpl(const LinearOptions& options_) : options(options_) {
  reset();
}

void LinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out(), options.in()}));
  if (options.with_bias()) {
    bias = register_parameter("bias", torch::empty(options.out()));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Linear(in=" << options.in()
         << ", out=" << options.out() << ", with_bias=" << options.with_bias()
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  AT_ASSERT(!options.with_bias() || bias.defined());
  return torch::linear(input, weight, bias);
}

// ============================================================================

BilinearImpl::BilinearImpl(const BilinearOptions& options_) : options(options_) {
  reset();
}

void BilinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out_features(), options.in1_features(), options.in2_features()}));
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }

  const auto bound = 1.0 / std::sqrt(weight.size(1));
  init::uniform_(weight, -bound, bound);
  if (bias.defined()) {
      init::uniform_(bias, -bound, bound);
  }
}

void BilinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Bilinear(in1_features=" << options.in1_features()
         << ", in2_features=" << options.in2_features() << ", out_features=" << options.out_features() << ", bias=" << options.bias()
         << ")";
}

Tensor BilinearImpl::forward(const Tensor& input1, const Tensor& input2) {
  return F::bilinear(input1, input2, weight, bias);
}
} // namespace nn
} // namespace torch
