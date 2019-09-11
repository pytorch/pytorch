#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace torch {
namespace nn {
LinearOptions::LinearOptions(int64_t in, int64_t out) : in_(in), out_(out) {}

LinearImpl::LinearImpl(LinearOptions options) : options(options) {
  reset();
}

void LinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out_, options.in_}));
  if (options.with_bias_) {
    bias = register_parameter("bias", torch::empty(options.out_));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Linear(in=" << options.in_
         << ", out=" << options.out_ << ", with_bias=" << options.with_bias_
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  AT_ASSERT(!options.with_bias_ || bias.defined());
  return torch::linear(input, weight, bias);
}

BilinearOptions::BilinearOptions(int64_t in1, int64_t in2, int64_t out) : in1_(in1), in2_(in2), out_(out) {}

BilinearImpl::BilinearImpl(BilinearOptions options) : options(options) {
  reset();
}

void BilinearImpl::reset() {
  weight =
      register_parameter("weight", torch::empty({options.out_, options.in1_, options.in2_}));
  if (options.with_bias_) {
    bias = register_parameter("bias", torch::empty(options.out_));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Bilinear(in1=" << options.in1_
         << ", in2=" << options.in2_ << ", out=" << options.out_ << ", with_bias=" << options.with_bias_
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input1, const Tensor& input2) {
  AT_ASSERT(!options.with_bias_ || bias.defined());
  return torch::bilinear(input1, input2, weight, bias);
}
} // namespace nn
} // namespace torch
