#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <numeric>

namespace torch {
namespace nn {
LinearOptions::LinearOptions(int64_t in, int64_t out)
    : in_({in}), out_({out}) {}

LinearOptions::LinearOptions(std::vector<int64_t> in, std::vector<int64_t> out)
    : in_(std::move(in)), out_(std::move(out)) {}

LinearImpl::LinearImpl(LinearOptions options)
    : options(options),
      in_prod_(std::accumulate(options.in_.begin(),
                               options.in_.end(),
                               1,
                               std::multiplies<int64_t>())),
      out_prod_(std::accumulate(options.out_.begin(),
                                options.out_.end(),
                                1,
                                std::multiplies<int64_t>())) {
  reset();
}

void LinearImpl::reset() {
  std::vector<int64_t> sizes(options.out_);
  sizes.insert(sizes.end(), options.in_.begin(), options.in_.end());
  weight =
      register_parameter("weight", torch::empty(sizes));
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
  auto input_view = input.view_as(input);
  if (options.in_.size() != 1) {
    // reshape input
    auto in_sizes = input.sizes().vec();
    auto in_sizes_length = in_sizes.size() - options.in_.size() + 1;
    in_sizes.resize(in_sizes_length);
    in_sizes.back() = in_prod_;
    input_view = input.view(in_sizes);
  }

  auto output = torch::linear(
      input_view, weight.view({out_prod_, in_prod_}), bias.view({out_prod_}));

  auto output_view = output.view_as(output);
  if (options.out_.size() != 1) {
    // reshape output
    auto out_sizes = output.sizes().vec();
    out_sizes.pop_back();
    out_sizes.insert(out_sizes.end(), options.out_.begin(), options.out_.end());
    output_view = output.view(out_sizes);
  }

  return output_view;
}
} // namespace nn
} // namespace torch
