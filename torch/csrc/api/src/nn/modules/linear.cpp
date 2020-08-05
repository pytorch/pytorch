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
  weight = register_parameter("weight",
    torch::empty({options.out_features(), options.in_features()}));
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", {}, /*requires_grad=*/false);
  }

  reset_parameters();
}

void LinearImpl::reset_parameters() {
  torch::nn::init::kaiming_uniform_(weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  if (bias.defined()) {
    int64_t fan_in, fan_out;
    std::tie(fan_in, fan_out) =
      torch::nn::init::_calculate_fan_in_and_fan_out(weight);
    const auto bound = 1 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Linear(in_features=" << options.in_features()
         << ", out_features=" << options.out_features()
         << ", bias=" << options.bias() << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  return F::linear(input, weight, bias);
}

// ============================================================================

FlattenImpl::FlattenImpl(const FlattenOptions& options_) : options(options_) {}

void FlattenImpl::reset() {}

void FlattenImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Flatten(start_dim=" << options.start_dim()
         << ", end_dim=" << options.end_dim() << ")";
}

Tensor FlattenImpl::forward(const Tensor& input) {
  return input.flatten(options.start_dim(), options.end_dim());
}

// ============================================================================

UnflattenImpl::UnflattenImpl(const UnflattenOptions& options_) : options(options_) {
  reset();
}

void UnflattenImpl::reset() {
  named = c10::get_if<std::string>(&options.dim()) != nullptr;
}

void UnflattenImpl::pretty_print(std::ostream& stream) const {
  if (named) {
    auto dim = c10::get<std::string>(options.dim());
    auto namedshape = c10::get<UnflattenOptions::namedshape_t>(options.unflattened_size());
    stream << "torch::nn::Unflatten(dim=\"" << dim << "\", unflattened_size={";
    size_t i;
    for (i = 0; i < namedshape.size() - 1; ++i) {
      stream << "{\"" << std::get<0>(namedshape[i]) << "\", " << std::get<1>(namedshape[i]) << "}, ";
    }
    stream << "{\"" << std::get<0>(namedshape[i]) << "\", " << std::get<1>(namedshape[i]) << "}})";
  } else {
    auto dim = c10::get<int64_t>(options.dim());
    auto sizes = c10::get<std::vector<int64_t>>(options.unflattened_size());
    stream << "torch::nn::Unflatten(dim=" << dim << ", unflattened_size={";
    size_t i;
    for (i = 0; i < sizes.size() - 1; ++i) {
      stream << sizes[i] << ", ";
    }
    stream << sizes[i] << "})";
  }
}

Tensor UnflattenImpl::forward(const Tensor& input) {
  if (named) {
    auto dim = c10::get<std::string>(options.dim());
    auto unflattened_size = c10::get<UnflattenOptions::namedshape_t>(options.unflattened_size());
    auto dimname = torch::Dimname::fromSymbol(torch::Symbol::dimname(dim));
    std::vector<int64_t> sizes;
    std::vector<torch::Dimname> names;
    for (auto i : unflattened_size) {
      names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(std::get<0>(i))));
      sizes.push_back(std::get<1>(i));
    }
    return input.unflatten(dimname, sizes, names);
  }
  auto dim = c10::get<int64_t>(options.dim());
  auto sizes = c10::get<std::vector<int64_t>>(options.unflattened_size());
  return input.unflatten(dim, sizes, torch::nullopt);
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

  reset_parameters();
}

void BilinearImpl::reset_parameters() {
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
