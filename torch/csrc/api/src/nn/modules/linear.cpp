#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace F = torch::nn::functional;

namespace torch::nn {

void IdentityImpl::reset() {}

void IdentityImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Identity()";
}

Tensor IdentityImpl::forward(const Tensor& input) {
  return input;
}

// ============================================================================

LinearImpl::LinearImpl(const LinearOptions& options_) : options(options_) {
  LinearImpl::reset();
}

void LinearImpl::reset() {
  weight = register_parameter(
      "weight", torch::empty({options.out_features(), options.in_features()}));
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", {}, /*requires_grad=*/false);
  }

  reset_parameters();
}

void LinearImpl::reset_parameters() {
  torch::nn::init::kaiming_uniform_(
      weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  if (bias.defined()) {
    auto [fan_in, fan_out] =
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

UnflattenImpl::UnflattenImpl(UnflattenOptions options_)
    : options(std::move(options_)) {}

void UnflattenImpl::reset() {}

void UnflattenImpl::pretty_print(std::ostream& stream) const {
  auto namedshape = options.namedshape();
  if (!namedshape.empty()) {
    stream << "torch::nn::Unflatten(dim=\"" << options.dimname()
           << "\", unflattened_size={";
    size_t i = 0;
    for (; i < namedshape.size() - 1; ++i) {
      stream << "{\"" << std::get<0>(namedshape[i]) << "\", "
             << std::get<1>(namedshape[i]) << "}, ";
    }
    stream << "{\"" << std::get<0>(namedshape[i]) << "\", "
           << std::get<1>(namedshape[i]) << "}})";
  } else {
    stream << "torch::nn::Unflatten(dim=" << options.dim()
           << ", unflattened_size={";
    auto sizes = options.sizes();
    size_t i = 0;
    for (; i < sizes.size() - 1; ++i) {
      stream << sizes[i] << ", ";
    }
    stream << sizes[i] << "})";
  }
}

Tensor UnflattenImpl::forward(const Tensor& input) {
  auto namedshape = options.namedshape();
  if (!namedshape.empty()) {
    auto dimname =
        torch::Dimname::fromSymbol(torch::Symbol::dimname(options.dimname()));
    std::vector<int64_t> sizes;
    std::vector<torch::Dimname> names;
    for (auto i : namedshape) {
      names.push_back(
          torch::Dimname::fromSymbol(torch::Symbol::dimname(std::get<0>(i))));
      sizes.push_back(std::get<1>(i));
    }
    return input.unflatten(dimname, sizes, names);
  }
  return input.unflatten(options.dim(), options.sizes());
}

// ============================================================================

BilinearImpl::BilinearImpl(const BilinearOptions& options_)
    : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void BilinearImpl::reset() {
  weight = register_parameter(
      "weight",
      torch::empty(
          {options.out_features(),
           options.in1_features(),
           options.in2_features()}));
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
  stream << std::boolalpha
         << "torch::nn::Bilinear(in1_features=" << options.in1_features()
         << ", in2_features=" << options.in2_features()
         << ", out_features=" << options.out_features()
         << ", bias=" << options.bias() << ")";
}

Tensor BilinearImpl::forward(const Tensor& input1, const Tensor& input2) {
  return F::bilinear(input1, input2, weight, bias);
}

} // namespace torch::nn
