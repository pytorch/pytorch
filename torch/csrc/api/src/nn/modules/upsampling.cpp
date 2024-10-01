#include <torch/nn/modules/upsampling.h>

#include <string>

namespace F = torch::nn::functional;

namespace torch::nn {

UpsampleImpl::UpsampleImpl(
    const UpsampleOptions& options_) // NOLINT(modernize-pass-by-value)
    : options(options_) {}

void UpsampleImpl::reset() {}

void UpsampleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Upsample(";
  if (options.scale_factor() != std::nullopt) {
    stream << "scale_factor=" << at::ArrayRef<double>(*options.scale_factor());
  } else {
    stream << "size=" << at::ArrayRef<int64_t>(*options.size());
  }
  stream << ", mode=" << enumtype::get_enum_name(options.mode()) << ")";
}

Tensor UpsampleImpl::forward(const Tensor& input) {
  F::InterpolateFuncOptions::mode_t mode;
  if (std::holds_alternative<enumtype::kNearest>(options.mode())) {
    mode = torch::kNearest;
  } else if (std::holds_alternative<enumtype::kLinear>(options.mode())) {
    mode = torch::kLinear;
  } else if (std::holds_alternative<enumtype::kBilinear>(options.mode())) {
    mode = torch::kBilinear;
  } else if (std::holds_alternative<enumtype::kBicubic>(options.mode())) {
    mode = torch::kBicubic;
  } else if (std::holds_alternative<enumtype::kTrilinear>(options.mode())) {
    mode = torch::kTrilinear;
  }

  return F::detail::interpolate(
      input,
      options.size(),
      options.scale_factor(),
      mode,
      options.align_corners(),
      std::nullopt,
      false);
}

} // namespace torch::nn
