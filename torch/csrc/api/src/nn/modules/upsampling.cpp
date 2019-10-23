#include <torch/nn/modules/upsampling.h>

#include <string>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

UpsampleImpl::UpsampleImpl(const UpsampleOptions& options_)
    : options(options_) {}

void UpsampleImpl::reset() {}

void UpsampleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Upsample(";
  if (!options.scale_factor().empty()) {
    stream << "scale_factor=" << at::ArrayRef<double>(options.scale_factor());
  } else {
    stream << "size=" << at::ArrayRef<int64_t>(options.size());
  }
  stream << ", mode=" << c10::visit(enumtype::enum_name{}, options.mode()) << ")";
}

Tensor UpsampleImpl::forward(const Tensor& input) {
  return F::interpolate(
      input,
      InterpolateOptions()
          .size(options.size())
          .scale_factor(options.scale_factor())
          .mode(decltype(InterpolateOptions().mode())(options.mode()))
          .align_corners(
              options.align_corners().has_value() ? options.align_corners()
                                                  : c10::nullopt));
}

} // namespace nn
} // namespace torch
