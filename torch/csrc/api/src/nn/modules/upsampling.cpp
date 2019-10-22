#include <torch/nn/modules/upsampling.h>

#include <string>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

namespace detail {

template <typename T>
std::string str(const std::vector<T>& items) {
  const auto comma = ", ";
  auto delimiter = "";
  std::ostringstream stream;
  stream << "[";
  for (const auto item : items) {
    stream << delimiter << item;
    delimiter = comma;
  }
  stream << "]";
  return stream.str();
}

} // namespace detail

UpsampleImpl::UpsampleImpl(const UpsampleOptions& options_)
    : options(options_) {}

void UpsampleImpl::reset() {}

void UpsampleImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Upsample(";
  if (options.scale_factor() != c10::nullopt) {
    stream << "scale_factor=" << detail::str<double>(*options.scale_factor());
  } else {
    stream << "size=" << detail::str<int64_t>(*options.size());
  }
  stream << ")";
}

Tensor UpsampleImpl::forward(const Tensor& input) {
  return F::interpolate(input, options);
}

} // namespace nn
} // namespace torch
