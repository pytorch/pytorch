#include <torch/nn/modules/padding.h>

#include <torch/expanding_array.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
ReflectionPadImpl<D, Derived>::ReflectionPadImpl(const ReflectionPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReflectionPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

Tensor ReflectionPad1dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(options.padding()).mode("reflect"));
}

Tensor ReflectionPad2dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(options.padding()).mode("reflect"));
}

template class ReflectionPadImpl<1, ReflectionPad1dImpl>;
template class ReflectionPadImpl<2, ReflectionPad2dImpl>;

} // namespace nn
} // namespace torch
