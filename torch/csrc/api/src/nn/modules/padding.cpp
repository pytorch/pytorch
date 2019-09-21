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

// ============================================================================

template <size_t D, typename Derived>
ReplicationPadImpl<D, Derived>::ReplicationPadImpl(const ReplicationPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReplicationPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

Tensor ReplicationPad1dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(options.padding()).mode("replicate"));
}

Tensor ReplicationPad2dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(options.padding()).mode("replicate"));
}

Tensor ReplicationPad3dImpl::forward(const Tensor& input) {
  return F::pad(input, PadOptions(options.padding()).mode("replicate"));
}

template class ReplicationPadImpl<1, ReplicationPad1dImpl>;
template class ReplicationPadImpl<2, ReplicationPad2dImpl>;
template class ReplicationPadImpl<3, ReplicationPad3dImpl>;

} // namespace nn
} // namespace torch
