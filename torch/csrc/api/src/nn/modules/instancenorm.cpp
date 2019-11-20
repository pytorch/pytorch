#include <torch/nn/functional/instancenorm.h>
#include <torch/nn/modules/instancenorm.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
void InstanceNormImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::InstanceNorm" << D << "d("
         << this->options.num_features() << ", "
         << "eps=" << this->options.eps() << ", "
         << "momentum=" << this->options.momentum() << ", "
         << "affine=" << this->options.affine() << ", "
         << "track_running_stats=" << this->options.track_running_stats() << ")";
}

void InstanceNorm1dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() == 2) {
    TORCH_CHECK(
      false,
      "InstanceNorm1d returns 0-filled tensor to 2D tensor.",
      "This is because InstanceNorm1d reshapes inputs to",
      "(1, N * C, ...) from (N, C,...) and this makes",
      "variances 0.");
  }
  if (input.dim() != 3) {
    TORCH_CHECK(
      false,
      "expected 3D input (got ", input.dim(), "D input)");
  }
}

void InstanceNorm2dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 4) {
    TORCH_CHECK(
      false,
      "expected 4D input (got ", input.dim(), "D input)");
  }
}

void InstanceNorm3dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() != 5) { // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    TORCH_CHECK(
      false,
      "expected 5D input (got ", input.dim(), "D input)");
  }
}

template class InstanceNormImpl<1, InstanceNorm1dImpl>;
template class InstanceNormImpl<2, InstanceNorm2dImpl>;
template class InstanceNormImpl<3, InstanceNorm3dImpl>;

} // namespace nn
} // namespace torch
