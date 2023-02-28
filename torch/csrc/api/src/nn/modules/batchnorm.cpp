#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::BatchNorm" << D << "d("
         << this->options.num_features() << ", "
         << "eps=" << this->options.eps() << ", "
         << "momentum=";

  if (this->options.momentum().has_value()) {
    stream << this->options.momentum().value();
  } else {
    stream << "None";
  }

  stream << ", "
         << "affine=" << this->options.affine() << ", "
         << "track_running_stats=" << this->options.track_running_stats()
         << ")";
}

void BatchNorm1dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "expected 2D or 3D input (got ",
      input.dim(),
      "D input)");
}

void BatchNorm2dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 4, "expected 4D input (got ", input.dim(), "D input)");
}

void BatchNorm3dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 5, "expected 5D input (got ", input.dim(), "D input)");
}

template class BatchNormImplBase<1, BatchNorm1dImpl>;
template class BatchNormImplBase<2, BatchNorm2dImpl>;
template class BatchNormImplBase<3, BatchNorm3dImpl>;

} // namespace nn
} // namespace torch
