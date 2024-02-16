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
