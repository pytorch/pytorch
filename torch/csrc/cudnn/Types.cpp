#include "Types.h"

#include <ATen/ATen.h>

namespace torch { namespace cudnn {

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  if (tensor.type().scalarType() == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (tensor.type().scalarType() == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (tensor.type().scalarType() == at::kHalf) {
    return CUDNN_DATA_HALF;
  }
  std::string msg("getCudnnDataType() not supported for ");
  msg += at::toString(tensor.type().scalarType());
  throw std::runtime_error(msg);
}

void _cudnn_assertContiguous(const at::Tensor& tensor, const std::string& name)
{
  static const std::string error_str = "cuDNN requires contiguous ";
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(error_str + name);
  }
}

}}  // namespace torch::cudnn
