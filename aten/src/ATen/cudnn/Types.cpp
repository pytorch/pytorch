#include "Types.h"

#include <ATen/ATen.h>

namespace at { namespace native {

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

int64_t cudnn_version() {
  return CUDNN_VERSION;
}

}}  // namespace at::cudnn
