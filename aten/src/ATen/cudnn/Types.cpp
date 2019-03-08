#include <ATen/cudnn/Types.h>

#include <ATen/ATen.h>

namespace at { namespace native {

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (tensor.scalar_type() == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (tensor.scalar_type() == at::kHalf) {
    return CUDNN_DATA_HALF;
  }
  std::string msg("getCudnnDataType() not supported for ");
  msg += toString(tensor.scalar_type());
  throw std::runtime_error(msg);
}

int64_t cudnn_version() {
  return CUDNN_VERSION;
}

}}  // namespace at::cudnn
