#include <ATen/cudnn/Types.h>

#include <ATen/ATen.h>

namespace at { namespace native {

cudnnDataType_t getCudnnDataTypeFromScalarType(const at::ScalarType dtype) {
  if (dtype == c10::kQInt8) {
    return CUDNN_DATA_INT8;
  } else if (dtype == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (dtype == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (dtype == at::kHalf) {
    return CUDNN_DATA_HALF;
  }
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
  else if (dtype == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (dtype == at::kInt) {
    return CUDNN_DATA_INT32;
  } else if (dtype == at::kByte) {
    return CUDNN_DATA_UINT8;
  } else if (dtype == at::kChar) {
    return CUDNN_DATA_INT8;
  }
#endif
  std::string msg("getCudnnDataTypeFromScalarType() not supported for ");
  msg += toString(dtype);
  throw std::runtime_error(msg);
}

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  return getCudnnDataTypeFromScalarType(tensor.scalar_type());
}

int64_t cudnn_version() {
  return CUDNN_VERSION;
}

}}  // namespace at::cudnn
