#include <ATen/cudnn/Types.h>


#include <c10/util/Exception.h>

namespace at::native {

cudnnDataType_t getCudnnDataTypeFromScalarType(const at::ScalarType dtype) {
  if (dtype == c10::kQInt8 || dtype == at::kChar) {
    return CUDNN_DATA_INT8;
  } else if (dtype == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (dtype == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (dtype == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (dtype == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (dtype == at::kInt) {
    return CUDNN_DATA_INT32;
  } else if (dtype == at::kByte) {
    return CUDNN_DATA_UINT8;
  }
  TORCH_CHECK(false,
    "getCudnnDataTypeFromScalarType() not supported for ",
    toString(dtype)
  );
}

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  return getCudnnDataTypeFromScalarType(tensor.scalar_type());
}

int64_t cudnn_version() {
  return CUDNN_VERSION;
}

} // namespace at::native
