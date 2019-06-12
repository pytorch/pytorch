#include "Types.h"

#include <ATen/ATen.h>
#include "miopen/version.h"

namespace at { namespace native {

miopenDataType_t getMiopenDataType(const at::Tensor& tensor) {
  if (tensor.type().scalarType() == at::kFloat) {
    return miopenFloat;
  } else if (tensor.type().scalarType() == at::kHalf) {
    return miopenHalf;
  }
  std::string msg("getMiopenDataType() not supported for ");
  msg += at::toString(tensor.type().scalarType());
  throw std::runtime_error(msg);
}

int64_t miopen_version() {
  return (MIOPEN_VERSION_MAJOR<<8) + (MIOPEN_VERSION_MINOR<<4) + MIOPEN_VERSION_PATCH;
}

}}  // namespace at::miopen
