#include <ATen/miopen/Types.h>

#include <ATen/ATen.h>
#include <miopen/version.h>

namespace at { namespace native {

miopenDataType_t getMiopenDataType(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kFloat) {
    return miopenFloat;
  } else if (tensor.scalar_type() == at::kHalf) {
    return miopenHalf;
  }  else if (tensor.scalar_type() == at::kBFloat16) {
    return miopenBFloat16;
  }
  std::string msg("getMiopenDataType() not supported for ");
  msg += toString(tensor.scalar_type());
  throw std::runtime_error(msg);
}

int64_t miopen_version() {
  return (MIOPEN_VERSION_MAJOR<<8) + (MIOPEN_VERSION_MINOR<<4) + MIOPEN_VERSION_PATCH;
}

}}  // namespace at::miopen
