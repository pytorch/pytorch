#pragma once

#include <ATen/Tensor.h>
#include <ATen/miopen/miopen-wrapper.h>

namespace at {
namespace native {

miopenDataType_t getMiopenDataType(const at::Tensor& tensor);

int64_t miopen_version();

} // namespace native
} // namespace at
