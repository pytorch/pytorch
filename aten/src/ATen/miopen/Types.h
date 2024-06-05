#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/Tensor.h>

namespace at { namespace native {

miopenDataType_t getMiopenDataType(const at::Tensor& tensor);

int64_t miopen_version();

}}  // namespace at::miopen
