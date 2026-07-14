#pragma once

#include <ATen/Tensor.h>
#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

TORCH_CUDA_CPP_API miopenDataType_t getMiopenDataType(const at::Tensor& tensor);

int64_t miopen_version();

} // namespace at::native
