#pragma once

#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at { namespace native {

TORCH_CUDA_CPP_API miopenHandle_t getMiopenHandle();

}} // namespace
