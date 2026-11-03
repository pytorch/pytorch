#pragma once

#include <ATen/core/Tensor.h>

namespace at::native {

TORCH_XPU_API Tensor& zero_xpu_(Tensor& self);

} // namespace at::native