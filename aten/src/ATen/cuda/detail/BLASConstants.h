#pragma once

#include <ATen/core/TensorBase.h>

namespace at::cuda::detail {

float *get_cublas_device_one();
float *get_cublas_device_zero();

} // namespace at::cuda::detail
