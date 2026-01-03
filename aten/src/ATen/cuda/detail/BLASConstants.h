#pragma once

#include <ATen/core/TensorBase.h>

namespace at::cuda::detail {

float *get_cublas_device_one();
float *get_cublas_device_zero();
float *get_user_alpha_ptr();

} // namespace at::cuda::detail
