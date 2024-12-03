#pragma once

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

namespace at::native {

void bgemm_kernel_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_intrawave_v3(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16));

}; // namespace at::native
