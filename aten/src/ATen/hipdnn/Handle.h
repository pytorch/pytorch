#pragma once

#include <c10/macros/Export.h>
#include <hipdnn_frontend.hpp>

namespace at::native {

TORCH_CUDA_CPP_API hipdnnHandle_t getHipdnnHandle();
} // namespace at::native
