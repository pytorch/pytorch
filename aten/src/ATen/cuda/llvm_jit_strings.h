#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace at {
namespace cuda {

TORCH_CUDA_CPP_API const std::string& get_traits_string();
TORCH_CUDA_CPP_API const std::string& get_cmath_string();
TORCH_CUDA_CPP_API const std::string& get_complex_body_string();
TORCH_CUDA_CPP_API const std::string& get_complex_half_body_string();
TORCH_CUDA_CPP_API const std::string& get_complex_math_string();

} // namespace cuda
} // namespace at
