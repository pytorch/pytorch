#pragma once

#include <c10/core/ScalarType.h>

#include <library_types.h>

namespace at {
namespace cuda {

inline cudaDataType toCudaDataType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
// HIP doesn't define integral types
#ifndef __HIP_PLATFORM_HCC__
    case c10::ScalarType::Byte:
      return CUDA_R_8U;
    case c10::ScalarType::Char:
      return CUDA_R_8I;
    case c10::ScalarType::Int:
      return CUDA_R_32I;
#endif
    case c10::ScalarType::Half:
      return CUDA_R_16F;
    case c10::ScalarType::Float:
      return CUDA_R_32F;
    case c10::ScalarType::Double:
      return CUDA_R_64F;
    case c10::ScalarType::ComplexHalf:
      return CUDA_C_16F;
    case c10::ScalarType::ComplexFloat:
      return CUDA_C_32F;
    case c10::ScalarType::ComplexDouble:
      return CUDA_C_64F;
#if !defined(__HIP_PLATFORM_HCC__) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    case c10::ScalarType::Short:
      return CUDA_R_16I;
    case c10::ScalarType::Long:
      return CUDA_R_64I;
    case c10::ScalarType::BFloat16:
      return CUDA_R_16BF;
#endif
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to cudaDataType.")
  }
}

} // namespace cuda
} // namespace at
