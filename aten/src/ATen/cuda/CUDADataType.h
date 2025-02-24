#pragma once

#include <c10/core/ScalarType.h>

#include <cuda.h>
#include <library_types.h>

namespace at::cuda {

template <typename scalar_t>
cudaDataType getCudaDataType() {
  static_assert(false && sizeof(scalar_t), "Cannot convert type to cudaDataType.");
  return {};
}

template<> inline cudaDataType getCudaDataType<at::Half>() {
  return CUDA_R_16F;
}
template<> inline cudaDataType getCudaDataType<float>() {
  return CUDA_R_32F;
}
template<> inline cudaDataType getCudaDataType<double>() {
  return CUDA_R_64F;
}
template<> inline cudaDataType getCudaDataType<c10::complex<c10::Half>>() {
  return CUDA_C_16F;
}
template<> inline cudaDataType getCudaDataType<c10::complex<float>>() {
  return CUDA_C_32F;
}
template<> inline cudaDataType getCudaDataType<c10::complex<double>>() {
  return CUDA_C_64F;
}

template<> inline cudaDataType getCudaDataType<uint8_t>() {
  return CUDA_R_8U;
}
template<> inline cudaDataType getCudaDataType<int8_t>() {
  return CUDA_R_8I;
}
template<> inline cudaDataType getCudaDataType<int>() {
  return CUDA_R_32I;
}

template<> inline cudaDataType getCudaDataType<int16_t>() {
  return CUDA_R_16I;
}
template<> inline cudaDataType getCudaDataType<int64_t>() {
  return CUDA_R_64I;
}
template<> inline cudaDataType getCudaDataType<at::BFloat16>() {
  return CUDA_R_16BF;
}

inline cudaDataType ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return CUDA_R_8U;
    case c10::ScalarType::Char:
      return CUDA_R_8I;
    case c10::ScalarType::Int:
      return CUDA_R_32I;
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
    case c10::ScalarType::Short:
      return CUDA_R_16I;
    case c10::ScalarType::Long:
      return CUDA_R_64I;
    case c10::ScalarType::BFloat16:
      return CUDA_R_16BF;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11080) || (defined(USE_ROCM) && ROCM_VERSION >= 60300)
    case c10::ScalarType::Float8_e4m3fn:
      return CUDA_R_8F_E4M3;
    case c10::ScalarType::Float8_e5m2:
      return CUDA_R_8F_E5M2;
#endif
#if defined(USE_ROCM)
    case c10::ScalarType::Float8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
    case c10::ScalarType::Float8_e5m2fnuz:
      return HIP_R_8F_E5M2_FNUZ;
#endif
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to cudaDataType.")
  }
}

} // namespace at::cuda
