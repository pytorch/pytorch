#pragma once

#include <c10/core/ScalarType.h>

#include <cuda.h>
#include <library_types.h>

namespace at {
namespace cuda {

template <typename scalar_t>
cudaDataType getCudaDataType() {
  TORCH_INTERNAL_ASSERT(false, "Cannot convert type ", typeid(scalar_t).name(), " to cudaDataType.")
}

template<> cudaDataType getCudaDataType<at::Half>() {
  return CUDA_R_16F;
}
template<> cudaDataType getCudaDataType<float>() {
  return CUDA_R_32F;
}
template<> cudaDataType getCudaDataType<double>() {
  return CUDA_R_64F;
}
template<> cudaDataType getCudaDataType<c10::complex<c10::Half>>() {
  return CUDA_C_16F;
}
template<> cudaDataType getCudaDataType<c10::complex<float>>() {
  return CUDA_C_32F;
}
template<> cudaDataType getCudaDataType<c10::complex<double>>() {
  return CUDA_C_64F;
}

// HIP doesn't define integral types
#ifndef __HIP_PLATFORM_HCC__
template<> cudaDataType getCudaDataType<uint8_t>() {
  return CUDA_R_8U;
}
template<> cudaDataType getCudaDataType<int8_t>() {
  return CUDA_R_8I;
}
template<> cudaDataType getCudaDataType<int>() {
  return CUDA_R_32I;
}
#endif

#if !defined(__HIP_PLATFORM_HCC__) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000
template<> cudaDataType getCudaDataType<int16_t>() {
  return CUDA_R_16I;
}
template<> cudaDataType getCudaDataType<int64_t>() {
  return CUDA_R_64I;
}
template<> cudaDataType getCudaDataType<at::BFloat16>() {
  return CUDA_R_16BF;
}
#endif

} // namespace cuda
} // namespace at
