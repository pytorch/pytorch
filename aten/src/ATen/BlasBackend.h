#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace at {

enum class BlasBackend : int8_t { Default, Cublas, Cublaslt, Ck };

inline std::string BlasBackendToString(at::BlasBackend backend) {
  switch (backend) {
    case BlasBackend::Default:
      return "at::BlasBackend::Default";
    case BlasBackend::Cublas:
      return "at::BlasBackend::Cublas";
    case BlasBackend::Cublaslt:
      return "at::BlasBackend::Cublaslt";
    case BlasBackend::Ck:
      return "at::BlasBackend::Ck";
    default:
      TORCH_CHECK(false, "Unknown blas backend");
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::BlasBackend backend) {
  return stream << BlasBackendToString(backend);
}

namespace blas {

enum class ScalingType : std::uint8_t {
  TensorWise, // fp32 scales
  RowWise, // fp32 scales
  BlockWise1x16, // fp8_e4m3fn scales
  BlockWise1x32, // fp8_e8m0fnu scales
  BlockWise1x128, // fp32 scales
  BlockWise128x128, // fp32 scales
};

enum class SwizzleType : std::uint8_t { NO_SWIZZLE = 0, SWIZZLE_32_4_4 = 1 };

} // namespace blas

} // namespace at
