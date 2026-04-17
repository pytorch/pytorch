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

inline std::string ScalingTypeToString(ScalingType scaling_type) {
  switch (scaling_type) {
    case ScalingType::TensorWise:
      return "TensorWise";
    case ScalingType::RowWise:
      return "RowWise";
    case ScalingType::BlockWise1x16:
      return "BlockWise1x16";
    case ScalingType::BlockWise1x32:
      return "BlockWise1x32";
    case ScalingType::BlockWise1x128:
      return "BlockWise1x128";
    case ScalingType::BlockWise128x128:
      return "BlockWise128x128";
    default:
      TORCH_CHECK(false, "Unknown scaling type");
  }
}

inline std::ostream& operator<<(
    std::ostream& stream,
    ScalingType scaling_type) {
  return stream << ScalingTypeToString(scaling_type);
}

enum class SwizzleType : std::uint8_t { NO_SWIZZLE = 0, SWIZZLE_32_4_4 = 1 };

} // namespace blas

} // namespace at
