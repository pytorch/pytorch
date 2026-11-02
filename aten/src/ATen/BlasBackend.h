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
  BlockWise1x128, // fp32 or fp8_e8m0fnu scales
  BlockWise128x128, // fp32 scales
};

// Memory layout of a blockwise scale tensor. NO_SWIZZLE is the backend's
// default layout for the recipe.
enum class SwizzleType : std::uint8_t {
  NO_SWIZZLE = 0,
  SWIZZLE_32_4_4 = 1, // cuBLAS 128x4-tiled block scales
  SWIZZLE_32_8 = 2, // hipBLASLt BLK32_UE8M0_32_8
  SWIZZLE_MNxK4 = 3, // cuBLAS MN-major scales with four values packed along K
};

inline const char* ScalingTypeToString(ScalingType value) {
  switch (value) {
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
      return "UnknownScalingType";
  }
}

inline const char* SwizzleTypeToString(SwizzleType value) {
  switch (value) {
    case SwizzleType::NO_SWIZZLE:
      return "NO_SWIZZLE";
    case SwizzleType::SWIZZLE_32_4_4:
      return "SWIZZLE_32_4_4";
    case SwizzleType::SWIZZLE_32_8:
      return "SWIZZLE_32_8";
    case SwizzleType::SWIZZLE_MNxK4:
      return "SWIZZLE_MNxK4";
    default:
      return "UnknownSwizzleType";
  }
}

} // namespace blas

} // namespace at
