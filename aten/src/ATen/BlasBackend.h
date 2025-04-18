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

} // namespace at
