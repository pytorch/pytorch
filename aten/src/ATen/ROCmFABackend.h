#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace at {

enum class ROCmFABackend : int8_t { Default, AOTriton, Ck };

inline std::string ROCmFABackendToString(at::ROCmFABackend backend) {
  switch (backend) {
    case ROCmFABackend::Default:
      return "at::ROCmFABackend::Default";
    case ROCmFABackend::AOTriton:
      return "at::ROCmFABackend::AOTriton";
    case ROCmFABackend::Ck:
      return "at::ROCmFABackend::Ck";
    default:
      TORCH_CHECK(false, "Unknown ROCm flash attention backend")
  }
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ROCmFABackend backend) {
  return stream << ROCmFABackendToString(backend);
}

} // namespace at
