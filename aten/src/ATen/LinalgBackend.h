#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace at {

enum class LinalgBackend : int8_t { Default, Cusolver, Magma };

inline std::string LinalgBackendToString(at::LinalgBackend backend) {
  switch (backend) {
    case LinalgBackend::Default:
      return "at::LinalgBackend::Default";
    case LinalgBackend::Cusolver:
      return "at::LinalgBackend::Cusolver";
    case LinalgBackend::Magma:
      return "at::LinalgBackend::Magma";
    default:
      TORCH_CHECK(false, "Unknown linalg backend");
  }
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::LinalgBackend backend) {
  return stream << LinalgBackendToString(backend);
}

} // namespace at
