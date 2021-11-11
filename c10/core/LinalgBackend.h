#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace c10 {

enum class LinalgBackend : int8_t { Default, Cusolver, Magma };

inline std::string LinalgBackendToString(at::LinalgBackend backend) {
  switch (backend) {
    case LinalgBackend::Default:
      return "linalg_default";
    case LinalgBackend::Cusolver:
      return "linalg_cusolver";
    case LinalgBackend::Magma:
      return "linalg_magma";
    default:
      TORCH_CHECK(false, "Unknown memory format");
  }
}

inline std::string LinalgBackendToRepr(at::LinalgBackend backend) {
  return std::string("torch.") + at::LinalgBackendToString(backend);
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::LinalgBackend backend) {
  return stream << LinalgBackendToString(backend);
}

} // namespace c10
