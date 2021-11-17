#pragma once

#include <c10/util/Exception.h>

#include <ostream>
#include <string>

namespace at {

enum class LinalgBackend : int8_t { Default, Cusolver, Magma };

// WARNING: These exact strings, e.g. "torch.linalg_default", are also used in
// python bindings. Modifying output strings is **very** likely to cause
// BC-breaking in python side.
inline std::string LinalgBackendToString(at::LinalgBackend backend) {
  switch (backend) {
    case LinalgBackend::Default:
      return "linalg_default";
    case LinalgBackend::Cusolver:
      return "linalg_cusolver";
    case LinalgBackend::Magma:
      return "linalg_magma";
    default:
      TORCH_CHECK(false, "Unknown linalg backend");
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
