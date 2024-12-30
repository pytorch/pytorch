#pragma once

#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

namespace at::native {
// These constants control the approximation behavior of gelu function.
enum class GeluType {
  None,             // Baseline Gelu
  Tanh,             // Tanh Gelu Approximation
  END
};

inline GeluType get_gelutype_enum(const std::string_view approximate) {
  if (approximate == "none") {
    return GeluType::None;
  } else if (approximate == "tanh") {
    return GeluType::Tanh;
  } else {
    TORCH_CHECK(false, "approximate argument must be either none or tanh.");
  }
}

inline std::string gelutype_to_string(const GeluType type) {
  switch(type) {
    case GeluType::None: return "none";
    case GeluType::Tanh: return "tanh";
    default: TORCH_CHECK(false, "unknown GELU type: ", static_cast<int>(type));
  }
}


} // namespace at::native
