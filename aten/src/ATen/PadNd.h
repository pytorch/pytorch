#pragma once

namespace at {

enum class padding_mode {
  reflect,
  replicate,
  circular,
  constant,
};

static inline c10::string_view padding_mode_string(padding_mode m) {
  switch (m) {
    case padding_mode::reflect:
      return "reflect";
    case padding_mode::replicate:
      return "replicate";
    case padding_mode::circular:
      return "circular";
    case padding_mode::constant:
      return "constant";
  }
  TORCH_CHECK(false, "Invalid padding mode (", static_cast<int64_t>(m), ")");
}

} // namespace at
