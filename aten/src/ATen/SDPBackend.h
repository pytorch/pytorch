#pragma once
#include <cstdint>

namespace at {

constexpr int32_t num_sdp_backends = 5;
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2,
  cudnn_attention = 3,
  overrideable = 4
};

} // namespace at
